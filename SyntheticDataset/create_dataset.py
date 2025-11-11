import os
import uuid
import cv2
import numpy as np
from typing import Tuple

# Input parent: each subfolder is a label (A–Z, 0–9). One seed image per folder.
SEED_ROOT = r"SeedRootFolder"

# Output root for synthetic variants (a folder per label will be created here)
OUTPUT_ROOT = r"OutputRootFolder"

# Canvas configuration
CANVAS_SIZE: Tuple[int, int] = (64, 64)
MIN_BORDER = 3  # pixels of black margin around the glyph


def _ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _ensure_binary_white_fg(img: np.ndarray) -> np.ndarray:
    """Return binary 0/255 with characters as white (255)."""
    img = _ensure_gray_u8(img)
    u = np.unique(img)
    if u.size <= 3 and set(u.tolist()).issubset({0, 255}):
        # Already binary; make sure foreground is white (assume larger area is bg=0)
        # If most pixels are white, assume white bg -> invert
        if np.count_nonzero(img) > img.size // 2:
            return 255 - img
        return img
    # Threshold and invert to get white text on black
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background came out white, invert
    if np.count_nonzero(bw) > bw.size // 2:
        bw = 255 - bw
    return bw


def _fit_to_canvas(tile: np.ndarray, size: Tuple[int, int] = CANVAS_SIZE, border: int = MIN_BORDER) -> np.ndarray:
    """Center tile on a black canvas, preserving aspect and minimum border."""
    out_h, out_w = int(size[0]), int(size[1])
    border = max(0, int(border))
    tile = _ensure_binary_white_fg(tile)
    h, w = tile.shape
    avail_w = max(1, out_w - 2 * border)
    avail_h = max(1, out_h - 2 * border)
    s = min(avail_w / float(w), avail_h / float(h))
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    resized = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _correlated_dropout(mask_fg: np.ndarray, strength: float = 0.03, sigma: float = 1.8, interior_only: bool = True) -> np.ndarray:
    """Create a boolean mask of pixels to drop (set to 0) inside the glyph.

    - strength: approximate fraction of foreground pixels to drop (0.01–0.08 typical)
    - sigma: Gaussian blur sigma to create spatial correlation in dropout regions
    - interior_only: avoid edges by eroding the glyph once
    """
    h, w = mask_fg.shape
    # Random noise field in [0,1], blur to correlate
    noise = np.random.rand(h, w).astype(np.float32)
    field = cv2.GaussianBlur(noise, (0, 0), sigmaX=float(sigma))
    field = (field - field.min()) / max(1e-6, (field.max() - field.min()))

    # Target proportion of drops within the foreground region
    fg_idx = mask_fg > 0
    if np.count_nonzero(fg_idx) == 0:
        return np.zeros_like(mask_fg, dtype=bool)
    # Choose threshold by percentile so roughly 'strength' fraction of fg might drop
    thr = np.quantile(field[fg_idx], 1.0 - float(strength))
    drop = field >= thr

    # Restrict to glyph
    drop &= fg_idx

    if interior_only:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        core = cv2.erode(mask_fg, ker, iterations=1) > 0
        drop &= core
    return drop


def _correlated_edge_perturb(
    fg: np.ndarray,
    shave_strength: float = 0.06,
    add_strength: float = 0.03,
    sigma: float = 1.4,
    ring_width: int = 1,
) -> np.ndarray:
    """Randomly shave and add pixels along the glyph outline to vary edge shape.

    - shave_strength: approx fraction of inner-edge pixels to remove (0.02–0.10)
    - add_strength: approx fraction of outer-edge ring pixels to add (0.0–0.08)
    - sigma: correlation for the noise fields; larger → chunkier changes
    """
    fg = _ensure_binary_white_fg(fg)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ring_width = max(1, int(ring_width))
    inner = (fg > 0) & (cv2.erode(fg, ker, ring_width) < 255)  # inner edge ring
    outer = (cv2.dilate(fg, ker, ring_width) > 0) & (fg == 0)  # outer edge ring

    h, w = fg.shape
    noise = np.random.rand(h, w).astype(np.float32)
    field = cv2.GaussianBlur(noise, (0, 0), sigmaX=float(sigma))
    field = (field - field.min()) / max(1e-6, (field.max() - field.min()))

    out = fg.copy()
    # Shave inner edge
    if shave_strength > 0 and np.any(inner):
        thr_in = np.quantile(field[inner], 1.0 - float(shave_strength))
        shave = (field >= thr_in) & inner
        out[shave] = 0
    # Add on outer ring
    if add_strength > 0 and np.any(outer):
        thr_out = np.quantile(field[outer], 1.0 - float(add_strength))
        addm = (field >= thr_out) & outer
        out[addm] = 255
    return out


def augment_one(tile: np.ndarray) -> np.ndarray:
    """Generate one augmented sample with varied outline and fine, lighter erosion."""
    base = _fit_to_canvas(tile, CANVAS_SIZE, MIN_BORDER)
    fg = base

    # 1) Correlated interior dropout (lighter)
    strength = np.random.uniform(0.008, 0.035)  # 0.8–3.5% of fg pixels
    sigma = np.random.uniform(1.2, 2.6)
    drop = _correlated_dropout(fg, strength=strength, sigma=sigma, interior_only=True)
    aug = fg.copy()
    aug[drop] = 0

    # 2) Edge perturbation to diversify outline shapes (emphasize shape variety)
    if np.random.rand() < 0.95:
        shave_s = np.random.uniform(0.01, 0.05)   # finer shaving
        add_s = np.random.uniform(0.01, 0.08)     # allow small outward bumps
        sigma_e = np.random.uniform(1.4, 2.8)
        ring_w = np.random.choice([1, 2])         # vary ring width
        aug = _correlated_edge_perturb(
            aug, shave_strength=shave_s, add_strength=add_s, sigma=sigma_e, ring_width=ring_w
        )

    # 3) Ensure binary and refit to protect border
    _, aug = cv2.threshold(aug, 127, 255, cv2.THRESH_BINARY)
    aug = _fit_to_canvas(aug, CANVAS_SIZE, MIN_BORDER)
    return aug


def save_variants_for_single_image(src_image_path: str, label: str, out_root: str = OUTPUT_ROOT, count: int = 100) -> int:
    """Load one seed image and write 'count' augmented variants to out_root/label."""
    img = cv2.imread(src_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {src_image_path}")
    seed = _fit_to_canvas(img, CANVAS_SIZE, MIN_BORDER)
    label = label.upper()
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for _ in range(int(count)):
        aug = augment_one(seed)
        name = f"{uuid.uuid4().hex}.png"
        cv2.imwrite(os.path.join(out_dir, name), aug)
        saved += 1
    return saved


def create_dataset_from_seeds(seed_root: str = SEED_ROOT, out_root: str = OUTPUT_ROOT, per_label: int = 1000) -> None:
    """Iterate label folders (A–Z except 'I', plus 0–9) and produce per_label variants each.

    Expects exactly one seed image per label folder; uses the first supported image found.
    """
    os.makedirs(out_root, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    labels = [d for d in os.listdir(seed_root) if os.path.isdir(os.path.join(seed_root, d))]
    total = 0
    for label in sorted(labels, key=str.upper):
        L = label.upper()
        if len(L) != 1 or (L not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") or L == "I":
            continue
        folder = os.path.join(seed_root, label)
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
        if not files:
            print(f"[skip] No image in {folder}")
            continue
        seed_path = os.path.join(folder, files[0])
        n = save_variants_for_single_image(seed_path, L, out_root=out_root, count=per_label)
        print(f"[{L}] saved {n} variants from {seed_path}")
        total += n
    print(f"Done. Wrote {total} augmented tiles into {out_root}")


if __name__ == "__main__":
    # Update SEED_ROOT and OUTPUT_ROOT above, then run:
    create_dataset_from_seeds(SEED_ROOT, OUTPUT_ROOT, per_label=1000)
