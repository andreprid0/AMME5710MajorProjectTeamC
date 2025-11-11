import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat 

# ---------------- Dataset configuration ----------------
# Parent directory containing one subfolder per label (A–Z except 'I', and 0–9)
FOLDER_ROOT_PLACES = "SyntheticDataset/Developed Training Dataset"

# All images are normalized/resized to (H, W)
H, W = 64, 64


# ---------------- Dataset helpers ----------------
def list_place_classes(root: str = FOLDER_ROOT_PLACES):
    labels = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    # Keep alphanumerics only and skip ambiguous 'I'
    keep = []
    for lab in labels:
        L = lab.upper()
        if len(L) == 1 and L in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" and L != "I":
            keep.append(lab)
    return sorted(keep)


def load_dataset(
    root: str = FOLDER_ROOT_PLACES,
    H_: int = H,
    W_: int = W,
    use_cache: bool = True,
    cache_dir: str = "cache",
):
    """
    Load number-plate character dataset from folder-per-label structure.
    Returns: image_pixel_data (ndarray uint8 BGR), label_data (ndarray str), place_classes (list[str])
    """
    place_classes = list_place_classes(root)

    image_pixel_data = []
    label_data = []
    loaded_count = 0

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Pre-count total files for a simple progress display
    total_files = 0
    for label in place_classes:
        class_dir = os.path.join(root, label)
        total_files += sum(
            1
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
            and os.path.splitext(f)[1].lower() in valid_exts
        )

    step = max(1, total_files // 20)  # ~5% steps

    for label in place_classes:
        class_dir = os.path.join(root, label)
        filenames = [
            f for f in sorted(os.listdir(class_dir))
            if os.path.isfile(os.path.join(class_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
        ]
        for fname in filenames:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # Normalize channels to BGR 3-channel for downstream HOG pipeline
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # Drop alpha if present
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img = cv2.resize(img, (W_, H_))
            image_pixel_data.append(img)
            label_data.append(label)
            loaded_count += 1
            if (loaded_count % step == 0) or (loaded_count == total_files):
                pct = int(round(loaded_count * 100.0 / max(1, total_files)))
                print(
                    f"Loading dataset: {loaded_count}/{total_files} ({pct}%)",
                    end="\r",
                    flush=True,
                )

    print("")
    print(loaded_count, "- number of total files")

    image_pixel_data = np.array(image_pixel_data)
    label_data = np.array(label_data)

    # No cache write-back: always return freshly loaded arrays
    return image_pixel_data, label_data, place_classes


def load_emnist_dataset(
    split: str = "train",
    H_: int = H,
    W_: int = W,
    max_per_class: int | None = None,
):
    """
    Load EMNIST ByClass split from the preloaded EMNIST .mat file.

    Returns: image_pixel_data (ndarray uint8 BGR), label_data (ndarray str), classes (list[str])

    Notes:
    - Uses the 'mapping' table inside the .mat to convert label indices to Unicode characters.
    - Uppercases letters and filters to digits 0-9 and letters A-Z; excludes ambiguous 'I' to
      match the folder dataset's class policy.
    - Resizes 28x28 grayscale to H_ x W_ and converts to 3-channel BGR for consistency.
    - Applies the standard EMNIST orientation fix (transpose + horizontal flip).
    """
    ds = EMNIST.get("dataset")
    if ds is None:
        raise ValueError("EMNIST .mat missing 'dataset' key")

    # Access train/test structs
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    split_struct = ds[split][0, 0]
    images = split_struct["images"][0, 0]
    labels = split_struct["labels"][0, 0].squeeze()
    mapping = ds["mapping"][0, 0]

    # Build label->unicode codepoint mapping
    try:
        map_dict = {int(row[0]): int(row[1]) for row in mapping}
    except Exception:
        # Fallback if mapping comes as a nested object array
        mapping_np = np.array(mapping)
        map_dict = {int(r[0]): int(r[1]) for r in mapping_np.reshape((-1, 2))}

    X_list = []
    y_list = []
    per_class_counts: dict[str, int] = {}
    keep_set = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") - {"I"}

    N = images.shape[0]
    step = max(1, N // 20)  # ~5% steps
    for i in range(N):
        img_row = images[i]
        lab = int(labels[i])
        code = map_dict.get(lab)
        if code is None:
            continue
        ch = chr(code).upper()
        if ch not in keep_set:
            continue
        # Enforce per-class cap if requested
        if max_per_class is not None:
            c = per_class_counts.get(ch, 0)
            if c >= max_per_class:
                if (i + 1) % step == 0 or (i + 1) == N:
                    pct = int(round((i + 1) * 100.0 / max(1, N)))
                    print(
                        f"Loading EMNIST({split}): {i+1}/{N} ({pct}%) kept={len(X_list)} (capped)",
                        end="\r",
                        flush=True,
                    )
                continue

        # 28x28 reshape and orientation correction
        g = img_row.reshape(28, 28).T  # transpose
        g = np.flip(g, axis=1)         # horizontal flip to correct EMNIST orientation
        g = g.astype(np.uint8)

        # Convert grayscale to BGR 3-channel and resize
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        bgr = cv2.resize(bgr, (W_, H_))

        X_list.append(bgr)
        y_list.append(ch)
        per_class_counts[ch] = per_class_counts.get(ch, 0) + 1

        if (i + 1) % step == 0 or (i + 1) == N:
            pct = int(round((i + 1) * 100.0 / max(1, N)))
            print(
                f"Loading EMNIST({split}): {i+1}/{N} ({pct}%) kept={len(X_list)}",
                end="\r",
                flush=True,
            )

    print("")

    image_pixel_data = np.array(X_list)
    label_data = np.array(y_list)
    classes = sorted(list(set(label_data.tolist())))
    return image_pixel_data, label_data, classes


def get_splits(image_pixel_data, label_data, n_splits: int = 4, shuffle: bool = True, random_state: int = 0):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(cv.split(image_pixel_data, label_data))
    train_index, test_index = splits[0]
    image_train = image_pixel_data[train_index]
    image_test = image_pixel_data[test_index]
    label_train = label_data[train_index]
    label_test = label_data[test_index]
    return splits, (image_train, image_test, label_train, label_test, train_index, test_index)
