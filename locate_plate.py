# ---------------------------------------------------------------------
# locate_plate.py  (commented-only version)
# Full pipeline: parallel-line focus → ROI merge → plate candidates →
# validation/scoring → numbered overlay → crops.
# ---------------------------------------------------------------------

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#   Detect parallel line pairs on mask (focus regions)
# ------------------------------------------------------------------------------------------
def detect_parallel_line_pairs_on_mask(
    white_mask,
    hough_rho=1,
    hough_theta=np.pi/180,
    hough_thresh=50,
    min_len=40,
    max_gap=6,
    angle_tol_deg=5.0,
    min_dist_px=5,
    roi="full"
):
    """
    Detect nearly-parallel line pairs over a (white) mask using Canny + Hough.

    Args:
        white_mask (ndarray): Single-channel grayscale/binary mask.
        hough_rho (float): Hough distance resolution (pixels).
        hough_theta (float): Hough angle resolution (radians).
        hough_thresh (int): Hough votes threshold.
        min_len (int): Min segment length to keep.
        max_gap (int): Max break allowed within a segment.
        angle_tol_deg (float): Angle difference for "parallel" (degrees).
        min_dist_px (int): Minimum perpendicular distance between lines.
        roi (str): "full" or "bottom_half" region to search.

    Returns:
        pairs (list[tuple]): List of ((x1,y1,x2,y2),(x1,y1,x2,y2)) parallel pairs.
        vis (ndarray): BGR visualization with lines drawn.
    """
    # Ensure single-channel input
    assert white_mask.ndim == 2, "white_mask must be single-channel (grayscale/binary)."

    # Optionally restrict to bottom half to reduce false positives
    ih, iw = white_mask.shape[:2]
    if roi == "bottom_half":
        y0 = ih // 2
        mask_roi = white_mask[y0:, :]
    else:
        y0 = 0
        mask_roi = white_mask

    # Light clean-up before edges
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Edge + Probabilistic Hough
    edges = cv2.Canny(mask_clean, 50, 150)
    linesP = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_thresh,
                             minLineLength=min_len, maxLineGap=max_gap)
    if linesP is None or len(linesP) == 0:
        return [], cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

    # Shift lines back to full-image coordinates if ROI used
    segs = []
    for l in linesP:
        x1, y1, x2, y2 = map(int, l[0])
        segs.append((x1, y1 + y0, x2, y2 + y0))

    # --- Helpers for geometry ---
    def seg_angle_deg(s):
        x1, y1, x2, y2 = s
        return abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0

    def seg_len(s):
        x1, y1, x2, y2 = s
        return math.hypot(x2 - x1, y2 - y1)

    def line_params(s):
        x1, y1, x2, y2 = s
        A = y1 - y2
        B = x2 - x1
        C = x1*y2 - x2*y1
        n = math.hypot(A, B)
        if n == 0: 
            return 0, 0, 0
        return A/n, B/n, C/n

    def perp_dist(s1, s2):
        # Perpendicular distance from mid-point of s2 to line of s1
        A, B, C = line_params(s1)
        xm = 0.5 * (s2[0] + s2[2])
        ym = 0.5 * (s2[1] + s2[3])
        return abs(A*xm + B*ym + C)

    # Keep only sufficiently long lines
    segs = [s for s in segs if seg_len(s) >= min_len]
    if not segs:
        return [], cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

    # Sort by angle so we can group near-parallel lines efficiently
    angs = np.array([seg_angle_deg(s) for s in segs], dtype=np.float32)
    order = np.argsort(angs)
    segs_sorted = [segs[i] for i in order]
    ang_sorted = angs[order]

    # Pair lines whose angles match and are not too close together
    pairs = []
    n = len(segs_sorted)
    for i in range(n):
        for j in range(i+1, n):
            if abs(ang_sorted[j] - ang_sorted[i]) > angle_tol_deg:
                break
            s1, s2 = segs_sorted[i], segs_sorted[j]
            if perp_dist(s1, s2) < min_dist_px:
                continue
            pairs.append((s1, s2))

    # Build visualization
    vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in segs:
        cv2.line(vis, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
    for s1, s2 in pairs:
        cv2.line(vis, (s1[0], s1[1]), (s1[2], s1[3]), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(vis, (s2[0], s2[1]), (s2[2], s2[3]), (0, 255, 0), 2, cv2.LINE_AA)

    return pairs, vis


# ------------------------------------------------------------------------------------------
#   Convert line pairs => rectangular ROIs (expanded paddings)
# ------------------------------------------------------------------------------------------
def rois_from_parallel_pairs(pairs, img_shape, pad_along=12, pad_across=24):
    """
    Convert matched line-pairs into rectangular ROIs, then merge overlaps.

    Args:
        pairs (list): Output from detect_parallel_line_pairs_on_mask.
        img_shape (tuple): (H, W, ...) shape of the original image/mask.
        pad_along (int): Padding along the line direction.
        pad_across (int): Padding perpendicular to line direction.

    Returns:
        rois (list[tuple]): List of merged rectangles (x0,y0,x1,y1).
    """
    # Build raw rectangles around each pair with padding
    ih, iw = img_shape[:2]
    rois = []
    for (a, b) in pairs:
        x1a, y1a, x2a, y2a = a
        x1b, y1b, x2b, y2b = b
        xs = [x1a, x2a, x1b, x2b]
        ys = [y1a, y2a, y1b, y2b]
        x0 = max(0, min(xs) - pad_across)
        y0 = max(0, min(ys) - pad_along)
        x1 = min(iw, max(xs) + pad_across)
        y1 = min(ih, max(ys) + pad_along)
        if x1 > x0 and y1 > y0:
            rois.append((int(x0), int(y0), int(x1), int(y1)))

    # Merge overlapping rectangles to reduce redundancy
    rois = merge_rois(rois)
    return rois


def merge_rois(rois, iou_thresh=0.2):
    """
    Merge overlapping rectangular ROIs using IoU thresholding.

    Args:
        rois (list[tuple]): Rectangles as (x0,y0,x1,y1).
        iou_thresh (float): Intersection-over-Union threshold to merge.

    Returns:
        merged (list[tuple]): Merged rectangles (x0,y0,x1,y1).
    """
    if not rois: 
        return rois

    # Sort for deterministic merging
    rois = sorted(rois, key=lambda r: (r[0], r[1]))
    merged = []
    for r in rois:
        x0, y0, x1, y1 = r
        placed = False
        for i,(X0,Y0,X1,Y1) in enumerate(merged):
            # Compute IoU with an existing merged rectangle
            inter = (max(0, min(x1,X1)-max(x0,X0))) * (max(0, min(y1,Y1)-max(y0,Y0)))
            a1 = (x1-x0)*(y1-y0)
            a2 = (X1-X0)*(Y1-Y0)
            iou = inter / float(a1 + a2 - inter + 1e-6)
            if iou > iou_thresh:
                # Replace with union box
                nx0, ny0 = min(x0,X0), min(y0,Y0)
                nx1, ny1 = max(x1,X1), max(y1,Y1)
                merged[i] = (nx0,ny0,nx1,ny1)
                placed = True
                break
        if not placed:
            merged.append(r)
    return merged


# ------------------------------------------------------------------------------------------
#   Plate candidate detection restricted to ROIs derived from parallel lines
# ------------------------------------------------------------------------------------------
def detect_plate_from_white_mask_in_rois(
    white_mask, rois,
    min_area=1200, aspect_lo=2.0, aspect_hi=6.5,
    bottom_ratio=0.5, middle_band_frac=0.40
):
    """
    Within each ROI, find rectangular plate-like blobs via morphology + contours.

    Args:
        white_mask (ndarray): Binary/gray mask (white highlights).
        rois (list[tuple]): Regions (x0,y0,x1,y1) in which to search.
        min_area (int): Minimum rectangle area for candidates.
        aspect_lo (float): Minimum aspect ratio w/h.
        aspect_hi (float): Maximum aspect ratio w/h.
        bottom_ratio (float): Emphasize bottom region of image (y >= H*ratio).
        middle_band_frac (float): Emphasize a central horizontal band.

    Returns:
        boxes (list[tuple]): Candidate rectangles as (x,y,w,h).
        all_cands (list[tuple]): Ranked tuples with scores and tier.
    """
    # Unify to grayscale if needed
    if white_mask.ndim == 3:
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_BGR2GRAY)

    ih, iw = white_mask.shape[:2]
    global_min_area = max(min_area, int(0.003 * iw * ih))

    # Precompute positional priorities (middle band + bottom region)
    mid_x = iw * 0.5
    half_band = (iw * middle_band_frac) * 0.5
    mid_left, mid_right = mid_x - half_band, mid_x + half_band
    bottom_y_threshold = ih * bottom_ratio

    def process_patch(patch, xoff, yoff):
        """Process one ROI patch; return scored local candidates."""
        # Light morphology to simplify shapes
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed  = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, k_close, iterations=1)
        mask_clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)

        # Extract external contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes_local = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            if rect_area < global_min_area:
                continue

            # Aspect ratio screening
            aspect = w / float(h) if h else 0.0
            if not (aspect_lo <= aspect <= aspect_hi):
                continue

            # Reject very large blobs versus whole image (likely not plates)
            if w > 0.10 * iw and h > 0.05 * ih:
                continue

            # Fill/solidity scoring
            area = cv2.contourArea(cnt)
            if rect_area == 0:
                continue
            fill = area / float(rect_area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area + 1e-6)
            score = 0.6 * fill + 0.4 * solidity

            # Tier by position: bottom & mid-band are preferred
            cx = xoff + x + 0.5 * w
            cy = yoff + y + 0.5 * h
            in_bottom = (cy >= bottom_y_threshold)
            in_middle = (mid_left <= cx <= mid_right)
            if in_bottom and in_middle:
                tier = 0
            elif in_bottom or in_middle:
                tier = 1
            else:
                tier = 2

            boxes_local.append((tier, -score, xoff + x, yoff + y, w, h, score, fill, solidity, aspect))
        return boxes_local

    # Process all ROIs; if none provided, fall back to whole image
    all_cands = []
    if not rois:
        rois = [(0,0,iw,ih)]
    for (x0,y0,x1,y1) in rois:
        patch = white_mask[y0:y1, x0:x1]
        all_cands.extend(process_patch(patch, x0, y0))

    # Rank by (tier asc, score desc)
    all_cands.sort(key=lambda t: (t[0], t[1]))

    # Strip metadata to (x,y,w,h) list for downstream steps
    boxes = [(x, y, w, h) for (_tier, _ns, x, y, w, h, *_rest) in all_cands]
    return boxes, all_cands


# ------------------------------------------------------------------------------------------
#   Validation & scoring for each candidate ROI
# ------------------------------------------------------------------------------------------
def validate_plate(roi_bgr):
    """
    Validate if an ROI likely contains a plate via multiple cues:
    stroke density, character-like components, contrast, and edge bands.

    Args:
        roi_bgr (ndarray): Candidate crop in BGR.

    Returns:
        score (float): Confidence [0..1].
        verdict (bool): Pass/fail based on core cues.
        info (dict): Diagnostic metrics (counts, statistics).
    """
    info = {}
    if roi_bgr is None or min(roi_bgr.shape[:2]) < 12:
        return 0.0, False, {"reason": "roi_too_small"}

    # Normalize height for stable metrics
    H_TARGET = 96
    h0, w0 = roi_bgr.shape[:2]
    scale = H_TARGET / float(max(1, h0))
    roi_bgr = cv2.resize(roi_bgr, (max(8, int(w0 * scale)), H_TARGET), interpolation=cv2.INTER_LINEAR)
    h, w = roi_bgr.shape[:2]

    # Edges + stroke density
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    stroke_density = cv2.countNonZero(edges) / float(max(1, w * h))
    info["stroke_density"] = stroke_density
    ok_strokes = stroke_density >= 0.014

    # Character-like blobs via adaptive threshold + CC analysis
    bin_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=8
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_clean, connectivity=8)
    char_boxes = []
    for i in range(1, num_labels):
        x, y, wcc, hcc, area = stats[i]
        if area < 12:
            continue
        ar = wcc / float(max(1, hcc))
        if 0.15 <= ar <= 1.2 and 8 <= hcc <= int(0.9 * h):
            char_boxes.append((x, y, wcc, hcc, area))
    char_boxes.sort(key=lambda b: b[0])

    # Merge horizontally adjacent components (split-stroke fixes)
    merged = []
    for cb in char_boxes:
        if not merged:
            merged.append(list(cb))
        else:
            x, y, wcc, hcc, area = cb
            px, py, pw, ph, pa = merged[-1]
            if x <= px + pw + 3 and abs((y + hcc/2) - (py + ph/2)) <= h * 0.25:
                nx  = min(px, x)
                ny  = min(py, y)
                nx2 = max(px + pw, x + wcc)
                ny2 = max(py + ph, y + hcc)
                merged[-1] = [nx, ny, nx2 - nx, ny2 - ny, pa + area]
            else:
                merged.append([x, y, wcc, hcc, area])

    # Character-count heuristic
    char_count = len(merged)
    info["char_count"] = char_count
    ok_chars = 4 <= char_count <= 10

    # Global contrast heuristic
    contrast = float(np.std(gray)) / 255.0
    info["contrast"] = contrast
    ok_contrast = contrast >= 0.075

    # Horizontal reflective-band heuristic (top/bottom vs mid)
    row_proj = edges.sum(axis=1) / 255.0
    top_band = np.mean(row_proj[:max(3, h // 8)])
    mid_band = np.mean(row_proj[h // 3: 2 * h // 3])
    bot_band = np.mean(row_proj[-max(3, h // 8):])
    horiz_candidates = (top_band > mid_band * 1.15) + (bot_band > mid_band * 1.15)
    ok_horiz = horiz_candidates >= 1
    info["horiz_bands"] = {"top": float(top_band), "mid": float(mid_band), "bot": float(bot_band)}

    # Core cues gating (need at least 2 of 3)
    core_passes = sum([ok_strokes, ok_chars, ok_contrast])
    verdict = core_passes >= 2

    # Weighted confidence score
    score = (
        0.45 * np.clip((stroke_density - 0.010) / 0.060, 0, 1) +
        0.35 * np.clip((char_count - 3) / 7.0, 0, 1) +
        0.20 * np.clip((contrast - 0.060) / 0.12, 0, 1)
    )
    if ok_horiz:
        score = min(1.0, score + 0.08)

    info["verdict_reasons"] = {
        "ok_strokes": ok_strokes,
        "ok_chars": ok_chars,
        "ok_contrast": ok_contrast,
        "ok_horiz": ok_horiz,
        "core_passes": int(core_passes)
    }
    return float(score), bool(verdict), info


# ------------------------------------------------------------------------------------------
#   Utilities: verify, NMS, crops aligned with results, numbered overlay
# ------------------------------------------------------------------------------------------
def verify_all_candidates(img_bgr, boxes, min_conf):
    """
    Score and filter each (x,y,w,h) candidate using validate_plate(); keep those ≥ min_conf.

    Args:
        img_bgr (ndarray): Original image in BGR.
        boxes (list[tuple]): Candidate rectangles (x,y,w,h).
        min_conf (float): Minimum score to keep.

    Returns:
        results (list): [((x,y,w,h), score, verdict, info), ...] sorted by pass then score.
    """
    results = []
    for (x, y, w, h) in boxes:
        roi = img_bgr[y:y+h, x:x+w]
        score, ok, info = validate_plate(roi)
        if score < min_conf:
            continue
        results.append(((x, y, w, h), score, ok, info))

    # Sort: passes first, then by score descending
    results.sort(key=lambda r: (not r[2], -r[1]))
    return results


def suppress_overlap_keep_larger(boxes, iou_thresh=0.30):
    """
    Simple NMS: keep larger rectangles and discard overlapping smaller ones.

    Args:
        boxes (list[tuple]): Rectangles (x,y,w,h).
        iou_thresh (float): IoU threshold for suppression.

    Returns:
        kept (list[tuple]): Non-overlapping rectangles.
    """
    if not boxes:
        return boxes

    # Sort by area desc so large boxes survive first
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)

    def iou(b1, b2):
        x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
        xa = max(x1, x2); ya = max(y1, y2)
        xb = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        a1 = w1*h1; a2 = w2*h2
        return inter / (a1 + a2 - inter + 1e-6)

    kept = []
    for b in boxes:
        if all(iou(b, k) <= iou_thresh for k in kept):
            kept.append(b)
    return kept


def draw_numbered_boxes(img_bgr, results, color=(0,255,0)):
    """
    Draw numbered rectangles for results list [((x,y,w,h), score, ok, info), ...].

    Args:
        img_bgr (ndarray): Original image.
        results (list): Output from verify_all_candidates.
        color (tuple): BGR color for rectangles/text.

    Returns:
        vis (ndarray): Annotated BGR image.
    """
    vis = img_bgr.copy()
    for i, (b, score, ok, _info) in enumerate(results, 1):
        x,y,w,h = b
        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
        cv2.putText(vis, f"#{i}", (x, max(0, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis


def crops_from_results(
    img_bgr,
    results,
    pad_left=0,
    pad_right=0,
    pad_top=0,
    pad_bottom=0
):
    """
    Crop each detection with independent padding for each side.

    Args:
        img_bgr (ndarray): original image (BGR)
        results (list): list of ((x, y, w, h), score, ok, info)
        pad_left/right/top/bottom (int): pixels to expand on each side

    Returns:
        crops (dict): {"candidate1": crop_image, ...}
        boxes_named (dict): {"candidate1": (x, y, w, h)}
    """
    # Image bounds for clamping pads
    ih, iw = img_bgr.shape[:2]
    crops, boxes_named = {}, {}

    for i, (b, _score, _ok, _info) in enumerate(results, 1):
        x, y, w, h = b

        # Manual per-side padding (clamped to image bounds)
        x0 = max(0, x - pad_left)
        y0 = max(0, y - pad_top)
        x1 = min(iw, x + w + pad_right)
        y1 = min(ih, y + h + pad_bottom)

        if x1 <= x0 or y1 <= y0:
            continue

        # Named output for deterministic ordering
        name = f"candidate{i}"
        crops[name] = img_bgr[y0:y1, x0:x1]
        boxes_named[name] = (x, y, w, h)

    return crops, boxes_named


# ------------------------------------------------------------------------------------------
def crop_plate(white_mask, img_bgr):
    """
    High-level: lines → ROIs → candidates → validate → numbered overlay → crops.

    Args:
        white_mask (ndarray): Binary mask guiding search.
        img_bgr (ndarray): Original BGR image.

    Returns:
        plate_candidates (list[ndarray]): Cropped candidate regions.
    """
    # 1) Find parallel line pairs over mask to focus the search
    pairs, vis_lines = detect_parallel_line_pairs_on_mask(
        white_mask, hough_thresh=60, min_len=40, max_gap=3,
        angle_tol_deg=6.0, min_dist_px=10, roi="bottom_half"
    )

    # 2) Convert those pairs into (merged) rectangular ROIs
    rois = rois_from_parallel_pairs(pairs, white_mask.shape, pad_along=14, pad_across=28)

    # 3) Detect plate-like blobs within these ROIs
    boxes, cand_tuples = detect_plate_from_white_mask_in_rois(
        white_mask, rois,
        min_area=1200, aspect_lo=2.0, aspect_hi=6.5,
        bottom_ratio=0.5, middle_band_frac=0.40
    )

    # 4) Suppress overlaps, keeping the larger rectangles
    boxes = suppress_overlap_keep_larger(boxes, iou_thresh=0.30)

    # (Optional) visualize boxes on the mask
    vis_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis_mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 5) Validate all candidates with a strict threshold first, then relax
    all_results = verify_all_candidates(img_bgr, boxes, min_conf=1.0)
    if not all_results:
        for thr in (0.95, 0.90, 0.85, 0.80, 0.75):
            all_results = verify_all_candidates(img_bgr, boxes, min_conf=thr)
            if all_results:
                print(f"[auto-relaxed] min_conf lowered to {thr}")
                break

    # Defensive: image and mask must match sizes
    assert img_bgr.shape[:2] == white_mask.shape[:2], "Size mismatch (img vs mask)!"

    # 6) Numbered overlay (kept for potential debugging/plots)
    vis_numbered = draw_numbered_boxes(img_bgr, all_results)

    # 7) Produce per-candidate crops (note: cropping from white_mask as in original)
    crops, boxes_named = crops_from_results(
        white_mask,       # (kept exactly as in your original code)
        all_results,
        pad_left=1,
        pad_right=1,
        pad_top=1,
        pad_bottom=1
    )

    # Collect crops into a simple list in candidate order
    plate_candidates = []
    for name, crop in crops.items():
        plate_candidates.append(crop)

    return plate_candidates
