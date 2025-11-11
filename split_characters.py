import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def convert_rgb_to_gray(plate):
    """
    Convert a BGR license plate image to grayscale.

    Args:
        plate (ndarray): Input plate image in BGR color space.

    Returns:
        gray (ndarray): Grayscale version of the input plate.
    """
    # Convert from BGR (OpenCV default) to single-channel grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    return gray


def normalise_lighting(gray): 
    """
    Normalize lighting conditions across a grayscale plate image.

    Suppresses uneven illumination or shadows by dividing 
    the image by a blurred version of itself.
    """
    # Apply Gaussian blur to estimate lighting field
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    # Divide original by blurred to flatten lighting variations
    normalised = cv2.divide(gray, blurred, scale=255)
    return normalised


def gray_morphology(gray, ksize=3, iterations=1):
    """
    Apply grayscale morphological closing to remove small dark specks.
    """
    # Create rectangular kernel for morphological operation
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    # Perform closing: dilation then erosion
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker, iterations=iterations)
    return closed


def adaptive_threshold(gray, block=31, C=2):
    """
    Convert a grayscale image to binary using adaptive Gaussian thresholding.
    """
    # Adaptive threshold considers local mean in each block region
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, C,
    )
    return bw


def open_morphology_plate(mask, erode_iterations=1, dilate_iterations=1):
    """
    Perform morphological opening on an inverted binary plate mask.
    """
    # Invert so characters become white (foreground)
    mask = cv2.bitwise_not(mask)
    # Build small kernels for erosion and dilation
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Erode to remove small bright specks
    eroded = cv2.erode(mask, kernel_erosion, iterations=erode_iterations)
    # Dilate to recover true character strokes
    opened = cv2.dilate(eroded, kernel_open, iterations=dilate_iterations)
    return opened


def find_all_contours(mask, margin):
    """
    Detect all external contours in a binary mask while discarding border-touching ones.
    """
    # Find all external contours from binary mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drop any contour that touches the image border
    h, w = mask.shape[:2]
    kept_contours = []
    for c in contours: 
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
            continue
        kept_contours.append(c)

    # Draw all kept contours for visualization
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    cv2.drawContours(image, kept_contours, -1, (0, 0, 255), 1)
    return image, kept_contours


def filter_contours(mask, contours):
    """
    Filter plate contours to retain the largest and other significant ones.
    """
    # Find largest contour (assumed to be full plate area)
    plate_chars = []
    plate_chars.append(max(contours, key=cv2.contourArea))
    largest_contour_area = cv2.contourArea(plate_chars[0])

    # Keep contours that are at least 1/5 of largest area
    for contour in contours: 
        if contour is plate_chars[0]:
            continue
        contour_area = cv2.contourArea(contour)
        if contour_area > (largest_contour_area / 5):
            plate_chars.append(contour)
    
    # Sort contours left-to-right (based on x position)
    plate_chars_sorted = sorted(plate_chars, key=lambda c: cv2.boundingRect(c)[0])

    # Visualize filtered contours
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    cv2.drawContours(image, plate_chars_sorted, -1, (0, 0, 255), 2)
    return image, plate_chars_sorted


def fit_char_to_box(mask, contours, size=(64, 64), border=6):
    """
    Fit each character contour into a fixed-size box (e.g., 64×64).
    """
    out_h, out_w = int(size[0]), int(size[1])
    tiles = []

    # Loop through each contour and normalize its bounding region
    for c in contours:
        if c is None or len(c) == 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        # Crop region of interest around contour
        roi = mask[y:y + h, x:x + w]

        # Compute scaling to fit inside output size with border
        avail_w = max(1, out_w - 2 * border)
        avail_h = max(1, out_h - 2 * border)
        s = min(avail_w / float(w), avail_h / float(h))
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create empty black canvas and paste resized character in center
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        x0 = (out_w - new_w) // 2
        y0 = (out_h - new_h) // 2

        # Compute paste coordinates (clamped to bounds)
        px0 = max(0, x0)
        py0 = max(0, y0)
        px1 = min(out_w, x0 + new_w)
        py1 = min(out_h, y0 + new_h)
        if px0 >= px1 or py0 >= py1:
            tiles.append(canvas)
            continue
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = sx0 + (px1 - px0)
        sy1 = sy0 + (py1 - py0)

        # Paste character into canvas
        canvas[py0:py1, px0:px1] = resized[sy0:sy1, sx0:sx1]
        tiles.append(canvas)
    return tiles


def _trim_uniform_border(img, tol=5, pad=0):
    """
    Crop uniform borders (solid background) from an image.
    """
    # Estimate background intensity from image edges
    arr = img
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return img

    # Compare pixel intensities to background and crop difference region
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]).astype(np.int16)
    bg = int(np.median(border))
    mask = np.abs(gray.astype(np.int16) - bg) > int(tol)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    return img[y0:y1, x0:x1]


def _show_borderless(img, cmap=None, figsize=(6, 3), dpi=120, trim=False, tol=5, pad=0):
    """
    Display an image without axes or borders (for visualization only).
    """
    # Create full-frame matplotlib figure with no axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    # Optionally trim uniform edges
    if trim:
        img = _trim_uniform_border(img, tol=tol, pad=pad)
    # Display grayscale or color
    if cmap is None:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation='nearest')
    ax.axis('off')
    plt.show()


def find_chars(cropped_plates):
    """
    Detect and extract individual character tiles from a cropped license plate.
    """
    # Convert plate to BGR then grayscale
    IMAGE = cv2.cvtColor(cropped_plates[0], cv2.COLOR_GRAY2BGR)
    gray_mask = convert_rgb_to_gray(IMAGE)

    # Normalize lighting across plate surface
    normalised = normalise_lighting(gray_mask)

    # Remove small noise using morphological closing
    gray_cleaned = gray_morphology(normalised)

    # Threshold adaptively to obtain binary mask
    binary_mask = adaptive_threshold(gray_cleaned)

    # Clean binary mask with morphological opening
    cleaned_mask = open_morphology_plate(binary_mask, erode_iterations=1, dilate_iterations=1)

    # Detect all contours (potential characters)
    contours_drawn, contours = find_all_contours(cleaned_mask, margin=0)

    # Filter to retain probable character contours only
    final_contours_drawn, chars_sorted = filter_contours(cleaned_mask, contours)

    # Fit each detected character into 64×64 boxes
    chars_boxed = fit_char_to_box(cleaned_mask, chars_sorted)
    return chars_boxed, final_contours_drawn
