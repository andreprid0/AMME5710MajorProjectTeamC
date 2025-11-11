import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#   Filter Image
# ------------------------------------------------------------------------------------------
def filter_image(img_bgr):
    """
    Apply multiple smoothing filters to an input BGR image.

    Filters applied:
      - Average filter (7×7)
      - Gaussian blur
      - Median blur
      - Bilateral filter (preserves edges)

    Args:
        img_bgr (ndarray): Input image in BGR color space.

    Returns:
        im_avg (ndarray): Averaged filter result.
        im_gauss (ndarray): Gaussian blur result.
        im_median (ndarray): Median blur result.
        im_bilateral (ndarray): Bilateral filter result.
    """
    # Convert BGR to RGB for consistent color interpretation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Create averaging kernel (7×7 window)
    kernel = (1 / 49.0) * np.ones((7, 7), np.float32)

    # Apply four filtering methods
    im_avg = cv2.filter2D(img_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    im_gauss = cv2.GaussianBlur(img_rgb, (7, 7), 0)
    im_median = cv2.medianBlur(img_rgb, 7)
    im_bilateral = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    return im_avg, im_gauss, im_median, im_bilateral


# ------------------------------------------------------------------------------------------
#   Contrast Enhance
# ------------------------------------------------------------------------------------------
def enhance_contrast(img_rgb, clip_limit, tile_grid_size):
    """
    Improve image contrast using CLAHE on the Value channel (HSV space).

    Args:
        img_rgb (ndarray): Input RGB image.
        clip_limit (float): CLAHE contrast limit.
        tile_grid_size (tuple): Size of grid tiles for local histogram equalization.

    Returns:
        contrast_rgb (ndarray): RGB image with enhanced contrast.
    """
    # Convert to HSV color space for brightness-based enhancement
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Apply CLAHE (adaptive histogram equalization) on V channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)

    # Merge enhanced V channel back into HSV and convert to RGB
    hsv_clahe = cv2.merge([h, s, v_clahe])
    contrast_rgb = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)
    return contrast_rgb


# ------------------------------------------------------------------------------------------
#   Vibrancy Enhance
# ------------------------------------------------------------------------------------------
def enhance_vibrancy(img_bgr, sat_boost, val_boost):
    """
    Enhance image vibrancy by scaling HSV saturation and brightness.

    Args:
        img_bgr (ndarray): Input image in BGR color space.
        sat_boost (float): Multiplicative factor for saturation.
        val_boost (float): Multiplicative factor for brightness (value channel).

    Returns:
        vibrant_bgr (ndarray): BGR image with boosted color vibrancy.
    """
    # Convert BGR → HSV for color adjustment
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Boost saturation and brightness
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_boost, 0, 255)

    # Convert back to uint8 BGR
    vibrant_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return vibrant_bgr


# ------------------------------------------------------------------------------------------
#   Sharpen Image
# ------------------------------------------------------------------------------------------
def sharpen_img(img_bgr, gamma, ksize):
    """
    Sharpen an image using unsharp masking (original + gamma × detail).

    Args:
        img_bgr (ndarray): Input image (BGR).
        gamma (float): Sharpening strength (multiplier for high-frequency detail).
        ksize (tuple): Gaussian blur kernel size for detail extraction.

    Returns:
        sharpened (ndarray): Sharpened RGB float image (0–1 range).
    """
    # Convert to RGB for consistent processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Normalize to 0–1 float range
    img_f = img_rgb.astype(np.float32) / 255.0

    # Compute blurred version to extract details
    blur = cv2.GaussianBlur(img_f, ksize, 0)
    detail = img_f - blur

    # Add weighted detail to enhance edges
    sharpened = np.clip(img_f + gamma * detail, 0, 1)
    return sharpened


# ------------------------------------------------------------------------------------------
#   Clean image (full preprocessing pipeline)
# ------------------------------------------------------------------------------------------
def clean_image(img_bgr):
    """
    Full preprocessing pipeline combining all enhancement steps.

    Steps:
      1. Apply smoothing filters to reduce noise.
      2. Enhance contrast using CLAHE on the value channel.
      3. Boost vibrancy by increasing HSV saturation and brightness.
      4. Sharpen final image using unsharp masking.

    Args:
        img_bgr (ndarray): Input image in BGR color space.

    Returns:
        img_sharp_rgb (ndarray): Preprocessed RGB image ready for masking/detection.
    """
    # 1) Filtering (average, gaussian, median, bilateral)
    im_avg, im_gauss, im_median, im_bilateral = filter_image(img_bgr)

    # 2) Contrast Enhancement (operate on bilateral result)
    img_contrast_rgb = enhance_contrast(im_bilateral, clip_limit=2.0, tile_grid_size=(8, 8))

    # 3) Vibrancy Enhancement (convert to BGR, then boost)
    img_contrast_bgr = cv2.cvtColor(img_contrast_rgb, cv2.COLOR_RGB2BGR)
    img_vibrant_bgr = enhance_vibrancy(img_contrast_bgr, sat_boost=1.6, val_boost=1.1)

    # 4) Sharpening (final polish)
    img_sharp_rgb = sharpen_img(img_vibrant_bgr, gamma=2.3, ksize=(7, 7))

    return img_sharp_rgb
