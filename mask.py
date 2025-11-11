import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#   Mask Black and White
# ------------------------------------------------------------------------------------------
def mask_image(img_rgb):
    """
    Generate binary black and white masks from an RGB image.

    Workflow:
      1) Ensure image is 8-bit (convert from float if needed).
      2) Convert RGB image to grayscale for intensity-based masking.
      3) Create two binary masks:
         - black_mask: highlights dark regions (intensity 0–80)
         - white_mask: highlights bright regions (intensity 175–255)
      4) Return both masks for downstream plate localization.

    Args:
        img_rgb (ndarray): Input RGB image.

    Returns:
        black_mask (ndarray): Binary mask of dark regions.
        white_mask (ndarray): Binary mask of bright regions.
    """
    # Convert to 8-bit if input is normalized float (0–1)
    if img_rgb.dtype != np.uint8:
        img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img_rgb

    # Convert RGB to grayscale
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    # Create binary masks for dark and bright intensity ranges
    black_mask = cv2.inRange(gray, 0, 80)
    white_mask = cv2.inRange(gray, 175, 255)
    
    # Optional debug: visualize black and white masks side-by-side
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1); plt.imshow(black_mask, cmap='gray'); plt.title("Black Mask"); plt.axis("off")
    # plt.subplot(1,2,2); plt.imshow(white_mask, cmap='gray'); plt.title("White Mask"); plt.axis("off")
    # plt.tight_layout(); plt.show()
    
    return black_mask, white_mask
