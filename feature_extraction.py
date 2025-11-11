import numpy as np
import cv2

#-----------Configuration Parameters------------------------------
H, W = 64, 64


# ---------------- Feature extraction ----------------
def extract_downsample(img, new_size=(7, 7)):
    """
    Downsample an image and flatten it into a 1D vector.

    Args:
        img (ndarray): Input image (BGR or grayscale).
        new_size (tuple): Target downsample size (width, height).

    Returns:
        flattened (ndarray): Flattened vector of the resized image.
    """
    # Resize to smaller spatial size
    small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    # Flatten into 1D feature vector
    return small.flatten()


def downsample_features(X, new_size=(7, 7)):
    """
    Apply extract_downsample() to a list/array of images.

    Args:
        X (list[ndarray]): Collection of images.
        new_size (tuple): Target size for each downsample.

    Returns:
        features (ndarray): Stacked feature array (N × flattened_length).
    """
    return np.vstack([extract_downsample(img, new_size) for img in X])


def extract_histogram(img, bins=32):
    """
    Extract normalized HSV histograms from an image.

    Args:
        img (ndarray): Input BGR image.
        bins (int): Number of bins per channel.

    Returns:
        hists_all_channels (ndarray): Flattened and normalized concatenated histogram.
    """
    # Convert to HSV for color-based features
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hists = []

    # Compute histogram for each HSV channel
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256]).flatten()
        hists.append(hist)

    # Concatenate and normalize
    hists_all_channels = np.concatenate(hists).astype(np.float32)
    hists_all_channels /= hists_all_channels.sum()
    return hists_all_channels


def hsv_hist_features(X, bins=32):
    """
    Compute HSV histograms for a collection of images.

    Args:
        X (list[ndarray]): List of images.
        bins (int): Number of bins per channel.

    Returns:
        features (ndarray): Stacked HSV histogram features (N × (3×bins)).
    """
    return np.vstack([extract_histogram(img, bins) for img in X])


def get_hog(H_: int = H, W_: int = W):
    """
    Create a preconfigured HOGDescriptor.

    Args:
        H_ (int): Target image height.
        W_ (int): Target image width.

    Returns:
        hog (cv2.HOGDescriptor): Configured HOG descriptor.
    """
    # Configure the HOG descriptor (cell, block, stride, bins)
    return cv2.HOGDescriptor(
        _winSize=(W_, H_),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def extract_hog(img, hog=None):
    """
    Compute HOG features for a single image.

    Args:
        img (ndarray): Input BGR image.
        hog (cv2.HOGDescriptor | None): Optional precreated HOG descriptor.

    Returns:
        feats (ndarray): Flattened HOG feature vector.
    """
    # Initialize HOG if not provided
    if hog is None:
        hog = get_hog()

    # Convert to grayscale for gradient-based descriptor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute and flatten HOG features
    feats = hog.compute(gray).ravel()
    return feats


def hog_features(X, H_: int = H, W_: int = W):
    """
    Compute HOG features for a batch of images.

    Args:
        X (list[ndarray]): List of input images.
        H_ (int): Target height for HOG.
        W_ (int): Target width for HOG.

    Returns:
        features (ndarray): Stacked HOG feature matrix (N × descriptor_length).
    """
    # Create a single shared HOG descriptor to reuse for all images
    hog = get_hog(H_, W_)
    return np.vstack([extract_hog(img, hog) for img in X])


def make_feature_sets(image_train, image_test, label_train, bins=32):
    """
    Construct multiple feature sets (HOG, Downsample, HSV Histogram)
    for training and testing data.

    Args:
        image_train (list[ndarray]): Training images.
        image_test (list[ndarray]): Test images.
        label_train (list): Labels for training images.
        bins (int): Number of bins per HSV channel for histogram features.

    Returns:
        train_sets (dict): {
            "HOG": (features, labels),
            "Down": (features, labels),
            "HSV": (features, labels)
        }
        test_sets (dict): {
            "HOG": features,
            "Down": features,
            "HSV": features
        }
    """
    # Extract HOG features
    hog_extracted_imgs = hog_features(image_train)
    # Extract spatially downsampled features
    downsamples_imgs = downsample_features(image_train)
    # Extract HSV color histogram features
    histogram_imgs = hsv_hist_features(image_train, bins=bins)

    # Pack training feature sets (with labels)
    train_sets = {
        "HOG": (hog_extracted_imgs, label_train),
        "Down": (downsamples_imgs, label_train),
        "HSV": (histogram_imgs, label_train),
    }

    # Extract corresponding features for test images (no labels)
    test_sets = {
        "HOG": hog_features(image_test),
        "Down": downsample_features(image_test),
        "HSV": hsv_hist_features(image_test, bins=bins),
    }

    return train_sets, test_sets
