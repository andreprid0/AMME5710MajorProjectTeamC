import os
from pathlib import Path
import numpy as np
import cv2
import joblib
from matplotlib import pyplot as plt

from feature_extraction import hog_features


def test_svm(cropped_plates, tiles, clf):
    print("\nPredicting characters for detected tiles...")

    for i, tile in enumerate(tiles):
        # Show the tile
        plt.figure(figsize=(2.5, 2.5), dpi=120)
        plt.imshow(tile, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        plt.title(f"Tile #{i}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Prepare features and predict
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        feat = hog_features(np.array([tile_bgr]))
        pred = clf.predict(feat)[0]

        # Print only the predicted character (no score)
        print(f"Tile {i}: predicted '{pred}'")
