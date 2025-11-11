import os
import sys

from load_data import load_dataset, load_emnist_dataset, get_splits
from feature_extraction import hog_features
from train_and_results import get_results_knn

from sklearn.svm import SVC
import joblib

set_kernel_parameter = "linear"  # one of: "linear", "rbf", "poly"
CACHE_DIR = "cache"
MODEL_PATH = os.path.join(CACHE_DIR, f"svm_{set_kernel_parameter}_emnist.joblib")

def main():
    # Load data (cap to 1000 samples per class for speed)
    X, y, classes = load_emnist_dataset(split="train", max_per_class=1000)
    print(f"Classes: {len(classes)} | X: {X.shape} | y: {y.shape}")

    # First stratified split
    _, (X_tr, X_te, y_tr, y_te, _, _) = get_splits(X, y, n_splits=4, shuffle=True, random_state=0)
    # HOG features with progress
    H_tr = hog_features(X_tr, progress=True)
    H_te = hog_features(X_te, progress=True)
    print(f"HOG train: {H_tr.shape} | HOG test: {H_te.shape}")

    if set_kernel_parameter == "linear":
        clf = SVC(kernel="linear", C=1.0)
    elif set_kernel_parameter == "rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    elif set_kernel_parameter == "poly":
        clf = SVC(kernel="poly", C=1.0, degree=3, gamma="scale")
    else:
        raise ValueError(f"Unsupported kernel parameter: {set_kernel_parameter}")

    clf.fit(H_tr, y_tr)

        # Evaluate using existing helper for consistency
    models = {1: clf}
    results = get_results_knn(models, H_te, y_te)
    print(f"SVM({set_kernel_parameter}) accuracy: {results['accuracies'][1]:.4f}")
    print("Confusion Matrix:\n", results["best_metrics"]["confusion_matrix"])
    print(results["best_metrics"]["classification_report"])

    # Save model
    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved SVM({set_kernel_parameter}) model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
