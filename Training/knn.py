import os
import sys

from load_data import load_dataset, get_splits
from feature_extraction import hog_features
from train_and_results import get_results_knn

from sklearn.neighbors import KNeighborsClassifier
import joblib

set_k_parameter = "5"  
CACHE_DIR = "cache"
MODEL_PATH = os.path.join(CACHE_DIR, f"knn_k_{set_k_parameter}.joblib")


def main():
    # Load data
    X, y, classes = load_dataset()
    print(f"Classes: {len(classes)} | X: {X.shape} | y: {y.shape}")

    # First stratified split
    _, (X_tr, X_te, y_tr, y_te, _, _) = get_splits(X, y, n_splits=4, shuffle=True, random_state=0)
    # HOG features
    H_tr = hog_features(X_tr)
    H_te = hog_features(X_te)
    print(f"HOG train: {H_tr.shape} | HOG test: {H_te.shape}")

    if set_k_parameter == "1": 
        # Train k=1
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(H_tr, y_tr)
    elif set_k_parameter == "5": 
        # Train k=5
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(H_tr, y_tr)

    # Evaluate using existing helper for consistency
    models = {1: clf}
    results = get_results_knn(models, H_te, y_te)
    print(f"k=1 accuracy: {results['accuracies'][1]:.4f}")
    print("Confusion Matrix:\n", results["best_metrics"]["confusion_matrix"])
    print(results["best_metrics"]["classification_report"])

    # Save model
    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved KNN model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
