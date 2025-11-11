Classic computer vision pipeline to find a plate, split characters, and predict them with a cached model.
Main scripts live in the repo root. Training code and synthetic data are in subfolders.

What it will do: 
You’ll see the input image, plate crops, character tiles, and predictions.


To Run: 
Run Main and tweak the following params: 
Test image: edit IMG_PATH in main.py (point it to something in TestImages/).
Cached model: edit MODEL_PATH in main.py (use one from cache/, e.g. svm_linear.joblib, svm_rbf.joblib, svm_poly.joblib, knn_k_1.joblib, knn_k_5.joblib).

Root pipeline: main.py, image_preprocess.py, mask.py, locate_plate.py, split_characters.py, feature_extraction.py, test_model.py, ground_truth_data.py
Models: cache/ (pre-trained .joblib files)
Sample images: TestImages/
Training: Training/ (KNN/SVM trainers + data loaders) to train -> Unzip SyntheticDataset/Developed Training Dataset.zip so it becomes SyntheticDataset/Developed Training Dataset/.
Synthetic data: SyntheticDataset/ (generator + a zip of the dataset) -> used here for dataset generation visualisation (doesn't run)

Other Notes: 
“Could not read image”: make sure IMG_PATH points to an existing file.
“Model not found”: pick a .joblib that exists in cache/.
Empty detections: try another image or swap models.
