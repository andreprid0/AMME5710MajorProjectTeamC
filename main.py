import cv2
import matplotlib.pyplot as plt
import image_preprocess
import locate_plate
import mask
import joblib
import os
import split_characters 
import test_model as test

#cached models in folder cache
MODEL_PATH = os.path.join("cache", "svm_linear.joblib")
#test images in TestImages folder
IMG_PATH = "TestImages/mazda.jpg"

def main():
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMG_PATH}")
    # Show original image using matplotlib (convert BGR to RGB)
    plt.figure(figsize=(6, 4), dpi=120)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- NEW unified cleaning step ---
    cleaned_image = image_preprocess.clean_image(img_bgr)

    # --- Continue the rest of the pipeline ---
    black_mask, white_mask = mask.mask_image(cleaned_image)

    plate_candidates = locate_plate.crop_plate(white_mask, img_bgr)
    
    # Access and show all

    for plate in plate_candidates:
        # cv2.imwrite("output_image.jpg", crop)
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    clf = joblib.load(MODEL_PATH)   
    character_boxed, contours = split_characters.find_chars(plate_candidates)
    # Display contours image from character detection
    plt.figure(figsize=(6, 3), dpi=120)
    plt.imshow(cv2.cvtColor(contours, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Contours on Plate")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    test.test_svm(plate_candidates, character_boxed, clf)

if __name__ == "__main__":
    main()
