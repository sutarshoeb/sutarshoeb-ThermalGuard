import os
import shutil
from tqdm import tqdm

# --------- PATHS ---------

FLIR_TRAIN_PATH = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\raw\flir_adas\images_thermal_train\data"
FLIR_VAL_PATH = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\raw\flir_adas\images_thermal_val\data"

PROCESSED_TRAIN_IMG = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\processed\images\train"
PROCESSED_VAL_IMG = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\processed\images\val"

PROCESSED_TRAIN_LABEL = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\processed\labels\train"
PROCESSED_VAL_LABEL = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\processed\labels\val"


def prepare_background(source_img_path, dest_img_path, dest_label_path):
    images = [f for f in os.listdir(source_img_path) if f.lower().endswith(".jpg")]

    print(f"Found {len(images)} images in {source_img_path}")

    for img_name in tqdm(images):
        src_img = os.path.join(source_img_path, img_name)
        dst_img = os.path.join(dest_img_path, img_name)

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Create empty label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(dest_label_path, label_name)

        open(label_path, "w").close()  # create empty file


if __name__ == "__main__":
    print("Preparing FLIR TRAIN background data...")
    prepare_background(FLIR_TRAIN_PATH, PROCESSED_TRAIN_IMG, PROCESSED_TRAIN_LABEL)

    print("Preparing FLIR VAL background data...")
    prepare_background(FLIR_VAL_PATH, PROCESSED_VAL_IMG, PROCESSED_VAL_LABEL)

    print("FLIR background preparation complete.")