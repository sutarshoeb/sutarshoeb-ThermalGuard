import os
import cv2
import numpy as np
from tqdm import tqdm

# 🔥 PROJECT ROOT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# INPUT DATASETS (RELATIVE — CORRECT)
DATASETS = [
    os.path.join(BASE_DIR, "data/raw/visible_animals/cattle_dataset"),
    os.path.join(BASE_DIR, "data/raw/visible_animals/road_animals_dataset")
]

# OUTPUT PATHS
OUT_IMG_TRAIN = os.path.join(BASE_DIR, "data/processed/visible_thermal/images/train")
OUT_LBL_TRAIN = os.path.join(BASE_DIR, "data/processed/visible_thermal/labels/train")

OUT_IMG_VAL = os.path.join(BASE_DIR, "data/processed/visible_thermal/images/val")
OUT_LBL_VAL = os.path.join(BASE_DIR, "data/processed/visible_thermal/labels/val")

# CREATE FOLDERS
for p in [OUT_IMG_TRAIN, OUT_LBL_TRAIN, OUT_IMG_VAL, OUT_LBL_VAL]:
    os.makedirs(p, exist_ok=True)

print("📁 Saving to:", OUT_IMG_TRAIN)


# ✅ REALISTIC THERMAL CONVERSION
def to_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize contrast (thermal effect)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # blur (thermal cameras are softer)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # add noise (sensor simulation)
    noise = np.random.normal(0, 5, gray.shape).astype(np.uint8)
    gray = cv2.add(gray, noise)

    return gray  # ✅ grayscale (NO colormap)


def process_split(split, out_img_dir, out_lbl_dir):
    for dataset in DATASETS:
        dataset_name = os.path.basename(dataset)

        img_dir = os.path.join(dataset, split, "images")
        lbl_dir = os.path.join(dataset, split, "labels")

        if not os.path.exists(img_dir):
            print(f"⚠️ Missing: {img_dir}")
            continue

        for img_name in tqdm(os.listdir(img_dir), desc=f"{dataset_name}-{split}"):

            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)
            if img is None:
                continue

            thermal_img = to_thermal(img)

            new_name = f"{dataset_name}_{img_name}"

            # SAVE IMAGE
            cv2.imwrite(os.path.join(out_img_dir, new_name), thermal_img)

            # SAVE LABEL
            save_lbl_path = os.path.join(out_lbl_dir, new_name.replace(".jpg", ".txt"))

            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = "0"  # force single class
                    new_lines.append(" ".join(parts))

                with open(save_lbl_path, "w") as f:
                    f.write("\n".join(new_lines))
            else:
                open(save_lbl_path, "w").close()


# RUN
process_split("train", OUT_IMG_TRAIN, OUT_LBL_TRAIN)
process_split("valid", OUT_IMG_VAL, OUT_LBL_VAL)

print("✅ DONE — Thermal dataset rebuilt correctly")