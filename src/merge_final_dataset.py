import os
import shutil
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# INPUT PATHS
ANIMAL_IMG = os.path.join(BASE_DIR, "data/processed/visible_thermal/images/train")
ANIMAL_LBL = os.path.join(BASE_DIR, "data/processed/visible_thermal/labels/train")

FLIR_IMG = os.path.join(BASE_DIR, "data/processed/images/train")

# OUTPUT PATHS
OUT_IMG_TRAIN = os.path.join(BASE_DIR, "data/processed/final_dataset/images/train")
OUT_LBL_TRAIN = os.path.join(BASE_DIR, "data/processed/final_dataset/labels/train")

OUT_IMG_VAL = os.path.join(BASE_DIR, "data/processed/final_dataset/images/val")
OUT_LBL_VAL = os.path.join(BASE_DIR, "data/processed/final_dataset/labels/val")

# CREATE FOLDERS
for p in [OUT_IMG_TRAIN, OUT_LBL_TRAIN, OUT_IMG_VAL, OUT_LBL_VAL]:
    os.makedirs(p, exist_ok=True)

# =========================
# 🔥 DATA COLLECTION
# =========================

data = []

# 1️⃣ ADD ANIMALS (ALL)
animal_images = os.listdir(ANIMAL_IMG)
print("Animal images:", len(animal_images))

for img_name in animal_images:
    data.append(("animal", img_name))

# 2️⃣ LIMIT FLIR BACKGROUND
MAX_BACKGROUND = 4000

flir_images = os.listdir(FLIR_IMG)
random.shuffle(flir_images)

flir_images = flir_images[:MAX_BACKGROUND]

print("FLIR images used:", len(flir_images))

for img_name in flir_images:
    data.append(("flir", img_name))

# =========================
# 🔀 SHUFFLE + SPLIT
# =========================

print("TOTAL DATA:", len(data))

random.shuffle(data)

split_idx = int(0.8 * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]

print("TRAIN SIZE:", len(train_data))
print("VAL SIZE:", len(val_data))

# =========================
# 📦 PROCESS FUNCTION
# =========================

def process(data_list, out_img, out_lbl, name):
    print(f"\n📦 Processing {name}... count = {len(data_list)}")

    for i, (dtype, img_name) in enumerate(tqdm(data_list)):

        # Debug first few samples
        if i < 3:
            print(f"{name} sample:", img_name)

        if dtype == "animal":
            src_img = os.path.join(ANIMAL_IMG, img_name)
            src_lbl = os.path.join(ANIMAL_LBL, img_name.replace(".jpg", ".txt"))
            new_name = "animal_" + img_name

        else:  # FLIR
            src_img = os.path.join(FLIR_IMG, img_name)
            src_lbl = None
            new_name = "flir_" + img_name

        if not os.path.exists(src_img):
            print("❌ Missing:", src_img)
            continue

        # SAVE IMAGE
        dst_img_path = os.path.join(out_img, new_name)
        shutil.copy(src_img, dst_img_path)

        # SAVE LABEL
        lbl_name = new_name.replace(".jpg", ".txt")
        dst_lbl_path = os.path.join(out_lbl, lbl_name)

        if src_lbl and os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl_path)
        else:
            open(dst_lbl_path, "w").close()

# =========================
# ▶️ RUN
# =========================

process(train_data, OUT_IMG_TRAIN, OUT_LBL_TRAIN, "TRAIN")
process(val_data, OUT_IMG_VAL, OUT_LBL_VAL, "VAL")

print("\n✅ DATASET REBUILT SUCCESSFULLY")