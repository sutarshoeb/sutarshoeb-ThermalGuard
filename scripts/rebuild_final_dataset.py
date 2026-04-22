import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")

# ── INPUT SOURCES ─────────────────────────────────────────────────
ORIGINAL_ANIMAL_IMG = BASE_DIR / "data/processed/visible_thermal/images/train"
ORIGINAL_ANIMAL_LBL = BASE_DIR / "data/processed/visible_thermal/labels/train"

NEW_ANIMAL_IMG      = BASE_DIR / "data/raw/additional_animals/images"
NEW_ANIMAL_LBL      = BASE_DIR / "data/raw/additional_animals/labels"

FLIR_IMG            = BASE_DIR / "data/processed/images/train"

# ── OUTPUT ────────────────────────────────────────────────────────
OUT_BASE  = BASE_DIR / "data/processed/final_dataset_v2"
OUT_TRAIN_IMG = OUT_BASE / "images/train"
OUT_TRAIN_LBL = OUT_BASE / "labels/train"
OUT_VAL_IMG   = OUT_BASE / "images/val"
OUT_VAL_LBL   = OUT_BASE / "labels/val"

for p in [OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

random.seed(42)

print("=" * 60)
print("REBUILDING FINAL DATASET v2")
print("=" * 60)

# ── COLLECT ALL ANIMAL IMAGES ─────────────────────────────────────
animal_data = []

# original thermal animals
orig_imgs = list(ORIGINAL_ANIMAL_IMG.glob("*.jpg")) + \
            list(ORIGINAL_ANIMAL_IMG.glob("*.png"))
for img in orig_imgs:
    lbl = ORIGINAL_ANIMAL_LBL / (img.stem + ".txt")
    if lbl.exists() and lbl.stat().st_size > 0:
        animal_data.append(("orig", img, lbl))

print(f"Original animal images (with labels): {len(animal_data)}")

# new Open Images animals
new_imgs = list(NEW_ANIMAL_IMG.glob("*.jpg"))
new_count = 0
for img in new_imgs:
    lbl = NEW_ANIMAL_LBL / (img.stem + ".txt")
    if lbl.exists() and lbl.stat().st_size > 0:
        animal_data.append(("new", img, lbl))
        new_count += 1

print(f"New Open Images animals (with labels): {new_count}")
print(f"Total animal images: {len(animal_data)}")

# ── COLLECT FLIR BACKGROUND ───────────────────────────────────────
# Match animal count for ~50/50 balance
TARGET_BG = len(animal_data)
flir_imgs = list(FLIR_IMG.glob("*.jpg"))
random.shuffle(flir_imgs)
flir_imgs = flir_imgs[:TARGET_BG]

print(f"\nFLIR background images (capped at {TARGET_BG}): {len(flir_imgs)}")

# ── BUILD COMBINED LIST ───────────────────────────────────────────
all_data = []
for src, img, lbl in animal_data:
    all_data.append(("animal", src, img, lbl))
for img in flir_imgs:
    all_data.append(("flir", None, img, None))

random.shuffle(all_data)

split_idx = int(0.8 * len(all_data))
train_data = all_data[:split_idx]
val_data   = all_data[split_idx:]

total_animal_train = sum(1 for d in train_data if d[0] == "animal")
total_flir_train   = sum(1 for d in train_data if d[0] == "flir")

print(f"\nTOTAL dataset    : {len(all_data)}")
print(f"Train            : {len(train_data)}")
print(f"  Animal (train) : {total_animal_train} ({100*total_animal_train/len(train_data):.1f}%)")
print(f"  FLIR   (train) : {total_flir_train} ({100*total_flir_train/len(train_data):.1f}%)")
print(f"Val              : {len(val_data)}")

# ── COPY FILES ────────────────────────────────────────────────────
def copy_split(data_list, img_dir, lbl_dir, split_name):
    print(f"\nCopying {split_name}...")
    for i, entry in enumerate(tqdm(data_list)):
        dtype, src, img_path, lbl_path = entry
        ext = img_path.suffix

        if dtype == "animal":
            prefix = "orig" if src == "orig" else "new"
            new_stem = f"{prefix}_{i:06d}"
        else:
            new_stem = f"flir_{i:06d}"

        # copy image
        dst_img = img_dir / f"{new_stem}{ext}"
        shutil.copy2(img_path, dst_img)

        # copy or create label
        dst_lbl = lbl_dir / f"{new_stem}.txt"
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
        else:
            dst_lbl.write_text("")  # empty label for background

copy_split(train_data, OUT_TRAIN_IMG, OUT_TRAIN_LBL, "TRAIN")
copy_split(val_data,   OUT_VAL_IMG,   OUT_VAL_LBL,   "VAL")

# ── WRITE data.yaml ───────────────────────────────────────────────
yaml_content = f"""path: {OUT_BASE.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: animal
"""

(OUT_BASE / "data.yaml").write_text(yaml_content)
print(f"\ndata.yaml written to {OUT_BASE / 'data.yaml'}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────
train_imgs = list(OUT_TRAIN_IMG.glob("*.jpg")) + list(OUT_TRAIN_IMG.glob("*.png"))
val_imgs   = list(OUT_VAL_IMG.glob("*.jpg"))   + list(OUT_VAL_IMG.glob("*.png"))

print(f"\n{'='*60}")
print(f"DATASET REBUILD COMPLETE")
print(f"{'='*60}")
print(f"Train images : {len(train_imgs)}")
print(f"Val images   : {len(val_imgs)}")
print(f"data.yaml    : {OUT_BASE / 'data.yaml'}")
print(f"\nNext step: update training config to use final_dataset_v2")
print(f"DONE — paste results back to Claude Chat")
print("=" * 60)
