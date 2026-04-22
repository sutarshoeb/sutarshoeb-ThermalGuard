import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")

# ── INPUT SOURCES ─────────────────────────────────────────────────
BETTER_THERMAL_IMG = BASE_DIR / "data/raw/better_thermal/images"
BETTER_THERMAL_LBL = BASE_DIR / "data/raw/better_thermal/labels"
FLIR_IMG_TRAIN     = BASE_DIR / "data/processed/images/train"
FLIR_IMG_VAL       = BASE_DIR / "data/processed/images/val"

# ── OUTPUT ────────────────────────────────────────────────────────
OUT_BASE      = BASE_DIR / "data/processed/final_dataset_v3"
OUT_TRAIN_IMG = OUT_BASE / "images/train"
OUT_TRAIN_LBL = OUT_BASE / "labels/train"
OUT_VAL_IMG   = OUT_BASE / "images/val"
OUT_VAL_LBL   = OUT_BASE / "labels/val"

for p in [OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

random.seed(42)

print("=" * 60)
print("REBUILDING DATASET v3 — better thermal simulation")
print("=" * 60)

# ── COLLECT ANIMAL IMAGES (better thermal only) ───────────────────
animal_imgs = list(BETTER_THERMAL_IMG.glob("*.jpg"))
animal_data = []
for img in animal_imgs:
    lbl = BETTER_THERMAL_LBL / (img.stem + ".txt")
    if lbl.exists():
        content = lbl.read_text().strip()
        if content:  # only images with actual annotations
            animal_data.append((img, lbl))

print(f"Animal images with annotations: {len(animal_data)}")

# ── COLLECT FLIR BACKGROUNDS ──────────────────────────────────────
# Match animal count for 50/50 balance
TARGET_BG = len(animal_data)
flir_all  = list(FLIR_IMG_TRAIN.glob("*.jpg")) + \
            list(FLIR_IMG_VAL.glob("*.jpg"))
random.shuffle(flir_all)
flir_bg = flir_all[:TARGET_BG]
print(f"FLIR background images: {len(flir_bg)}")

# ── COMBINE AND SPLIT ─────────────────────────────────────────────
all_data = []
for img, lbl in animal_data:
    all_data.append(("animal", img, lbl))
for img in flir_bg:
    all_data.append(("flir", img, None))

random.shuffle(all_data)

split_idx  = int(0.8 * len(all_data))
train_data = all_data[:split_idx]
val_data   = all_data[split_idx:]

n_animal_train = sum(1 for d in train_data if d[0] == "animal")
n_flir_train   = sum(1 for d in train_data if d[0] == "flir")

print(f"\nTOTAL  : {len(all_data)}")
print(f"TRAIN  : {len(train_data)}")
print(f"  animal : {n_animal_train} ({100*n_animal_train/len(train_data):.1f}%)")
print(f"  flir   : {n_flir_train} ({100*n_flir_train/len(train_data):.1f}%)")
print(f"VAL    : {len(val_data)}")

# ── COPY FILES ────────────────────────────────────────────────────
def copy_split(data_list, img_dir, lbl_dir, name):
    print(f"\nCopying {name}...")
    for i, (dtype, img_path, lbl_path) in enumerate(tqdm(data_list)):
        stem = f"{dtype}_{i:06d}"
        shutil.copy2(img_path, img_dir / f"{stem}.jpg")
        dst_lbl = lbl_dir / f"{stem}.txt"
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
        else:
            dst_lbl.write_text("")

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

# ── SUMMARY ───────────────────────────────────────────────────────
train_count = len(list(OUT_TRAIN_IMG.glob("*.jpg")))
val_count   = len(list(OUT_VAL_IMG.glob("*.jpg")))

print(f"\n{'='*60}")
print(f"DATASET v3 COMPLETE")
print(f"Train images : {train_count}")
print(f"Val images   : {val_count}")
print(f"data.yaml    : {OUT_BASE / 'data.yaml'}")
print(f"\nDONE — paste results back to Claude Chat")
print("=" * 60)
