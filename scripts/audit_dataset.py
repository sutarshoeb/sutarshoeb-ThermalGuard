import os
from pathlib import Path
import cv2

BASE_DIR = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")

SPLITS = {
    "final_train_images":  BASE_DIR / "data/processed/final_dataset/images/train",
    "final_train_labels":  BASE_DIR / "data/processed/final_dataset/labels/train",
    "final_val_images":    BASE_DIR / "data/processed/final_dataset/images/val",
    "final_val_labels":    BASE_DIR / "data/processed/final_dataset/labels/val",
    "animal_images_train": BASE_DIR / "data/processed/visible_thermal/images/train",
    "animal_labels_train": BASE_DIR / "data/processed/visible_thermal/labels/train",
    "flir_images_train":   BASE_DIR / "data/processed/images/train",
    "flir_images_val":     BASE_DIR / "data/processed/images/val",
}

print("=" * 60)
print("THERMAL ANIMAL DETECTION — DATASET AUDIT")
print("=" * 60)

for name, path in SPLITS.items():
    if not path.exists():
        print(f"\n[MISSING]  {name}: {path}")
        continue
    files = list(path.glob("*"))
    images = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    labels = [f for f in files if f.suffix == ".txt"]
    print(f"\n[{name}]")
    print(f"  Total files : {len(files)}")
    print(f"  Images      : {len(images)}")
    print(f"  Labels      : {len(labels)}")

print("\n" + "=" * 60)
print("FINAL DATASET BREAKDOWN")
print("=" * 60)

train_img_dir = SPLITS["final_train_images"]
train_lbl_dir = SPLITS["final_train_labels"]
val_img_dir   = SPLITS["final_val_images"]

if train_img_dir.exists() and train_lbl_dir.exists():
    train_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    val_images   = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))

    animal_train = [f for f in train_images if f.name.startswith("animal_")]
    flir_train   = [f for f in train_images if f.name.startswith("flir_")]
    other_train  = [f for f in train_images if not f.name.startswith("animal_") and not f.name.startswith("flir_")]

    print(f"\n  TRAIN total     : {len(train_images)}")
    print(f"  TRAIN animal    : {len(animal_train)}  ({100*len(animal_train)/max(len(train_images),1):.1f}%)")
    print(f"  TRAIN flir bg   : {len(flir_train)}   ({100*len(flir_train)/max(len(train_images),1):.1f}%)")
    print(f"  TRAIN other     : {len(other_train)}")
    print(f"\n  VAL total       : {len(val_images)}")

    total = len(train_images) + len(val_images)
    print(f"\n  OVERALL total   : {total}")
    print(f"  Train/Val split : {100*len(train_images)/max(total,1):.1f}% / {100*len(val_images)/max(total,1):.1f}%")

print("\n" + "=" * 60)
print("LABEL QUALITY CHECK")
print("=" * 60)

empty_labels   = 0
missing_labels = 0
corrupt_images = 0
checked        = 0

for img_path in train_images[:500]:
    lbl_path = train_lbl_dir / (img_path.stem + ".txt")
    if not lbl_path.exists():
        missing_labels += 1
    else:
        content = lbl_path.read_text().strip()
        if content == "":
            empty_labels += 1
    img = cv2.imread(str(img_path))
    if img is None:
        corrupt_images += 1
    checked += 1

print(f"\n  Checked (first 500 train images)")
print(f"  Missing label files : {missing_labels}")
print(f"  Empty label files   : {empty_labels}  (background images — expected)")
print(f"  Corrupt images      : {corrupt_images}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE — paste these results back to Claude Chat")
print("=" * 60)
