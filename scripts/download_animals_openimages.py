import fiftyone as fo
import fiftyone.zoo as foz
import cv2
import numpy as np
from pathlib import Path
import time

BASE_DIR = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")
OUT_DIR  = BASE_DIR / "data/raw/additional_animals"
OUT_IMG  = OUT_DIR / "images"
OUT_LBL  = OUT_DIR / "labels"
PROGRESS = OUT_DIR / "progress.txt"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Cattle", "Horse", "Sheep", "Dog", "Deer",
    "Elephant", "Zebra", "Goat", "Camel"
]
MAX_PER_CLASS = 300

# ── load already-done classes from progress file ──────────────────
done = set()
if PROGRESS.exists():
    done = set(PROGRESS.read_text().strip().splitlines())
    print(f"Resuming — already done: {done}")

# ── count already saved images ────────────────────────────────────
saved_total = len(list(OUT_IMG.glob("*.jpg")))
print(f"Images already saved: {saved_total}")

def to_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.random.normal(0, 5, gray.shape).astype(np.uint8)
    return cv2.add(gray, noise)

print("=" * 60)
print("DOWNLOADING FROM GOOGLE OPEN IMAGES V7")
print("=" * 60)

for cls in CLASSES:
    if cls in done:
        print(f"\n[SKIP] {cls} — already completed")
        continue

    print(f"\n[START] Downloading class: {cls} ...")

    # retry up to 3 times on connection errors
    ds = None
    for attempt in range(1, 4):
        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=[cls],
                max_samples=MAX_PER_CLASS,
                dataset_name=f"oi_{cls.lower()}",
                overwrite=True,
            )
            print(f"  Downloaded {len(ds)} samples for {cls}")
            break
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < 3:
                print(f"  Waiting 10 seconds before retry...")
                time.sleep(10)
            else:
                print(f"  All retries failed for {cls} — skipping")

    if ds is None:
        continue

    cls_saved = 0
    cls_skipped = 0

    for sample in ds:
        # FiftyOne stores images here — use the actual filepath directly
        img_path = Path(sample.filepath)

        if not img_path.exists():
            cls_skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            cls_skipped += 1
            continue

        # collect bounding boxes for this class only
        lines = []
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                if det.label != cls:
                    continue
                bx, by, bw, bh = det.bounding_box
                cx = bx + bw / 2
                cy = by + bh / 2
                # clamp to [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            cls_skipped += 1
            continue

        stem = f"oi_{cls.lower()}_{saved_total:05d}"
        thermal = to_thermal(img)
        cv2.imwrite(str(OUT_IMG / f"{stem}.jpg"), thermal)
        (OUT_LBL / f"{stem}.txt").write_text("\n".join(lines))

        saved_total += 1
        cls_saved += 1

    print(f"  {cls}: saved {cls_saved}, skipped {cls_skipped}")

    # mark this class as done
    with open(PROGRESS, "a") as f:
        f.write(cls + "\n")

print(f"\n{'='*60}")
print(f"TOTAL SAVED : {saved_total}")
print(f"Output      : {OUT_IMG}")
print(f"\nDONE — paste results back to Claude Chat")
print("=" * 60)
