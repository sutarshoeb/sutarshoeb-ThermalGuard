import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")

INPUT_DIRS = [
    BASE_DIR / "data/raw/additional_animals/images",
    BASE_DIR / "data/processed/visible_thermal/images/train",
]

OUTPUT_DIR = BASE_DIR / "data/raw/better_thermal/images"
OUTPUT_LBL = BASE_DIR / "data/raw/better_thermal/labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LBL.mkdir(parents=True, exist_ok=True)

# ── copy labels from source dirs ──────────────────────────────────
SOURCE_LBL_DIRS = [
    BASE_DIR / "data/raw/additional_animals/labels",
    BASE_DIR / "data/processed/visible_thermal/labels/train",
]


def to_realistic_thermal(img):
    """
    Simulate real FLIR-style thermal image from visible light.

    Real thermal characteristics:
    - Animals appear BRIGHT (warm bodies = high IR emission)
    - Sky and road appear DARK (cold surfaces)
    - Edges are soft and bloomed (thermal cameras have lower resolution)
    - High contrast between warm bodies and cold background
    - Grayscale with characteristic thermal noise pattern
    - Vignette effect (edges slightly darker)
    - Hot spots on animal core (belly, head) brighter than limbs
    """

    h, w = img.shape[:2]

    # ── STEP 1: Convert to grayscale luminance ────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── STEP 2: Estimate "warmth" of each pixel ───────────────────
    # In thermal, warm objects (animals) appear bright
    # We use skin/fur tone detection + overall brightness as proxy
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue channel — organic warm tones (browns, tans, greens of grass)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Warm organic regions (animal fur/skin tones) get boosted
    # These hue ranges cover browns, tans, ochres, greens
    warm_mask = (
        ((hue >= 0) & (hue <= 30)) |    # reds, oranges
        ((hue >= 10) & (hue <= 50)) |   # oranges, yellows
        ((hue >= 60) & (hue <= 90))     # yellows, light greens
    ).astype(np.float32)

    # Saturated regions are usually organic (animal, vegetation)
    # Desaturated = sky, road, concrete (cold in thermal)
    organic_boost = (sat / 255.0) * warm_mask * 60.0

    # ── STEP 3: Build thermal map ─────────────────────────────────
    # Base: inverted brightness (dark areas in visible = cold in thermal)
    # but animals often have mid-tone fur that should appear bright
    thermal = gray.copy()

    # Boost warm organic regions significantly
    thermal = np.clip(thermal + organic_boost, 0, 255)

    # Dark areas (road, sky) pushed darker in thermal
    dark_mask = (gray < 80).astype(np.float32)
    thermal = thermal - dark_mask * 30.0
    thermal = np.clip(thermal, 0, 255)

    # ── STEP 4: Apply thermal-style contrast enhancement ─────────
    # CLAHE — real thermal cameras have high local contrast
    thermal_uint8 = thermal.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    thermal_uint8 = clahe.apply(thermal_uint8)
    thermal = thermal_uint8.astype(np.float32)

    # ── STEP 5: Thermal bloom effect ─────────────────────────────
    # Warm objects "bloom" in thermal — bright areas spread slightly
    bright_mask = (thermal > 180).astype(np.float32)
    bloom = cv2.GaussianBlur(bright_mask * 40, (15, 15), 0)
    thermal = np.clip(thermal + bloom, 0, 255)

    # ── STEP 6: Soft focus (thermal cameras have lower resolution) ─
    # Two-stage blur: fine detail lost, edges soft
    thermal = cv2.GaussianBlur(thermal.astype(np.uint8), (5, 5), 1.2)
    thermal = cv2.bilateralFilter(thermal, 9, 75, 75)
    thermal = thermal.astype(np.float32)

    # ── STEP 7: Realistic thermal noise ──────────────────────────
    # Fixed pattern noise (FPN) — characteristic of thermal sensors
    fpn_row = np.random.normal(0, 1.5, (h, 1)).repeat(w, axis=1)
    fpn_col = np.random.normal(0, 1.5, (1, w)).repeat(h, axis=0)
    # Gaussian random noise
    random_noise = np.random.normal(0, 3.0, (h, w))
    thermal = thermal + fpn_row + fpn_col + random_noise
    thermal = np.clip(thermal, 0, 255)

    # ── STEP 8: Vignette (thermal lens falloff) ───────────────────
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - 0.15 * (dist / max_dist)
    thermal = thermal * vignette
    thermal = np.clip(thermal, 0, 255)

    # ── STEP 9: Final contrast stretch ───────────────────────────
    # Real thermal cameras auto-stretch contrast across the scene
    p_low = np.percentile(thermal, 2)
    p_high = np.percentile(thermal, 98)
    if p_high > p_low:
        thermal = (thermal - p_low) / (p_high - p_low) * 255.0
    thermal = np.clip(thermal, 0, 255).astype(np.uint8)

    return thermal


# ── PROCESS ALL IMAGES ────────────────────────────────────────────
saved = 0
skipped = 0

for src_img_dir, src_lbl_dir in zip(INPUT_DIRS, SOURCE_LBL_DIRS):
    if not src_img_dir.exists():
        print(f"SKIP (not found): {src_img_dir}")
        continue

    images = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
    print(f"\nProcessing {len(images)} images from {src_img_dir.name}...")

    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        thermal = to_realistic_thermal(img)

        stem = f"bt_{saved:06d}"
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}.jpg"), thermal)

        # Copy label
        lbl_src = src_lbl_dir / (img_path.stem + ".txt")
        lbl_dst = OUTPUT_LBL / f"{stem}.txt"

        if lbl_src.exists():
            lbl_dst.write_text(lbl_src.read_text())
        else:
            lbl_dst.write_text("")

        saved += 1

print(f"\n{'='*60}")
print(f"Saved  : {saved} realistic thermal images")
print(f"Skipped: {skipped}")
print(f"Output : {OUTPUT_DIR}")
print(f"\nDONE — paste results back to Claude Chat")
print("=" * 60)import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")

INPUT_DIRS = [
    BASE_DIR / "data/raw/additional_animals/images",
    BASE_DIR / "data/processed/visible_thermal/images/train",
]

OUTPUT_DIR = BASE_DIR / "data/raw/better_thermal/images"
OUTPUT_LBL = BASE_DIR / "data/raw/better_thermal/labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LBL.mkdir(parents=True, exist_ok=True)

# ── copy labels from source dirs ──────────────────────────────────
SOURCE_LBL_DIRS = [
    BASE_DIR / "data/raw/additional_animals/labels",
    BASE_DIR / "data/processed/visible_thermal/labels/train",
]


def to_realistic_thermal(img):
    """
    Simulate real FLIR-style thermal image from visible light.

    Real thermal characteristics:
    - Animals appear BRIGHT (warm bodies = high IR emission)
    - Sky and road appear DARK (cold surfaces)
    - Edges are soft and bloomed (thermal cameras have lower resolution)
    - High contrast between warm bodies and cold background
    - Grayscale with characteristic thermal noise pattern
    - Vignette effect (edges slightly darker)
    - Hot spots on animal core (belly, head) brighter than limbs
    """

    h, w = img.shape[:2]

    # ── STEP 1: Convert to grayscale luminance ────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── STEP 2: Estimate "warmth" of each pixel ───────────────────
    # In thermal, warm objects (animals) appear bright
    # We use skin/fur tone detection + overall brightness as proxy
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue channel — organic warm tones (browns, tans, greens of grass)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Warm organic regions (animal fur/skin tones) get boosted
    # These hue ranges cover browns, tans, ochres, greens
    warm_mask = (
        ((hue >= 0) & (hue <= 30)) |    # reds, oranges
        ((hue >= 10) & (hue <= 50)) |   # oranges, yellows
        ((hue >= 60) & (hue <= 90))     # yellows, light greens
    ).astype(np.float32)

    # Saturated regions are usually organic (animal, vegetation)
    # Desaturated = sky, road, concrete (cold in thermal)
    organic_boost = (sat / 255.0) * warm_mask * 60.0

    # ── STEP 3: Build thermal map ─────────────────────────────────
    # Base: inverted brightness (dark areas in visible = cold in thermal)
    # but animals often have mid-tone fur that should appear bright
    thermal = gray.copy()

    # Boost warm organic regions significantly
    thermal = np.clip(thermal + organic_boost, 0, 255)

    # Dark areas (road, sky) pushed darker in thermal
    dark_mask = (gray < 80).astype(np.float32)
    thermal = thermal - dark_mask * 30.0
    thermal = np.clip(thermal, 0, 255)

    # ── STEP 4: Apply thermal-style contrast enhancement ─────────
    # CLAHE — real thermal cameras have high local contrast
    thermal_uint8 = thermal.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    thermal_uint8 = clahe.apply(thermal_uint8)
    thermal = thermal_uint8.astype(np.float32)

    # ── STEP 5: Thermal bloom effect ─────────────────────────────
    # Warm objects "bloom" in thermal — bright areas spread slightly
    bright_mask = (thermal > 180).astype(np.float32)
    bloom = cv2.GaussianBlur(bright_mask * 40, (15, 15), 0)
    thermal = np.clip(thermal + bloom, 0, 255)

    # ── STEP 6: Soft focus (thermal cameras have lower resolution) ─
    # Two-stage blur: fine detail lost, edges soft
    thermal = cv2.GaussianBlur(thermal.astype(np.uint8), (5, 5), 1.2)
    thermal = cv2.bilateralFilter(thermal, 9, 75, 75)
    thermal = thermal.astype(np.float32)

    # ── STEP 7: Realistic thermal noise ──────────────────────────
    # Fixed pattern noise (FPN) — characteristic of thermal sensors
    fpn_row = np.random.normal(0, 1.5, (h, 1)).repeat(w, axis=1)
    fpn_col = np.random.normal(0, 1.5, (1, w)).repeat(h, axis=0)
    # Gaussian random noise
    random_noise = np.random.normal(0, 3.0, (h, w))
    thermal = thermal + fpn_row + fpn_col + random_noise
    thermal = np.clip(thermal, 0, 255)

    # ── STEP 8: Vignette (thermal lens falloff) ───────────────────
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - 0.15 * (dist / max_dist)
    thermal = thermal * vignette
    thermal = np.clip(thermal, 0, 255)

    # ── STEP 9: Final contrast stretch ───────────────────────────
    # Real thermal cameras auto-stretch contrast across the scene
    p_low = np.percentile(thermal, 2)
    p_high = np.percentile(thermal, 98)
    if p_high > p_low:
        thermal = (thermal - p_low) / (p_high - p_low) * 255.0
    thermal = np.clip(thermal, 0, 255).astype(np.uint8)

    return thermal


# ── PROCESS ALL IMAGES ────────────────────────────────────────────
saved = 0
skipped = 0

for src_img_dir, src_lbl_dir in zip(INPUT_DIRS, SOURCE_LBL_DIRS):
    if not src_img_dir.exists():
        print(f"SKIP (not found): {src_img_dir}")
        continue

    images = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
    print(f"\nProcessing {len(images)} images from {src_img_dir.name}...")

    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        thermal = to_realistic_thermal(img)

        stem = f"bt_{saved:06d}"
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}.jpg"), thermal)

        # Copy label
        lbl_src = src_lbl_dir / (img_path.stem + ".txt")
        lbl_dst = OUTPUT_LBL / f"{stem}.txt"

        if lbl_src.exists():
            lbl_dst.write_text(lbl_src.read_text())
        else:
            lbl_dst.write_text("")

        saved += 1

print(f"\n{'='*60}")
print(f"Saved  : {saved} realistic thermal images")
print(f"Skipped: {skipped}")
print(f"Output : {OUTPUT_DIR}")
print(f"\nDONE — paste results back to Claude Chat")
print("=" * 60)