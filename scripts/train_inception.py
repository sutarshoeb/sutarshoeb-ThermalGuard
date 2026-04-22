"""
Training script for Inception-YOLOv8.
IMPORTANT: inception_backbone must be imported BEFORE YOLO
loads the YAML so InceptionModule is registered first.
"""

from pathlib import Path

# -- CRITICAL: import this FIRST to register InceptionModule ----------
# This must happen before YOLO() is called
import sys
PROJECT_ROOT = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")
sys.path.insert(0, str(PROJECT_ROOT))
from models.inception_backbone import InceptionModule  # triggers registration

# Force registration into parse_model scope
import ultralytics.nn.tasks
ultralytics.nn.tasks.__dict__['InceptionModule'] = InceptionModule

from ultralytics import YOLO

# -- Paths -------------------------------------------------------------
DATA_YAML  = PROJECT_ROOT / "data/processed/final_dataset_v3/data.yaml"
MODEL_YAML = PROJECT_ROOT / "models/inception_yolo.yaml"
SAVE_DIR   = Path(r"C:\Users\kupak\runs\detect")


def train():
    print("=" * 60)
    print("INCEPTION-YOLOV8 THERMAL ANIMAL DETECTION")
    print("Genuine multi-scale parallel backbone active")
    print("=" * 60)

    # Build model from YAML — InceptionModule already registered
    model = YOLO(str(MODEL_YAML))

    # Count and display parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\nArchitecture: Inception-YOLOv8")
    print(f"Total parameters: {total_params:,}")
    print(f"Data: {DATA_YAML}")
    print(f"Saving to: {SAVE_DIR}/train_v7_inception_yolo/\n")

    results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    patience=30,
    batch=4,           # ← halved from 8 — biggest VRAM reduction
    imgsz=512,         # ← reduced from 640 — cuts memory by 36%
    device=0,
    workers=2,         # ← reduced from 4 — less CPU/RAM pressure
    amp=True,          # ← keep — automatic mixed precision saves ~30% VRAM
    cache=False,       # ← do NOT cache images — saves RAM
    lr0=0.005,
    lrf=0.001,
    cos_lr=False,
    optimizer="AdamW",
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_bias_lr=0.05,
    momentum=0.937,
    box=8.0,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.1,
    mosaic=0.5,        # ← reduced from 1.0 — less augmentation = less compute
    mixup=0.1,         # ← reduced from 0.3
    copy_paste=0.0,    # ← disabled — saves compute
    degrees=5.0,
    translate=0.1,
    scale=0.4,
    erasing=0.3,
    close_mosaic=10,
    single_cls=True,
    save=True,
    save_period=5,
    plots=True,
    name="train_v7_inception_yolo",
    project=str(SAVE_DIR),
    exist_ok=True,     # ← True so it resumes into same folder
    pretrained=False
)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Weights: {SAVE_DIR}/train_v7_inception_yolo/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    train()
