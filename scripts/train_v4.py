from multiprocessing import freeze_support

def main():
    from ultralytics import YOLO
    from pathlib import Path
    import sys
    import os

    # ── suppress all YOLO verbose output at the source ───────────
    os.environ["YOLO_VERBOSE"] = "False"

    BASE_DIR  = Path(r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection")
    DATA_YAML = BASE_DIR / "data/processed/final_dataset_v2/data.yaml"
    PREV_WEIGHTS = BASE_DIR / "models/yolov8s/best.pt"

    # ── clean segment labels ──────────────────────────────────────
    print("Cleaning segment data from labels...")
    labels_dirs = [
        BASE_DIR / "data/processed/final_dataset_v2/labels/train",
        BASE_DIR / "data/processed/final_dataset_v2/labels/val",
    ]
    cleaned = 0
    for lbl_dir in labels_dirs:
        for lbl_file in lbl_dir.glob("*.txt"):
            text = lbl_file.read_text().strip()
            if not text:
                continue
            new_lines = []
            changed = False
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) > 5:
                    new_lines.append(" ".join(parts[:5]))
                    changed = True
                else:
                    new_lines.append(line.strip())
            if changed:
                lbl_file.write_text("\n".join(new_lines))
                cleaned += 1
    print(f"Cleaned {cleaned} label files")

    # ── delete stale cache ────────────────────────────────────────
    for cache in [
        BASE_DIR / "data/processed/final_dataset_v2/labels/train.cache",
        BASE_DIR / "data/processed/final_dataset_v2/labels/val.cache",
    ]:
        if cache.exists():
            cache.unlink()
            print(f"Deleted cache: {cache.name}")

    # ── load model ────────────────────────────────────────────────
    if PREV_WEIGHTS.exists():
        model = YOLO(str(PREV_WEIGHTS))
        print(f"Resuming from: {PREV_WEIGHTS}")
    else:
        model = YOLO("yolov8s.pt")
        print("Starting from pretrained yolov8s.pt")

    print(f"Dataset: {DATA_YAML}")
    print("="*65)
    print(f"{'Epoch':>8}  {'GPU_mem':>7}  {'box':>7}  {'cls':>7}  {'dfl':>7}  {'mAP50':>7}")
    print("="*65)
    sys.stdout.flush()

    # ── callback: one line per epoch ──────────────────────────────
    def on_train_epoch_end(trainer):
        e     = trainer.epoch + 1
        total = trainer.epochs

        try:
            import torch
            gpu = f"{torch.cuda.memory_reserved(0)/1e9:.2f}G"
        except Exception:
            gpu = "-"

        li = trainer.loss_items
        box = f"{float(li[0]):.4f}" if li is not None and len(li) > 0 else "---"
        cls = f"{float(li[1]):.4f}" if li is not None and len(li) > 1 else "---"
        dfl = f"{float(li[2]):.4f}" if li is not None and len(li) > 2 else "---"

        m     = trainer.metrics or {}
        map50 = m.get("metrics/mAP50(B)", 0.0)

        print(f"{e:>4}/{total:<4}  {gpu:>7}  {box:>7}  {cls:>7}  {dfl:>7}  {map50:>7.4f}")
        sys.stdout.flush()

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # ── redirect stdout during training to suppress batch lines ───
    class EpochOnlyFilter:
        def __init__(self, real_stdout):
            self.real = real_stdout
            self.buffer = ""

        def write(self, _text):
            # only pass through lines that come from our callback
            # (they start with a digit — epoch number)
            # block everything else during training
            pass

        def flush(self):
            self.real.flush()

    real_stdout = sys.stdout

    # ── run training ──────────────────────────────────────────────
    sys.stdout = EpochOnlyFilter(real_stdout)

    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=100,
            patience=30,
            batch=16,
            imgsz=640,
            device="0",
            workers=4,
            amp=True,
            verbose=False,

            lr0=0.001,
            lrf=0.01,
            cos_lr=True,

            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            flipud=0.1,
            mosaic=1.0,
            mixup=0.3,
            copy_paste=0.1,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            erasing=0.4,

            optimizer="AdamW",
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,

            close_mosaic=15,
            single_cls=True,

            box=8.0,
            cls=0.5,
            dfl=1.5,

            save=True,
            save_period=10,
            plots=True,
            project=str(BASE_DIR / "runs/detect"),
            name="train_v4_balanced",
            exist_ok=True,
        )
    finally:
        sys.stdout = real_stdout

    print("="*65)
    print("TRAINING COMPLETE")
    print(f"Best mAP50    : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"Best mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Weights       : {BASE_DIR}/runs/detect/train_v4_balanced/weights/best.pt")
    print("="*65)
    print("DONE — paste final results back to Claude Chat")


if __name__ == "__main__":
    freeze_support()
    main()
