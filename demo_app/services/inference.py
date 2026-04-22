from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .assets import save_image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at scaffold time
    YOLO = None


def model_available(model_path: str) -> bool:
    return Path(model_path).exists()


def _register_inception_module() -> None:
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "models" / "inception_backbone.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Custom module not found: {module_path}")

    module_name = "thermal_inception_backbone"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load custom module from: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    inception_cls = getattr(module, "InceptionModule", None)
    if inception_cls is None:
        raise RuntimeError("InceptionModule is missing from the custom backbone file.")

    # Some checkpoints were serialized with __main__.InceptionModule.
    sys.modules["__main__"].InceptionModule = inception_cls


@lru_cache(maxsize=8)
def load_model(model_path: str, model_key: str = ""):
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Run `pip install -r requirements.txt` first.")
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    if model_key == "inception_yolov8":
        _register_inception_module()
    return YOLO(model_path)


def _run_single_model(model_key: str, metadata: dict, image_path: str, output_dir: Path) -> dict:
    model_path = metadata["path"]
    if not model_available(model_path):
        return {
            "key": model_key,
            "label": metadata["label"],
            "family": metadata["family"],
            "available": False,
            "message": f"Weight file not found: {model_path}",
        }

    try:
        started = time.perf_counter()
        model = load_model(model_path, model_key)
        results = model.predict(image_path, verbose=False, conf=0.08, imgsz=960)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        first = results[0]
        annotated = first.plot()
        output_path = save_image(annotated, output_dir, f"{model_key}_annotated.jpg")

        detections = []
        boxes = getattr(first, "boxes", None)
        if boxes is not None:
            names = getattr(first, "names", {})
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                detections.append(
                    {
                        "class_name": names.get(int(cls_id), f"class_{int(cls_id)}"),
                        "confidence": round(float(conf), 4),
                    }
                )

        detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)

        return {
            "key": model_key,
            "label": metadata["label"],
            "family": metadata["family"],
            "available": True,
            "elapsed_ms": round(elapsed_ms, 2),
            "detections": detections[:8],
            "detection_count": len(detections),
            "image_path": output_path,
        }
    except Exception as exc:
        return {
            "key": model_key,
            "label": metadata["label"],
            "family": metadata["family"],
            "available": False,
            "message": f"Inference failed: {exc}",
        }


def run_model_suite(model_registry: dict[str, dict[str, str]], selected_models: list[str], image_path: str, output_dir: Path) -> list[dict]:
    active_keys = [key for key in selected_models if key in model_registry]
    if not active_keys:
        active_keys = list(model_registry.keys())

    with ThreadPoolExecutor(max_workers=min(4, len(active_keys))) as executor:
        futures = [
            executor.submit(_run_single_model, key, model_registry[key], image_path, output_dir)
            for key in active_keys
        ]
        return [future.result() for future in futures]
