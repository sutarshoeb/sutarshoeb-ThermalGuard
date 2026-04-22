from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DemoConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    app_artifacts_dir: Path = field(init=False)
    upload_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    benchmark_csv: Path = field(init=False)
    secret_key: str = "thermal-animal-detection-demo"
    max_content_length: int = 16 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __post_init__(self) -> None:
        self.app_artifacts_dir = self.base_dir / "app_artifacts"
        self.upload_dir = self.app_artifacts_dir / "uploads"
        self.output_dir = self.app_artifacts_dir / "outputs"
        self.benchmark_csv = self.base_dir / "research" / "benchmarks" / "paper_results_table.csv"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_registry(self) -> dict[str, dict[str, str]]:
        results_root = self.base_dir / "final_assets" / "colab_export" / "results"
        return {
            "rtdetr_l": {
                "label": "RT-DETR-L",
                "family": "Transformer",
                "path": str(results_root / "rtdetr_l" / "weights" / "best.pt"),
            },
            "yolov8s": {
                "label": "YOLOv8s",
                "family": "CNN Baseline",
                "path": str(results_root / "yolov8s" / "weights" / "best.pt"),
            },
            "yolov9c": {
                "label": "YOLOv9c",
                "family": "CNN + PGI",
                "path": str(results_root / "yolov9c" / "weights" / "best.pt"),
            },
            "inception_yolov8": {
                "label": "Inception-YOLOv8",
                "family": "Multi-Scale CNN",
                "path": str(results_root / "inception_yolov8" / "weights" / "best.pt"),
            },
        }


config = DemoConfig()
