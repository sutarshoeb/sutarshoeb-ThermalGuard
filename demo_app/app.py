from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for

from .config import config
from .services.assets import create_session_dir, save_image, save_upload
from .services.benchmark import load_benchmark_table
from .services.inference import run_model_suite
from .services.pipeline import load_image, to_demo_thermal


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            benchmarks=load_benchmark_table(config.benchmark_csv),
            model_registry=config.model_registry,
            result=None,
            error=None,
        )

    @app.post("/analyze")
    def analyze():
        upload = request.files.get("image")
        source_type = request.form.get("source_type", "visible")
        selected_models = request.form.getlist("models")

        if upload is None or not upload.filename:
            return render_template(
                "index.html",
                benchmarks=load_benchmark_table(config.benchmark_csv),
                model_registry=config.model_registry,
                result=None,
                error="Please upload an image before running the demo.",
            )

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in config.allowed_extensions:
            return render_template(
                "index.html",
                benchmarks=load_benchmark_table(config.benchmark_csv),
                model_registry=config.model_registry,
                result=None,
                error=f"Unsupported file type: {suffix}",
            )

        upload_session = create_session_dir(config.upload_dir)
        output_session = create_session_dir(config.output_dir)
        uploaded_path = save_upload(upload, upload_session)

        pipeline_views = []
        inference_path = uploaded_path
        inference_url = None

        if source_type == "visible":
            image = load_image(str(uploaded_path))
            stages = to_demo_thermal(image)
            for stage in stages:
                if len(stage.image.shape) == 2:
                    stage_image = stage.image
                else:
                    stage_image = stage.image
                saved_path = save_image(stage_image, output_session, f"{stage.key}.jpg")
                pipeline_views.append(
                    {
                        "key": stage.key,
                        "title": stage.title,
                        "description": stage.description,
                        "url": url_for("artifact", kind="outputs", session=output_session.name, filename=saved_path.name),
                    }
                )
            inference_path = output_session / "thermal.jpg"
            thermal_stage = next((stage for stage in pipeline_views if stage["key"] == "thermal"), None)
            preview_stage = next((stage for stage in pipeline_views if stage["key"] == "colorized"), None)
            inference_url = thermal_stage["url"] if thermal_stage else None
            pipeline_preview_url = preview_stage["url"] if preview_stage else inference_url

        model_results = run_model_suite(
            config.model_registry,
            selected_models,
            str(inference_path),
            output_session,
        )

        for item in model_results:
            if item.get("image_path"):
                image_path = Path(item["image_path"])
                item["image_url"] = url_for(
                    "artifact",
                    kind="outputs",
                    session=output_session.name,
                    filename=image_path.name,
                )

        available_results = [item for item in model_results if item.get("available")]
        fastest_model = min(available_results, key=lambda item: item["elapsed_ms"], default=None)
        busiest_model = max(available_results, key=lambda item: item["detection_count"], default=None)

        uploaded_url = url_for("artifact", kind="uploads", session=upload_session.name, filename=uploaded_path.name)
        if source_type != "visible":
            inference_url = uploaded_url
            pipeline_preview_url = uploaded_url
        else:
            pipeline_preview_url = pipeline_preview_url or inference_url or uploaded_url
            inference_url = inference_url or uploaded_url

        return render_template(
            "index.html",
            benchmarks=load_benchmark_table(config.benchmark_csv),
            model_registry=config.model_registry,
            error=None,
            result={
                "source_type": source_type,
                "uploaded_url": uploaded_url,
                "inference_url": inference_url,
                "pipeline_preview_url": pipeline_preview_url,
                "pipeline_views": pipeline_views,
                "model_results": sorted(
                    model_results,
                    key=lambda item: item.get("elapsed_ms", 999999),
                ),
                "summary": {
                    "fastest": fastest_model,
                    "most_detections": busiest_model,
                    "recommended_demo_model": "YOLOv8s",
                    "recommended_accuracy_model": "RT-DETR-L",
                },
            },
        )

    @app.route("/artifacts/<kind>/<session>/<filename>")
    def artifact(kind: str, session: str, filename: str):
        base = config.upload_dir if kind == "uploads" else config.output_dir
        return send_from_directory(base / session, filename)

    return app
