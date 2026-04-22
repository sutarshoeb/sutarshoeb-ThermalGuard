from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PipelineStage:
    key: str
    title: str
    image: np.ndarray
    description: str


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _colorize_thermal(gray_image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(gray_image, cv2.COLORMAP_INFERNO)


def to_demo_thermal(image: np.ndarray) -> list[PipelineStage]:
    """
    Stronger FLIR-style thermal simulation adapted from the project's earlier
    better_thermal_simulation workflow. The inference image remains grayscale,
    while a colorized view is also generated for presentation.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    warm_mask = (
        ((hue >= 0) & (hue <= 30)) |
        ((hue >= 10) & (hue <= 50)) |
        ((hue >= 60) & (hue <= 90))
    ).astype(np.float32)

    organic_boost = (sat / 255.0) * warm_mask * 60.0
    thermal = np.clip(gray + organic_boost, 0, 255)

    dark_mask = (gray < 80).astype(np.float32)
    thermal = np.clip(thermal - dark_mask * 30.0, 0, 255)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(thermal.astype(np.uint8))
    thermal = contrast.astype(np.float32)

    bright_mask = (thermal > 180).astype(np.float32)
    bloom = cv2.GaussianBlur(bright_mask * 40, (15, 15), 0)
    thermal = np.clip(thermal + bloom, 0, 255)
    bloom_stage = thermal.astype(np.uint8)

    softened = cv2.GaussianBlur(bloom_stage, (5, 5), 1.2)
    softened = cv2.bilateralFilter(softened, 9, 75, 75)
    thermal = softened.astype(np.float32)

    fpn_row = np.random.normal(0, 1.5, (h, 1)).repeat(w, axis=1)
    fpn_col = np.random.normal(0, 1.5, (1, w)).repeat(h, axis=0)
    random_noise = np.random.normal(0, 3.0, (h, w))
    thermal = np.clip(thermal + fpn_row + fpn_col + random_noise, 0, 255)

    cx, cy = w // 2, h // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1.0 - 0.15 * (dist / max_dist)
    vignette_stage = np.clip(thermal * vignette, 0, 255)

    p_low = np.percentile(vignette_stage, 2)
    p_high = np.percentile(vignette_stage, 98)
    if p_high > p_low:
        normalized = (vignette_stage - p_low) / (p_high - p_low) * 255.0
    else:
        normalized = vignette_stage
    inference_thermal = np.clip(normalized, 0, 255).astype(np.uint8)
    colorized = _colorize_thermal(inference_thermal)

    return [
        PipelineStage("original", "Original Visible Frame", image, "The uploaded visible-light input frame."),
        PipelineStage("luminance", "Luminance Extraction", gray.astype(np.uint8), "The visible image is reduced to a grayscale structure map."),
        PipelineStage("contrast", "Thermal Contrast Build", contrast, "Warm organic regions are boosted while darker road-like regions are suppressed."),
        PipelineStage("bloom", "Heat Bloom Simulation", bloom_stage, "Bright thermal regions are allowed to spread slightly to mimic thermal bloom."),
        PipelineStage("thermal", "Inference Thermal Frame", inference_thermal, "This grayscale thermal-like frame is used for model inference."),
        PipelineStage("colorized", "Presentation Thermal View", colorized, "A colorized thermal rendering for easier visual presentation in the demo."),
    ]
