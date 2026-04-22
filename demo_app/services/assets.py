from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


def create_session_dir(root: Path) -> Path:
    session_dir = root / uuid.uuid4().hex
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_upload(file: FileStorage, destination: Path) -> Path:
    filename = secure_filename(file.filename or "upload.png")
    target = destination / filename
    file.save(target)
    return target


def save_image(image: np.ndarray, destination: Path, filename: str) -> Path:
    target = destination / filename
    cv2.imwrite(str(target), image)
    return target
