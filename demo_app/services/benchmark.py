from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_benchmark_table(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    records = pd.read_csv(csv_path).to_dict(orient="records")
    return records
