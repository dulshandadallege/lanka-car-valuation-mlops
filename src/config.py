"""Paths and parameters shared by every pipeline step."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "vehicle_price_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
REPO = "dulshandadallege/lanka-car-valuation-mlops"
RELEASE_MODEL_URL = f"https://github.com/{REPO}/releases/latest/download/vehicle_price_model.joblib"


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
