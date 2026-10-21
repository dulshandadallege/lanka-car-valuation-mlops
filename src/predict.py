"""Load the released model and predict prices.

    python -m src.predict data/new_listings_unseen.csv     # prints predictions for a CSV of listings
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH, RELEASE_MODEL_URL


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Use the local model; otherwise download the latest GitHub Release asset; otherwise train one."""
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
    try:
        print(f"Downloading model from {RELEASE_MODEL_URL}")
        urllib.request.urlretrieve(RELEASE_MODEL_URL, path)
        return path
    except Exception as exc:  # no release yet, or offline: fall back to training locally
        print(f"Download failed ({exc}); training a model locally instead")
        from .pipeline import run
        run("local")
        return path


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(ensure_model(path))


def predict(df: pd.DataFrame, bundle: dict | None = None) -> pd.Series:
    b = bundle or load_bundle()
    return pd.Series(b["model"].predict(df[b["features"]]), index=df.index, name="predicted_price")


if __name__ == "__main__":
    frame = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/new_listings_unseen.csv")
    out = frame.assign(predicted_price=predict(frame).round(-4))
    cols = [c for c in ["brand", "model_family", "age", "mileage_km", "price", "predicted_price"] if c in out]
    print(out[cols].to_string(index=False))
