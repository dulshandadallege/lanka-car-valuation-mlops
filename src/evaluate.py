"""Metrics and the quality gate that decides whether a trained model may be released."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from .config import load_params
from .features import FEATURES


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape": float(100 * mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "within_10pct": float(100 * np.mean(np.abs(y_pred / y_true - 1) <= 0.10)),
        "n": int(len(y_true)),
    }


def evaluate(model, X_train, y_train, X_test, y_test, holdout) -> dict:
    hold_pred = model.predict(holdout[FEATURES])
    return {
        "train": regression_metrics(y_train, model.predict(X_train)),
        "test": regression_metrics(y_test, model.predict(X_test)),
        "holdout": {"median_ape": float(100 * np.median(np.abs(hold_pred / holdout["price"].values - 1))),
                    "n": int(len(holdout))},
    }


def quality_gate(metrics: dict, params: dict | None = None) -> list[str]:
    """Return a list of failures; an empty list means the model passes."""
    g = (params or load_params())["quality_gate"]
    fails = []
    if metrics["test"]["mape"] > g["max_test_mape"]:
        fails.append(f"test MAPE {metrics['test']['mape']:.2f}% > {g['max_test_mape']}%")
    if metrics["test"]["r2"] < g["min_test_r2"]:
        fails.append(f"test R² {metrics['test']['r2']:.3f} < {g['min_test_r2']}")
    if metrics["holdout"]["median_ape"] > g["max_holdout_median_ape"]:
        fails.append(f"hold-out median error {metrics['holdout']['median_ape']:.2f}% > {g['max_holdout_median_ape']}%")
    return fails
