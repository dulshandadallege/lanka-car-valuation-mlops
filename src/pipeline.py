"""End-to-end training pipeline: prepare -> train -> evaluate -> quality gate -> save.

    python -m src.pipeline                 # train, evaluate, save model + reports/metrics.json
    python -m src.pipeline --version v1.0.0
Exits with code 1 if the model fails the quality gate in params.yaml, so CI blocks the release.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import METRICS_PATH, REPORT_DIR, ROOT, load_params
from .data import build_datasets, data_fingerprint
from .evaluate import evaluate, quality_gate
from .train import build_pipeline, make_bundle, save_bundle


def run(version: str = "dev") -> dict:
    params = load_params()
    t0 = time.time()
    data, log, X_train, X_test, y_train, y_test, holdout = build_datasets(params)
    print(f"[prepare] {len(data):,} clean listings | train {len(X_train):,} | test {len(X_test):,} | "
          f"holdout {len(holdout)}")
    for step, n in log:
        print(f"          {step:<28}{n:>6,}")

    model = build_pipeline(params).fit(X_train, y_train)
    print(f"[train]   {params['model']['name']} fitted in {time.time() - t0:.0f}s")

    metrics = evaluate(model, X_train, y_train, X_test, y_test, holdout)
    t = metrics["test"]
    print(f"[eval]    test MAE Rs {t['mae']:,.0f} | MAPE {t['mape']:.2f}% | R² {t['r2']:.3f} | "
          f"within ±10% {t['within_10pct']:.1f}% | hold-out median error {metrics['holdout']['median_ape']:.1f}%")

    data_hash = data_fingerprint(params)
    fails = quality_gate(metrics, params)
    report = {"version": version, "data_sha256_16": data_hash, "cleaning_log": log, "metrics": metrics,
              "quality_gate": {"passed": not fails, "failures": fails, **params["quality_gate"]}}
    REPORT_DIR.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    # A copy of the training-data profile, used by the drift monitor as its reference.
    data.to_csv(REPORT_DIR / "training_reference.csv", index=False)
    holdout.to_csv(ROOT / "data" / "new_listings_unseen.csv", index=False)

    if fails:
        print("[gate]    FAILED: " + "; ".join(fails))
        return report
    path = save_bundle(make_bundle(model, data, X_train, metrics, data_hash, version, params))
    print(f"[gate]    passed -> saved {path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="dev")
    report = run(ap.parse_args().version)
    sys.exit(0 if report["quality_gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
