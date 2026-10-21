from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .config import REPORT_DIR, load_params
from .features import CAT, NUM

EPS = 1e-4


def psi_numeric(ref: pd.Series, new: pd.Series, bins: int = 10) -> float:
    ref, new = ref.dropna().astype(float), new.dropna().astype(float)
    if len(ref) == 0 or len(new) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, EPS, None), np.clip(n, EPS, None)
    return float(np.sum((n - r) * np.log(n / r)))


def psi_categorical(ref: pd.Series, new: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(new.dropna()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    n = new.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(np.sum((n - r) * np.log(n / r)))


def drift_report(reference: pd.DataFrame, new: pd.DataFrame, params: dict | None = None) -> dict:
    m = (params or load_params())["monitoring"]
    rows = {}
    for f in NUM:
        rows[f] = psi_numeric(reference[f], new[f])
    for f in CAT:
        rows[f] = psi_categorical(reference[f].astype(str), new[f].astype(str))
    status = {f: ("alert" if v > m["psi_alert"] else "warn" if v > m["psi_warn"] else "ok") for f, v in rows.items()}
    return {"psi": rows, "status": status, "n_reference": len(reference), "n_new": len(new),
            "retrain_recommended": any(s == "alert" for s in status.values())}


def to_markdown(rep: dict) -> str:
    icon = {"ok": "🟢", "warn": "🟡", "alert": "🔴"}
    lines = [f"## Data drift report ({rep['n_new']:,} new vs {rep['n_reference']:,} training listings)", "",
             "| Feature | PSI | Status |", "|---|---|---|"]
    for f, v in sorted(rep["psi"].items(), key=lambda kv: -np.nan_to_num(kv[1])):
        lines.append(f"| {f} | {v:.3f} | {icon[rep['status'][f]]} {rep['status'][f]} |")
    lines += ["", "**Retraining recommended.**" if rep["retrain_recommended"] else "No significant drift."]
    return "\n".join(lines)


def main() -> None:
    ref_path = REPORT_DIR / "training_reference.csv"
    if not ref_path.exists():
        sys.exit("Run `python -m src.pipeline` first to create reports/training_reference.csv")
    data = pd.read_csv(ref_path, parse_dates=["ad_date"])
    if len(sys.argv) > 1:
        reference, new = data, pd.read_csv(sys.argv[1])
    else:
        cutoff = data["ad_date"].max().to_period("M").to_timestamp()
        reference, new = data[data["ad_date"] < cutoff], data[data["ad_date"] >= cutoff]
        print(f"Reference: ads before {cutoff:%b %Y} ({len(reference):,}) | new: {cutoff:%b %Y} ({len(new):,})")
    rep = drift_report(reference, new)
    (REPORT_DIR / "drift.json").write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep)
    (REPORT_DIR / "drift.md").write_text(md)
    print(md)
    sys.exit(2 if rep["retrain_recommended"] else 0)


if __name__ == "__main__":
    main()
