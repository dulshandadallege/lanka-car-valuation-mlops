from __future__ import annotations

import datetime as dt
import platform

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODEL_DIR, MODEL_PATH, load_params
from .features import CAT, NUM


def build_pipeline(params: dict | None = None) -> TransformedTargetRegressor:
    p = params or load_params()
    m = p["model"]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=m["ohe_min_frequency"],
                              sparse_output=False), CAT),
    ])
    rf = RandomForestRegressor(n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                               min_samples_leaf=m["min_samples_leaf"], max_features=m["max_features"],
                               random_state=p["seed"], n_jobs=-1)
    return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("model", rf)]),
                                      func=np.log, inverse_func=np.exp)


def make_bundle(model, data, X_train, metrics: dict, data_hash: str, version: str, params: dict) -> dict:
    brand_models = {b: sorted(data.loc[data["brand"] == b, "model_family"].unique())
                    for b in sorted(data["brand"].unique())}
    defaults = (data.groupby("model_family").agg(
        engine_cc=("engine_cc", "median"), body_type=("body_type", lambda s: s.mode()[0]),
        fuel_type=("fuel_type", lambda s: s.mode()[0]), origin_country=("origin_country", lambda s: s.mode()[0]),
    ).to_dict("index"))
    return {
        "model": model, "model_name": "Random Forest (tuned)", "features": CAT + NUM, "cat": CAT, "num": NUM,
        "ref_year": params["reference_year"], "brand_models": brand_models, "model_defaults": defaults,
        "options": {c: sorted(data[c].unique()) for c in CAT},
        "test_mae": metrics["test"]["mae"], "test_mape": metrics["test"]["mape"],
        "trained_on": int(len(X_train)),
        "data_period": f"{data['ad_date'].min():%b %Y} – {data['ad_date'].max():%b %Y}",
        "metadata": {
            "version": version, "data_sha256_16": data_hash, "trained_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "params": params["model"], "metrics": metrics, "sklearn": sklearn.__version__,
            "python": platform.python_version(),
        },
    }


def save_bundle(bundle: dict) -> str:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    return str(MODEL_PATH)
