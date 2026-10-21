"""Behavioural tests on the trained model (run after `python -m src.pipeline`)."""
import numpy as np
import pandas as pd

from src.evaluate import quality_gate
from src.features import make_row

CAR = {"brand": "Toyota", "model_family": "Aqua", "gear": "Automatic", "fuel_type": "Hybrid",
       "condition": "Registered (Used)", "body_type": "Hatchback", "origin_country": "Japan", "district": "Colombo",
       "year": 2014, "mileage_km": 90_000, "engine_cc": 1500, "power_options": 3}


def price(bundle, **kw):
    return bundle["model"].predict(make_row({**CAR, **kw}, bundle["ref_year"]))[0]


def test_bundle_contract(bundle):
    for key in ["model", "features", "options", "brand_models", "test_mape", "metadata"]:
        assert key in bundle
    assert bundle["metadata"]["data_sha256_16"]


def test_prediction_is_plausible(bundle):
    assert 3_000_000 < price(bundle) < 15_000_000


def test_directional_expectations(bundle):
    assert price(bundle, year=2006) < price(bundle, year=2018), "older cars must be cheaper"
    bmw = dict(brand="BMW", model_family="5 Series", origin_country="Germany", body_type="Sedan",
               fuel_type="Diesel", engine_cc=2000)
    assert price(bundle, **bmw) > price(bundle), "a 5 Series should cost more than an Aqua"


def test_unseen_listings(bundle, datasets):
    holdout = datasets[-1]
    pred = bundle["model"].predict(holdout[bundle["features"]])
    assert np.median(np.abs(pred / holdout["price"].values - 1)) < 0.15


def test_quality_gate_logic():
    good = {"test": {"mape": 8.0, "r2": 0.92}, "holdout": {"median_ape": 8.0}}
    bad = {"test": {"mape": 14.0, "r2": 0.80}, "holdout": {"median_ape": 20.0}}
    assert quality_gate(good) == []
    assert len(quality_gate(bad)) == 3


def test_batch_prediction_shape(bundle, datasets):
    X_test = datasets[3].head(50)
    out = pd.Series(bundle["model"].predict(X_test))
    assert out.notna().all() and (out > 0).all() and len(out) == 50
