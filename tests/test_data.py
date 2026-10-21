"""Data-validation tests: the raw schema and the cleaned modelling table."""
import pandas as pd

from src.data import REQUIRED_RAW_COLUMNS, load_raw
from src.features import CAT, FEATURES, NUM


def test_raw_schema():
    raw = load_raw()
    missing = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)
    assert not missing, f"raw files are missing columns: {missing}"
    assert len(raw) >= 5000


def test_clean_table_contract(datasets):
    data, log, *_ = datasets
    assert list(data[FEATURES].columns) == FEATURES
    assert data["price"].between(300_000, 100_000_000).all()
    assert data["age"].between(0, 60).all()
    assert data["engine_cc"].dropna().between(600, 6000).all()
    assert (data["mileage_km"].dropna() >= 0).all()
    assert data["district"].ne("nan").all(), "every town must map to a district"
    assert not data.duplicated(["brand", "Model", "year", "mileage_km", "price", "Location"]).any()
    for c in CAT:
        assert data[c].map(type).eq(str).all()
    for c in NUM:
        assert pd.api.types.is_numeric_dtype(data[c])
    assert [n for _, n in log] == sorted([n for _, n in log], reverse=True)


def test_split_has_no_leakage(datasets):
    _, _, X_train, X_test, _, _, holdout = datasets
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(holdout.index).isdisjoint(set(X_train.index) | set(X_test.index))
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
