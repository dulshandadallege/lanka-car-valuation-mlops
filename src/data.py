from __future__ import annotations

import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, load_params
from .features import CAT, FEATURES, FILE_MODEL, TOWN_DISTRICT, family_from_text

REQUIRED_RAW_COLUMNS = ["Brand", "Model", "YOM", "Mileage", "Gear", "Fuel Type", "Engine (cc)", "Condition",
                        "Ad Date", "Location", "Price", "Body Type", "Power Steering", "Manufacturing Country"]


def parse_number(s) -> float:
    if pd.isna(s):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(s))
    return float(digits) if digits else np.nan


def raw_files(params: dict | None = None) -> list[str]:
    params = params or load_params()
    return sorted(glob.glob(str(ROOT / params["data"]["raw_glob"])))


def data_fingerprint(params: dict | None = None) -> str:
    h = hashlib.sha256()
    for f in raw_files(params):
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def load_raw(params: dict | None = None) -> pd.DataFrame:
    frames = []
    for f in raw_files(params):
        d = pd.read_csv(f)
        d["source"] = os.path.basename(f).replace("riyasewana_", "").replace(".csv", "")
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No raw listing files found - check data.raw_glob in params.yaml")
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame, params: dict | None = None) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    p = params or load_params()
    d, ref_year = p["data"], p["reference_year"]
    df = raw.copy()
    for a, b in [("Power Shutters", "Power Shutter"), ("Power Mirrors", "Power Mirror"),
                 ("Number of Cylinders", "Number of Cylinder")]:
        if b in df:
            df[a] = df[a].fillna(df[b])
            df = df.drop(columns=b)

    df["price"] = df["Price"].apply(parse_number)
    df["mileage_km"] = df["Mileage"].apply(parse_number)
    df["ad_date"] = pd.to_datetime(df["Ad Date"].str.split(",").str[0], format="%Y %b %d", errors="coerce")
    df["model_family"] = [FILE_MODEL.get(s) or family_from_text(s, m) for s, m in zip(df["source"], df["Model"], strict=True)]
    df["model_family"] = df["model_family"].fillna(df["Brand"] + " other")
    df["district"] = df["Location"].map(TOWN_DISTRICT)

    df["year"] = df["YOM"]
    df["age"] = (ref_year - df["year"]).clip(lower=0)
    lo, hi = d["engine_cc_range"]
    df["engine_cc"] = df["Engine (cc)"].where(df["Engine (cc)"].between(lo, hi))
    df.loc[df["Condition"].eq("Brand New") & df["mileage_km"].isna(), "mileage_km"] = 0
    df["mileage_km"] = df["mileage_km"].where(df["mileage_km"] <= d["max_mileage_km"])
    df["km_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    power_cols = ["Power Steering", "Power Shutters", "Power Mirrors"]
    df["power_options"] = df[power_cols].eq("Yes").sum(axis=1).where(df["Power Shutters"].notna())
    df["origin_country"] = df["Manufacturing Country"].where(
        df["Manufacturing Country"].isin(["Japan", "Germany", "Thailand", "India"]), "Other")
    df = df.rename(columns={"Brand": "brand", "Gear": "gear", "Fuel Type": "fuel_type",
                            "Condition": "condition", "Body Type": "body_type"})

    log = [("raw_listings", len(df))]
    df = df[df["price"].notna()]
    log.append(("drop_negotiable_no_price", len(df)))
    df = df[df["condition"] != "Antique"]
    log.append(("drop_antique", len(df)))
    df = df[df["price"].between(d["min_price"], d["max_price"])]
    log.append(("drop_impossible_prices", len(df)))
    df = df.drop_duplicates(subset=["brand", "Model", "year", "mileage_km", "price", "Location"])
    log.append(("drop_reposted_duplicates", len(df)))
    counts = df["model_family"].value_counts()
    rare = counts[counts < d["rare_family_min_count"]].index
    df.loc[df["model_family"].isin(rare), "model_family"] = df["brand"] + " other"

    data = df[FEATURES + ["price", "year", "ad_date", "Model", "Location", "URL"]].reset_index(drop=True)
    data[CAT] = data[CAT].astype(str)
    return data, log


def split(data: pd.DataFrame, params: dict | None = None):
    p = params or load_params()
    d = p["data"]
    holdout = data.sample(d["holdout_n"], random_state=d["holdout_seed"])
    model_data = data.drop(holdout.index)
    X, y = model_data[FEATURES], model_data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=d["test_size"], random_state=p["seed"])
    return X_train, X_test, y_train, y_test, holdout


def build_datasets(params: dict | None = None):
    params = params or load_params()
    data, log = clean(load_raw(params), params)
    return (data, log, *split(data, params))
