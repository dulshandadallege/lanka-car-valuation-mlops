import math

import pytest

from src.features import FEATURES, TOWN_DISTRICT, family_from_text, make_row


@pytest.mark.parametrize("source,text,expected", [
    ("bmw_sedan_only", "520d M Sport", "5 Series"),
    ("bmw_sedan_only", "i8", "i8"),
    ("mercedes_benz_edited_cleaned", "C200 AMG", "C-Class"),
    ("mercedes_benz_edited_cleaned", "W210", "E-Class"),
    ("mercedes_benz_edited_cleaned", "CLA 180", "CLA"),
    ("audi_edited", "A3 Sline", "A3"),
    ("mazda_with_country", "3", "Axela"),
    ("nissan_cleaned", "Roox", "Dayz"),
    ("mitsubishi_with_country", "Unknown thing", None),
])
def test_family_from_text(source, text, expected):
    assert family_from_text(source, text) == expected


def test_town_mapping():
    assert TOWN_DISTRICT["Nugegoda"] == "Colombo"
    assert TOWN_DISTRICT["Negombo"] == "Gampaha"
    assert len(set(TOWN_DISTRICT.values())) == 24


def test_make_row():
    row = make_row({"brand": "Toyota", "model_family": "Aqua", "gear": "Automatic", "fuel_type": "Hybrid",
                    "condition": "Registered (Used)", "body_type": "Hatchback", "origin_country": "Japan",
                    "district": "Colombo", "year": 2016, "mileage_km": 100_000, "engine_cc": 1500,
                    "power_options": 3}, ref_year=2026)
    assert list(row.columns) == FEATURES
    assert row["age"].iloc[0] == 10 and row["km_per_year"].iloc[0] == 10_000
    brand_new = make_row({**row.iloc[0].to_dict(), "year": 2026, "mileage_km": 0, "engine_cc": None}, 2026)
    assert brand_new["km_per_year"].iloc[0] == 0 and math.isnan(brand_new["engine_cc"].iloc[0])
