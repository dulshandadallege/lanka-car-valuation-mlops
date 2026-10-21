import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import make_row as _make_row  # noqa: E402
from src.predict import load_bundle  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"

st.set_page_config(page_title="Sri Lanka Car Price Estimator", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def get_bundle():
    return load_bundle()


@st.cache_resource
def load_explainer(_bundle):
    inner = _bundle["model"].regressor_
    pre, rf = inner.named_steps["pre"], inner.named_steps["model"]
    cats = sorted(_bundle["cat"], key=len, reverse=True)

    def owner(name):
        kind, rest = name.split("__", 1)
        return rest if kind == "num" else next(c for c in cats if rest.startswith(c + "_"))

    groups = [owner(n) for n in pre.get_feature_names_out()]
    explainer = shap.TreeExplainer(rf)
    return pre, explainer, groups, float(np.ravel(explainer.expected_value)[0])


def explain(row: pd.DataFrame) -> tuple[pd.Series, float]:
    pre, explainer, groups, base = load_explainer(B)
    raw = explainer.shap_values(pre.transform(row))[0]
    return pd.Series(raw, index=groups).groupby(level=0).sum().reindex(FEATURES), base


@st.cache_data
def load_examples():
    p = ROOT / "data" / "new_listings_unseen.csv"
    if p.exists():
        return pd.read_csv(p)
    from src.data import build_datasets
    return build_datasets()[-1].reset_index(drop=True)


B = get_bundle()
pipe = B["model"]
FEATURES, CAT, NUM, REF_YEAR = B["features"], B["cat"], B["num"], B["ref_year"]
opts = B["options"]
VERSION = B.get("metadata", {}).get("version", "dev")


def make_row(v: dict) -> pd.DataFrame:
    return _make_row(v, REF_YEAR)


def predict(row: pd.DataFrame) -> float:
    return float(pipe.predict(row)[0])


st.sidebar.header("Car details")
examples = load_examples()
preset = {}
if examples is not None:
    labels = ["(enter my own car)"] + [
        f"{r.brand} {r.model_family} · {2026 - int(r.age)} · {int(r.mileage_km):,} km" for r in examples.itertuples()]
    pick = st.sidebar.selectbox("Or load an unseen test listing", labels)
    if pick != labels[0]:
        preset = examples.iloc[labels.index(pick) - 1].to_dict()


def idx(options, value, fallback=0):
    return options.index(value) if value in options else fallback


brand = st.sidebar.selectbox("Brand", list(B["brand_models"]), index=idx(list(B["brand_models"]), preset.get("brand"),
                             idx(list(B["brand_models"]), "Toyota")))
models = B["brand_models"][brand]
model_family = st.sidebar.selectbox("Model", models, index=idx(models, preset.get("model_family")))
d = B["model_defaults"].get(model_family, {})
c1, c2 = st.sidebar.columns(2)
year = c1.number_input("Year of manufacture", 1985, REF_YEAR,
                       int(REF_YEAR - preset["age"]) if preset else 2015)
mileage = c2.number_input("Mileage (km)", 0, 800_000, int(preset.get("mileage_km", 80_000) or 0), step=5_000)
engine_default = preset.get("engine_cc") if preset and not pd.isna(preset.get("engine_cc")) else d.get("engine_cc", 1500)
engine = c1.number_input("Engine (cc)", 0, 6000, int(engine_default or 0), step=50)
gear = c2.selectbox("Gear", opts["gear"], index=idx(opts["gear"], preset.get("gear", "Automatic")))
fuel = c1.selectbox("Fuel", opts["fuel_type"], index=idx(opts["fuel_type"], preset.get("fuel_type", d.get("fuel_type"))))
body = c2.selectbox("Body type", opts["body_type"], index=idx(opts["body_type"], preset.get("body_type", d.get("body_type"))))
condition = st.sidebar.selectbox("Condition", opts["condition"],
                                 index=idx(opts["condition"], preset.get("condition", "Registered (Used)")))
origin = st.sidebar.selectbox("Manufactured in", opts["origin_country"],
                              index=idx(opts["origin_country"], preset.get("origin_country", d.get("origin_country"))))
district = st.sidebar.selectbox("District", opts["district"], index=idx(opts["district"], preset.get("district", "Colombo")))
power = st.sidebar.slider("Power options (steering, shutters, mirrors)", 0, 3,
                          int(preset.get("power_options", 3)) if preset and not pd.isna(preset.get("power_options")) else 3)
asking = st.sidebar.number_input("Seller's asking price (Rs, optional)", 0, 150_000_000,
                                 int(preset.get("price", 0) or 0), step=50_000)

values = dict(brand=brand, model_family=model_family, year=year, mileage_km=mileage, engine_cc=engine, gear=gear,
              fuel_type=fuel, body_type=body, condition=condition, origin_country=origin, district=district,
              power_options=power)
row = make_row(values)
price = predict(row)
mape = B["test_mape"] / 100
low, high = price * (1 - mape), price * (1 + mape)

st.title("🚗 Sri Lanka Used-Car Fair-Price Estimator")
st.caption(f"{B['model_name']} {VERSION} trained on {B['trained_on']:,} riyasewana.com listings ({B['data_period']}). "
           f"Typical error on unseen test cars: ±{B['test_mape']:.1f}%.")

m1, m2, m3 = st.columns(3)
m1.metric("Estimated market price", f"Rs {price:,.0f}")
m2.metric("Fair range (± typical error)", f"Rs {low/1e6:,.2f}M – {high/1e6:,.2f}M")
if asking:
    diff = (asking / price - 1) * 100
    verdict = "🟢 Fair price" if abs(diff) <= B["test_mape"] else ("🔴 Looks overpriced" if diff > 0 else "🔵 Looks underpriced")
    m3.metric("Asking price vs estimate", f"{diff:+.1f}%", help="Compared with the model's estimate")
    m3.markdown(f"**{verdict}**")
else:
    m3.info("Enter an asking price to get a fair / over / under verdict.")

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Why this price? (SHAP)")
    with st.spinner("Computing SHAP explanation…"):
        contrib, base_value = explain(row)
    shown = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])[::-1]
    labels = [f"{f} = {row[f].iloc[0]:,.0f}" if f in NUM and pd.notna(row[f].iloc[0]) else f"{f} = {row[f].iloc[0]}"
              for f in shown.index]
    pct = (np.exp(shown.values) - 1) * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(labels, pct, color=[BLUE if v > 0 else ORANGE for v in pct], height=0.6)
    ax.axvline(0, color=GREY, lw=0.8)
    span = max(abs(pct).max(), 5)
    ax.set_xlim(min(pct.min(), 0) - 0.25 * span, max(pct.max(), 0) + 0.25 * span)
    for i, v in enumerate(pct):
        ax.text(v, i, f" {v:+.0f}% ", va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax.set_xlabel("Effect on price vs an average listing (%)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    st.caption(f"Average listing (SHAP base value): Rs {np.exp(base_value):,.0f}. Blue pushes the price up, "
               "orange pulls it down. Effects multiply, so they don't add up exactly in %.")

with right:
    st.subheader("What if? (same car, one change)")
    what = st.radio("Vary", ["Mileage", "Year of manufacture"], horizontal=True)
    if what == "Mileage":
        xs = np.linspace(0, 250_000, 26)
        rows = pd.concat([make_row({**values, "mileage_km": x}) for x in xs])
        xlab, xv, cur = "Mileage ('000 km)", xs / 1000, mileage / 1000
    else:
        xs = np.arange(max(1990, year - 12), REF_YEAR + 1)
        rows = pd.concat([make_row({**values, "year": int(x)}) for x in xs])
        xlab, xv, cur = "Year of manufacture", xs, year
    ys = pipe.predict(rows) / 1e6
    fig2, ax2 = plt.subplots(figsize=(5.6, 3.6))
    ax2.plot(xv, ys, color=BLUE, lw=2)
    ax2.axvline(cur, color=GREY, ls="--", lw=1)
    ax2.set(xlabel=xlab, ylabel="Predicted price (Rs. M)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.25)
    st.pyplot(fig2, width="stretch")
    st.caption("Tree ensembles predict in steps, so the curve can wobble when one feature changes on its own. "
               f"Treat bumps smaller than the typical error (±{B['test_mape']:.0f}%) as noise.")

with st.expander("Model input row"):
    st.dataframe(row, width="stretch")
st.caption("Educational project for IT5514 Applied Machine Learning (University of Moratuwa). "
           "Estimates reflect asking prices on classified ads, not final sale prices.")
