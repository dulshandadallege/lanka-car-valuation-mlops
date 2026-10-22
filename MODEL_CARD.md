# Model card: Sri Lanka used-car fair-price estimator

| | |
|---|---|
| **Model** | scikit-learn `RandomForestRegressor` (500 trees, `max_features=0.5`, no depth limit) inside a preprocessing pipeline, trained on log(price) |
| **Version** | Set by the release tag (e.g. `v1.0.0`); stored in the model file's `metadata` together with a SHA-256 fingerprint of the training data |
| **Task** | Predict the market **asking** price (Rs.) of a used car from its listing details |
| **Live demo** | https://lanka-car-valuation.streamlit.app |
| **Owner** | Dulshan, MSc in AI (B19), University of Moratuwa, IT5514 Applied Machine Learning |

## Intended use
- A buyer or seller checks whether an asking price looks fair, and why (a SHAP explanation per car).
- **Not** for lending, insurance or legal valuation decisions. It estimates what sellers ask, not what cars sell for.

## Data
- 6,436 riyasewana.com ads (May–Aug 2026), 11 brands, 30 popular models. After cleaning, **4,912** listings remain.
- Removed: ads with no price ("Negotiable", 22%), antiques, impossible prices and re-posted duplicates.
- Split: 10 listings held out as unseen "new ads", then 80/20 train/test (seed 42).
- 13 features: brand, model family, gear, fuel, condition, body, country of manufacture, district, age, mileage,
  km per year, engine cc, power options.

## Performance (test set, 981 listings)
| MAE | RMSE | MAPE | R² | Priced within ±10% |
|---|---|---|---|---|
| Rs 764,419 | Rs 1,802,582 | 8.07% | 0.917 | 78.9% |

The median error on the 10 hidden listings is 8.4%. A linear-regression baseline reaches 11.1% MAPE, and a
stacking ensemble 8.0%. The single Random Forest was chosen for consistency (best 5-fold CV MAE) and
exact explainability.

**Release quality gate** (`params.yaml`): test MAPE ≤ 10%, R² ≥ 0.88, hold-out median error ≤ 15%.

## Explainability
Global SHAP importance: age (≈35% typical effect) ≫ country of manufacture (European premium) > body type >
gear > engine size > mileage. District, fuel type and listed condition each contribute under 1%.

## Limitations and risks
- Asking prices, not transaction prices. Excluding "Negotiable" ads may bias the model towards the priced segment.
- About three months of data from one site, in a market still adjusting after the 2025 import reopening.
  **Expect drift.** The weekly PSI monitor flags it and opens a retraining issue.
- Accident history, number of owners and service records are not observed.
- Random Forest predictions are piecewise-constant, so a single-feature what-if curve can wobble within the error band.
