# Diamond Dynamics: Price Prediction & Market Segmentation

> **53,794 diamonds · 7 input features · 6 ML models · XGBoost champion (R²=0.9879) · FastAPI + Streamlit deployed on Render**

![CI](https://github.com/priya2359/Diamond-Dynamics-Price-Prediction-and-Market-Segmentation/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

Predict diamond prices in USD and INR, and classify diamonds into market segments using only the 4 Cs and physical dimensions. Built as a production ML system covering the complete lifecycle: raw data → cleaning → feature engineering → 6 trained models → async REST API → interactive Streamlit dashboard.

**Engineering highlights:** dual-pipeline design (regression + clustering share preprocessing) · Streamlit as thin HTTP client (no sklearn/joblib in UI image) · KS-test drift detection on live inference · SHAP explainability in log-space · OOD detection via centroid distance p95 · 78 automated tests · GitHub Actions CI

**Live demo:** [Streamlit App](https://diamond-dynamics-app.onrender.com) · [FastAPI Docs](https://diamond-dynamics-api.onrender.com/docs)

---

## What This Project Does

| Layer | What's Built |
|-------|-------------|
| **Data Pipeline** | 8-notebook workflow: cleaning → EDA → preprocessing → feature engineering → encoding → regression → clustering → inference |
| **Cleaning** | Ratio-based decimal error detection (dim > 3× sibling avg → divide by 10); zero→NaN conversion; regression-based imputation for missing x/y/z |
| **Feature Engineering** | 5 derived features: Volume (x·y·z), Price-per-Carat, Dimension Ratio, Carat Category (Light/Medium/Heavy), USD→INR conversion |
| **Preprocessing** | IQR + Z-Score outlier capping with physical bounds; log1p transform on price; sqrt transform on carat/volume/table |
| **Encoding** | OrdinalEncoder for cut/color/clarity/carat_category with gemologically correct ordering (Fair→Ideal, J→D, I1→IF) |
| **Regression** | 6 models: Linear Regression, Decision Tree, KNN, Random Forest, XGBoost, ANN (TF/Keras) · XGBoost champion (R²=0.9879) |
| **Clustering** | K-Means (K=2, silhouette=0.265) + DBSCAN comparison · Elbow + silhouette K-selection · ARI stability check across 5 seeds |
| **API** | FastAPI with Pydantic validation, error counters, CORS, request-ID tracing, rate limiting (60 req/min), KS-test drift endpoint |
| **Dashboard** | Streamlit app with 3 pages: Price Prediction, Market Segment (with OOD flag), Visual Insights (PCA scatter, live Plotly charts) |
| **Monitoring** | Prometheus metrics (request count, latency, price distribution, drift score) + Grafana dashboards + Prometheus alert rules |
| **Tests** | 78 automated tests across 7 modules · 5-layer ordering (utils → clean → preprocess → features → inference → API) · GitHub Actions CI |

---

## Architecture

```
                    ┌──────────────┐
                    │  Streamlit   │  (HTTP client — no sklearn/joblib)
                    │  :8501       │
                    └──────┬───────┘
                           │ POST /v1/predict/{price,segment}
                           ▼
                    ┌──────────────┐     ┌──────────────┐
                    │   FastAPI    │────▶│  Prometheus  │
                    │   :8000      │     │  :9090       │
                    │  /metrics    │     └──────┬───────┘
                    └──────────────┘            │
                           │                    ▼
                    loads .pkl at        ┌──────────────┐
                    startup              │   Grafana    │
                           │             │   :3000      │
                    ┌──────┴───────┐     └──────────────┘
                    │  artifacts/  │
                    │  best_model  │ ← Pipeline A: preprocessor + XGBoost
                    │  kmeans_model│ ← Pipeline B: preprocessor + KMeans
                    └──────────────┘
```

**Dual-Pipeline Design:**
- **Pipeline A (Regression):** 7 raw inputs → feature engineering → encode → scale → XGBoost → price (USD/INR)
- **Pipeline B (Clustering):** same preprocessed data → drop price → encode → scale → K-Means → market segment + OOD flag

---

## ML Models

### Regression — XGBoost *(Champion — R² = 0.9879)*
- Full sklearn `Pipeline` (ColumnTransformer + XGBoost) saved as `best_model.pkl`
- Trained on 80% split; XGBoost used 64%/16% train/val carve-out for early stopping only
- 5-fold cross-validation R² = 0.9912 ± 0.0003
- All hyperparameters in `config.py` — nothing hardcoded inline

### Clustering — K-Means *(K = 2)*
- K selected by silhouette score (K=2..10); ARI stability = 0.9924 across 5 random seeds
- Two segments: **Affordable Compact Diamonds** (avg $1,159 · avg 0.42ct) and **Premium Heavy Diamonds** (avg $6,194 · avg 1.15ct)
- OOD detection: `centroid_distance > p95` (3.31 / 4.10) triggers warning flag

---

## Results

| Model | MAE (USD) | RMSE (USD) | R² | MAPE (%) |
|-------|-----------|------------|-----|----------|
| **XGBoost** | **$212** | **$372** | **0.9879** | **5.88** |
| Random Forest | $220 | $398 | 0.9861 | 6.21 |
| Decision Tree | $265 | $467 | 0.9809 | 7.31 |
| KNN | $291 | $503 | 0.9778 | 8.23 |
| ANN (Keras) | $327 | $548 | 0.9737 | 8.88 |
| Linear Regression | $830 | $1806 | 0.7140 | 32.15 |

Cross-validation (5-fold, XGBoost): R² = 0.9912 ± 0.0003

---

## Production Readiness

| Category | What's Implemented |
|----------|-------------------|
| **Input Validation** | Pydantic `DiamondInput` validates numeric ranges (ge/le) and category membership; case normalisation for cut/color/clarity before OrdinalEncoder |
| **Error Handling** | `ValueError`/`KeyError` → 400; unexpected exceptions → 500; `ERROR_COUNT` Prometheus counter per endpoint + error type |
| **Observability** | Prometheus: request count, latency histogram (p50/p95/p99), price distribution, drift KS statistic per feature · Grafana 6-panel dashboard |
| **Drift Detection** | KS-test on sliding deque (100 predictions) vs. 2000-row reference sample; `/v1/drift` endpoint; Prometheus alert rule fires at p < 0.05 |
| **Request Tracing** | UUID `X-Request-ID` injected per request via `RequestIDMiddleware` |
| **Rate Limiting** | SlowAPI 60 req/min per IP; disabled in test suite via `TESTING` env var (set at `Limiter` init time) |
| **Graceful Degradation** | `load_drift_reference()` wrapped in try/except — app starts and serves predictions even if monitoring artifact is missing |
| **Reproducibility** | `random_state=42` throughout; all dependencies pinned to exact versions in 4 split requirements files |
| **Tests** | 78 automated tests; 5-layer dependency ordering ensures failures are instantly locatable by layer |
| **CI/CD** | GitHub Actions: flake8 lint → mypy type-check → pytest (78 tests) → docker compose build |

---

## Key Design Decisions

- **Log-transform price, then expm1 at serving time:** The model is trained on `log1p(price)` (right-skewed raw price → near-normal). Predictions are inverted via `np.expm1()` before display. SHAP values are in log-space — displayed as **percentage contributions** (`shap_val / Σ|shap_vals| × 100`) rather than raw values, since `expm1` is nonlinear and can't be applied per-feature.

- **Dropped x, y, z raw dimensions:** VIF > 400 for all three (r(x, volume) = 0.9973). Volume = x·y·z absorbs all spatial information. Dropping raw dimensions eliminates extreme multicollinearity without losing information, and reduces the regression feature set from 11 to 8.

- **price_per_carat: banned in regression, kept in clustering:** In regression, `price / carat` directly encodes the target (leakage). In clustering, `price` itself is dropped (target leakage), but `price_per_carat` captures per-unit value-density — a legitimate market signal once the absolute price is excluded.

- **Regression-based imputation for zero dimensions:** 20 diamonds have `x`/`y`/`z = 0`. Since `carat → dimension` correlation is r = 0.95–0.98 (R² ≥ 0.95), linear regression gives accurate imputation. Simple median would be wrong — a 0.3 ct and 2.5 ct stone have fundamentally different correct dimensions.

- **Decimal error detection by ratio:** 3 diamonds have one dimension ~10× larger than siblings (e.g. y = 58.9 instead of 5.89). Detected by flagging `dim > 3× sibling avg`, corrected by dividing by 10. No manual inspection needed.

- **Streamlit as thin HTTP client:** No sklearn/joblib/TensorFlow in the Streamlit Docker image. All inference calls `POST /v1/predict/{price,segment}` on the FastAPI service. This removes ~300 MB from the UI image and keeps all feature engineering logic in a single source of truth (`src/inference/prepare_input.py`).

- **depth computed, table defaulted:** `depth = 200·z / (x + y)` is the dataset's own definition — no extra user input needed. `table` has negligible RF importance and defaults to the training median (57.0). This keeps the prediction form at exactly 7 inputs.

- **ARI stability check before accepting K:** Before committing to K=2, K-Means is refit with 5 different seeds and pairwise Adjusted Rand Index is computed. Mean ARI = 0.9924 confirms the clustering is stable, not a lucky seed.

---

## API Reference

**Base URL (live):** `https://diamond-dynamics-api.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/predict/price` | Predict price in USD + INR |
| POST | `/v1/predict/segment` | Predict market segment + price + OOD flag |
| GET | `/v1/health` | Service health + model load status |
| GET | `/v1/drift` | KS-test drift status per feature |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Interactive Swagger UI |

**Request body (both prediction endpoints):**

| Field | Type | Valid range |
|-------|------|------------|
| `carat` | float | 0.20 – 5.01 |
| `x` | float | 3.73 – 10.74 mm |
| `y` | float | 3.18 – 10.54 mm |
| `z` | float | 1.07 – 8.06 mm |
| `cut` | string | Fair / Good / Very Good / Premium / Ideal |
| `color` | string | D / E / F / G / H / I / J |
| `clarity` | string | IF / VVS1 / VVS2 / VS1 / VS2 / SI1 / SI2 / I1 |

---

## Quick Start

### Option A — Use the live deployment (no setup needed)

```bash
curl -X POST https://diamond-dynamics-api.onrender.com/v1/predict/price \
  -H "Content-Type: application/json" \
  -d '{
    "carat": 0.7,
    "x": 5.70,
    "y": 5.71,
    "z": 3.53,
    "cut": "Ideal",
    "color": "E",
    "clarity": "VS2"
  }'
```

> **Note:** First request on Render free tier may take 30–60s (cold start — service wakes from sleep).

---

### Option B — Local development

**Prerequisites:** Python 3.11.9, git

```bash
git clone https://github.com/priya2359/Diamond-Dynamics-Price-Prediction-and-Market-Segmentation.git
cd Diamond-Dynamics-Price-Prediction-and-Market-Segmentation

python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/ -v               # 78 tests

# Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger UI → http://localhost:8000/docs

# Start Streamlit (separate terminal)
streamlit run streamlit_app/Home.py
# Dashboard → http://localhost:8501
```

---

### Option C — Docker Compose (full 4-service stack)

```bash
docker compose up --build
```

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000 | REST API + Swagger at /docs |
| Streamlit | http://localhost:8501 | Interactive UI |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

---

### Re-run the full ML pipeline

Requires `data/raw/diamonds.csv`:

```bash
python notebooks/02_data_cleaning.py
python notebooks/03_eda.py
python notebooks/05_preprocessing.py
python notebooks/06_feature_engineering.py
python notebooks/07_encoding.py
python notebooks/08_regression.py
python notebooks/09_clustering.py
```

---

## Project Structure

```
├── api/
│   ├── main.py                 # Endpoints, middleware, rate limiting
│   ├── schemas.py              # Pydantic request/response models
│   ├── monitoring.py           # Prometheus metrics + KS-test drift
│   └── Dockerfile
├── src/
│   ├── data/                   # clean.py, preprocess.py
│   ├── features/               # engineer.py, encode.py
│   ├── models/                 # regression.py, clustering.py
│   ├── inference/              # prepare_input.py (single serving source of truth)
│   └── utils/                  # helpers.py (NumpyEncoder, save_json, save_figure)
├── streamlit_app/
│   ├── Home.py
│   ├── pages/                  # 1_Price_Prediction, 2_Market_Segment, 3_Visual_Insights
│   ├── utils.py                # HTTP client helpers (calls FastAPI, no sklearn)
│   └── Dockerfile
├── notebooks/                  # Sections 2–9 (sequential .py notebooks)
├── tests/                      # 78 pytest tests across 7 modules
│   ├── conftest.py             # Shared fixtures — single source of truth
│   ├── test_utils.py
│   ├── test_data_clean.py
│   ├── test_data_preprocess.py
│   ├── test_data_quality.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_model_golden.py
│   └── test_api.py
├── artifacts/
│   ├── regression/             # best_model.pkl, metrics.json, error_stratification.json
│   ├── clustering/             # kmeans_model.pkl, cluster_profiles.json
│   ├── preprocessing/          # encoding_map.json
│   └── monitoring/             # drift_reference.csv
├── monitoring/
│   ├── prometheus/             # prometheus.yml, alert_rules.yml
│   └── grafana/                # provisioning (datasource + dashboard)
├── docs/
│   ├── figures/                # EDA + model figures (03_xx, 08_xx, 09_xx)
│   ├── API.md                  # Endpoint docs with curl examples
│   ├── MODEL_CARD.md           # Model details, limitations, ethical considerations
│   └── RUNBOOK.md              # Monitoring, troubleshooting, retraining guide
├── config.py                   # Centralised configuration (paths, hyperparams, encoding orders)
├── docker-compose.yml          # 4-service orchestration
├── render.yaml                 # Render deployment blueprint
└── .github/workflows/ci.yml    # CI: lint → type-check → test → docker build
```

---

## Tech Stack

```
Python 3.11.9        scikit-learn 1.9.0   XGBoost 3.2.0
TensorFlow 2.21.0    SHAP                 MLflow
FastAPI 0.136.3      Pydantic 2.13.4      SlowAPI 0.1.10
Streamlit 1.58.0     Plotly 6.8.0         Requests
Prometheus           Grafana              scipy 1.17.1
Docker Compose       pytest 9.0.3         mypy 2.1.0
```

---

## Dataset

| Source | Rows | Features | Domain |
|--------|------|----------|--------|
| [ggplot2 `diamonds`](https://ggplot2.tidyverse.org/reference/diamonds.html) | 53,794 | 10 raw → 8 selected | Gemology |

**Features:** carat, cut (Fair→Ideal), color (J→D), clarity (I1→IF), depth %, table %, x/y/z dimensions

---

## Documentation

- [API Reference](docs/API.md) — Endpoint docs with curl examples and error formats
- [Model Card](docs/MODEL_CARD.md) — Model details, training data, limitations, ethical considerations
- [Operations Runbook](docs/RUNBOOK.md) — Monitoring, drift response, retraining, troubleshooting

---

## Author

**Priya Neha** · [GitHub](https://github.com/priya2359)
