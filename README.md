# Diamond Dynamics: Price Prediction & Market Segmentation

A production-grade ML platform for diamond price prediction (XGBoost regression, R2=0.9879) and market segmentation (K-Means clustering, K=2) built on the diamonds dataset (53,794 records). Served via FastAPI + Streamlit with Prometheus/Grafana monitoring and Docker Compose orchestration.

## Features

- **Price Prediction** — XGBoost regression pipeline predicts diamond price in USD/INR from 7 physical and quality attributes (carat, x, y, z, cut, color, clarity)
- **Market Segmentation** — K-Means clustering assigns diamonds to market segments (Affordable Compact / Premium Heavy) with OOD detection
- **REST API** — FastAPI service with Pydantic validation, structured error responses, CORS, and request tracing
- **Interactive UI** — Streamlit multi-page app (Price Prediction, Market Segment, Visual Insights)
- **Monitoring** — Prometheus metrics (request count, latency, price distribution, KS-test drift detection) + Grafana dashboards
- **Explainability** — SHAP analysis for per-feature contribution to price predictions

## Quick Start

### Prerequisites
- Python 3.11.9
- Docker & Docker Compose (for containerized deployment)

### Local Development

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start Streamlit (in another terminal)
streamlit run streamlit_app/Home.py
```

### Docker Compose (Full Stack)

```bash
docker compose up --build
```

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000 | REST API + Swagger docs at /docs |
| Streamlit | http://localhost:8501 | Interactive UI |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

## Architecture

```
                    ┌──────────────┐
                    │  Streamlit   │ (HTTP client, no sklearn)
                    │  :8501       │
                    └──────┬───────┘
                           │ POST /v1/predict/{price,segment}
                           v
                    ┌──────────────┐     ┌──────────────┐
                    │   FastAPI    │────>│  Prometheus   │
                    │   :8000      │     │  :9090        │
                    │  /metrics    │     └──────┬───────┘
                    └──────────────┘            │
                           │                    v
                    loads .pkl at        ┌──────────────┐
                    startup              │   Grafana    │
                           │             │   :3000      │
                    ┌──────┴───────┐     └──────────────┘
                    │  artifacts/  │
                    │  best_model  │
                    │  kmeans_model│
                    └──────────────┘
```

**Dual-Pipeline Design:**
- **Pipeline A (Regression):** Raw data -> Preprocess -> Feature Engineer -> Encode -> Scale -> XGBoost -> Price (USD/INR)
- **Pipeline B (Clustering):** Same preprocessed data -> Drop price -> Encode -> Scale -> K-Means -> Market Segment

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/predict/price` | Predict diamond price (USD + INR) |
| POST | `/v1/predict/segment` | Predict market segment + price |
| GET | `/v1/health` | Service health check |
| GET | `/v1/drift` | KS-test drift detection status |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI (auto-generated) |

See [docs/API.md](docs/API.md) for request/response examples.

## Model Performance

| Model | MAE (USD) | RMSE (USD) | R2 | MAPE (%) |
|-------|-----------|------------|-----|----------|
| **XGBoost** | **$212** | **$372** | **0.9879** | **5.88** |
| Random Forest | $220 | $398 | 0.9861 | 6.21 |
| Decision Tree | $265 | $467 | 0.9809 | 7.31 |
| KNN | $291 | $503 | 0.9778 | 8.23 |
| ANN (Keras) | $327 | $548 | 0.9737 | 8.88 |
| Linear Regression | $830 | $1806 | 0.7140 | 32.15 |

Cross-validation (5-fold): XGBoost R2 = 0.9912 +/- 0.0003

## Project Structure

```
Diamond Price Prediction And Market Segmentation/
├── api/                        # FastAPI serving layer
│   ├── main.py                 # Endpoints + middleware
│   ├── schemas.py              # Pydantic models
│   ├── monitoring.py           # Prometheus + drift detection
│   └── Dockerfile
├── src/                        # Core ML pipeline
│   ├── data/                   # clean.py, preprocess.py
│   ├── features/               # engineer.py, encode.py
│   ├── models/                 # regression.py, clustering.py
│   ├── inference/              # prepare_input.py
│   └── utils/                  # helpers.py
├── streamlit_app/              # Streamlit UI
│   ├── Home.py
│   ├── pages/                  # Price, Segment, Visual Insights
│   ├── utils.py                # HTTP client helpers
│   └── Dockerfile
├── notebooks/                  # Training notebooks (Section 2-9)
├── tests/                      # pytest test suite (78 tests)
├── artifacts/                  # Saved models, metrics, profiles
├── monitoring/                 # Prometheus + Grafana configs
├── config.py                   # Centralized configuration
├── docker-compose.yml          # 4-service orchestration
└── docs/                       # Figures, MODEL_CARD, API docs
```

## Tech Stack

Python 3.11.9 | scikit-learn | XGBoost | TensorFlow/Keras | SHAP | FastAPI | Streamlit | Prometheus | Grafana | Docker | MLflow | pytest

## Documentation

- [API Reference](docs/API.md) — Endpoint docs with curl examples
- [Model Card](docs/MODEL_CARD.md) — Model details, limitations, ethical considerations
- [Operations Runbook](docs/RUNBOOK.md) — Monitoring, troubleshooting, retraining
