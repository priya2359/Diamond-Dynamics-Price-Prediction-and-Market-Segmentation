# filename: config.py
# purpose:  Centralized configuration — paths, constants, hyperparameters, encoding maps
# version:  1.0

# stdlib
import os
from pathlib import Path

# third-party
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REGRESSION_ARTIFACTS_DIR = ARTIFACTS_DIR / "regression"
CLUSTERING_ARTIFACTS_DIR = ARTIFACTS_DIR / "clustering"
EDA_ARTIFACTS_DIR = ARTIFACTS_DIR / "eda"
MLFLOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "mlflow"

DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------
USD_TO_INR = float(os.getenv("USD_TO_INR", "83.5"))

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Ordinal encoding orders (FIXED — see CLAUDE.md "Exact Encoding Orders")
# Lower number = lower quality/value, higher number = higher quality/value.
# Wrong order silently corrupts model training — do not change without re-validating.
# ---------------------------------------------------------------------------
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# ---------------------------------------------------------------------------
# Carat category bins (per GUVI spec)
# ---------------------------------------------------------------------------
CARAT_CATEGORY_BINS = [0, 0.5, 1.5, float("inf")]
CARAT_CATEGORY_LABELS = ["Light", "Medium", "Heavy"]

# ---------------------------------------------------------------------------
# Outlier handling hyperparameters
# ---------------------------------------------------------------------------
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
# Physical constraints from gemological standards — values outside these
# ranges are measurement errors regardless of Z-score.
DEPTH_PHYSICAL_BOUNDS = (50.0, 75.0)   # depth % valid range
TABLE_PHYSICAL_BOUNDS = (50.0, 70.0)   # table % valid range

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_ARTIFACTS_DIR / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT_REGRESSION = "diamond_price_regression"

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Section 8 — Regression: train/val/test split
# 80% train_full / 20% test (random_state=RANDOM_STATE).
# train_full is further split 80/20 -> 64% train / 16% val (overall),
# val used only for XGBoost and ANN early stopping. Linear/DT/RF/KNN train
# on the full 80% train_full. All models evaluated on the same 20% test set.
# ---------------------------------------------------------------------------
VAL_SIZE = 0.2

# ---------------------------------------------------------------------------
# Section 8 — Regression model hyperparameters (all models, RANDOM_STATE fixed)
# ---------------------------------------------------------------------------
LINEAR_PARAMS: dict = {}  # fit_intercept=True (sklearn default)

DT_PARAMS = {
    "max_depth": 10,
    "min_samples_leaf": 20,
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

KNN_PARAMS = {
    "n_neighbors": 10,
    "weights": "distance",
    "metric": "euclidean",
}

# ---------------------------------------------------------------------------
# Section 8 — ANN (TensorFlow/Keras) architecture and training config
# ---------------------------------------------------------------------------
ANN_ARCHITECTURE = {
    "hidden_layers": [
        {"units": 64, "activation": "relu", "batch_norm": True, "dropout": 0.2},
        {"units": 32, "activation": "relu", "batch_norm": True, "dropout": 0.2},
    ],
    "output_activation": "linear",
}

ANN_TRAINING = {
    "learning_rate": 0.001,
    "loss": "mse",
    "metrics": ["mae"],
    "batch_size": 256,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 5,
}

# ---------------------------------------------------------------------------
# Section 9 -- Clustering (K-Means + DBSCAN comparison) hyperparameters
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_CLUSTERING = "diamond_market_segmentation"

K_RANGE = range(2, 11)

KMEANS_PARAMS = {
    "n_init": 10,
    "random_state": RANDOM_STATE,
}

SILHOUETTE_SAMPLE_SIZE = 5000

# DBSCAN comparison only (not the saved/serving model). min_samples = 2 * n_features
# (9 clustering features) per the standard rule of thumb; eps chosen via KneeLocator
# on the k-distance plot.
DBSCAN_MIN_SAMPLES = 18

# K-Means stability check: fit at the chosen K with several seeds, compare labelings
# via Adjusted Rand Index. Below ARI_STABILITY_THRESHOLD => prefer a smaller, more
# reproducible K even if its silhouette score is slightly lower.
KMEANS_STABILITY_SEEDS = [0, 7, 13, 21, 42]
ARI_STABILITY_THRESHOLD = 0.85

# Dynamic cluster naming: tiers are assigned RELATIVE to the other clusters in the
# same run (tertile rank of each cluster's mean among all cluster means), not on
# absolute thresholds -- so naming works for any K in K_RANGE.
CLUSTER_NAME_PRICE_TIERS = ["Affordable", "Mid-range", "Premium"]
CLUSTER_NAME_SIZE_TIERS = ["Compact", "Balanced", "Heavy"]

# ---------------------------------------------------------------------------
# Section 10 -- Streamlit input ranges/defaults
# Bounds and defaults from data/processed/diamonds_clean.csv .describe().
# `table` is defaulted (not collected) -- RF importance is negligible (Section 6),
# keeping Module 1/2 forms aligned with the GUVI spec inputs (carat, x, y, z,
# cut, color, clarity). `depth` is computed exactly as 200*z/(x+y) (the
# diamonds dataset's own definition of depth%), also not collected.
# ---------------------------------------------------------------------------
INPUT_FEATURE_RANGES = {
    "carat": {"min": 0.20, "max": 5.01, "default": 0.70, "step": 0.01},
    "x": {"min": 3.73, "max": 10.74, "default": 5.70, "step": 0.01},
    "y": {"min": 3.18, "max": 10.54, "default": 5.71, "step": 0.01},
    "z": {"min": 1.07, "max": 8.06, "default": 3.53, "step": 0.01},
}
TABLE_DEFAULT = 57.0  # training-set median table %

# ---------------------------------------------------------------------------
# Section 11 -- FastAPI serving (Phase 2A)
# Streamlit calls the FastAPI service over HTTP for all predictions.
# Default targets a locally-running `uvicorn api.main:app` process; in
# docker-compose this is overridden to http://fastapi:8000 (service name).
# ---------------------------------------------------------------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")

# ---------------------------------------------------------------------------
# Section 12 -- Drift detection (Phase 2B)
# KS-test on Section-5-transformed numeric features, comparing a saved
# reference sample (from diamonds_processed.csv) against a sliding window of
# live inference values. "Apples to apples" -- both sides are already
# sqrt/log-transformed.
# ---------------------------------------------------------------------------
DRIFT_REFERENCE_SAMPLE_SIZE = 2000
DRIFT_WINDOW_SIZE = 100
DRIFT_KS_ALPHA = 0.05
DRIFT_NUMERIC_FEATURES = ["carat", "volume", "depth", "table"]
MONITORING_ARTIFACTS_DIR = ARTIFACTS_DIR / "monitoring"
