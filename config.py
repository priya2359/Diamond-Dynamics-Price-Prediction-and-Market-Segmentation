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
