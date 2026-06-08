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
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_ARTIFACTS_DIR / 'mlflow.db'}"
)
