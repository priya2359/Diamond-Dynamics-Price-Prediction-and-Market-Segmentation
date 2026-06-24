# filename: tests/conftest.py
# purpose:  Shared fixtures for all test modules -- single source of truth
# version:  1.0

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    return PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="session")
def processed_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def valid_diamond() -> dict:
    """Canonical test diamond used across all test modules."""
    return {
        "carat": 0.7,
        "x": 5.7,
        "y": 5.71,
        "z": 3.53,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS2",
    }


@pytest.fixture(scope="module")
def regression_pipeline(artifacts_dir):
    import joblib
    return joblib.load(artifacts_dir / "regression" / "best_model.pkl")


@pytest.fixture(scope="module")
def clustering_pipeline(artifacts_dir):
    import joblib
    return joblib.load(artifacts_dir / "clustering" / "kmeans_model.pkl")


@pytest.fixture(scope="module")
def cluster_profiles(artifacts_dir):
    import json
    with open(artifacts_dir / "clustering" / "cluster_profiles.json") as f:
        return json.load(f)
