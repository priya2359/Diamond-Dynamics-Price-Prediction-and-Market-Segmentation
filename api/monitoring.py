# filename: api/monitoring.py
# purpose:  Prometheus metrics and KS-test drift detection for the Diamond
#           Dynamics FastAPI service. Drift is measured on Section-5-transformed
#           numeric features (carat, volume, depth, table) against a reference
#           sample from diamonds_processed.csv -- apples to apples.
# version:  1.0

# stdlib
import logging
import time
from collections import deque
from contextlib import contextmanager
from typing import Generator

# third-party
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from scipy.stats import ks_2samp

# internal
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint"],
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_VALUE = Histogram(
    "prediction_value_distribution",
    "Predicted price (USD) distribution",
    buckets=(100, 500, 1000, 2500, 5000, 10000, 20000, 50000),
)
DRIFT_SCORE = Gauge(
    "data_drift_score",
    "KS statistic for feature drift (0=no drift, 1=total drift)",
    ["feature"],
)
DRIFT_PVALUE = Gauge(
    "data_drift_pvalue",
    "KS test p-value for feature drift",
    ["feature"],
)
DRIFT_DETECTED = Gauge(
    "data_drift_detected",
    "1 if drift detected (p < alpha), 0 otherwise",
    ["feature"],
)

# ---------------------------------------------------------------------------
# Drift detection state
# ---------------------------------------------------------------------------
_drift_windows: dict[str, deque] = {
    feat: deque(maxlen=config.DRIFT_WINDOW_SIZE)
    for feat in config.DRIFT_NUMERIC_FEATURES
}

_reference_data: dict[str, np.ndarray] = {}


def load_drift_reference() -> None:
    """Load the reference sample at startup (called from lifespan)."""
    ref_path = config.MONITORING_ARTIFACTS_DIR / "drift_reference.csv"
    df = pd.read_csv(ref_path)
    for feat in config.DRIFT_NUMERIC_FEATURES:
        _reference_data[feat] = df[feat].to_numpy()
    logger.info(
        "Drift reference loaded: %d rows, features=%s",
        len(df),
        config.DRIFT_NUMERIC_FEATURES,
    )


def record_features(transformed_values: dict[str, float]) -> None:
    """Append a single observation's transformed feature values to the deques."""
    for feat in config.DRIFT_NUMERIC_FEATURES:
        if feat in transformed_values:
            _drift_windows[feat].append(transformed_values[feat])


def check_drift() -> dict[str, dict]:
    """
    Run KS-test per feature if the window is full. Updates Prometheus gauges
    and returns a summary dict.
    """
    results: dict[str, dict] = {}
    for feat in config.DRIFT_NUMERIC_FEATURES:
        window = _drift_windows[feat]
        if len(window) < config.DRIFT_WINDOW_SIZE:
            continue
        if feat not in _reference_data:
            continue

        stat, pvalue = ks_2samp(np.array(window), _reference_data[feat])
        is_drifted = bool(pvalue < config.DRIFT_KS_ALPHA)

        DRIFT_SCORE.labels(feature=feat).set(stat)
        DRIFT_PVALUE.labels(feature=feat).set(pvalue)
        DRIFT_DETECTED.labels(feature=feat).set(int(is_drifted))

        results[feat] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pvalue), 4),
            "is_drifted": is_drifted,
        }

        if is_drifted:
            logger.warning(
                "Drift detected on %s: KS=%.4f, p=%.4f < alpha=%.4f",
                feat, stat, pvalue, config.DRIFT_KS_ALPHA,
            )

    return results


@contextmanager
def track_request(endpoint: str) -> Generator[None, None, None]:
    """Context manager that counts and times a prediction request."""
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
