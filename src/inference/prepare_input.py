# filename: src/inference/prepare_input.py
# purpose:  Transform a single raw diamond (Streamlit/API input) into the
#           engineered, Section-5-transformed feature DataFrames expected by
#           artifacts/regression/best_model.pkl and artifacts/clustering/kmeans_model.pkl,
#           and run end-to-end price / segment predictions.
# version:  1.1

# stdlib
import logging

# third-party
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# internal
import config

logger = logging.getLogger(__name__)

# Feature orders mirror artifacts/regression/selected_features.json and
# artifacts/clustering/selected_features.json. Defined locally (not imported
# from src.models.*) so this serving-time module doesn't pull in TensorFlow.
NUMERIC_FEATURES = ["carat", "volume", "depth", "table"]
CATEGORICAL_FEATURES = ["carat_category", "clarity", "color", "cut"]
REGRESSION_FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLUSTERING_FEATURE_ORDER = NUMERIC_FEATURES + ["price_per_carat"] + CATEGORICAL_FEATURES


def _carat_category(carat: float) -> str:
    """Bucket carat into Light/Medium/Heavy per config.CARAT_CATEGORY_BINS/LABELS."""
    category = pd.cut(
        [carat],
        bins=config.CARAT_CATEGORY_BINS,
        labels=config.CARAT_CATEGORY_LABELS,
        right=False,
    )[0]
    if pd.isna(category):
        raise ValueError(
            f"carat={carat} falls outside category bins {config.CARAT_CATEGORY_BINS}"
        )
    return str(category)


def build_regression_input(
    carat: float,
    x: float,
    y: float,
    z: float,
    cut: str,
    color: str,
    clarity: str,
    table: float = config.TABLE_DEFAULT,
) -> pd.DataFrame:
    """
    Engineer + Section-5-transform a single raw diamond into the 8-feature
    DataFrame expected by best_model.pkl. depth = 200*z/(x+y) (the diamonds
    dataset's own definition of depth%); table defaults to its training median
    (config.TABLE_DEFAULT) -- both per the Section 10 input-handling decision.
    """
    if (x + y) == 0:
        raise ValueError(f"x + y = 0 (x={x}, y={y}); cannot compute depth percentage")
    volume = x * y * z
    depth = 200.0 * z / (x + y)

    row = {
        "carat": np.sqrt(carat),
        "volume": np.sqrt(volume),
        "depth": depth,
        "table": np.sqrt(table),
        "carat_category": _carat_category(carat),
        "clarity": clarity,
        "color": color,
        "cut": cut,
    }
    return pd.DataFrame([row])[REGRESSION_FEATURE_ORDER]


def build_clustering_input(
    carat: float,
    x: float,
    y: float,
    z: float,
    cut: str,
    color: str,
    clarity: str,
    price_usd: float,
    table: float = config.TABLE_DEFAULT,
) -> pd.DataFrame:
    """
    Engineer + Section-5-transform a single raw diamond into the 9-feature
    DataFrame expected by kmeans_model.pkl. price_usd (from predict_price)
    is used to derive price_per_carat -- price itself is not a clustering
    feature -- price itself is not a clustering feature.
    """
    if (x + y) == 0:
        raise ValueError(f"x + y = 0 (x={x}, y={y}); cannot compute depth percentage")
    if carat == 0:
        raise ValueError(f"carat must be > 0 for price_per_carat computation, got {carat}")
    volume = x * y * z
    depth = 200.0 * z / (x + y)
    price_per_carat = price_usd / carat

    row = {
        "carat": np.sqrt(carat),
        "volume": np.sqrt(volume),
        "depth": depth,
        "table": np.sqrt(table),
        "price_per_carat": np.sqrt(price_per_carat),
        "carat_category": _carat_category(carat),
        "clarity": clarity,
        "color": color,
        "cut": cut,
    }
    return pd.DataFrame([row])[CLUSTERING_FEATURE_ORDER]


def predict_price(
    pipeline: Pipeline,
    carat: float,
    x: float,
    y: float,
    z: float,
    cut: str,
    color: str,
    clarity: str,
    table: float = config.TABLE_DEFAULT,
    return_features: bool = False,
) -> dict:
    """Predict diamond price in USD and INR from raw physical/quality attributes."""
    df = build_regression_input(carat, x, y, z, cut, color, clarity, table)
    log_price_usd = pipeline.predict(df)[0]
    price_usd = float(np.expm1(log_price_usd))  # inverse of log1p -- NOT np.exp
    if not np.isfinite(price_usd) or price_usd < 0:
        raise ValueError(
            f"Model returned invalid price: {price_usd} (log_pred={log_price_usd})"
        )
    price_inr = price_usd * config.USD_TO_INR
    logger.info("Predicted price: $%.2f / Rs%.2f", price_usd, price_inr)
    result: dict = {"price_usd": price_usd, "price_inr": price_inr}
    if return_features:
        result["_features_df"] = df
    return result


def predict_segment(
    pipeline: Pipeline,
    cluster_profiles: dict,
    carat: float,
    x: float,
    y: float,
    z: float,
    cut: str,
    color: str,
    clarity: str,
    price_usd: float,
    table: float = config.TABLE_DEFAULT,
) -> dict:
    """
    Predict the market segment (cluster) for a diamond given its
    regression-predicted price_usd. Returns cluster id/name/profile plus an
    out-of-distribution flag based on centroid_distance_p95 (Section 9's
    deferred OOD check, closed out here).
    """
    df = build_clustering_input(carat, x, y, z, cut, color, clarity, price_usd, table)

    preprocessor = pipeline.named_steps["preprocessor"]
    kmeans = pipeline.named_steps["model"]

    x_scaled = preprocessor.transform(df)
    cluster_id = int(kmeans.predict(x_scaled)[0])

    clusters = cluster_profiles.get("clusters", {})
    if str(cluster_id) not in clusters:
        raise ValueError(
            f"Cluster {cluster_id} not found in cluster_profiles; "
            f"available: {list(clusters.keys())}"
        )

    centroid = kmeans.cluster_centers_[cluster_id]
    centroid_distance = float(np.linalg.norm(x_scaled[0] - centroid))

    profile = clusters[str(cluster_id)]
    centroid_distance_p95 = profile["centroid_distance_p95"]

    logger.info(
        "Predicted cluster %d (%s), centroid_distance=%.4f (p95=%.4f)",
        cluster_id, profile["cluster_name"], centroid_distance, centroid_distance_p95,
    )

    return {
        "cluster_id": cluster_id,
        "cluster_name": profile["cluster_name"],
        "profile": profile,
        "centroid_distance": centroid_distance,
        "centroid_distance_p95": centroid_distance_p95,
        "is_ood": centroid_distance > centroid_distance_p95,
    }
