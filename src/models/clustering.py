# filename: src/models/clustering.py
# purpose:  Section 9 -- Preprocessing pipeline, K-Means model selection
#           (elbow/silhouette/stability), DBSCAN eps detection, dynamic cluster
#           naming, and cluster profiling helpers for diamond market segmentation
# version:  1.0

# stdlib
import logging

# third-party
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# internal
import config
from src.features.encode import build_ordinal_encoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature spec -- from artifacts/clustering/selected_features.json (Section 6).
# price and price-derived columns other than price_per_carat are dropped before
# this point. CATEGORICAL_FEATURES order matches sorted(ORDINAL_FEATURE_MAP.keys())
# so it lines up with the category order returned by build_ordinal_encoder().
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["carat", "volume", "depth", "table", "price_per_carat"]
CATEGORICAL_FEATURES = ["carat_category", "clarity", "color", "cut"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Preprocessing pipeline -- same structure as Section 8's regression
# preprocessor (numeric passthrough + ordinal-encode -> StandardScaler), but
# fit on the clustering feature set (includes price_per_carat, excludes price).
# ---------------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    """ColumnTransformer (passthrough numeric + ordinal-encode categorical) -> StandardScaler."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", build_ordinal_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("encode", column_transformer),
        ("scale", StandardScaler()),
    ])


def build_full_pipeline(preprocessor: Pipeline, model) -> Pipeline:
    """Wrap an already-fitted preprocessor and an already-fitted clustering model."""
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# K selection -- elbow (inertia, full data) + silhouette (subsample, since
# silhouette_score is O(n^2) and 53,794 rows is too slow at full size).
# ---------------------------------------------------------------------------
def compute_elbow_silhouette(X: np.ndarray, k_range, sample_size: int, random_state: int) -> dict:
    """For each K in k_range: fit KMeans on full X, return inertia + silhouette (on a subsample)."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_sample = X[sample_idx]

    results: dict[int, dict] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, **config.KMEANS_PARAMS)
        km.fit(X)
        sil = silhouette_score(X_sample, km.predict(X_sample))
        results[k] = {"inertia": float(km.inertia_), "silhouette": float(sil)}
        logger.info("K=%d  inertia=%.2f  silhouette=%.4f", k, km.inertia_, sil)
    return results


# ---------------------------------------------------------------------------
# Stability check -- fit KMeans at the chosen K with several seeds and compare
# labelings via Adjusted Rand Index. Low mean ARI => clusters are seed-dependent
# and not reproducible in production.
# ---------------------------------------------------------------------------
def compute_stability_ari(X: np.ndarray, k: int, seeds: list[int]) -> dict:
    """Fit KMeans with several random seeds at a fixed K, compute pairwise ARI."""
    label_sets = []
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init=config.KMEANS_PARAMS["n_init"], random_state=seed)
        label_sets.append(km.fit_predict(X))

    pairwise_ari = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            pairwise_ari.append(float(adjusted_rand_score(label_sets[i], label_sets[j])))

    mean_ari = float(np.mean(pairwise_ari))
    return {
        "k": k,
        "seeds": seeds,
        "pairwise_ari": [round(a, 4) for a in pairwise_ari],
        "mean_ari": round(mean_ari, 4),
        "stable": mean_ari >= config.ARI_STABILITY_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# DBSCAN eps selection -- k-distance plot (k = min_samples) + automated knee
# detection via kneed.KneeLocator. DBSCAN itself is for comparison only (not
# the saved/serving model), so kneed is a requirements-dev.txt-only dependency
# and imported lazily here to avoid a hard import-time dependency for callers
# that only need build_preprocessor / compute_cluster_profiles in production.
# ---------------------------------------------------------------------------
def find_dbscan_eps(X: np.ndarray, min_samples: int) -> dict:
    """Return sorted k-distances plus the KneeLocator-suggested eps."""
    from kneed import KneeLocator

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    x = np.arange(len(k_distances))
    knee = KneeLocator(x, k_distances, curve="convex", direction="increasing")
    eps = float(k_distances[knee.knee]) if knee.knee is not None else float(np.median(k_distances))

    return {"k_distances": k_distances, "eps": eps, "knee_index": knee.knee}


# ---------------------------------------------------------------------------
# Dynamic cluster naming -- tiers are RELATIVE (tertile rank among the other
# clusters in the same run), so this works for any K in config.K_RANGE without
# forcing K=3 to match the illustrative 3-cluster example.
# ---------------------------------------------------------------------------
def tertile_label(value: float, all_values: list[float], labels: list[str]) -> str:
    """Rank `value` among `all_values` (ascending) and bucket into low/mid/high labels."""
    sorted_vals = sorted(all_values)
    rank = sorted_vals.index(value) / max(len(sorted_vals) - 1, 1)
    if rank < 1 / 3:
        return labels[0]
    if rank > 2 / 3:
        return labels[2]
    return labels[1]


def generate_cluster_name(
    avg_price_per_carat: float,
    avg_carat: float,
    all_price_per_carat: list[float],
    all_carat: list[float],
) -> str:
    """Combine a price tier and a size tier (each relative to other clusters) into a name."""
    price_tier = tertile_label(avg_price_per_carat, all_price_per_carat, config.CLUSTER_NAME_PRICE_TIERS)
    size_tier = tertile_label(avg_carat, all_carat, config.CLUSTER_NAME_SIZE_TIERS)
    return f"{price_tier} {size_tier} Diamonds"


# ---------------------------------------------------------------------------
# Cluster profiling -- per-cluster stats in interpretable units (carat,
# price_per_carat inverse-sqrt'd; price_inr already in raw INR per
# transform_params.json) plus the centroid-distance p95 used to flag
# out-of-distribution inference inputs (Phase 2 / Section 10 risk).
# ---------------------------------------------------------------------------
def compute_centroid_distances(X_scaled: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to its assigned cluster's centroid."""
    return np.linalg.norm(X_scaled - centroids[labels], axis=1)


def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    kmeans: KMeans,
) -> dict:
    """
    Build per-cluster profiles for naming and Streamlit display.

    df must contain the original (untransformed-where-noted) columns:
    carat (sqrt), price_per_carat (sqrt), price (log1p USD), price_inr (raw INR),
    cut, color, clarity -- aligned row-for-row with `labels` and `X_scaled`.
    """
    distances = compute_centroid_distances(X_scaled, labels, kmeans.cluster_centers_)

    raw: dict[int, dict] = {}
    for c in sorted(np.unique(labels)):
        mask = labels == c
        sub = df[mask]
        raw[int(c)] = {
            "count": int(mask.sum()),
            "avg_carat": float((sub["carat"] ** 2).mean()),
            "avg_price_per_carat_usd": float((sub["price_per_carat"] ** 2).mean()),
            "avg_price_usd": float(np.expm1(sub["price"]).mean()),
            "avg_price_inr": float(sub["price_inr"].mean()),
            "dominant_cut": str(sub["cut"].mode().iat[0]),
            "dominant_color": str(sub["color"].mode().iat[0]),
            "dominant_clarity": str(sub["clarity"].mode().iat[0]),
            "centroid_distance_p95": float(np.percentile(distances[mask], 95)),
        }

    all_price_per_carat = [v["avg_price_per_carat_usd"] for v in raw.values()]
    all_carat = [v["avg_carat"] for v in raw.values()]

    for c, stats in raw.items():
        stats["cluster_name"] = generate_cluster_name(
            stats["avg_price_per_carat_usd"], stats["avg_carat"], all_price_per_carat, all_carat
        )

    return raw
