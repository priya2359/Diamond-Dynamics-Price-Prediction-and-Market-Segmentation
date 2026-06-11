# filename: notebooks/09_clustering.py
# purpose:  Section 9 -- Market segmentation: K-Means (primary, with elbow +
#           silhouette K-selection and ARI stability check), DBSCAN (comparison),
#           PCA 2D/3D visualization, dynamic cluster naming, MLflow logging,
#           save clustering Pipeline + profiles.
# version:  1.0

# %% [markdown]
# # Section 9 -- Clustering / Market Segmentation
#
# Input: data/processed/diamonds_processed.csv, clustering feature set from
# artifacts/clustering/selected_features.json (price and price-derived columns
# other than price_per_carat are dropped before fitting -- price_per_carat is
# safe here because price itself is excluded).
#
# ## Pipeline design
# - Preprocessor: ColumnTransformer (numeric passthrough [carat, volume, depth,
#   table, price_per_carat] + OrdinalEncoder [carat_category, clarity, color, cut],
#   fixed categories from config.py) -> StandardScaler. Fit on the FULL dataset --
#   no train/test split for clustering (locked architecture decision).
# - K selection: elbow (inertia) + silhouette score over K=2..10 (config.K_RANGE).
#   Silhouette computed on a 5,000-row subsample (silhouette_score is O(n^2)).
# - Stability check: at the silhouette-best K, refit KMeans with 5 different seeds
#   (config.KMEANS_STABILITY_SEEDS) and compute pairwise Adjusted Rand Index. If
#   mean ARI < config.ARI_STABILITY_THRESHOLD (0.85), step down to K-1 and retry --
#   stable, reproducible clusters are preferred over marginally-higher-silhouette
#   but seed-dependent ones.
# - DBSCAN: comparison only (NOT the saved model). eps chosen via KneeLocator on
#   the k-distance plot (k = min_samples = 2 * n_features = 18).
# - Hierarchical clustering: SKIPPED. Full agglomerative clustering is O(n^2)
#   memory/time at 53,794 rows -- infeasible. A sample-based dendrogram (e.g. 2,000
#   rows = 3.7% of data) would reflect the sample, not the dataset, and is not
#   reliable evidence. DBSCAN already satisfies the GUVI doc's optional "try
#   different clustering techniques" requirement.
# - Cluster naming: dynamic, NOT forced to K=3. Each cluster gets a price tier
#   (Affordable/Mid-range/Premium) and a size tier (Compact/Balanced/Heavy), each
#   a tertile RANK relative to the OTHER clusters in this run -- works for any K.
#   See src/models/clustering.py generate_cluster_name().
# - PCA(2) and PCA(3) for visualization only -- clustering itself runs on the
#   scaled 9-feature space, never on PCA components.
#
# ## Risk documented (data computed here, runtime check deferred to Section 10 /
# ## Phase 2 FastAPI)
# KMeans.predict() always returns a label, even for an out-of-distribution input
# (e.g. an 8-carat stone when training data maxed out near 5 carat) -- it just
# picks the "least wrong" centroid. cluster_profiles.json saves
# centroid_distance_p95 per cluster (95th percentile of training-point distances
# to their assigned centroid). Section 10/Phase 2 should compare a new point's
# distance to its assigned centroid against this threshold and surface a
# "low-confidence cluster assignment" warning if it's exceeded.

# %% Setup
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import config
from src.models.clustering import (
    FEATURE_ORDER,
    build_preprocessor,
    build_full_pipeline,
    compute_elbow_silhouette,
    compute_stability_ari,
    find_dbscan_eps,
    compute_cluster_profiles,
)
from src.utils.helpers import save_figure, save_json

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR

df = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_processed.csv")
print("Loaded:", df.shape)
print("Clustering features:", FEATURE_ORDER)

# %% [markdown]
# ## 1. Preprocessing (full dataset, no split)

# %% Fit preprocessor
preprocessor = build_preprocessor()
X = df[FEATURE_ORDER]
X_scaled = preprocessor.fit_transform(X)
print("Scaled feature matrix:", X_scaled.shape)

# %% [markdown]
# ## 2. MLflow Setup

# %% MLflow
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_CLUSTERING)
print("MLflow tracking URI:", config.MLFLOW_TRACKING_URI)

# %% [markdown]
# ## 3. K Selection -- Elbow + Silhouette

# %% Elbow + silhouette sweep
k_results = compute_elbow_silhouette(
    X_scaled, config.K_RANGE, config.SILHOUETTE_SAMPLE_SIZE, config.RANDOM_STATE
)

with mlflow.start_run(run_name="k_selection"):
    for k, m in k_results.items():
        mlflow.log_metrics({"inertia": m["inertia"], "silhouette": m["silhouette"]}, step=k)

best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
print(f"\nBest K by silhouette: {best_k}  (silhouette={k_results[best_k]['silhouette']:.4f})")

# %% [markdown]
# ## 4. Stability Check (Adjusted Rand Index across seeds)

# %% Stability check, step down K if unstable
chosen_k = best_k
stability = compute_stability_ari(X_scaled, chosen_k, config.KMEANS_STABILITY_SEEDS)
print(f"K={chosen_k}: mean_ARI={stability['mean_ari']:.4f}  stable={stability['stable']}")

while not stability["stable"] and chosen_k > 2:
    chosen_k -= 1
    stability = compute_stability_ari(X_scaled, chosen_k, config.KMEANS_STABILITY_SEEDS)
    print(f"  -> retry K={chosen_k}: mean_ARI={stability['mean_ari']:.4f}  stable={stability['stable']}")

print(f"\nFinal chosen K: {chosen_k}  (silhouette={k_results[chosen_k]['silhouette']:.4f}, "
      f"mean_ARI={stability['mean_ari']:.4f})")

# %% [markdown]
# ## 5. Final K-Means Fit

# %% Fit final KMeans
kmeans = KMeans(n_clusters=chosen_k, **config.KMEANS_PARAMS)
labels = kmeans.fit_predict(X_scaled)
cluster_sizes = pd.Series(labels).value_counts().sort_index()
print("Cluster sizes:\n", cluster_sizes)

with mlflow.start_run(run_name="kmeans_final"):
    mlflow.log_params({"k": chosen_k, **config.KMEANS_PARAMS})
    mlflow.log_metrics({
        "inertia": k_results[chosen_k]["inertia"],
        "silhouette": k_results[chosen_k]["silhouette"],
        "stability_mean_ari": stability["mean_ari"],
    })

# %% [markdown]
# ## 6. PCA Visualization (2D + 3D)

# %% PCA
pca_2d = PCA(n_components=2, random_state=config.RANDOM_STATE)
coords_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3, random_state=config.RANDOM_STATE)
coords_3d = pca_3d.fit_transform(X_scaled)

print("PCA(2) explained variance ratio:", pca_2d.explained_variance_ratio_)
print("PCA(3) explained variance ratio:", pca_3d.explained_variance_ratio_)

# %% [markdown]
# ## 7. DBSCAN (comparison only -- not the saved model)

# %% DBSCAN eps via KneeLocator
dbscan_eps_info = find_dbscan_eps(X_scaled, config.DBSCAN_MIN_SAMPLES)
print(f"DBSCAN suggested eps: {dbscan_eps_info['eps']:.4f}  (knee_index={dbscan_eps_info['knee_index']})")

dbscan = DBSCAN(eps=dbscan_eps_info["eps"], min_samples=config.DBSCAN_MIN_SAMPLES)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
noise_pct = float((dbscan_labels == -1).mean() * 100)
print(f"DBSCAN clusters: {n_dbscan_clusters}  noise: {noise_pct:.2f}%")

dbscan_silhouette = None
if n_dbscan_clusters >= 2:
    non_noise = dbscan_labels != -1
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = np.where(non_noise)[0]
    sample_idx = rng.choice(idx, size=min(config.SILHOUETTE_SAMPLE_SIZE, len(idx)), replace=False)
    dbscan_silhouette = float(silhouette_score(X_scaled[sample_idx], dbscan_labels[sample_idx]))
    print(f"DBSCAN silhouette (non-noise subsample): {dbscan_silhouette:.4f}")

with mlflow.start_run(run_name="dbscan_comparison"):
    mlflow.log_params({"eps": dbscan_eps_info["eps"], "min_samples": config.DBSCAN_MIN_SAMPLES})
    mlflow.log_metrics({
        "n_clusters": n_dbscan_clusters,
        "noise_pct": noise_pct,
        **({"silhouette": dbscan_silhouette} if dbscan_silhouette is not None else {}),
    })

# %% [markdown]
# ## 8. Cluster Profiling & Naming

# %% Profiles
profiles = compute_cluster_profiles(df, labels, X_scaled, kmeans)
for c, p in profiles.items():
    print(f"Cluster {c} ({p['cluster_name']}): n={p['count']}, "
          f"avg_carat={p['avg_carat']:.2f}, avg_price=${p['avg_price_usd']:,.0f}, "
          f"avg_price_inr=Rs{p['avg_price_inr']:,.0f}")

# %% [markdown]
# ## 9. Figures

# %% Figure 1: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ks = list(k_results.keys())

axes[0].plot(ks, [k_results[k]["inertia"] for k in ks], "o-", color="#3B82F6")
axes[0].axvline(chosen_k, color="red", linestyle="--", linewidth=1, label=f"chosen K={chosen_k}")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method (Inertia)")
axes[0].legend()

axes[1].plot(ks, [k_results[k]["silhouette"] for k in ks], "o-", color="#10B981")
axes[1].axvline(chosen_k, color="red", linestyle="--", linewidth=1, label=f"chosen K={chosen_k}")
axes[1].axhline(0.4, color="gray", linestyle=":", linewidth=1, label="target=0.4")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette Score")
axes[1].legend()

fig.suptitle("Section 9 -- K Selection", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "09_01_elbow_silhouette.png", FIGURES)

# %% Figure 2: PCA 2D
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab10", alpha=0.3, s=5)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title(f"Section 9 -- PCA 2D (K={chosen_k})")
legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
ax.add_artist(legend)
plt.tight_layout()
save_figure(fig, "09_02_pca_2d_clusters.png", FIGURES)

# %% Figure 3: PCA 3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], c=labels, cmap="tab10", alpha=0.3, s=5)
ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title(f"Section 9 -- PCA 3D (K={chosen_k})")
legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
ax.add_artist(legend)
plt.tight_layout()
save_figure(fig, "09_03_pca_3d_clusters.png", FIGURES)

# %% Figure 4: DBSCAN k-distance plot with knee
fig, ax = plt.subplots(figsize=(8, 5))
k_dist = dbscan_eps_info["k_distances"]
ax.plot(k_dist, color="#6366F1")
if dbscan_eps_info["knee_index"] is not None:
    ax.axvline(dbscan_eps_info["knee_index"], color="red", linestyle="--",
               label=f"KneeLocator eps={dbscan_eps_info['eps']:.3f}")
    ax.axhline(dbscan_eps_info["eps"], color="red", linestyle="--")
ax.set_xlabel("Points sorted by k-distance")
ax.set_ylabel(f"Distance to {config.DBSCAN_MIN_SAMPLES}th nearest neighbor")
ax.set_title("Section 9 -- DBSCAN k-distance Plot (eps selection)")
ax.legend()
plt.tight_layout()
save_figure(fig, "09_04_dbscan_kdist.png", FIGURES)

# %% Figure 5: Cluster profiles (avg carat / avg price by cluster)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cluster_ids = sorted(profiles.keys())
names = [profiles[c]["cluster_name"] for c in cluster_ids]

axes[0].bar(names, [profiles[c]["avg_carat"] for c in cluster_ids], color="#F59E0B")
axes[0].set_ylabel("Avg carat")
axes[0].set_title("Average Carat by Cluster")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(names, [profiles[c]["avg_price_usd"] for c in cluster_ids], color="#EC4899")
axes[1].set_ylabel("Avg price (USD)")
axes[1].set_title("Average Price by Cluster")
axes[1].tick_params(axis="x", rotation=30)

fig.suptitle("Section 9 -- Cluster Profiles", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "09_05_cluster_profiles.png", FIGURES)

# %% [markdown]
# ## 10. Save Artifacts

# %% Save KMeans pipeline + labels
kmeans_pipeline = build_full_pipeline(preprocessor, kmeans)
joblib.dump(kmeans_pipeline, config.CLUSTERING_ARTIFACTS_DIR / "kmeans_model.pkl")

labels_df = pd.DataFrame({
    "cluster": labels,
    "cluster_name": [profiles[c]["cluster_name"] for c in labels],
})
labels_df.to_csv(config.CLUSTERING_ARTIFACTS_DIR / "cluster_labels.csv", index=False)
print("Saved: kmeans_model.pkl, cluster_labels.csv")

# %% Save PCA coordinates
pd.DataFrame({"PC1": coords_2d[:, 0], "PC2": coords_2d[:, 1], "cluster": labels}).to_csv(
    config.CLUSTERING_ARTIFACTS_DIR / "pca_2d.csv", index=False
)
pd.DataFrame({
    "PC1": coords_3d[:, 0], "PC2": coords_3d[:, 1], "PC3": coords_3d[:, 2], "cluster": labels,
}).to_csv(config.CLUSTERING_ARTIFACTS_DIR / "pca_3d.csv", index=False)
print("Saved: pca_2d.csv, pca_3d.csv")

# %% Save cluster profiles
save_json(
    {"k": chosen_k, "clusters": profiles},
    config.CLUSTERING_ARTIFACTS_DIR / "cluster_profiles.json",
)

# %% Save metrics
metrics_artifact = {
    "n_samples": int(X_scaled.shape[0]),
    "features": FEATURE_ORDER,
    "k_selection": {
        "k_range": list(config.K_RANGE),
        "results": {str(k): v for k, v in k_results.items()},
        "best_k_by_silhouette": best_k,
    },
    "stability": stability,
    "chosen_k": chosen_k,
    "final_kmeans": {
        "inertia": k_results[chosen_k]["inertia"],
        "silhouette": k_results[chosen_k]["silhouette"],
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
    },
    "pca": {
        "explained_variance_ratio_2d": pca_2d.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_3d": pca_3d.explained_variance_ratio_.tolist(),
    },
    "dbscan_comparison": {
        "min_samples": config.DBSCAN_MIN_SAMPLES,
        "eps": dbscan_eps_info["eps"],
        "n_clusters": n_dbscan_clusters,
        "noise_pct": noise_pct,
        "silhouette": dbscan_silhouette,
    },
    "hierarchical": "skipped -- O(n^2) infeasible at 53,794 rows; DBSCAN comparison "
                    "satisfies the optional 'try different clustering techniques' requirement",
    "targets": {
        "silhouette_target": 0.4,
        "silhouette_met": k_results[chosen_k]["silhouette"] > 0.4,
        "dbscan_noise_target_pct": 10,
        "dbscan_noise_met": noise_pct < 10,
    },
}
save_json(metrics_artifact, config.CLUSTERING_ARTIFACTS_DIR / "metrics.json")

# %% [markdown]
# ## 11. Final Summary

# %% Summary
print("\n=== Section 9 Summary ===")
print(f"Chosen K: {chosen_k}")
print(f"K-Means silhouette: {k_results[chosen_k]['silhouette']:.4f} "
      f"(target>0.4: {'MET' if k_results[chosen_k]['silhouette'] > 0.4 else 'NOT MET'})")
print(f"Stability mean ARI: {stability['mean_ari']:.4f} (stable={stability['stable']})")
print(f"DBSCAN: {n_dbscan_clusters} clusters, noise={noise_pct:.2f}% "
      f"(target<10%: {'MET' if noise_pct < 10 else 'NOT MET'})")
for c, p in profiles.items():
    print(f"  Cluster {c}: {p['cluster_name']} (n={p['count']})")
