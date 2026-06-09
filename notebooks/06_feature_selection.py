# filename: notebooks/06_feature_selection.py
# purpose:  Section 6 -- Feature selection (correlation + VIF + RF importance)
# version:  1.0

# %% [markdown]
# # Section 6 -- Feature Selection
# Three-lens analysis: Pearson correlation -> VIF -> RF importance.
#
# VIF analysis on numeric features only -- ordinal-encoded categoricals are statistically
# meaningless inputs to VIF (OLS regression on ordinal codes has no geometric interpretation).
# All VIF calls use `add_constant()` to avoid the forcing-through-origin inflation that
# occurs when the design matrix has no intercept term.

# %% Setup
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from sklearn.ensemble import RandomForestRegressor
import config
from src.utils.helpers import save_figure, save_json

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR

df = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_processed.csv")
print("Loaded:", df.shape)
print("Columns:", list(df.columns))

# %% [markdown]
# ## 1. Correlation Matrix -- All Numeric Features

# %% Correlation heatmap
numeric_cols = ["carat", "depth", "table", "x", "y", "z",
                "volume", "dimension_ratio", "price"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix (transformed features -- post Section 5)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "06_01_correlation_matrix.png", FIGURES)

# %% Print key correlations that drive feature selection decisions
print("\nKey correlations:")
print(f"  r(carat, volume)          = {df['carat'].corr(df['volume']):.4f}   <- near-identical after sqrt transforms")
print(f"  r(x, volume)              = {df['x'].corr(df['volume']):.4f}")
print(f"  r(y, volume)              = {df['y'].corr(df['volume']):.4f}")
print(f"  r(z, volume)              = {df['z'].corr(df['volume']):.4f}")
print(f"  r(depth, dimension_ratio) = {df['depth'].corr(df['dimension_ratio']):.4f}  <- critical anti-collinearity")
print(f"  r(carat, price)           = {df['carat'].corr(df['price']):.4f}")
print(f"  r(volume, price)          = {df['volume'].corr(df['price']):.4f}")

# %% [markdown]
# ## 2. VIF Analysis -- Three Progressive Runs
#
# VIF = 1/(1 - R^2_j), where R^2_j is the OLS R^2 from regressing feature j on all others.
# Threshold: VIF > 10 = high collinearity; VIF > 100 = critical, at least one feature must be dropped.
# `add_constant()` is required -- without it, OLS forced through origin inflates all VIFs.

# %% Helper
def run_vif(df: pd.DataFrame, cols: list, label: str) -> dict:
    X = add_constant(df[cols].values.astype(float))
    results = {}
    for i, col in enumerate(cols):
        v = variance_inflation_factor(X, i + 1)  # i+1: skip constant at index 0
        results[col] = round(float(v), 2)
    print(f"\nVIF -- {label}")
    print(f"  {'Feature':<22} {'VIF':>10}  Status")
    print(f"  {'-'*50}")
    for col, v in results.items():
        flag = ("CRITICAL (>100)" if v > 100
                else ("HIGH (>10)" if v > 10
                      else ("moderate (>5)" if v > 5 else "OK")))
        print(f"  {col:<22} {v:>10.2f}  {flag}")
    return results


# %% Run 1: All 8 numeric features
vif_run1 = run_vif(
    df,
    ["carat", "depth", "table", "x", "y", "z", "volume", "dimension_ratio"],
    "Run 1 -- all 8 numeric features"
)
print("\n  Findings: x/y/z (VIF 434-538) and carat/volume (VIF 861/1215) are critical.")
print("  x, y, z are absorbed by volume = x*y*z. depth (VIF=41) is anti-collinear")
print("  with dimension_ratio (VIF=37). table (VIF=1.4) is the only clean feature.")

# %% Run 2: Drop x, y, z -> isolate carat/volume and depth/dimension_ratio pairs
vif_run2 = run_vif(
    df,
    ["carat", "depth", "table", "volume", "dimension_ratio"],
    "Run 2 -- drop x, y, z"
)
print("\n  Findings: carat/volume (VIF ~800) -- genuine collinearity r=0.9991.")
print("  depth/dimension_ratio (VIF ~37) -- geometric duplicate: dimension_ratio ~ 100/depth.")
print("  Geometric proof: dimension_ratio=(x+y)/2z, depth_pct=200z/(x+y) -> reciprocals.")
print("  Decision: DROP dimension_ratio (RF importance 0.13%), KEEP depth.")

# %% Run 3: Final clean set -- drop x/y/z and dimension_ratio
vif_run3 = run_vif(
    df,
    ["carat", "depth", "table", "volume"],
    "Run 3 -- final numeric set (after drops)"
)
print("\n  Findings: depth (1.44) and table (1.38) are now clean -- confirmed by dropping")
print("  their respective collinear partners. carat/volume (VIF ~800) persist -- genuine")
print("  r=0.9991 collinearity. Decision: keep both for tree-based models (immune to")
print("  collinearity); for linear regression, use Ridge regularization.")

# %% [markdown]
# ## 3. RF Feature Importance

# %% Temporary ordinal encoding of categoricals for RF
df_enc = df.copy()
for col, cats in [("cut", config.CUT_ORDER),
                   ("color", config.COLOR_ORDER),
                   ("clarity", config.CLARITY_ORDER),
                   ("carat_category", config.CARAT_CATEGORY_LABELS)]:
    df_enc[col] = pd.Categorical(df_enc[col], categories=cats, ordered=True).codes

# Run RF with dimension_ratio included so we can report its importance as evidence
feats_with_dim = ["carat", "volume", "depth", "table", "dimension_ratio",
                   "cut", "color", "clarity", "carat_category"]
rf_full = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    random_state=config.RANDOM_STATE, n_jobs=-1
)
rf_full.fit(df_enc[feats_with_dim], df_enc["price"])
imp_full = pd.Series(rf_full.feature_importances_, index=feats_with_dim).sort_values(ascending=False)

# RF on final feature set (no x/y/z, no dimension_ratio)
final_feats = ["carat", "volume", "depth", "table", "cut", "color", "clarity", "carat_category"]
rf_final = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    random_state=config.RANDOM_STATE, n_jobs=-1
)
rf_final.fit(df_enc[final_feats], df_enc["price"])
imp_final = pd.Series(rf_final.feature_importances_, index=final_feats).sort_values(ascending=False)

print("\nRF importances (with dimension_ratio -- evidence for dropping it):")
for col, v in imp_full.items():
    marker = "  <- drop" if col == "dimension_ratio" else ""
    print(f"  {col:<22} {v:.4f}{marker}")

print("\nRF importances (final feature set):")
for col, v in imp_final.items():
    print(f"  {col:<22} {v:.4f}")

# %% Importance bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# With dimension_ratio
colors_full = ["#EF4444" if col == "dimension_ratio" else "#3B82F6"
               for col in imp_full.index]
axes[0].barh(imp_full.index, imp_full.values, color=colors_full)
axes[0].invert_yaxis()
axes[0].set_xlabel("Importance")
axes[0].set_title("RF Importances (incl. dimension_ratio)", fontweight="bold")
axes[0].axvline(0.01, color="gray", linestyle="--", linewidth=1, label="1% threshold")
axes[0].legend()

# Final set
colors_final = ["#3B82F6" if v > 0.01 else "#D1D5DB" for v in imp_final.values]
axes[1].barh(imp_final.index, imp_final.values, color=colors_final)
axes[1].invert_yaxis()
axes[1].set_xlabel("Importance")
axes[1].set_title("RF Importances -- Final Feature Set", fontweight="bold")

for ax in axes:
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8)

fig.suptitle("Feature Importance -- RF Regressor (200 trees, depth=15)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "06_02_rf_importances.png", FIGURES)

# %% [markdown]
# ## 4. Feature Relationships -- Final Set

# %% Scatter matrix for final numeric features vs price
sample_cols = ["carat", "volume", "depth", "table", "price"]
pairplot = sns.pairplot(
    df[sample_cols].sample(3000, random_state=config.RANDOM_STATE),
    plot_kws={"alpha": 0.15, "s": 8},
    diag_kind="kde"
)
pairplot.figure.suptitle("Pairplot -- Final Numeric Features vs Price (n=3000 sample)",
                         y=1.02, fontweight="bold")
save_figure(pairplot.figure, "06_03_pairplot_final_features.png", FIGURES)

# %% [markdown]
# ## 5. Feature Selection Summary

# %% Print decision table
print("\n" + "=" * 65)
print("SECTION 6 -- FEATURE SELECTION DECISIONS")
print("=" * 65)
print("\nDROPPED from all pipelines:")
print("  x, y, z          -- VIF 434-538 (Run 1). Absorbed by volume=x*y*z.")
print("                     r(x,volume)=0.9973, r(y,volume)=0.9957, r(z,volume)=0.9870")
print("  dimension_ratio  -- VIF 37 (Run 2), r(depth,dim_ratio)=-0.9864.")
print("                     Geometric duplicate of depth. RF importance=0.0013 (0.13%).")
print("\nKEPT -- regression pipeline:")
print("  carat, volume, depth, table, cut, color, clarity, carat_category")
print("\nKEPT -- clustering pipeline (adds price_per_carat):")
print("  carat, volume, depth, table, cut, color, clarity, carat_category, price_per_carat")
print("  (price_per_carat = target leak for regression; valid for clustering)")
print("\nResidual collinearity (addressed in Section 8 training):")
print("  carat/volume: r=0.9991, VIF~800 -- tree models immune; Ridge for linear regression")

# %% [markdown]
# ## 6. Save Artifacts

# %% Build and save evidence-rich JSONs
dropped_features = {
    "x": {
        "reason": "Absorbed by volume (x*y*z). Mutual r>0.97 with y/z. VIF=538 in Run 1.",
        "vif_run1": vif_run1.get("x"),
        "corr_with_volume": round(float(df["x"].corr(df["volume"])), 4),
    },
    "y": {
        "reason": "Absorbed by volume (x*y*z). Mutual r>0.97 with x/z. VIF=469 in Run 1.",
        "vif_run1": vif_run1.get("y"),
        "corr_with_volume": round(float(df["y"].corr(df["volume"])), 4),
    },
    "z": {
        "reason": "Absorbed by volume (x*y*z). Mutual r>0.97 with x/y. VIF=434 in Run 1.",
        "vif_run1": vif_run1.get("z"),
        "corr_with_volume": round(float(df["z"].corr(df["volume"])), 4),
    },
    "dimension_ratio": {
        "reason": (
            "r(depth, dimension_ratio)=-0.9864 -- near-perfect anti-collinearity. "
            "Geometric duplicate of depth: dimension_ratio=(x+y)/2z, depth_pct=200z/(x+y). "
            "RF importance=0.0013 (0.13%) -- negligible predictive value. "
            "KEEP depth: standard gemological metric, interpretable, VIF=1.44 after this drop."
        ),
        "pearson_r_with_depth": round(float(df["depth"].corr(df["dimension_ratio"])), 4),
        "vif_run2": vif_run2.get("dimension_ratio"),
        "rf_importance_with_dimension_ratio": round(float(imp_full["dimension_ratio"]), 4),
    },
}

collinearity_notes = {
    "carat_volume": {
        "pearson_r": round(float(df["carat"].corr(df["volume"])), 4),
        "vif_run3_carat": vif_run3.get("carat"),
        "vif_run3_volume": vif_run3.get("volume"),
        "decision": (
            "Keep both for tree-based models (RF, XGBoost) and ANN -- these are immune to "
            "collinearity between features. For linear regression: apply Ridge regularization."
        ),
        "rationale": (
            "Both carry genuine signal. Tree models select the more informative split; "
            "collinearity between predictors does not bias their predictions. "
            "Linear regression requires Ridge (L2 regularization) to shrink collinear "
            "coefficient magnitudes and stabilize estimates."
        ),
    },
}

base_payload = {
    "analysis_methods": ["pearson_correlation", "vif_with_add_constant", "rf_importance"],
    "vif_evidence": {
        "run1_all_8_numerics": vif_run1,
        "run2_drop_xyz": vif_run2,
        "run3_final_set": vif_run3,
        "note": (
            "VIF computed with statsmodels add_constant() on transformed features "
            "(sqrt/log1p applied per Section 5). add_constant is required to avoid "
            "artificially inflated VIFs from OLS forced through origin."
        ),
    },
    "rf_importance_top5": {
        col: round(float(imp_final[col]), 4) for col in imp_final.index[:5]
    },
    "dropped_features": dropped_features,
    "collinearity_notes": collinearity_notes,
}

# Regression artifact
reg_payload = {
    **base_payload,
    "pipeline": "regression",
    "features": ["carat", "volume", "depth", "table", "cut", "color", "clarity", "carat_category"],
    "note": (
        "price_per_carat excluded -- computed from the regression target (price), "
        "making it a direct target-leaking feature."
    ),
}

# Clustering artifact
clust_payload = {
    **base_payload,
    "pipeline": "clustering",
    "features": ["carat", "volume", "depth", "table", "cut", "color", "clarity",
                 "carat_category", "price_per_carat"],
    "note": (
        "price_per_carat included -- price is dropped before clustering so "
        "price_per_carat is not leaking the target."
    ),
}

config.REGRESSION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
config.CLUSTERING_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

reg_path = config.REGRESSION_ARTIFACTS_DIR / "selected_features.json"
clust_path = config.CLUSTERING_ARTIFACTS_DIR / "selected_features.json"

save_json(reg_payload, reg_path)
save_json(clust_payload, clust_path)

print(f"\nSaved: {reg_path}")
print(f"Saved: {clust_path}")
print(f"\nFigures saved: 06_01, 06_02, 06_03 -> {FIGURES}")
