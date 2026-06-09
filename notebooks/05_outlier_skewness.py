# filename: notebooks/05_outlier_skewness.py
# purpose:  Section 5 — Outlier handling + skewness treatment (audit + visualisation)
# version:  1.0

# %% [markdown]
# # Section 5 — Outlier Handling + Skewness Treatment
# **Order (locked):** outlier capping → skewness transforms
# **Methods:** IQR Winsorization (skewed cols) + Z-score/physical-bound cap (near-normal cols)

# %% Setup
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import config
from src.data.preprocess import preprocess, save_preprocess_artifacts, TRANSFORM_PARAMS
from src.utils.helpers import save_figure

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR

df_feat = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_featured.csv")
print("Input shape:", df_feat.shape)

# %% [markdown]
# ## 1. Pre-processing Skewness Audit (data-driven transform decisions)

# %% Skew table before any transforms
_audit_cols = ["carat", "depth", "table", "price", "x", "y", "z",
               "volume", "price_per_carat", "dimension_ratio"]
skew_before = df_feat[_audit_cols].skew().round(4)

def _classify(s):
    if abs(s) > 1.5:  return "log1p"
    if abs(s) >= 0.5: return "sqrt"
    return "none"

skew_table = pd.DataFrame({
    "Actual Skew": skew_before,
    "Transform":   skew_before.apply(_classify),
})
# Override: price is LOCKED to log1p regardless of threshold
skew_table.loc["price", "Transform"] = "log1p (LOCKED)"
# price_per_carat override note
skew_table.loc["price_per_carat", "Transform"] = "sqrt (clustering/EDA only)"

print("\nSkewness audit — transform decisions:")
print(skew_table.to_string())

# %% [markdown]
# ## 2. Run Preprocessing Pipeline

# %% Execute
df_proc, outlier_params, transform_params = preprocess(df_feat)
save_preprocess_artifacts(df_proc, outlier_params, transform_params)

# %% [markdown]
# ## 3. Before / After Comparison

# %% Skew after transforms
skew_after = df_proc[_audit_cols].skew().round(4)
comparison = pd.DataFrame({
    "Skew Before": skew_before,
    "Skew After":  skew_after,
    "Transform":   skew_table["Transform"],
    "Improved":    (skew_after.abs() < skew_before.abs()),
})
print("\nBefore / after skewness comparison:")
print(comparison.to_string())

# %% Row count preserved (Winsorization, not dropping)
print(f"\nInput rows: {len(df_feat):,}   Output rows: {len(df_proc):,}   Dropped: {len(df_feat)-len(df_proc)}")
print("Remaining nulls:", df_proc.isnull().sum().sum())

# %% [markdown]
# ## 4. Visualisations

# %% Before/after boxplots for IQR-treated columns
iqr_cols = ["carat", "price", "x", "y", "z", "volume"]
fig, axes = plt.subplots(2, 6, figsize=(20, 8))
for i, col in enumerate(iqr_cols):
    sns.boxplot(y=df_feat[col],  ax=axes[0, i], color="#93C5FD")
    axes[0, i].set_title(f"{col}\nbefore", fontsize=9)
    sns.boxplot(y=df_proc[col],  ax=axes[1, i], color="#6EE7B7")
    axes[1, i].set_title(f"{col}\nafter",  fontsize=9)

axes[0, 0].set_ylabel("Before (raw)")
axes[1, 0].set_ylabel("After (capped + transformed)")
fig.suptitle("Outlier Boxplots — Before vs After IQR Winsorization + Transform",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "05_01_boxplots_before_after.png", FIGURES)

# %% Price distribution: raw → log1p
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_feat["price"],    kde=True, ax=ax1, color="#3B82F6")
ax1.set_title("price — before (skew=1.62)")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))

sns.histplot(df_proc["price"],    kde=True, ax=ax2, color="#10B981")
ax2.set_title("log1p(price) — after (target for regression)")
ax2.set_xlabel("log1p(price_usd)")

fig.suptitle("Price Transform: log1p (forward) | expm1 (inverse at inference)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "05_02_price_log1p_transform.png", FIGURES)

# %% Skewness improvement bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(_audit_cols))
w = 0.38
ax.bar(x - w/2, skew_before.abs(), width=w, label="Before", color="#93C5FD")
ax.bar(x + w/2, skew_after.abs(),  width=w, label="After",  color="#6EE7B7")
ax.set_xticks(x)
ax.set_xticklabels(_audit_cols, rotation=30, ha="right")
ax.set_ylabel("|Skewness|")
ax.axhline(1.5, color="red",    linestyle="--", linewidth=1, label="|skew|=1.5 (log1p threshold)")
ax.axhline(0.5, color="orange", linestyle="--", linewidth=1, label="|skew|=0.5 (sqrt threshold)")
ax.legend()
ax.set_title("Skewness Reduction — Before vs After Transforms", fontweight="bold")
plt.tight_layout()
save_figure(fig, "05_03_skewness_comparison.png", FIGURES)

# %% [markdown]
# ## 5. Inverse Transform Reference (for inference)

# %% Confirm expm1 is the correct inverse
x_test = np.array([5.0, 7.0, 9.0, 11.23])
print("Inverse transform verification — expm1(log1p(x)) == x:")
for v in x_test:
    rt = np.expm1(np.log1p(v))
    print(f"  log1p({v:.2f}) = {np.log1p(v):.4f}  ->  expm1({np.log1p(v):.4f}) = {rt:.4f}  (match: {np.isclose(v, rt)})")

print("\nInference formula:")
print("  model output    = log1p(price_usd)")
print("  price_usd       = expm1(model_output)")
print(f"  price_inr       = price_usd * {config.USD_TO_INR}")
print("  NOT: np.exp(model_output) -- that adds +1 to every prediction")
