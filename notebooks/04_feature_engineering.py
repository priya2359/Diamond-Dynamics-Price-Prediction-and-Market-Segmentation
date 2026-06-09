# filename: notebooks/04_feature_engineering.py
# purpose:  Section 4 — Feature engineering (adds 5 derived features, saves output)
# version:  1.0

# %% [markdown]
# # Section 4 — Feature Engineering
# Adds: volume, price_per_carat, dimension_ratio, carat_category, price_inr

# %% Setup
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from src.features.engineer import engineer_features
from src.utils.helpers import save_figure

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR

# %% Load cleaned data
df = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_clean.csv")
print("Input shape:", df.shape)

# %% [markdown]
# ## Run Feature Engineering

# %% Engineer and save
df_feat = engineer_features(df)

out_path = config.PROCESSED_DATA_DIR / "diamonds_featured.csv"
df_feat.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  shape={df_feat.shape}")
print("\nNew columns:", [c for c in df_feat.columns if c not in df.columns])

# %% [markdown]
# ## Validation — New Feature Distributions

# %% Distribution of volume
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.histplot(df_feat["volume"], kde=True, ax=axes[0], color="#3B82F6")
axes[0].set_title("Volume (x·y·z)")
axes[0].set_xlabel("mm³")

sns.histplot(df_feat["dimension_ratio"], kde=True, ax=axes[1], color="#10B981")
axes[1].set_title("Dimension Ratio ((x+y)/2z)")
axes[1].set_xlabel("ratio")

sns.histplot(df_feat["price_per_carat"], kde=True, ax=axes[2], color="#F59E0B")
axes[2].set_title("Price per Carat (EDA/Clustering only)")
axes[2].set_xlabel("USD / carat")

fig.suptitle("Engineered Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "04_01_engineered_distributions.png", FIGURES)

# %% Carat category counts
print("\nCarat category counts:")
print(df_feat["carat_category"].value_counts().reindex(config.CARAT_CATEGORY_LABELS))

fig, ax = plt.subplots(figsize=(7, 4))
counts = df_feat["carat_category"].value_counts().reindex(config.CARAT_CATEGORY_LABELS)
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False,
            ax=ax, palette="muted")
ax.set_title("Carat Category Distribution")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
save_figure(fig, "04_02_carat_category_counts.png", FIGURES)

# %% Volume vs price (replaces x/y/z in regression)
sample = df_feat.sample(5000, random_state=config.RANDOM_STATE)
fig, ax = plt.subplots(figsize=(9, 6))
sns.regplot(data=sample, x="volume", y="price", ax=ax,
            scatter_kws={"alpha": 0.25, "s": 12, "color": "#3B82F6"},
            line_kws={"color": "#EF4444", "linewidth": 2})
ax.set_title("Volume vs Price", fontsize=13, fontweight="bold")
ax.set_xlabel("Volume (mm³)")
ax.set_ylabel("Price (USD)")
corr = df_feat["volume"].corr(df_feat["price"])
ax.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=11)
plt.tight_layout()
save_figure(fig, "04_03_volume_vs_price.png", FIGURES)

# %% [markdown]
# ## Final State

# %% Summary
print("\nFinal dataframe columns:")
for col in df_feat.columns:
    dtype = df_feat[col].dtype
    nulls = df_feat[col].isnull().sum()
    print(f"  {col:<20} {str(dtype):<12} nulls={nulls}")

print(f"\nShape: {df_feat.shape}")
print("\nSample row:")
print(df_feat.head(1).T)
