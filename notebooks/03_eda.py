# filename: notebooks/03_eda.py
# purpose:  Section 3 — Exploratory Data Analysis (8 GUVI required + 3 extras)
# version:  1.0

# %% [markdown]
# # Section 3 — Exploratory Data Analysis
# All figures saved to `docs/figures/` with prefix `03_`.

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
from src.utils.helpers import save_figure

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR

df = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_clean.csv")
# Carat category used in several plots — compute inline for EDA only
df["carat_category"] = pd.cut(
    df["carat"],
    bins=config.CARAT_CATEGORY_BINS,
    labels=config.CARAT_CATEGORY_LABELS,
    right=False,
)
print("Loaded cleaned data:", df.shape)

# %% [markdown]
# ## Plot 1 — Distribution of Numeric Features
# GUVI requirement: distribution plots for price, carat, x, y, z

# %% Plot 1
numeric_cols = ["price", "carat", "x", "y", "z"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("muted")[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

axes[-1].set_visible(False)
fig.suptitle("Distribution of Numeric Features", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
save_figure(fig, "03_01_distributions_numeric.png", FIGURES)

# %% [markdown]
# ## Plot 2 — Count Plots for Categorical Features
# GUVI requirement: count plots for cut, color, clarity

# %% Plot 2
cat_cols = ["cut", "color", "clarity"]
cat_orders = {
    "cut": config.CUT_ORDER,
    "color": list(reversed(config.COLOR_ORDER)),
    "clarity": list(reversed(config.CLARITY_ORDER)),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, cat_cols):
    order = cat_orders[col]
    counts = df[col].value_counts().reindex(order)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                legend=False, ax=ax, palette="muted")
    ax.set_title(f"Count by {col.capitalize()}")
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

fig.suptitle("Frequency of Categorical Features", fontsize=15, fontweight="bold")
plt.tight_layout()
save_figure(fig, "03_02_countplots_categorical.png", FIGURES)

# %% [markdown]
# ## Plot 3 — Price Variation by Categorical Features (Boxplots)
# GUVI requirement: price variation by cut, color, clarity

# %% Plot 3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col in zip(axes, cat_cols):
    order = cat_orders[col]
    sns.boxplot(data=df, x=col, y="price", order=order, hue=col,
                legend=False, ax=ax, palette="muted")
    ax.set_title(f"Price by {col.capitalize()}")
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel("Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))
    ax.tick_params(axis="x", rotation=25)

fig.suptitle("Price Variation by Cut, Color, Clarity", fontsize=15, fontweight="bold")
plt.tight_layout()
save_figure(fig, "03_03_price_boxplots_categorical.png", FIGURES)

# %% [markdown]
# ## Plot 4 — Correlation Heatmap
# GUVI requirement: correlation heatmap of numerical features

# %% Plot 4
numeric_df = df[["carat", "depth", "table", "price", "x", "y", "z"]]
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 11},
)
ax.set_title("Correlation Heatmap — Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "03_04_correlation_heatmap.png", FIGURES)

# %% [markdown]
# ## Plot 5 — Scatter Matrix (carat, x, y, z, price)
# GUVI requirement: scatterplot matrix

# %% Plot 5 (sample 4000 rows for render speed)
sample = df[["carat", "x", "y", "z", "price"]].sample(4000, random_state=config.RANDOM_STATE)
fig, axes = plt.subplots(5, 5, figsize=(14, 14))
pd.plotting.scatter_matrix(sample, ax=axes, alpha=0.3, diagonal="kde", color="#3B82F6")
fig.suptitle("Scatter Matrix — carat, x, y, z, price", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save_figure(fig, "03_05_scatter_matrix.png", FIGURES)

# %% [markdown]
# ## Plot 6 — Pairplot
# GUVI requirement: sns.pairplot

# %% Plot 6 (sample 3000 rows; pairplot is expensive on 53K rows)
sample_pp = df[["carat", "price", "depth", "table", "cut"]].sample(
    3000, random_state=config.RANDOM_STATE
)
g = sns.pairplot(sample_pp, hue="cut", hue_order=config.CUT_ORDER,
                 plot_kws={"alpha": 0.4, "s": 15}, diag_kind="kde")
g.figure.suptitle("Pairplot — key features, coloured by cut quality",
                   fontsize=13, fontweight="bold", y=1.01)
save_figure(g.figure, "03_06_pairplot.png", FIGURES)

# %% [markdown]
# ## Plot 7 — Carat vs Price Regression Lineplot
# GUVI requirement: carat vs price regression lineplot

# %% Plot 7 (sample 5000 for speed)
sample_reg = df[["carat", "price"]].sample(5000, random_state=config.RANDOM_STATE)
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=sample_reg, x="carat", y="price", ax=ax,
            scatter_kws={"alpha": 0.25, "s": 15, "color": "#3B82F6"},
            line_kws={"color": "#EF4444", "linewidth": 2})
ax.set_title("Carat vs Price — Regression Lineplot", fontsize=14, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Price (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))
plt.tight_layout()
save_figure(fig, "03_07_carat_vs_price_regplot.png", FIGURES)

# %% [markdown]
# ## Plot 8 — Average Price per Cut, Color, Clarity (Bar Plots)
# GUVI requirement: average price bar plots

# %% Plot 8
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col in zip(axes, cat_cols):
    order = cat_orders[col]
    avg_price = df.groupby(col, observed=False)["price"].mean().reindex(order)
    bars = sns.barplot(x=avg_price.index, y=avg_price.values, hue=avg_price.index,
                       legend=False, ax=ax, palette="muted")
    ax.set_title(f"Avg Price by {col.capitalize()}")
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel("Avg Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))
    ax.tick_params(axis="x", rotation=25)
    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"${bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=8,
        )

fig.suptitle("Average Price by Cut, Color, Clarity", fontsize=15, fontweight="bold")
plt.tight_layout()
save_figure(fig, "03_08_avg_price_bars.png", FIGURES)

# %% [markdown]
# ## Extra 1 — Price Distribution: Raw vs Log Scale
# Motivates the log-transform decision in Section 5

# %% Extra 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["price"], kde=True, ax=ax1, color="#3B82F6")
ax1.set_title("Price — Raw (right-skewed)")
ax1.set_xlabel("Price (USD)")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))

sns.histplot(np.log(df["price"]), kde=True, ax=ax2, color="#10B981")
ax2.set_title("log(Price) — Near-normal after transform")
ax2.set_xlabel("log(Price)")

skew_raw = df["price"].skew()
skew_log = np.log(df["price"]).skew()
fig.suptitle(
    f"Price Skewness: raw={skew_raw:.2f}  →  log-transformed={skew_log:.2f}",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
save_figure(fig, "03_09_price_log_distribution.png", FIGURES)

# %% [markdown]
# ## Extra 2 — Outlier Boxplots (carat, price, x, y, z)
# Preview for Section 5 IQR/Z-Score outlier handling

# %% Extra 2
fig, axes = plt.subplots(1, 5, figsize=(18, 6))
for ax, col in zip(axes, ["carat", "price", "x", "y", "z"]):
    sns.boxplot(y=df[col], ax=ax, color=sns.color_palette("muted")[3])
    ax.set_title(col)
    ax.set_ylabel("")

fig.suptitle("Outlier Boxplots — carat, price, x, y, z", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, "03_10_outlier_boxplots.png", FIGURES)

# %% [markdown]
# ## Extra 3 — Carat vs Price, Coloured by Cut Quality
# Shows how cut quality stratifies value at the same carat weight

# %% Extra 3
sample_cut = df.sample(5000, random_state=config.RANDOM_STATE)
palette = sns.color_palette("RdYlGn", n_colors=len(config.CUT_ORDER))

fig, ax = plt.subplots(figsize=(11, 7))
for i, cut_val in enumerate(config.CUT_ORDER):
    subset = sample_cut[sample_cut["cut"] == cut_val]
    ax.scatter(subset["carat"], subset["price"], alpha=0.4, s=12,
               color=palette[i], label=cut_val)

ax.set_xlabel("Carat")
ax.set_ylabel("Price (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))
ax.set_title("Carat vs Price — Coloured by Cut Quality", fontsize=14, fontweight="bold")
ax.legend(title="Cut", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
save_figure(fig, "03_11_carat_price_by_cut.png", FIGURES)

# %% [markdown]
# ## Summary
# All 8 GUVI-required figures + 3 extras saved to `docs/figures/`.

# %% Summary
import os
figs = sorted(f for f in os.listdir(FIGURES) if f.startswith("03_"))
print(f"Section 3 figures saved ({len(figs)} total):")
for f in figs:
    size_kb = os.path.getsize(FIGURES / f) // 1024
    print(f"  {f}  ({size_kb} KB)")
