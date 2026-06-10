# filename: notebooks/07_encoding.py
# purpose:  Section 7 -- Ordinal encoding for cut / color / clarity / carat_category
# version:  1.0

# %% [markdown]
# # Section 7 -- Ordinal Encoding
# Replaces string quality-grade columns with integer codes preserving grade order.
# Encoding orders are fixed by gemological standards (CLAUDE.md "Exact Encoding Orders").
#
# | Feature       | Order (ascending quality)                          | Codes |
# |---------------|----------------------------------------------------|-------|
# | cut           | Fair < Good < Very Good < Premium < Ideal          | 0-4   |
# | color         | J < I < H < G < F < E < D  (D=best)               | 0-6   |
# | clarity       | I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF    | 0-7   |
# | carat_category| Light < Medium < Heavy                             | 0-2   |

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
import config
from src.features.encode import encode_ordinal, ORDINAL_FEATURE_MAP
from src.utils.helpers import save_figure, save_json

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR

df = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_processed.csv")
print("Loaded:", df.shape)
print("\nCategorical columns before encoding:")
for col in ORDINAL_FEATURE_MAP:
    print(f"  {col}: {df[col].unique().tolist()}")

# %% [markdown]
# ## 1. Apply Ordinal Encoding

# %% Encode
df_enc, encoding_map = encode_ordinal(df)

print("\nEncoding map (string -> int code):")
for col, mapping in encoding_map.items():
    print(f"  {col}: {mapping}")

# %% Validate -- all categorical cols are now integer
print("\nColumn dtypes after encoding (categorical features):")
for col in ORDINAL_FEATURE_MAP:
    print(f"  {col}: {df_enc[col].dtype}  unique={sorted(df_enc[col].unique().tolist())}")

print("\nNull counts after encoding:", df_enc[list(ORDINAL_FEATURE_MAP.keys())].isnull().sum().to_dict())

# %% [markdown]
# ## 2. Validate -- Price Signal Preserved by Encoding

# %% Price-grade relationship -- unconditional vs partial (controlling for carat)
#
# Known confound: higher-quality grades (Ideal cut, D color, IF clarity) correlate
# with SMALLER diamonds. The market for small stones is dominated by high-quality cuts
# while large stones are more often Fair/Good cut. This means unconditional median price
# DECREASES with quality code -- the opposite of what encoding direction implies.
#
# This is NOT an encoding error. The ordinal mapping is correct (0=lowest, 7=highest).
# carat_category (3 broad buckets) is too coarse to isolate the quality effect --
# carat still varies widely within "Medium (0.5-1.5ct)". The correct validation is
# the PARTIAL effect: regress price ~ carat + quality_code and check the sign of
# the quality_code coefficient. This mirrors what the Section 8 model does.

print("\nPrice by quality grade -- unconditional (expect carat confound):")
for col in ["cut", "color", "clarity"]:
    medians = df_enc.groupby(col)["price"].median().sort_index()
    original_labels = {v: k for k, v in encoding_map[col].items()}
    vals = [f"{original_labels[c]}({round(m, 2)})" for c, m in medians.items()]
    direction = "DECREASING (carat confound)" if medians.is_monotonic_decreasing else "mixed"
    print(f"  {col}: {' < '.join(vals)}  => {direction}")

# %% Partial effect: OLS price ~ carat + quality_code (sign check)
import statsmodels.api as sm

print("\nPartial effect of quality grade on price, controlling for carat:")
print("  OLS: log1p(price) ~ carat + quality_code  -- expect positive quality_code coefficient\n")
for col in ["cut", "color", "clarity"]:
    X = sm.add_constant(df_enc[["carat", col]].astype(float))
    model = sm.OLS(df_enc["price"], X).fit()
    coef = model.params[col]
    pval = model.pvalues[col]
    direction = "positive (correct)" if coef > 0 else "negative (unexpected)"
    print(f"  {col}: coef={coef:+.4f}  p={pval:.2e}  => {direction}")

# %% [markdown]
# ## 3. Visualisations

# %% Figure 1: Before/After value distribution for each categorical
cat_cols = list(ORDINAL_FEATURE_MAP.keys())
fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for i, col in enumerate(cat_cols):
    cats = ORDINAL_FEATURE_MAP[col]

    # Before: string value counts in original order
    before_counts = df[col].value_counts().reindex(cats).fillna(0)
    axes[0, i].bar(range(len(cats)), before_counts.values, color="#93C5FD")
    axes[0, i].set_xticks(range(len(cats)))
    axes[0, i].set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    axes[0, i].set_title(f"{col}\n(before -- string)", fontsize=9)
    axes[0, i].set_ylabel("Count")

    # After: encoded int value counts
    after_counts = df_enc[col].value_counts().sort_index()
    axes[1, i].bar(after_counts.index, after_counts.values, color="#6EE7B7")
    axes[1, i].set_xticks(range(len(cats)))
    axes[1, i].set_title(f"{col}\n(after -- ordinal code)", fontsize=9)
    axes[1, i].set_xlabel("Encoded value (0 = lowest quality)")
    axes[1, i].set_ylabel("Count")

fig.suptitle("Ordinal Encoding -- Before vs After (distribution shape must be identical)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "07_01_encoding_before_after.png", FIGURES)

# %% Figure 2: Price distribution by encoded grade (box plots)
# Shows that encoding order captures price signal
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
quality_cols = ["cut", "color", "clarity"]

for i, col in enumerate(quality_cols):
    cats = ORDINAL_FEATURE_MAP[col]
    # Use original string labels for x-axis readability, but order them by code
    temp = df.copy()
    temp[col] = pd.Categorical(temp[col], categories=cats, ordered=True)
    sns.boxplot(data=temp, x=col, y="price", ax=axes[i],
                order=cats, color="#93C5FD",
                hue=col, legend=False,
                palette="Blues")
    axes[i].set_title(f"log1p(price) by {col} grade", fontweight="bold")
    axes[i].set_xlabel(f"{col}  (left = lowest quality, right = highest)")
    axes[i].set_ylabel("log1p(price_usd)" if i == 0 else "")
    axes[i].tick_params(axis="x", rotation=30)

fig.suptitle(
    "Marginal Price Distribution by Quality Grade\n"
    "(Decreasing trend = carat-size confound, NOT an encoding error -- "
    "see OLS partial-effect check above)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
save_figure(fig, "07_02_price_by_grade.png", FIGURES)

# %% [markdown]
# ## 4. Save Artifacts

# %% Save encoded CSV
out_path = config.PROCESSED_DATA_DIR / "diamonds_encoded.csv"
df_enc.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  shape={df_enc.shape}")
print("Remaining string columns:", [c for c in df_enc.columns if df_enc[c].dtype == object])

# %% Save encoding map as artifact
encoding_artifact = {
    "description": (
        "Ordinal encoding map for quality-grade categorical features. "
        "All codes are 0-indexed (0 = lowest quality). "
        "Ordering is fixed by gemological standards -- see CLAUDE.md Exact Encoding Orders."
    ),
    "features": {
        col: {
            "categories_in_order": ORDINAL_FEATURE_MAP[col],
            "code_map": encoding_map[col],
            "n_levels": len(ORDINAL_FEATURE_MAP[col]),
        }
        for col in ORDINAL_FEATURE_MAP
    },
    "notes": {
        "indexing": "0-indexed. Fair=0, Ideal=4. J=0, D=6. I1=0, IF=7. Light=0, Heavy=2.",
        "sklearn_pipeline": (
            "In Section 8 ColumnTransformer, build_ordinal_encoder() from "
            "src/features/encode.py produces an OrdinalEncoder with these same "
            "category orders. Fit on train set only -- result identical since "
            "categories are explicitly specified, not inferred from data."
        ),
    },
}

artifacts_dir = config.ARTIFACTS_DIR / "preprocessing"
artifacts_dir.mkdir(parents=True, exist_ok=True)
save_json(encoding_artifact, artifacts_dir / "encoding_map.json")
print(f"Saved: {artifacts_dir / 'encoding_map.json'}")

# %% [markdown]
# ## 5. Final State

# %% Summary
print("\nFinal encoded dataframe:")
print(f"  Shape: {df_enc.shape}")
print(f"  All numeric: {all(df_enc[col].dtype != object for col in df_enc.columns)}")
print(f"\nColumn summary:")
for col in df_enc.columns:
    dtype = df_enc[col].dtype
    print(f"  {col:<22} {str(dtype):<12} min={df_enc[col].min():.3f}  max={df_enc[col].max():.3f}")
