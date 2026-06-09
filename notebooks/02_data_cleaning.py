# filename: notebooks/02_data_cleaning.py
# purpose:  Section 2 — Data loading and cleaning pipeline (presentable deliverable)
# version:  1.0

# %% [markdown]
# # Section 2 — Data Loading & Cleaning
# **Pipeline:** Load → Drop duplicates → Fix decimal errors → Zero→NaN → Regression impute → Save

# %% Setup
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

import pandas as pd
import numpy as np
import config
from src.data.clean import load_raw, clean_dataframe, save_artifacts

# %% [markdown]
# ## 1. Raw Data — Audit Before Cleaning

# %% Load raw and audit
df_raw = load_raw()

print("Shape:", df_raw.shape)
print("\nDtypes:\n", df_raw.dtypes)
print("\nNull counts:\n", df_raw.isnull().sum())
print("\nDuplicate rows:", df_raw.duplicated().sum())

# %% Zero-value audit in physical dimensions
zero_counts = (df_raw[["x", "y", "z"]] == 0).sum()
print("Zero values in x/y/z:\n", zero_counts)

# %% Decimal-placement errors — rows where one dimension is ~10x its siblings
print("\nSuspect decimal-error rows (y > 15 or z > 15):")
print(df_raw[(df_raw["y"] > 15) | (df_raw["z"] > 15)][["carat", "x", "y", "z", "price"]])

# %% [markdown]
# ## 2. Run Cleaning Pipeline

# %% Execute and save
df_clean, imputation_params = clean_dataframe(df_raw)
save_artifacts(df_clean, imputation_params)

# %% [markdown]
# ## 3. Before / After Comparison

# %% Shape and null comparison
comparison = pd.DataFrame({
    "Before": [df_raw.shape[0], df_raw.duplicated().sum(),
               int((df_raw[["x", "y", "z"]] == 0).sum().sum()),
               df_raw.isnull().sum().sum()],
    "After":  [df_clean.shape[0], df_clean.duplicated().sum(),
               int((df_clean[["x", "y", "z"]] == 0).sum().sum()),
               df_clean.isnull().sum().sum()],
}, index=["Row count", "Duplicate rows", "Zero values (x/y/z)", "Null values"])

print(comparison)

# %% Verify decimal corrections — all three rows should now have y/z ≤ 10
print("\nPost-clean: rows with carat~2.00 and Premium/H/SI2 (was y=58.9):")
print(df_clean[
    (df_clean["carat"] == 2.00) & (df_clean["cut"] == "Premium") & (df_clean["clarity"] == "SI2")
][["carat", "x", "y", "z"]])

# %% Imputation parameters (stored for inference-time reuse)
print("\nImputation regression parameters:")
for col, params in imputation_params.items():
    predicted_example = params["coef"] * 1.0 + params["intercept"]
    print(f"  {col}: coef={params['coef']:.4f}, intercept={params['intercept']:.4f}"
          f"  =>  predicted dim for 1.0ct = {predicted_example:.4f} mm")

# %% [markdown]
# ## 4. Final State Validation

# %% Final checks
print("Final shape:", df_clean.shape)
print("Remaining nulls:\n", df_clean.isnull().sum())
print("\nDescriptive stats (x, y, z — confirm no zeros, no extreme values):")
print(df_clean[["x", "y", "z"]].describe().round(3))

# %% Cleaned data preview
print("\nCleaned data head:")
print(df_clean.head())
