# filename: src/data/clean.py
# purpose:  Data loading and cleaning pipeline for the diamonds dataset
# version:  1.0

# stdlib
import logging
from typing import Tuple

# third-party
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# internal
import config
from src.utils.helpers import save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    path = config.RAW_DATA_DIR / "diamonds.csv"
    df = pd.read_csv(path)
    logger.info("Loaded raw data: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Drop duplicates
# ---------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d exact duplicate rows: %d → %d", before - len(df), before, len(df))
    return df


# ---------------------------------------------------------------------------
# Step 3 — Fix decimal-placement errors in x / y / z
# ---------------------------------------------------------------------------

def fix_decimal_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify rows where one dimension is >3x the average of the other two.
    These are data-entry decimal errors (value is off by exactly 10x).
    Verified against raw data: y=58.9, z=31.8 rows — siblings confirm ÷10.
    """
    df = df.copy()
    total_fixed = 0
    for col in ["x", "y", "z"]:
        siblings = [c for c in ["x", "y", "z"] if c != col]
        avg_siblings = df[siblings].mean(axis=1)
        # Both sibling avg and the column must be non-zero — zero rows are handled later
        mask = (df[col] > 3 * avg_siblings) & (df[col] > 0) & (avg_siblings > 0)
        if mask.sum() > 0:
            df.loc[mask, col] = df.loc[mask, col] / 10.0
            logger.info("  %s: corrected %d decimal-placement error(s)", col, int(mask.sum()))
            total_fixed += mask.sum()
    logger.info("Decimal-placement corrections total: %d", total_fixed)
    return df


# ---------------------------------------------------------------------------
# Step 4 — Zero → NaN
# ---------------------------------------------------------------------------

def zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["x", "y", "z"]:
        n = int((df[col] == 0).sum())
        df.loc[df[col] == 0, col] = np.nan
        logger.info("  %s: %d zero(s) → NaN", col, n)
    return df


# ---------------------------------------------------------------------------
# Step 5 — Regression-based imputation
# ---------------------------------------------------------------------------

def impute_dimensions(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Impute missing x/y/z using linear regression on carat (corr ~0.95–0.98).
    Residual analysis confirmed homoscedastic structure — plain linear model
    is appropriate. No log transform needed (|residual|~carat corr = 0.15–0.28).

    Returns cleaned DataFrame and regression params (coef + intercept per
    dimension) for inference-time reuse — same params must be applied to
    new diamonds at serving time to avoid training-serving skew.
    """
    df = df.copy()
    params = {}

    for col in ["x", "y", "z"]:
        missing_mask = df[col].isna()
        if missing_mask.sum() == 0:
            continue

        valid_mask = df[col].notna()
        X_train = df.loc[valid_mask, "carat"].values.reshape(-1, 1)
        y_train = df.loc[valid_mask, col].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        coef = float(model.coef_[0])
        intercept = float(model.intercept_)
        params[col] = {"coef": coef, "intercept": intercept}

        X_missing = df.loc[missing_mask, "carat"].values.reshape(-1, 1)
        imputed = np.maximum(model.predict(X_missing), 0.0)
        df.loc[missing_mask, col] = imputed

        logger.info(
            "  %s: imputed %d value(s) — coef=%.4f, intercept=%.4f, R²=%.4f",
            col,
            int(missing_mask.sum()),
            coef,
            intercept,
            model.score(X_train, y_train),
        )

    return df, params


# ---------------------------------------------------------------------------
# Step 6 — Save
# ---------------------------------------------------------------------------

def save_artifacts(df: pd.DataFrame, imputation_params: dict) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = config.PROCESSED_DATA_DIR / "diamonds_clean.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved cleaned CSV: %s  shape=%s", csv_path, df.shape)

    json_path = config.PROCESSED_DATA_DIR / "imputation_params.json"
    save_json(imputation_params, json_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline.

    Steps:
      1. Drop exact duplicate rows
      2. Fix decimal-placement errors in x / y / z (÷10 where dim > 3× siblings)
      3. Convert remaining zero values in x / y / z to NaN
      4. Impute NaNs via linear regression on carat

    Returns:
      df_clean: cleaned DataFrame
      imputation_params: regression coefficients to reuse at inference time
    """
    df = drop_duplicates(df)
    df = fix_decimal_errors(df)
    df = zero_to_nan(df)
    df, params = impute_dimensions(df)

    remaining_nulls = int(df.isnull().sum().sum())
    logger.info(
        "Cleaning complete — final shape: %s | remaining nulls: %d",
        df.shape,
        remaining_nulls,
    )
    return df, params
