# filename: src/data/preprocess.py
# purpose:  Section 5 — Outlier handling (IQR + Z-score Winsorization) and
#           skewness transforms (log1p / sqrt). Produces diamonds_processed.csv.
# version:  1.0

# stdlib
import logging
from typing import Tuple

# third-party
import numpy as np
import pandas as pd

# internal
import config
from src.features.engineer import add_volume, add_dimension_ratio
from src.utils.helpers import save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transform specification — data-driven decisions from Section 5 skew audit.
# Changing a column's transform here requires re-running this notebook and
# retraining all downstream models.
# ---------------------------------------------------------------------------
_LOG1P_COLS = ["price", "dimension_ratio"]
_SQRT_COLS  = ["carat", "volume", "table", "price_per_carat"]
# x, y, z, depth: skew < 0.5 after cleaning — no transform needed.
# price_inr: display column only, kept in natural units.

TRANSFORM_PARAMS = {
    "log1p_columns": _LOG1P_COLS,
    "sqrt_columns": _SQRT_COLS,
    "no_transform_columns": ["x", "y", "z", "depth", "price_inr", "carat_category"],
    "price_transform": {
        "forward": "log1p",
        "inverse": "expm1",
        "apply_before_training": True,
        "inverse_apply_at_inference": True,
        "note": (
            "Model trained on log1p(price_usd). "
            "At inference: price_usd = expm1(model_output), "
            "then price_inr = price_usd * USD_TO_INR. "
            "Do NOT use np.exp — that adds +1 to every prediction."
        ),
    },
}


# ---------------------------------------------------------------------------
# Part A — Outlier handling
# ---------------------------------------------------------------------------

def _iqr_fences(series: pd.Series) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - config.IQR_MULTIPLIER * iqr, q3 + config.IQR_MULTIPLIER * iqr


def cap_iqr_outliers(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, dict]:
    """Winsorize columns using IQR fences. Returns df and fence dict for JSON."""
    df = df.copy()
    fences = {}
    for col in columns:
        lo, hi = _iqr_fences(df[col])
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        fences[col] = {"lower": round(lo, 6), "upper": round(hi, 6)}
        logger.info("  IQR %s: fence=[%.3f, %.3f]  capped %d values", col, lo, hi, n_clipped)
    return df, fences


def cap_zscore_outliers(
    df: pd.DataFrame,
    columns: list,
) -> Tuple[pd.DataFrame, dict]:
    """
    Winsorize columns using Z-score threshold (config.ZSCORE_THRESHOLD).
    Physical bounds from config override where tighter — e.g. a depth of 72%
    is technically within |Z|<3 but is still physically valid; we only cap
    values that cross BOTH the Z-score fence AND the physical bound.
    """
    df = df.copy()
    physical = {
        "depth": config.DEPTH_PHYSICAL_BOUNDS,
        "table": config.TABLE_PHYSICAL_BOUNDS,
    }
    fences = {}
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        z_lo = mean - config.ZSCORE_THRESHOLD * std
        z_hi = mean + config.ZSCORE_THRESHOLD * std

        phys_lo, phys_hi = physical.get(col, (-np.inf, np.inf))
        # Cap only where the value crosses both the Z-fence and physical bound.
        # Values that are statistically extreme but physically plausible are kept.
        lo = max(z_lo, phys_lo)
        hi = min(z_hi, phys_hi)

        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        fences[col] = {
            "lower": round(lo, 6),
            "upper": round(hi, 6),
            "z_lower": round(z_lo, 6),
            "z_upper": round(z_hi, 6),
            "physical_lower": phys_lo,
            "physical_upper": phys_hi,
        }
        logger.info(
            "  Z-score %s: effective fence=[%.2f, %.2f]  capped %d values",
            col, lo, hi, n_clipped,
        )
    return df, fences


# ---------------------------------------------------------------------------
# Part B — Skewness transforms
# ---------------------------------------------------------------------------

def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p / sqrt per the data-driven transform spec."""
    df = df.copy()
    for col in _LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            logger.info("  log1p applied to %s", col)
    for col in _SQRT_COLS:
        if col in df.columns:
            df[col] = np.sqrt(df[col])
            logger.info("  sqrt applied to %s", col)
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Full outlier + skewness preprocessing pipeline.

    Steps:
      1. IQR Winsorization on carat, price, x, y, z
      2. Recompute volume and dimension_ratio from capped x/y/z
      3. IQR Winsorization on derived columns (volume, dimension_ratio)
      4. Z-score + physical-bound Winsorization on depth, table
      5. Apply skewness transforms (log1p / sqrt per TRANSFORM_PARAMS)

    Returns:
      df_processed, outlier_params, transform_params
    """
    logger.info("--- Outlier handling ---")
    df, raw_fences = cap_iqr_outliers(df, ["carat", "price", "x", "y", "z"])

    # Recompute derived features from capped x/y/z so they stay consistent
    df = add_volume(df)
    df = add_dimension_ratio(df)
    logger.info("  volume and dimension_ratio recomputed from capped x/y/z")

    df, derived_fences = cap_iqr_outliers(df, ["volume", "dimension_ratio"])
    df, z_fences = cap_zscore_outliers(df, ["depth", "table"])

    outlier_params = {
        "method": "winsorize",
        "iqr_multiplier": config.IQR_MULTIPLIER,
        "zscore_threshold": config.ZSCORE_THRESHOLD,
        "iqr_fences": {**raw_fences, **derived_fences},
        "zscore_fences": z_fences,
    }

    logger.info("--- Skewness transforms ---")
    df = apply_transforms(df)

    remaining_nulls = int(df.isnull().sum().sum())
    logger.info(
        "Preprocessing complete — shape: %s | remaining nulls: %d",
        df.shape, remaining_nulls,
    )
    return df, outlier_params, TRANSFORM_PARAMS


def save_preprocess_artifacts(
    df: pd.DataFrame,
    outlier_params: dict,
    transform_params: dict,
) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = config.PROCESSED_DATA_DIR / "diamonds_processed.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved processed CSV: %s  shape=%s", csv_path, df.shape)

    save_json(outlier_params, config.PROCESSED_DATA_DIR / "outlier_params.json")
    save_json(transform_params, config.PROCESSED_DATA_DIR / "transform_params.json")
