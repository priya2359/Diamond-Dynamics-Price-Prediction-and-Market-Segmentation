# filename: src/features/encode.py
# purpose:  Ordinal encoding for quality-grade categorical features
# version:  1.0

# stdlib
import logging
from typing import Tuple

# third-party
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# internal
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature -> ordered category list (ascending quality/value).
# Sourced from config.py which holds the single source of truth.
# Wrong order silently corrupts model training -- verified in Section 6
# via RF importance and VIF analysis.
# ---------------------------------------------------------------------------
ORDINAL_FEATURE_MAP: dict[str, list] = {
    "cut":           config.CUT_ORDER,           # Fair -> Ideal  (5 levels)
    "color":         config.COLOR_ORDER,          # J -> D         (7 levels)
    "clarity":       config.CLARITY_ORDER,        # I1 -> IF       (8 levels)
    "carat_category": config.CARAT_CATEGORY_LABELS,  # Light -> Heavy (3 levels)
}


def encode_ordinal(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Apply ordinal encoding to all quality-grade categorical features.

    Returns:
        df_encoded: DataFrame with string categoricals replaced by integer codes (0-indexed).
        encoding_map: {feature: {string_label: int_code}} for audit and Streamlit display.
    """
    df = df.copy()
    encoding_map: dict[str, dict] = {}

    for col, cats in ORDINAL_FEATURE_MAP.items():
        if col not in df.columns:
            logger.warning("Column %s not found -- skipping", col)
            continue

        df[col] = pd.Categorical(df[col], categories=cats, ordered=True).codes
        encoding_map[col] = {label: int(code) for code, label in enumerate(cats)}
        logger.info(
            "  Encoded %s: %s",
            col,
            " < ".join(f"{k}={v}" for k, v in encoding_map[col].items()),
        )

    return df, encoding_map


def build_ordinal_encoder() -> OrdinalEncoder:
    """
    Build a pre-configured OrdinalEncoder for use inside an sklearn ColumnTransformer.

    Column order follows sorted(ORDINAL_FEATURE_MAP.keys()) to match the
    ColumnTransformer feature order set in Section 8.
    """
    sorted_cols = sorted(ORDINAL_FEATURE_MAP.keys())
    categories = [ORDINAL_FEATURE_MAP[col] for col in sorted_cols]
    return OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=float,
    )
