# filename: src/features/engineer.py
# purpose:  Feature engineering — computes all derived features for the diamonds dataset
# version:  1.0

# stdlib
import logging

# third-party
import pandas as pd

# internal
import config

logger = logging.getLogger(__name__)


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume"] = df["x"] * df["y"] * df["z"]
    logger.info("  volume: x * y * z  (range %.2f – %.2f)", df["volume"].min(), df["volume"].max())
    return df


def add_price_per_carat(df: pd.DataFrame) -> pd.DataFrame:
    # TARGET LEAK for regression — valid only in clustering and EDA
    df = df.copy()
    df["price_per_carat"] = df["price"] / df["carat"]
    logger.info("  price_per_carat: price / carat  (DO NOT use in regression pipeline)")
    return df


def add_dimension_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dimension_ratio"] = (df["x"] + df["y"]) / (2 * df["z"])
    logger.info(
        "  dimension_ratio: (x+y)/(2z)  (range %.2f – %.2f)",
        df["dimension_ratio"].min(),
        df["dimension_ratio"].max(),
    )
    return df


def add_carat_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["carat_category"] = pd.cut(
        df["carat"],
        bins=config.CARAT_CATEGORY_BINS,
        labels=config.CARAT_CATEGORY_LABELS,
        right=False,
    )
    counts = df["carat_category"].value_counts().to_dict()
    logger.info("  carat_category: %s", counts)
    return df


def add_price_inr(df: pd.DataFrame) -> pd.DataFrame:
    # Display column only — not a training feature in either pipeline
    df = df.copy()
    df["price_inr"] = df["price"] * config.USD_TO_INR
    logger.info(
        "  price_inr: price * %.1f  (range %.0f – %.0f)",
        config.USD_TO_INR,
        df["price_inr"].min(),
        df["price_inr"].max(),
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all five engineered features to the dataframe.

    Caller is responsible for pipeline-specific exclusions:
      - Regression: exclude price_per_carat (target leak), price_inr (display only)
      - Clustering: exclude price, price_inr before fitting
      - Both: volume replaces x, y, z (drop originals in feature selection step)
    """
    logger.info("Engineering features:")
    df = add_volume(df)
    df = add_price_per_carat(df)
    df = add_dimension_ratio(df)
    df = add_carat_category(df)
    df = add_price_inr(df)
    logger.info("Feature engineering complete — shape: %s", df.shape)
    return df
