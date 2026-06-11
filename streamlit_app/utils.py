# filename: streamlit_app/utils.py
# purpose:  Cached pipeline/artifact loaders + shared input form for the
#           Diamond Dynamics Streamlit app (Phase B -- load-only, no model.fit()).
# version:  1.0

# stdlib
import json
import sys
from pathlib import Path

# third-party
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# internal
import config
from src.inference import prepare_input


@st.cache_resource(show_spinner="Loading price prediction model...")
def load_regression_pipeline():
    return joblib.load(config.REGRESSION_ARTIFACTS_DIR / "best_model.pkl")


@st.cache_resource(show_spinner="Loading market segmentation model...")
def load_clustering_pipeline():
    return joblib.load(config.CLUSTERING_ARTIFACTS_DIR / "kmeans_model.pkl")


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> dict:
    with open(config.CLUSTERING_ARTIFACTS_DIR / "cluster_profiles.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_pca_2d() -> pd.DataFrame:
    return pd.read_csv(config.CLUSTERING_ARTIFACTS_DIR / "pca_2d.csv")


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(config.REGRESSION_ARTIFACTS_DIR / "model_comparison.csv", index_col=0)


def render_diamond_input_form(key_prefix: str) -> dict:
    """
    Shared diamond attribute form -- carat, x, y, z, cut, color, clarity
    (GUVI Module 1/2 inputs). depth and table are not collected: depth is
    derived exactly as 200*z/(x+y), table defaults to its training median
    (config.TABLE_DEFAULT) -- see Section 10 input-handling decision.
    """
    ranges = config.INPUT_FEATURE_RANGES
    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input(
            "Carat",
            min_value=ranges["carat"]["min"], max_value=ranges["carat"]["max"],
            value=ranges["carat"]["default"], step=ranges["carat"]["step"],
            key=f"{key_prefix}_carat",
        )
        x = st.number_input(
            "Length - x (mm)",
            min_value=ranges["x"]["min"], max_value=ranges["x"]["max"],
            value=ranges["x"]["default"], step=ranges["x"]["step"],
            key=f"{key_prefix}_x",
        )
        y = st.number_input(
            "Width - y (mm)",
            min_value=ranges["y"]["min"], max_value=ranges["y"]["max"],
            value=ranges["y"]["default"], step=ranges["y"]["step"],
            key=f"{key_prefix}_y",
        )
        z = st.number_input(
            "Depth - z (mm)",
            min_value=ranges["z"]["min"], max_value=ranges["z"]["max"],
            value=ranges["z"]["default"], step=ranges["z"]["step"],
            key=f"{key_prefix}_z",
        )

    with col2:
        cut = st.selectbox("Cut", config.CUT_ORDER, index=4, key=f"{key_prefix}_cut")
        color = st.selectbox("Color", config.COLOR_ORDER, index=3, key=f"{key_prefix}_color")
        clarity = st.selectbox("Clarity", config.CLARITY_ORDER, index=2, key=f"{key_prefix}_clarity")

    return {"carat": carat, "x": x, "y": y, "z": z, "cut": cut, "color": color, "clarity": clarity}
