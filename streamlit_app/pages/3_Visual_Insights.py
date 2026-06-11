# filename: streamlit_app/pages/3_Visual_Insights.py
# purpose:  Module 3 -- interactive visual insights (PCA segments, segment
#           profiles, regression model comparison) + supplementary EDA gallery.
# version:  1.0

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils import config, load_cluster_profiles, load_model_comparison, load_pca_2d

st.set_page_config(page_title="Visual Insights - Diamond Dynamics", page_icon="\U0001F4CA", layout="wide")

st.title("\U0001F4CA Module 3: Visual Insights")

cluster_profiles = load_cluster_profiles()
cluster_names = {int(k): v["cluster_name"] for k, v in cluster_profiles["clusters"].items()}

tab_segments, tab_models, tab_eda = st.tabs(["Market Segments", "Model Performance", "EDA Gallery"])

with tab_segments:
    st.subheader("PCA Projection of Market Segments")
    pca_df = load_pca_2d()
    pca_df["Segment"] = pca_df["cluster"].map(cluster_names)
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="Segment",
        title="K-Means Segments (K=2) -- PCA 2D Projection (58.8% variance explained)",
        opacity=0.5,
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Segment Profiles")
    profile_rows = [
        {
            "Segment": profile["cluster_name"],
            "Count": profile["count"],
            "Avg Carat": profile["avg_carat"],
            "Avg Price/Carat (USD)": profile["avg_price_per_carat_usd"],
            "Avg Price (INR)": profile["avg_price_inr"],
            "Dominant Cut": profile["dominant_cut"],
            "Dominant Color": profile["dominant_color"],
            "Dominant Clarity": profile["dominant_clarity"],
        }
        for profile in cluster_profiles["clusters"].values()
    ]
    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, width='stretch', hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.bar(profile_df, x="Segment", y="Avg Carat", color="Segment", title="Average Carat by Segment"),
            width='stretch',
        )
    with col2:
        st.plotly_chart(
            px.bar(profile_df, x="Segment", y="Avg Price (INR)", color="Segment", title="Average Price (INR) by Segment"),
            width='stretch',
        )

with tab_models:
    st.subheader("Regression Model Comparison (Test Set, USD)")
    comparison_df = load_model_comparison().reset_index().rename(columns={"index": "model"})
    st.dataframe(comparison_df, width='stretch', hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.bar(comparison_df, x="model", y="r2", title="R2 Score by Model (USD)", range_y=[0, 1]),
            width='stretch',
        )
    with col2:
        st.plotly_chart(
            px.bar(comparison_df, x="model", y="mae", title="MAE (USD) by Model"),
            width='stretch',
        )

with tab_eda:
    st.subheader("Exploratory Data Analysis & Model Diagnostics")
    gallery = [
        ("03_08_avg_price_bars.png", "Average Price by Cut / Color / Clarity"),
        ("03_11_carat_price_by_cut.png", "Carat vs. Price by Cut"),
        ("06_02_rf_importances.png", "Feature Importance (Random Forest)"),
        ("08_01_model_comparison.png", "Regression Model Comparison"),
        ("08_02_actual_vs_predicted.png", "XGBoost: Actual vs. Predicted Price"),
        ("09_05_cluster_profiles.png", "Cluster Profiles (Avg. Carat / Price)"),
    ]
    for filename, caption in gallery:
        path = Path(config.FIGURES_DIR) / filename
        if path.exists():
            st.image(str(path), caption=caption, width='stretch')
