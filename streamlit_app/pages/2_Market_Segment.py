# filename: streamlit_app/pages/2_Market_Segment.py
# purpose:  Module 2 -- Diamond market segment prediction (K-Means cluster).
# version:  1.0

import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils import (
    load_clustering_pipeline,
    load_cluster_profiles,
    load_regression_pipeline,
    prepare_input,
    render_diamond_input_form,
)

st.set_page_config(page_title="Market Segment - Diamond Dynamics", page_icon="\U0001F48E")

st.title("\U0001F48E Module 2: Market Segment Prediction")
st.caption(
    "Same inputs as Module 1. Internally, a price is predicted first "
    "(to derive price-per-carat), then the diamond is assigned to a market "
    "segment via K-Means."
)

raw_input = render_diamond_input_form(key_prefix="segment")

if st.button("Predict Cluster", type="primary"):
    regression_pipeline = load_regression_pipeline()
    clustering_pipeline = load_clustering_pipeline()
    cluster_profiles = load_cluster_profiles()

    price_result = prepare_input.predict_price(regression_pipeline, **raw_input)
    segment_result = prepare_input.predict_segment(
        clustering_pipeline, cluster_profiles, price_usd=price_result["price_usd"], **raw_input,
    )

    st.success(
        f"**Segment: {segment_result['cluster_name']}** (Cluster {segment_result['cluster_id']})"
    )

    profile = segment_result["profile"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Price (INR)", f"₹{price_result['price_inr']:,.2f}")
    col2.metric("Segment Avg. Price (INR)", f"₹{profile['avg_price_inr']:,.2f}")
    col3.metric("Segment Avg. Carat", f"{profile['avg_carat']:.2f}")

    st.markdown(
        f"**Segment profile** -- typical cut: {profile['dominant_cut']}, "
        f"color: {profile['dominant_color']}, clarity: {profile['dominant_clarity']} "
        f"(n={profile['count']:,} diamonds)."
    )

    if segment_result["is_ood"]:
        st.warning(
            "This diamond's attributes are unusual relative to its segment "
            f"(distance to segment center = {segment_result['centroid_distance']:.2f}, "
            f"95th percentile = {segment_result['centroid_distance_p95']:.2f}). "
            "The cluster assignment may be less reliable for this diamond."
        )
