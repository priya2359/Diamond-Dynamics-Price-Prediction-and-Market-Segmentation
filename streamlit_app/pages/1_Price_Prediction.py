# filename: streamlit_app/pages/1_Price_Prediction.py
# purpose:  Module 1 -- Diamond price prediction (USD / INR).
# version:  1.0

import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import requests

from utils import predict_price, render_diamond_input_form

st.set_page_config(page_title="Price Prediction - Diamond Dynamics", page_icon="\U0001F4B0")

st.title("\U0001F4B0 Module 1: Diamond Price Prediction")
st.caption(
    "Enter a diamond's physical and quality attributes to predict its market price. "
    "depth% is computed automatically as 200*z/(x+y); table% defaults to its "
    "training-set median (57.0) -- both have negligible impact on price (Section 6 RF importance)."
)

raw_input = render_diamond_input_form(key_prefix="price")

if st.button("Predict Price", type="primary"):
    try:
        result = predict_price(raw_input)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Predicted Price (USD)", f"${result['price_usd']:,.2f}")
        col2.metric("Predicted Price (INR)", f"₹{result['price_inr']:,.2f}")
