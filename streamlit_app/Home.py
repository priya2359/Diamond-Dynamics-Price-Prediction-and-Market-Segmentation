# filename: streamlit_app/Home.py
# purpose:  Diamond Dynamics landing page -- project overview and navigation.
# version:  1.0

import streamlit as st

st.set_page_config(page_title="Diamond Dynamics", page_icon="\U0001F48E", layout="wide")

st.title("\U0001F48E Diamond Dynamics")
st.subheader("Diamond Price Prediction & Market Segmentation")

st.markdown(
    """
Welcome to **Diamond Dynamics** -- a capstone project applying both
**supervised regression** and **unsupervised clustering** to the diamonds dataset
(53,940 records).

### Modules (see sidebar)
- **Price Prediction** -- predicts a diamond's market price (USD / INR) from its
  carat, dimensions (x, y, z), cut, color and clarity using an XGBoost model.
- **Market Segment** -- assigns a diamond to a market segment using K-Means
  clustering, by first predicting its price and deriving its price-per-carat.
- **Visual Insights** -- interactive charts covering segment composition, PCA
  visualization of the segments, and regression model comparison.

---
*Phase B (this app) loads pre-trained models from `artifacts/` -- no training happens here.*
"""
)

with st.expander("Dataset & Model Summary"):
    st.markdown(
        """
| | |
|---|---|
| Dataset | 53,794 diamonds (post-cleaning) |
| Regression target | Price (USD), log1p-transformed during training |
| Best regression model | XGBoost (R2 = 0.9879, MAE = $212.04 on test set) |
| Clustering | K-Means, K=2 (silhouette = 0.2650) |
| Segments | Affordable Compact Diamonds (n=26,291) / Premium Heavy Diamonds (n=27,503) |
"""
    )
