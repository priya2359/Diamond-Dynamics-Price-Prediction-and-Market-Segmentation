# Model Card: Diamond Price Prediction (XGBoost)

## Model Details

| | |
|---|---|
| **Model type** | XGBoost Regressor inside sklearn Pipeline (ColumnTransformer + StandardScaler + XGBRegressor) |
| **Version** | 1.0 |
| **Framework** | scikit-learn 1.9.0, XGBoost 3.2.0 |
| **Python** | 3.11.9 |
| **Target** | log1p(price_usd) — predictions are inverse-transformed via expm1 |
| **Features** | 8 (4 numeric: carat, volume, depth, table; 4 ordinal: cut, color, clarity, carat_category) |

## Intended Use

- **Primary:** Predicting diamond prices for e-commerce platforms, jewelry retailers, and market analysis
- **Secondary:** Market segmentation via K-Means clustering for targeted marketing
- **Out of scope:** Non-diamond gemstones, synthetic/lab-grown diamonds, auction pricing, investment advice

## Training Data

- **Source:** ggplot2/seaborn diamonds dataset (53,940 raw records)
- **Post-cleaning:** 53,794 records (146 exact duplicates removed, 3 decimal errors corrected, 35 zero dimensions imputed)
- **Split:** 80% train / 20% test (random_state=42); train further split 80/20 for XGBoost early stopping
- **Preprocessing:** IQR Winsorization (carat, price, x, y, z, volume, dimension_ratio), Z-score + physical-bound capping (depth, table), log1p transform (price, dimension_ratio), sqrt transform (carat, volume, table, price_per_carat)

## Evaluation Results

| Model | MAE (USD) | RMSE (USD) | R2 | MAPE (%) | CV R2 (5-fold) |
|-------|-----------|------------|-----|----------|----------------|
| **XGBoost** | **$212** | **$372** | **0.9879** | **5.88** | **0.9912 +/- 0.0003** |
| Random Forest | $220 | $398 | 0.9861 | 6.21 | 0.9894 +/- 0.0002 |
| Decision Tree | $265 | $467 | 0.9809 | 7.31 | 0.9865 +/- 0.0003 |
| KNN | $291 | $503 | 0.9778 | 8.23 | 0.9783 +/- 0.0006 |
| ANN (Keras) | $327 | $548 | 0.9737 | 8.88 | N/A |
| Linear Regression | $830 | $1806 | 0.7140 | 32.15 | 0.9547 +/- 0.0006 |

### Stratified Performance (XGBoost, by price quartile)

| Price Tier | MAE (USD) | MAPE (%) | Count |
|------------|-----------|----------|-------|
| Q1 (cheap, < ~$700) | $62 | 8.90% | 2,690 |
| Q2 (~$700-$1,500) | $105 | 6.74% | 2,691 |
| Q3 (~$1,500-$3,800) | $250 | 6.72% | 2,688 |
| Q4 (expensive, > $3,800) | $431 | 5.35% | 2,690 |

MAE increases with price (expected — absolute errors scale with price magnitude), but MAPE decreases, showing the model is proportionally more accurate for expensive diamonds.

### Clustering (K-Means, K=2)

| Metric | Value | Target |
|--------|-------|--------|
| Silhouette | 0.2650 | > 0.4 (NOT MET) |
| Davies-Bouldin | 1.4425 | lower is better |
| Calinski-Harabasz | 23,459 | higher is better |
| Stability (mean ARI) | 0.9924 | > 0.85 (MET) |
| DBSCAN noise | 0.06% | < 10% (MET) |

Silhouette target not met — documented honestly. The diamond dataset is a continuous price/size spectrum, not naturally clustered. Corroborated by DBSCAN finding only 1 cluster.

## Limitations

1. **Dataset vintage:** Training data is from ~2008 (ggplot2 diamonds dataset). Current diamond market prices may differ significantly
2. **No synthetic diamonds:** Lab-grown diamonds are not represented in training data
3. **Currency conversion:** USD to INR rate is configurable (default 83.5) but does not update in real-time
4. **Clustering is a continuum:** The 2-cluster market segmentation is a convenient business abstraction, not a natural data partition
5. **No prediction intervals:** The model returns point estimates only — no uncertainty quantification

## Ethical Considerations

- **Pricing bias:** The model reflects historical pricing patterns which may embed market biases. It should not be used as the sole determinant of diamond value
- **No demographic data:** The model operates on physical diamond attributes only — no risk of human demographic bias
- **Transparency:** SHAP analysis provides per-prediction explanations. SHAP values are in log(price) space; display as percentage contributions to avoid misleading arithmetic

## Monitoring

- **Drift detection:** KS-test on transformed features (carat, volume, depth, table) against a 2,000-row reference sample. Alert if p-value < 0.05
- **Performance monitoring:** Prometheus metrics (request count, latency p50/p95/p99, price distribution, error rate)
- **Dashboard:** Auto-provisioned Grafana dashboard at http://localhost:3000
