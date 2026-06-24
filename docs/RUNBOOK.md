# Diamond Dynamics — Operations Runbook

## Starting the Stack

```bash
# Full stack (FastAPI + Streamlit + Prometheus + Grafana)
docker compose up --build -d

# Check all services are running
docker compose ps

# View logs
docker compose logs -f fastapi
docker compose logs -f streamlit
```

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI API | http://localhost:8000/docs | N/A |
| Streamlit UI | http://localhost:8501 | N/A |
| Prometheus | http://localhost:9090 | N/A |
| Grafana | http://localhost:3000 | admin / (GRAFANA_ADMIN_PASSWORD env var, default: admin) |

## Health Checks

```bash
# API health
curl http://localhost:8000/v1/health

# Prometheus targets (should show fastapi:8000 as UP)
# Visit: http://localhost:9090/targets

# Quick prediction test
curl -X POST http://localhost:8000/v1/predict/price \
  -H "Content-Type: application/json" \
  -d '{"carat":0.7,"x":5.7,"y":5.71,"z":3.53,"cut":"Ideal","color":"E","clarity":"VS2"}'
```

## Monitoring

### Grafana Dashboard
The "Diamond Dynamics" dashboard is auto-provisioned with 6 panels:
1. **Request Rate** — predictions per second
2. **Latency p50/p95/p99** — response time percentiles
3. **Price Distribution** — histogram of predicted prices
4. **Total Predictions** — cumulative count
5. **Drift KS Statistic** — per-feature drift score
6. **Drift Detected** — binary drift status

### Key Metrics to Watch
- `prediction_latency_seconds` p95 > 2s — investigate model performance or resource contention
- `prediction_errors_total` increasing — check FastAPI logs for stack traces
- `data_drift_detected == 1` sustained > 10min — incoming data distribution has shifted

### Drift Detection
The API compares incoming prediction features against a 2,000-row reference sample from training data using the Kolmogorov-Smirnov test. Features monitored: carat, volume, depth, table (all in Section-5-transformed space).

**Check drift status:**
```bash
curl http://localhost:8000/v1/drift
```

**When drift is detected:**
1. Check whether the shift is in one feature or all — single-feature drift may indicate a data quality issue, not a genuine market shift
2. Review recent inputs — are they within the expected ranges defined in `config.INPUT_FEATURE_RANGES`?
3. If legitimate market shift: retrain the model (see Retraining below)
4. Drift state resets on container restart (in-memory deque)

## Troubleshooting

### FastAPI not responding
```bash
docker compose logs fastapi | tail -50
docker compose restart fastapi
```
Common causes:
- **Model file missing or corrupt:** Check that `artifacts/regression/best_model.pkl` and `artifacts/clustering/kmeans_model.pkl` exist
- **sklearn version mismatch:** The `.pkl` files were pickled with scikit-learn 1.9.0. If the Docker image installs a different version, deserialization fails
- **Memory exhaustion:** XGBoost model + preprocessing pipeline require ~500MB RAM

### Streamlit shows "Could not reach the prediction API"
- FastAPI service must be healthy before Streamlit starts (enforced by `depends_on: condition: service_healthy`)
- Check: `docker compose ps` — is the fastapi service showing "healthy"?
- Check: `curl http://localhost:8000/v1/health` from the host

### Prometheus not scraping
- Verify target status: http://localhost:9090/targets
- Check: `docker compose logs prometheus | tail -20`
- Ensure `monitoring/prometheus/prometheus.yml` references `fastapi:8000` (Docker service name, not localhost)

### Grafana dashboard not showing data
- Verify Prometheus datasource: Grafana > Settings > Data Sources > Prometheus
- URL should be `http://prometheus:9090` (Docker internal)
- Check: has the API received any predictions? Dashboard needs at least one data point

## Retraining

When drift is detected or new data is available:

1. Add new data to `data/raw/`
2. Run the full pipeline:
   ```bash
   python notebooks/02_data_cleaning.py
   python notebooks/04_feature_engineering.py
   python notebooks/05_outlier_skewness.py
   python notebooks/07_encoding.py
   python notebooks/08_regression.py
   python notebooks/09_clustering.py
   ```
3. Verify new model performance against previous metrics in `artifacts/regression/metrics.json`
4. If improved: rebuild Docker images and redeploy
   ```bash
   docker compose down
   docker compose up --build -d
   ```
5. Update `artifacts/monitoring/drift_reference.csv` with a fresh sample from the new processed data

## Updating USD/INR Conversion Rate

The conversion rate is configured via environment variable:

```bash
# docker-compose.yml or .env file
USD_TO_INR=85.0

# Restart services to pick up the new rate
docker compose restart fastapi streamlit
```

## Backup & Recovery

### Grafana dashboards
Dashboards are provisioned from `monitoring/grafana/provisioning/dashboards/diamond_dynamics.json` — no runtime backup needed. The JSON file is version-controlled.

### Prometheus data
Prometheus stores time-series data in its container volume. To persist across restarts, add a named volume in `docker-compose.yml`:
```yaml
prometheus:
  volumes:
    - prometheus_data:/prometheus
```

### Model artifacts
All `.pkl` files, metrics JSONs, and cluster profiles are in `artifacts/` and tracked in git. To rollback to a previous model: `git checkout <commit-hash> -- artifacts/`
