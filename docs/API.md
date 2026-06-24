# Diamond Dynamics API Reference

Base URL: `http://localhost:8000` (local) or `http://fastapi:8000` (Docker internal)

Auto-generated Swagger UI: `GET /docs`

## Endpoints

### POST /v1/predict/price

Predict diamond price in USD and INR.

**Request:**
```json
{
  "carat": 0.7,
  "x": 5.7,
  "y": 5.71,
  "z": 3.53,
  "cut": "Ideal",
  "color": "E",
  "clarity": "VS2"
}
```

**Response (200):**
```json
{
  "price_usd": 2847.53,
  "price_inr": 237768.76
}
```

**curl:**
```bash
curl -X POST http://localhost:8000/v1/predict/price \
  -H "Content-Type: application/json" \
  -d '{"carat":0.7,"x":5.7,"y":5.71,"z":3.53,"cut":"Ideal","color":"E","clarity":"VS2"}'
```

### POST /v1/predict/segment

Predict market segment (internally predicts price first to derive price-per-carat).

**Request:** Same as `/v1/predict/price`

**Response (200):**
```json
{
  "cluster_id": 0,
  "cluster_name": "Affordable Compact Diamonds",
  "price_usd": 2847.53,
  "price_inr": 237768.76,
  "profile": {
    "cluster_name": "Affordable Compact Diamonds",
    "count": 26291,
    "avg_carat": 0.42,
    "avg_price_per_carat_usd": 2620.97,
    "avg_price_usd": 1158.63,
    "avg_price_inr": 96745.62,
    "dominant_cut": "Ideal",
    "dominant_color": "E",
    "dominant_clarity": "VS2"
  },
  "centroid_distance": 2.15,
  "centroid_distance_p95": 3.31,
  "is_ood": false
}
```

### GET /v1/health

**Response (200):**
```json
{
  "status": "ok",
  "regression_model_loaded": true,
  "clustering_model_loaded": true
}
```

### GET /v1/drift

KS-test drift detection status. Returns empty `features` dict until the sliding window (100 predictions) is full.

**Response (200):**
```json
{
  "window_size": 100,
  "alpha": 0.05,
  "features": {
    "carat": {"ks_statistic": 0.85, "p_value": 0.0001, "is_drifted": true},
    "volume": {"ks_statistic": 0.12, "p_value": 0.42, "is_drifted": false}
  }
}
```

### GET /metrics

Prometheus metrics endpoint (text format). Scraped by Prometheus every 15s.

## Input Validation

| Field | Type | Constraints |
|-------|------|-------------|
| carat | float | 0.20 - 5.01 |
| x | float | 3.73 - 10.74 |
| y | float | 3.18 - 10.54 |
| z | float | 1.07 - 8.06 |
| cut | string | Fair, Good, Very Good, Premium, Ideal (case-normalized) |
| color | string | D, E, F, G, H, I, J (case-normalized) |
| clarity | string | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF (case-normalized) |

## Error Responses

**422 (Validation Error):** Input fails Pydantic validation (out of range, invalid category).

**400 (Bad Request):** Input passes validation but causes a prediction error (e.g., model returns NaN).

**500 (Internal Server Error):** Unexpected failure in model inference.

All error responses include a `detail` field with a human-readable message:
```json
{"detail": "color must be one of ['J', 'I', 'H', 'G', 'F', 'E', 'D'], got 'Z'"}
```

## Headers

- `X-Request-ID`: Every response includes a unique request ID for tracing. Send your own `X-Request-ID` header to propagate it.

## Monitoring

Prometheus metrics exposed at `/metrics`:
- `prediction_requests_total` (counter, by endpoint)
- `prediction_latency_seconds` (histogram, by endpoint)
- `prediction_value_distribution` (histogram, USD buckets)
- `prediction_errors_total` (counter, by endpoint + error_type)
- `data_drift_score`, `data_drift_pvalue`, `data_drift_detected` (gauges, by feature)
