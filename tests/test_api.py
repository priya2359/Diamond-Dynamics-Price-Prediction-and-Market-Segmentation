# filename: tests/test_api.py
# purpose:  FastAPI endpoint tests -- /v1/health, /v1/predict/price,
#           /v1/predict/segment, and Pydantic input validation (422s).
# version:  1.0

# stdlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# third-party
import pytest
from fastapi.testclient import TestClient

# internal
import config
from api.main import app

VALID_DIAMOND = {
    "carat": 0.7,
    "x": 5.7,
    "y": 5.71,
    "z": 3.53,
    "cut": "Ideal",
    "color": "E",
    "clarity": "VS2",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["regression_model_loaded"] is True
    assert body["clustering_model_loaded"] is True


def test_predict_price(client):
    response = client.post("/v1/predict/price", json=VALID_DIAMOND)
    assert response.status_code == 200
    body = response.json()
    assert body["price_usd"] > 0
    assert body["price_inr"] == pytest.approx(body["price_usd"] * config.USD_TO_INR)


def test_predict_segment(client):
    response = client.post("/v1/predict/segment", json=VALID_DIAMOND)
    assert response.status_code == 200
    body = response.json()
    assert body["cluster_id"] in (0, 1)
    assert body["cluster_name"]
    assert body["price_usd"] > 0
    assert body["price_inr"] == pytest.approx(body["price_usd"] * config.USD_TO_INR)
    assert set(body["profile"]) >= {
        "cluster_name", "count", "avg_carat", "avg_price_per_carat_usd",
        "avg_price_usd", "avg_price_inr", "dominant_cut", "dominant_color",
        "dominant_clarity",
    }
    assert isinstance(body["is_ood"], bool)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("carat", 100.0),      # exceeds config.INPUT_FEATURE_RANGES max
        ("cut", "Excellent"),  # not in config.CUT_ORDER
        ("color", "Z"),        # not in config.COLOR_ORDER
        ("clarity", "FL"),     # not in config.CLARITY_ORDER
    ],
)
def test_predict_price_validation_errors(client, field, bad_value):
    payload = {**VALID_DIAMOND, field: bad_value}
    response = client.post("/v1/predict/price", json=payload)
    assert response.status_code == 422


def test_predict_price_normalizes_case(client):
    """color/clarity are upper-cased, cut is title-cased before encoding."""
    payload = {**VALID_DIAMOND, "color": "e", "clarity": "vs2", "cut": "ideal"}
    response = client.post("/v1/predict/price", json=payload)
    assert response.status_code == 200
