# filename: api/main.py
# purpose:  FastAPI service exposing diamond price prediction (/v1/predict/price)
#           and market segment prediction (/v1/predict/segment), plus health
#           check (/v1/health), drift status (/v1/drift), and Prometheus metrics
#           (/metrics). Loads saved Pipelines once at startup and reuses
#           src/inference/prepare_input.py for all feature engineering and
#           inference -- no model.fit() and no duplicated logic.
# version:  2.0

# stdlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# third-party
import joblib
from fastapi import FastAPI
from prometheus_client import make_asgi_app

# internal
import config
from api.monitoring import (
    PREDICTION_VALUE,
    check_drift,
    load_drift_reference,
    record_features,
    track_request,
)
from api.schemas import (
    DiamondInput,
    DriftResponse,
    HealthResponse,
    PricePredictionResponse,
    SegmentPredictionResponse,
)
from src.inference import prepare_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading regression and clustering pipelines...")
    models["regression"] = joblib.load(config.REGRESSION_ARTIFACTS_DIR / "best_model.pkl")
    models["clustering"] = joblib.load(config.CLUSTERING_ARTIFACTS_DIR / "kmeans_model.pkl")
    with open(config.CLUSTERING_ARTIFACTS_DIR / "cluster_profiles.json") as f:
        models["cluster_profiles"] = json.load(f)
    load_drift_reference()
    logger.info("Models and drift reference loaded.")
    yield
    models.clear()


app = FastAPI(title="Diamond Dynamics API", version="2.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        regression_model_loaded="regression" in models,
        clustering_model_loaded="clustering" in models,
    )


@app.get("/v1/drift", response_model=DriftResponse)
def drift_status() -> DriftResponse:
    results = check_drift()
    return DriftResponse(
        window_size=config.DRIFT_WINDOW_SIZE,
        alpha=config.DRIFT_KS_ALPHA,
        features=results,
    )


def _record_prediction(diamond: DiamondInput, price_usd: float) -> None:
    """Record prediction metrics and transformed features for drift detection."""
    PREDICTION_VALUE.observe(price_usd)
    df = prepare_input.build_regression_input(**diamond.model_dump())
    transformed = {feat: float(df[feat].iloc[0]) for feat in config.DRIFT_NUMERIC_FEATURES}
    record_features(transformed)


@app.post("/v1/predict/price", response_model=PricePredictionResponse)
def predict_price(diamond: DiamondInput) -> PricePredictionResponse:
    with track_request("price"):
        result = prepare_input.predict_price(models["regression"], **diamond.model_dump())
    _record_prediction(diamond, result["price_usd"])
    return PricePredictionResponse(**result)


@app.post("/v1/predict/segment", response_model=SegmentPredictionResponse)
def predict_segment(diamond: DiamondInput) -> SegmentPredictionResponse:
    raw_input = diamond.model_dump()

    with track_request("segment"):
        price_result = prepare_input.predict_price(models["regression"], **raw_input)
        segment_result = prepare_input.predict_segment(
            models["clustering"],
            models["cluster_profiles"],
            price_usd=price_result["price_usd"],
            **raw_input,
        )

    _record_prediction(diamond, price_result["price_usd"])

    return SegmentPredictionResponse(
        cluster_id=segment_result["cluster_id"],
        cluster_name=segment_result["cluster_name"],
        price_usd=price_result["price_usd"],
        price_inr=price_result["price_inr"],
        profile=segment_result["profile"],
        centroid_distance=segment_result["centroid_distance"],
        centroid_distance_p95=segment_result["centroid_distance_p95"],
        is_ood=segment_result["is_ood"],
    )
