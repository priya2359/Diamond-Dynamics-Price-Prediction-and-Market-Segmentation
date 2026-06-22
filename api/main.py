# filename: api/main.py
# purpose:  FastAPI service exposing diamond price prediction (/v1/predict/price)
#           and market segment prediction (/v1/predict/segment), plus a health
#           check (/v1/health). Loads the saved Pipelines once at startup and
#           reuses src/inference/prepare_input.py for all feature engineering
#           and inference -- no model.fit() and no duplicated logic.
# version:  1.0

# stdlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json

# third-party
import joblib
from fastapi import FastAPI

# internal
import config
from api.schemas import (
    DiamondInput,
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
    logger.info("Models loaded.")
    yield
    models.clear()


app = FastAPI(title="Diamond Dynamics API", version="1.0", lifespan=lifespan)


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        regression_model_loaded="regression" in models,
        clustering_model_loaded="clustering" in models,
    )


@app.post("/v1/predict/price", response_model=PricePredictionResponse)
def predict_price(diamond: DiamondInput) -> PricePredictionResponse:
    result = prepare_input.predict_price(models["regression"], **diamond.model_dump())
    return PricePredictionResponse(**result)


@app.post("/v1/predict/segment", response_model=SegmentPredictionResponse)
def predict_segment(diamond: DiamondInput) -> SegmentPredictionResponse:
    raw_input = diamond.model_dump()

    price_result = prepare_input.predict_price(models["regression"], **raw_input)
    segment_result = prepare_input.predict_segment(
        models["clustering"],
        models["cluster_profiles"],
        price_usd=price_result["price_usd"],
        **raw_input,
    )

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
