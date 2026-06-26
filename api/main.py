# filename: api/main.py
# purpose:  FastAPI service exposing diamond price prediction (/v1/predict/price)
#           and market segment prediction (/v1/predict/segment), plus health
#           check (/v1/health), drift status (/v1/drift), and Prometheus metrics
#           (/metrics). Loads saved Pipelines once at startup and reuses
#           src/inference/prepare_input.py for all feature engineering and
#           inference -- no model.fit() and no duplicated logic.
# version:  3.0

# stdlib
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# third-party
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

# internal
import config
from api.monitoring import (
    ERROR_COUNT,
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


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


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


limiter = Limiter(
    key_func=get_remote_address,
    enabled=not os.environ.get("TESTING"),
)

app = FastAPI(title="Diamond Dynamics API", version="3.0", lifespan=lifespan)
app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

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


def _record_prediction(
    diamond: DiamondInput,
    price_usd: float,
    features_df: pd.DataFrame | None = None,
) -> None:
    """Record prediction metrics and transformed features for drift detection."""
    try:
        PREDICTION_VALUE.observe(price_usd)
        if features_df is None:
            features_df = prepare_input.build_regression_input(**diamond.model_dump())
        transformed = {
            feat: float(features_df[feat].iloc[0])
            for feat in config.DRIFT_NUMERIC_FEATURES
        }
        record_features(transformed)
    except Exception:
        logger.warning("Failed to record prediction metrics", exc_info=True)


@app.post("/v1/predict/price", response_model=PricePredictionResponse)
@limiter.limit(config.RATE_LIMIT)
def predict_price(request: Request, diamond: DiamondInput) -> PricePredictionResponse:
    try:
        with track_request("price"):
            result = prepare_input.predict_price(
                models["regression"], **diamond.model_dump(), return_features=True,
            )
        features_df = result.pop("_features_df", None)
        _record_prediction(diamond, result["price_usd"], features_df)
        return PricePredictionResponse(**result)
    except (ValueError, KeyError) as exc:
        ERROR_COUNT.labels(endpoint="price", error_type=type(exc).__name__).inc()
        logger.warning("predict_price client error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="price", error_type=type(exc).__name__).inc()
        logger.exception("predict_price failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/predict/segment", response_model=SegmentPredictionResponse)
@limiter.limit(config.RATE_LIMIT)
def predict_segment(request: Request, diamond: DiamondInput) -> SegmentPredictionResponse:
    try:
        raw_input = diamond.model_dump()

        with track_request("segment"):
            price_result = prepare_input.predict_price(
                models["regression"], **raw_input, return_features=True,
            )
            features_df = price_result.pop("_features_df", None)
            segment_result = prepare_input.predict_segment(
                models["clustering"],
                models["cluster_profiles"],
                price_usd=price_result["price_usd"],
                **raw_input,
            )

        _record_prediction(diamond, price_result["price_usd"], features_df)

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
    except (ValueError, KeyError) as exc:
        ERROR_COUNT.labels(endpoint="segment", error_type=type(exc).__name__).inc()
        logger.warning("predict_segment client error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="segment", error_type=type(exc).__name__).inc()
        logger.exception("predict_segment failed")
        raise HTTPException(status_code=500, detail=str(exc))
