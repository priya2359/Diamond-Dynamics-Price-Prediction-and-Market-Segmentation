# filename: api/schemas.py
# purpose:  Pydantic request/response models for the Diamond Dynamics API.
#           Validates and normalizes raw diamond attributes before they reach
#           the OrdinalEncoder (closes the unseen-category risk noted in
#           INTERVIEW_PREP.md Q25b -- handle_unknown=-1 is silent otherwise).
# version:  1.0

# third-party
from pydantic import BaseModel, Field, field_validator

# internal
import config

_RANGES = config.INPUT_FEATURE_RANGES


class DiamondInput(BaseModel):
    """Raw diamond attributes -- the 7 standard inputs shared by both modules."""

    carat: float = Field(..., ge=_RANGES["carat"]["min"], le=_RANGES["carat"]["max"])
    x: float = Field(..., ge=_RANGES["x"]["min"], le=_RANGES["x"]["max"])
    y: float = Field(..., ge=_RANGES["y"]["min"], le=_RANGES["y"]["max"])
    z: float = Field(..., ge=_RANGES["z"]["min"], le=_RANGES["z"]["max"])
    cut: str
    color: str
    clarity: str

    @field_validator("cut")
    @classmethod
    def validate_cut(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in config.CUT_ORDER:
            raise ValueError(f"cut must be one of {config.CUT_ORDER}, got {value!r}")
        return normalized

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in config.COLOR_ORDER:
            raise ValueError(f"color must be one of {config.COLOR_ORDER}, got {value!r}")
        return normalized

    @field_validator("clarity")
    @classmethod
    def validate_clarity(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in config.CLARITY_ORDER:
            raise ValueError(f"clarity must be one of {config.CLARITY_ORDER}, got {value!r}")
        return normalized


class PricePredictionResponse(BaseModel):
    price_usd: float
    price_inr: float


class ClusterProfile(BaseModel):
    cluster_name: str
    count: int
    avg_carat: float
    avg_price_per_carat_usd: float
    avg_price_usd: float
    avg_price_inr: float
    dominant_cut: str
    dominant_color: str
    dominant_clarity: str


class SegmentPredictionResponse(BaseModel):
    cluster_id: int
    cluster_name: str
    price_usd: float
    price_inr: float
    profile: ClusterProfile
    centroid_distance: float
    centroid_distance_p95: float
    is_ood: bool


class FeatureDrift(BaseModel):
    ks_statistic: float
    p_value: float
    is_drifted: bool


class DriftResponse(BaseModel):
    window_size: int
    alpha: float
    features: dict[str, FeatureDrift]


class HealthResponse(BaseModel):
    status: str
    regression_model_loaded: bool
    clustering_model_loaded: bool
