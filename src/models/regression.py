# filename: src/models/regression.py
# purpose:  Section 8 -- Preprocessing pipeline, model builders, and evaluation
#           helpers for diamond price regression (5 ML models + ANN)
# version:  1.0

# stdlib
import logging
from typing import Any

# third-party
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers

# internal
import config
from src.features.encode import build_ordinal_encoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature spec -- from artifacts/regression/selected_features.json (Section 6).
# CATEGORICAL_FEATURES order matches sorted(ORDINAL_FEATURE_MAP.keys()) so it
# lines up with the category order returned by build_ordinal_encoder().
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["carat", "volume", "depth", "table"]
CATEGORICAL_FEATURES = ["carat_category", "clarity", "color", "cut"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"  # already log1p-transformed in diamonds_processed.csv (Section 5)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """
    80% train_full / 20% test, then train_full split 80/20 -> 64% train / 16% val
    (overall). val is used only for XGBoost and ANN early stopping; Linear/DT/RF/KNN
    train on train_full. All models are evaluated on the same test set.
    """
    X = df[FEATURE_ORDER]
    y = df[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
    )
    logger.info(
        "Split sizes -- train: %d, val: %d, train_full: %d, test: %d",
        len(X_train), len(X_val), len(X_train_full), len(X_test),
    )
    return {
        "X_train_full": X_train_full, "y_train_full": y_train_full,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Preprocessing pipeline (shared by Linear / DT / RF / XGBoost / KNN)
#
# Risk (documented for Phase 2 FastAPI, NOT handled here): OrdinalEncoder
# raises ValueError on an unseen category string (e.g. "premium" vs "Premium").
# FastAPI request validation must normalise/validate cut/color/clarity/
# carat_category against config.py's *_ORDER lists before calling .predict().
# ---------------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    """ColumnTransformer (passthrough numeric + ordinal-encode categorical) -> StandardScaler."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", build_ordinal_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("encode", column_transformer),
        ("scale", StandardScaler()),
    ])


def build_full_pipeline(preprocessor: Pipeline, model) -> Pipeline:
    """Wrap an already-fitted preprocessor and an already-fitted model for saving/serving."""
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Sklearn-API model builders (hyperparameters from config.py)
# ---------------------------------------------------------------------------
def get_sklearn_models() -> dict:
    """Unfit estimators for the 5 required ML models."""
    return {
        "linear_regression": LinearRegression(**config.LINEAR_PARAMS),
        "decision_tree": DecisionTreeRegressor(**config.DT_PARAMS),
        "random_forest": RandomForestRegressor(**config.RF_PARAMS),
        "knn": KNeighborsRegressor(**config.KNN_PARAMS),
        "xgboost": XGBRegressor(**config.XGB_PARAMS),
    }


# ---------------------------------------------------------------------------
# Evaluation -- metrics computed in USD price space (expm1 of log1p target)
# ---------------------------------------------------------------------------
def evaluate_regression(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """MAE/MSE/RMSE/R2/MAPE in USD price space, plus R2 in log1p space for reference."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred_safe = np.maximum(y_pred, 0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_log_scale": float(r2_score(y_true_log, y_pred_log)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred_safe) * 100),
    }


# ---------------------------------------------------------------------------
# ANN (TensorFlow/Keras) -- separate StandardScaler, separate from sklearn Pipeline
# ---------------------------------------------------------------------------
def build_ann(input_dim: int) -> keras.Model:
    """64 -> 32 -> 1 with BatchNorm + Dropout(0.2) per hidden layer (config.ANN_ARCHITECTURE)."""
    arch = config.ANN_ARCHITECTURE
    train_cfg = config.ANN_TRAINING

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    hidden_layers: list[dict[str, Any]] = arch["hidden_layers"]  # type: ignore[assignment]
    for layer_cfg in hidden_layers:
        x = layers.Dense(layer_cfg["units"], activation=layer_cfg["activation"])(x)
        if layer_cfg.get("batch_norm"):
            x = layers.BatchNormalization()(x)
        if layer_cfg.get("dropout"):
            x = layers.Dropout(layer_cfg["dropout"])(x)
    outputs = layers.Dense(1, activation=arch["output_activation"])(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss=train_cfg["loss"],
        metrics=train_cfg["metrics"],
    )
    return model


def get_ann_callbacks() -> list:
    train_cfg = config.ANN_TRAINING
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["reduce_lr_factor"],
            patience=train_cfg["reduce_lr_patience"],
        ),
    ]
