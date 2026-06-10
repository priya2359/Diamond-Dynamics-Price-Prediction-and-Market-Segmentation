# filename: notebooks/08_regression.py
# purpose:  Section 8 -- Regression: 5 ML models (Linear, DT, RF, XGBoost, KNN)
#           + ANN (TensorFlow/Keras), evaluation (MAE/MSE/RMSE/R2), MLflow logging,
#           save best sklearn Pipeline + ANN.
# version:  1.0

# %% [markdown]
# # Section 8 -- Regression Models
#
# Input: data/processed/diamonds_processed.csv (8 selected features from Section 6,
# carat/volume/table already sqrt-transformed, price already log1p-transformed --
# see data/processed/transform_params.json).
#
# ## Pipeline design
# - Preprocessor: ColumnTransformer (numeric passthrough + OrdinalEncoder for
#   cut/color/clarity/carat_category, fixed categories from config.py) -> StandardScaler.
# - Split: 80% train_full / 20% test (random_state=42). train_full further split
#   80/20 -> 64% train / 16% val (overall). val used ONLY for XGBoost and ANN
#   early stopping. Linear/DT/RF/KNN train on train_full. All 6 models evaluated
#   on the same test set.
# - Hyperparameters: config.py (LINEAR_PARAMS, DT_PARAMS, RF_PARAMS, XGB_PARAMS,
#   KNN_PARAMS, ANN_ARCHITECTURE, ANN_TRAINING).
# - ANN gets its OWN preprocessor instance (separate StandardScaler), per
#   CLAUDE.md rule -- not shared with the sklearn Pipeline.
# - Metrics computed in USD price space (expm1 of the log1p target), per
#   transform_params.json price_transform.inverse = expm1.
#
# ## Risk documented (NOT implemented here -- Phase 2 / FastAPI concern)
# OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) will silently
# map an unseen category string (e.g. "premium" vs "Premium") to -1 instead of
# raising. FastAPI request validation must normalise/validate cut/color/clarity/
# carat_category against config.py's *_ORDER / CARAT_CATEGORY_LABELS lists
# BEFORE calling pipeline.predict().

# %% Setup
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import config
from src.models.regression import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    TARGET,
    split_data,
    build_preprocessor,
    build_full_pipeline,
    get_sklearn_models,
    evaluate_regression,
    build_ann,
    get_ann_callbacks,
)
from src.utils.helpers import save_figure, save_json

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES = config.FIGURES_DIR
tf.random.set_seed(config.RANDOM_STATE)

df = pd.read_csv(config.PROCESSED_DATA_DIR / "diamonds_processed.csv")
print("Loaded:", df.shape)
print("Features:", FEATURE_ORDER)
print("Target:", TARGET, "(log1p-transformed price_usd)")

# %% [markdown]
# ## 1. Train / Val / Test Split

# %% Split
splits = split_data(df)
for k, v in splits.items():
    print(f"  {k:<14} {v.shape}")

# %% [markdown]
# ## 2. Preprocessing
#
# Two separate fitted preprocessor instances (sklearn models vs ANN), both
# fit on X_train_full (80%) -- OrdinalEncoder categories are fixed from
# config.py (no leakage), StandardScaler stats from 80% vs 64% are
# negligibly different.

# %% Fit preprocessors
preprocessor = build_preprocessor()
X_train_full_t = preprocessor.fit_transform(splits["X_train_full"], splits["y_train_full"])
X_train_t = preprocessor.transform(splits["X_train"])
X_val_t = preprocessor.transform(splits["X_val"])
X_test_t = preprocessor.transform(splits["X_test"])

ann_preprocessor = build_preprocessor()
X_train_full_ann = ann_preprocessor.fit_transform(splits["X_train_full"], splits["y_train_full"])
X_train_ann = ann_preprocessor.transform(splits["X_train"])
X_val_ann = ann_preprocessor.transform(splits["X_val"])
X_test_ann = ann_preprocessor.transform(splits["X_test"])

print("Transformed shapes -- train_full:", X_train_full_t.shape, " test:", X_test_t.shape)
print("Feature order after preprocessing:", FEATURE_ORDER)

# %% [markdown]
# ## 3. MLflow Setup

# %% MLflow
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_REGRESSION)
print("MLflow tracking URI:", config.MLFLOW_TRACKING_URI)

# %% [markdown]
# ## 4. Train 5 ML Models (Linear, Decision Tree, Random Forest, XGBoost, KNN)

# %% Train + evaluate loop
y_train_full = splits["y_train_full"].values
y_train = splits["y_train"].values
y_val = splits["y_val"].values
y_test = splits["y_test"].values

models = get_sklearn_models()
results: dict[str, dict] = {}
fitted_models: dict = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")

    if name == "xgboost":
        model.fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
            verbose=False,
        )
        print(f"  best_iteration: {model.best_iteration}")
    else:
        model.fit(X_train_full_t, y_train_full)

    y_pred_log = model.predict(X_test_t)
    metrics = evaluate_regression(y_test, y_pred_log)
    results[name] = metrics
    fitted_models[name] = model

    print(
        f"  MAE=${metrics['mae']:.2f}  RMSE=${metrics['rmse']:.2f}  "
        f"R2={metrics['r2']:.4f}  R2(log)={metrics['r2_log_scale']:.4f}"
    )

    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
        mlflow.log_metrics(metrics)

# %% [markdown]
# ## 5. ANN (TensorFlow/Keras)
#
# Architecture: 8 -> Dense(64, relu) -> BatchNorm -> Dropout(0.2)
#                 -> Dense(32, relu) -> BatchNorm -> Dropout(0.2) -> Dense(1, linear)
# Optimizer: Adam(lr=0.001) | loss: mse | metrics: mae | batch_size: 256 | epochs<=100
# Callbacks: EarlyStopping(patience=10, restore_best_weights=True),
#            ReduceLROnPlateau(factor=0.5, patience=5)

# %% Build ANN
ann = build_ann(input_dim=X_train_ann.shape[1])
ann.summary()

# %% Train ANN
history = ann.fit(
    X_train_ann, y_train,
    validation_data=(X_val_ann, y_val),
    batch_size=config.ANN_TRAINING["batch_size"],
    epochs=config.ANN_TRAINING["epochs"],
    callbacks=get_ann_callbacks(),
    verbose=2,
)

# %% Evaluate ANN
y_pred_log_ann = ann.predict(X_test_ann, verbose=0).flatten()
ann_metrics = evaluate_regression(y_test, y_pred_log_ann)
results["ann"] = ann_metrics
print(
    f"\nANN: MAE=${ann_metrics['mae']:.2f}  RMSE=${ann_metrics['rmse']:.2f}  "
    f"R2={ann_metrics['r2']:.4f}  R2(log)={ann_metrics['r2_log_scale']:.4f}"
)
print(f"Epochs trained (early stopping): {len(history.history['loss'])}")

with mlflow.start_run(run_name="ann"):
    mlflow.log_params({
        "architecture": "8-64-32-1",
        "batch_norm": True,
        "dropout": 0.2,
        **{k: str(v) for k, v in config.ANN_TRAINING.items()},
    })
    mlflow.log_metrics(ann_metrics)
    mlflow.log_metric("epochs_trained", len(history.history["loss"]))

# %% [markdown]
# ## 6. Model Comparison

# %% Comparison table
comparison_df = pd.DataFrame(results).T[["mae", "mse", "rmse", "r2", "r2_log_scale"]].round(4)
comparison_df = comparison_df.sort_values("r2", ascending=False)
print(comparison_df)

best_overall = comparison_df.index[0]
sklearn_only = comparison_df.drop(index="ann", errors="ignore")
best_sklearn = sklearn_only.index[0]
print(f"\nBest overall (incl. ANN): {best_overall}  R2={comparison_df.loc[best_overall, 'r2']:.4f}")
print(f"Best sklearn model:       {best_sklearn}  R2={comparison_df.loc[best_sklearn, 'r2']:.4f}")

# %% [markdown]
# ## 7. Figures

# %% Figure 1: Model comparison (R2 and RMSE)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = comparison_df.index.tolist()

sns.barplot(x=comparison_df["r2"], y=order, ax=axes[0], hue=order, legend=False, palette="Blues_r")
axes[0].set_title("R2 by model (USD price space)")
axes[0].set_xlabel("R2")
axes[0].axvline(0.95, color="red", linestyle="--", linewidth=1, label="target R2=0.95")
axes[0].legend()

sns.barplot(x=comparison_df["rmse"], y=order, ax=axes[1], hue=order, legend=False, palette="Oranges_r")
axes[1].set_title("RMSE by model (USD)")
axes[1].set_xlabel("RMSE ($)")

fig.suptitle("Section 8 -- Model Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "08_01_model_comparison.png", FIGURES)

# %% Figure 2: Actual vs Predicted (best sklearn model + ANN)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

best_pred_usd = np.expm1(fitted_models[best_sklearn].predict(X_test_t))
ann_pred_usd = np.expm1(y_pred_log_ann)
actual_usd = np.expm1(y_test)

for ax, pred, title in zip(axes, [best_pred_usd, ann_pred_usd], [best_sklearn, "ann"]):
    ax.scatter(actual_usd, pred, alpha=0.15, s=8, color="#3B82F6")
    lims = [0, max(actual_usd.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Predicted price (USD)")
    ax.set_title(f"{title}: Actual vs Predicted (R2={results[title]['r2']:.4f})")

fig.suptitle("Section 8 -- Actual vs Predicted (Test Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "08_02_actual_vs_predicted.png", FIGURES)

# %% Figure 3: Residuals (best sklearn model + ANN)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, pred, title in zip(axes, [best_pred_usd, ann_pred_usd], [best_sklearn, "ann"]):
    residuals = actual_usd - pred
    ax.scatter(pred, residuals, alpha=0.15, s=8, color="#10B981")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted price (USD)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{title}: Residuals")

fig.suptitle("Section 8 -- Residual Plots (Test Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "08_03_residuals.png", FIGURES)

# %% Figure 4: Feature contribution -- RF / XGBoost importances + Linear coefficients
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

rf_importances = fitted_models["random_forest"].feature_importances_
axes[0].barh(FEATURE_ORDER, rf_importances, color="#6366F1")
axes[0].set_title("Random Forest -- feature importance")
axes[0].invert_yaxis()

xgb_importances = fitted_models["xgboost"].feature_importances_
axes[1].barh(FEATURE_ORDER, xgb_importances, color="#F59E0B")
axes[1].set_title("XGBoost -- feature importance")
axes[1].invert_yaxis()

linear_coefs = np.abs(fitted_models["linear_regression"].coef_)
axes[2].barh(FEATURE_ORDER, linear_coefs, color="#EC4899")
axes[2].set_title("Linear Regression -- |standardized coef|")
axes[2].invert_yaxis()

fig.suptitle("Section 8 -- Feature Contribution by Model", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "08_04_feature_importance.png", FIGURES)

# %% Figure 5: ANN training curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("ANN -- Loss (MSE, log1p price space)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()

axes[1].plot(history.history["mae"], label="train")
axes[1].plot(history.history["val_mae"], label="val")
axes[1].set_title("ANN -- MAE (log1p price space)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()

fig.suptitle("Section 8 -- ANN Training Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure(fig, "08_05_ann_training_curves.png", FIGURES)

# %% [markdown]
# ## 8. Save Artifacts

# %% Save metrics + comparison table
metrics_artifact = {
    "features": FEATURE_ORDER,
    "target": f"{TARGET} (log1p(price_usd))",
    "split": {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        "random_state": config.RANDOM_STATE,
        "n_train": len(splits["X_train"]),
        "n_val": len(splits["X_val"]),
        "n_train_full": len(splits["X_train_full"]),
        "n_test": len(splits["X_test"]),
    },
    "metrics": {name: {k: round(v, 4) for k, v in m.items()} for name, m in results.items()},
    "best_overall": best_overall,
    "best_sklearn": best_sklearn,
}
save_json(metrics_artifact, config.REGRESSION_ARTIFACTS_DIR / "metrics.json")

comparison_df.to_csv(config.REGRESSION_ARTIFACTS_DIR / "model_comparison.csv")
print(f"Saved: {config.REGRESSION_ARTIFACTS_DIR / 'model_comparison.csv'}")

# %% Save best sklearn pipeline (preprocessor + model) and shared preprocessor
best_pipeline = build_full_pipeline(preprocessor, fitted_models[best_sklearn])
joblib.dump(best_pipeline, config.REGRESSION_ARTIFACTS_DIR / "best_model.pkl")
joblib.dump(preprocessor, config.REGRESSION_ARTIFACTS_DIR / "preprocessor.pkl")
print(f"Saved: best_model.pkl ({best_sklearn})")
print(f"Saved: preprocessor.pkl")

# %% Save ANN (own preprocessor, .keras format)
ann.save(config.REGRESSION_ARTIFACTS_DIR / "ann_model.keras")
joblib.dump(ann_preprocessor, config.REGRESSION_ARTIFACTS_DIR / "ann_preprocessor.pkl")
print(f"Saved: ann_model.keras, ann_preprocessor.pkl")

# %% [markdown]
# ## 9. Final Summary

# %% Summary
print("\n=== Section 8 Summary ===")
print(comparison_df)
print(f"\nBest overall:  {best_overall}")
print(f"Best sklearn:  {best_sklearn}")
print(f"Target R2 >= 0.95: {'MET' if comparison_df.loc[best_overall, 'r2'] >= 0.95 else 'NOT MET'}")
