from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

TRAIN_FEATURE_ORDER = ["distance_to_built", "distance_to_roads"]


def validate_feature_order(prediction_feature_order: list[str], training_feature_order: list[str]) -> None:
    """Ensure prediction feature order matches training feature order exactly."""
    if prediction_feature_order != training_feature_order:
        raise ValueError(
            "Prediction feature order does not match training feature order. "
            f"Prediction={prediction_feature_order}, Training={training_feature_order}"
        )


def build_prediction_matrix(
    distance_to_built_2024: np.ndarray,
    distance_to_roads: np.ndarray,
    built_2024: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build prediction matrix only for 2024 non-built candidate pixels."""
    candidate_pred = built_2024 == 0
    rows, cols = np.where(candidate_pred)

    X_pred = np.column_stack(
        [
            distance_to_built_2024[rows, cols],
            distance_to_roads[rows, cols],
        ]
    ).astype(np.float32)

    if X_pred.ndim != 2 or X_pred.shape[1] != 2:
        raise ValueError(f"Prediction X must have exactly 2 columns, got shape {X_pred.shape}.")

    if np.isnan(X_pred).any():
        raise ValueError("Prediction X contains NaN values.")

    return X_pred, rows.astype(np.int32), cols.astype(np.int32), candidate_pred


def predict_probability_map(
    model: XGBClassifier,
    X_pred: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    raster_shape: tuple[int, int],
) -> np.ndarray:
    """Predict candidate probabilities and place them back on full raster."""
    probs = model.predict_proba(X_pred)[:, 1].astype(np.float32)
    pred_map = np.full(raster_shape, np.nan, dtype=np.float32)
    pred_map[rows, cols] = probs
    return pred_map
