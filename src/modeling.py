from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBClassifier


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": 4,
}


def _base_xgb_params(
    random_state: int,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared model hyperparameters required by specification."""
    params = DEFAULT_XGB_PARAMS.copy()
    if xgb_params:
        params.update(xgb_params)
    params["random_state"] = random_state
    return params


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute required binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(np.uint8)
    roc_auc = _safe_roc_auc(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def select_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Select threshold that maximizes F1 on provided validation labels/probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, 0.0

    f1_scores = (2.0 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train_model_with_internal_validation(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    xgb_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 20,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """Train model with 20% stratified split and early stopping."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    params = _base_xgb_params(random_state=random_state, xgb_params=xgb_params)
    model = XGBClassifier(**params, early_stopping_rounds=int(early_stopping_rounds))
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    y_prob_valid = model.predict_proba(X_valid)[:, 1]
    selected_threshold, selected_f1 = select_threshold_by_f1(y_valid, y_prob_valid)
    metrics = compute_binary_metrics(y_valid, y_prob_valid, threshold=selected_threshold)
    metrics_at_05 = compute_binary_metrics(y_valid, y_prob_valid, threshold=0.5)
    metrics.update(
        {
            "selected_threshold_f1": selected_threshold,
            "selected_threshold_f1_score": selected_f1,
            "metrics_at_threshold_0_5": metrics_at_05,
            "n_train": int(X_train.shape[0]),
            "n_valid": int(X_valid.shape[0]),
        }
    )

    return model, metrics


def run_spatial_group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    block_size: int = 64,
    n_splits: int = 5,
    random_state: int = 42,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run spatial GroupKFold using row/col block groups."""
    row_group = rows // block_size
    col_group = cols // block_size
    groups = row_group.astype(np.int64) * 1_000_000 + col_group.astype(np.int64)

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Not enough spatial groups for GroupKFold cross-validation.")

    n_actual_splits = int(min(n_splits, unique_groups.size))
    gkf = GroupKFold(n_splits=n_actual_splits)

    fold_results: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        fold_model = XGBClassifier(
            **_base_xgb_params(random_state + fold_idx, xgb_params=xgb_params)
        )
        fold_model.fit(X[train_idx], y[train_idx], verbose=False)

        y_prob = fold_model.predict_proba(X[test_idx])[:, 1]
        fold_metrics = compute_binary_metrics(y[test_idx], y_prob)
        fold_metrics.update(
            {
                "fold": fold_idx,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
            }
        )
        fold_results.append(fold_metrics)

    metric_keys = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    mean_metrics: dict[str, Any] = {}
    std_metrics: dict[str, Any] = {}
    for key in metric_keys:
        vals = [np.nan if result[key] is None else float(result[key]) for result in fold_results]
        vals_np = np.asarray(vals, dtype=float)
        mean_metrics[key] = None if np.isnan(vals_np).all() else float(np.nanmean(vals_np))
        std_metrics[key] = None if np.isnan(vals_np).all() else float(np.nanstd(vals_np))

    return {
        "n_splits": n_actual_splits,
        "block_size": int(block_size),
        "folds": fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }
