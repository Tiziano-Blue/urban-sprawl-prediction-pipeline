from __future__ import annotations

"""LEGACY MODULE: not used by the clean tutorial pipeline entrypoint (run_pipeline.py)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier


def default_xgb_params() -> dict:
    """Required model settings from project spec."""
    return {
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


def scale_pos_weight_from_labels(y: np.ndarray) -> float:
    """Compute scale_pos_weight = negative / positive."""
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, prefix: str = "valid") -> dict:
    """Compute ROC AUC, precision, recall, F1."""
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_prob)),
        f"{prefix}_pr_auc": float(average_precision_score(y_true, y_prob)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    early_stopping_rounds: int = 20,
    use_gpu: bool = False,
) -> tuple[XGBClassifier, dict]:
    """Train XGBoost with early stopping and validation monitoring."""
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )

    params = default_xgb_params()
    if use_gpu:
        params["tree_method"] = "gpu_hist"
    params["random_state"] = random_state
    params["scale_pos_weight"] = scale_pos_weight_from_labels(y_fit)

    model = XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_eval, y_eval)],
        verbose=False,
    )

    fit_prob = model.predict_proba(X_fit)[:, 1]
    eval_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "scale_pos_weight": float(params["scale_pos_weight"]),
        "best_iteration": int(getattr(model, "best_iteration", params["n_estimators"])),
    }
    metrics.update(compute_metrics(y_fit, fit_prob, prefix="train_internal"))
    metrics.update(compute_metrics(y_eval, eval_prob, prefix="eval_internal"))
    return model, metrics


def export_feature_importance(model: XGBClassifier, feature_names: list[str], csv_path: str | Path, png_path: str | Path) -> None:
    """Export feature importance to CSV and PNG."""
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp.astype(float)}).sort_values(
        "importance", ascending=False
    )

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(df["feature"], df["importance"], color="#4C78A8")
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()


def export_shap_importance(
    model: XGBClassifier,
    X_reference: np.ndarray,
    feature_names: list[str],
    csv_path: str | Path,
    sample_size: int = 5000,
    random_state: int = 42,
) -> None:
    """Export mean absolute TreeSHAP importance to CSV."""
    rng = np.random.default_rng(random_state)
    n = min(sample_size, X_reference.shape[0])
    idx = rng.choice(X_reference.shape[0], size=n, replace=False)
    Xs = X_reference[idx]

    contrib = model.get_booster().predict(DMatrix(Xs), pred_contribs=True)
    shap_mean = np.abs(contrib[:, :-1]).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_mean.astype(float)})
    df = df.sort_values("mean_abs_shap", ascending=False)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def save_json(path: str | Path, obj: dict) -> None:
    """Save dictionary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
