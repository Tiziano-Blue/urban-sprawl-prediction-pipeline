from __future__ import annotations

"""LEGACY MODULE: not used by the clean tutorial pipeline entrypoint (run_pipeline.py)."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


def spatial_group_ids(rows: np.ndarray, cols: np.ndarray, block_size: int) -> np.ndarray:
    """Generate group id from row/col blocks for spatial CV."""
    return (rows // block_size).astype(np.int64) * 1_000_000 + (cols // block_size).astype(np.int64)


def spatial_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    model_params: dict,
    block_size: int = 32,
    n_splits: int = 5,
) -> dict:
    """Spatial GroupKFold CV with ROC AUC / precision / recall / F1."""
    groups = spatial_group_ids(rows, cols, block_size)
    gkf = GroupKFold(n_splits=n_splits)

    aucs, precs, recs, f1s = [], [], [], []

    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        clf = XGBClassifier(**model_params)
        clf.fit(X[tr_idx], y[tr_idx], verbose=False)
        prob = clf.predict_proba(X[te_idx])[:, 1]
        pred = (prob >= 0.5).astype(np.uint8)

        aucs.append(roc_auc_score(y[te_idx], prob))
        precs.append(precision_score(y[te_idx], pred, zero_division=0))
        recs.append(recall_score(y[te_idx], pred, zero_division=0))
        f1s.append(f1_score(y[te_idx], pred, zero_division=0))

    return {
        "spatial_cv_roc_auc": float(np.mean(aucs)),
        "spatial_cv_precision": float(np.mean(precs)),
        "spatial_cv_recall": float(np.mean(recs)),
        "spatial_cv_f1": float(np.mean(f1s)),
    }
