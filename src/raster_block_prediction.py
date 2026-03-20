from __future__ import annotations

"""LEGACY MODULE: not used by the clean tutorial pipeline entrypoint (run_pipeline.py)."""

import numpy as np
from xgboost import XGBClassifier


def predict_probability_blocks(
    model: XGBClassifier,
    feature_stack: np.ndarray,
    block_size: int = 256,
    batch_size: int = 65536,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict probabilities safely in raster tiles.

    Args:
        model: trained XGBoost classifier
        feature_stack: HxWxC features
        block_size: tile size for memory-safe prediction
        batch_size: max rows per predict call inside each tile
        nodata_mask: optional HxW bool mask
    """
    h, w, c = feature_stack.shape
    out = np.full((h, w), np.nan, dtype=np.float32)

    for r0 in range(0, h, block_size):
        for c0 in range(0, w, block_size):
            r1 = min(h, r0 + block_size)
            c1 = min(w, c0 + block_size)

            tile = feature_stack[r0:r1, c0:c1, :]
            flat = tile.reshape(-1, c)

            valid = np.all(np.isfinite(flat), axis=1)
            if nodata_mask is not None:
                valid &= ~nodata_mask[r0:r1, c0:c1].reshape(-1)

            if not np.any(valid):
                continue

            idx = np.where(valid)[0]
            pred_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
            for i in range(0, len(idx), batch_size):
                chunk = idx[i : i + batch_size]
                pred_flat[chunk] = model.predict_proba(flat[chunk])[:, 1].astype(np.float32)

            out[r0:r1, c0:c1] = pred_flat.reshape(r1 - r0, c1 - c0)

    return out
