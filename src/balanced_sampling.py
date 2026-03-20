from __future__ import annotations

"""LEGACY MODULE: kept for backward compatibility; not used by run_pipeline.py tutorial flow."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BalancedSample:
    """Balanced sample container for training."""

    X_train: np.ndarray
    y_train: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


@dataclass
class SampledTrainingData:
    """Legacy-compatible sampled training data container."""

    X: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def build_change_00_10(built_2000: np.ndarray, built_2010: np.ndarray) -> np.ndarray:
    """Positive label: changed from non-built to built between 2000 and 2010."""
    return ((built_2000 == 0) & (built_2010 == 1)).astype(np.uint8)


def build_non_expansion_00_10(built_2000: np.ndarray, built_2010: np.ndarray) -> np.ndarray:
    """Negative label candidate: remained non-built between 2000 and 2010."""
    return ((built_2000 == 0) & (built_2010 == 0)).astype(np.uint8)


def balanced_sampling(
    feature_stack_2000: np.ndarray,
    built_2000: np.ndarray,
    built_2010: np.ndarray,
    random_state: int = 42,
) -> BalancedSample:
    """Create balanced training set.

    Rules:
    - training domain: only non-built pixels in 2000 (built_2000 == 0)
    - label y: built_2010 (1=built, 0=non-built) within that domain
    - positives: all y==1 pixels
    - negatives: random sample at 2x positives from y==0 pixels
    """
    rng = np.random.default_rng(random_state)

    candidate_mask = built_2000 == 0
    label_2000_2010 = (built_2010 == 1) & candidate_mask
    pos_mask = label_2000_2010
    neg_mask = (~label_2000_2010) & candidate_mask

    pos_idx = np.argwhere(pos_mask)
    neg_idx = np.argwhere(neg_mask)

    if len(pos_idx) == 0:
        raise ValueError("No positive expansion pixels found in 2000->2010 labels.")
    if len(neg_idx) == 0:
        raise ValueError("No negative non-expansion pixels found in 2000->2010 labels.")

    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), 2 * n_pos)

    neg_sel = neg_idx[rng.choice(len(neg_idx), size=n_neg, replace=False)]
    all_idx = np.vstack([pos_idx, neg_sel])
    y = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])

    order = rng.permutation(len(all_idx))
    all_idx = all_idx[order]
    y = y[order]

    rows = all_idx[:, 0]
    cols = all_idx[:, 1]
    X = feature_stack_2000[rows, cols, :].astype(np.float32)

    return BalancedSample(X_train=X, y_train=y, rows=rows, cols=cols)


def sample_training_pixels(
    feature_stack_2000: np.ndarray | None = None,
    feature_stack: np.ndarray | None = None,
    built_2000: np.ndarray | None = None,
    built_2010: np.ndarray | None = None,
    built_reference: np.ndarray | None = None,
    change_label: np.ndarray | None = None,
    dist_to_built: np.ndarray | None = None,
    config=None,
    output_csv: str | Path | None = None,
) -> SampledTrainingData:
    """Legacy wrapper kept for old tests and earlier pipeline interfaces.

    Supported calling patterns:
    1) New style: provide `feature_stack_2000`, `built_2000`, `built_2010`
    2) Old style: provide `feature_stack`, `change_label`, `built_reference`
    """
    stack = feature_stack_2000 if feature_stack_2000 is not None else feature_stack
    if stack is None:
        raise ValueError("feature stack is required.")

    if config is None:
        random_state = 42
        sample_pos = 10_000
        sample_neg = 10_000
    elif isinstance(config, dict):
        random_state = int(config.get("project", {}).get("random_state", 42))
        sample_pos = int(config.get("sampling", {}).get("sample_pos", 10_000))
        sample_neg = int(config.get("sampling", {}).get("sample_neg", 10_000))
    else:
        random_state = int(getattr(config, "random_state", 42))
        sample_pos = int(getattr(config, "sample_pos", 10_000))
        sample_neg = int(getattr(config, "sample_neg", 10_000))

    rng = np.random.default_rng(random_state)

    if built_2000 is not None and built_2010 is not None:
        pos_mask = build_change_00_10(built_2000, built_2010) == 1
        neg_mask = build_non_expansion_00_10(built_2000, built_2010) == 1
    elif change_label is not None and built_reference is not None:
        pos_mask = change_label == 1
        neg_mask = (change_label == 0) & (built_reference == 0)
    else:
        raise ValueError("Either (built_2000,built_2010) or (change_label,built_reference) must be provided.")

    pos_idx = np.argwhere(pos_mask)
    neg_idx = np.argwhere(neg_mask)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Insufficient positive/negative samples for training.")

    n_pos = min(sample_pos, len(pos_idx))
    n_neg = min(sample_neg, len(neg_idx))
    pos_sel = pos_idx[rng.choice(len(pos_idx), size=n_pos, replace=False)]
    neg_sel = neg_idx[rng.choice(len(neg_idx), size=n_neg, replace=False)]

    selected = np.vstack([pos_sel, neg_sel])
    y = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])
    order = rng.permutation(len(selected))
    selected = selected[order]
    y = y[order]

    rows = selected[:, 0]
    cols = selected[:, 1]
    X = stack[rows, cols, :].astype(np.float32)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df.insert(0, "col", cols)
        df.insert(0, "row", rows)
        df["label"] = y
        df.to_csv(output_csv, index=False)

    return SampledTrainingData(X=X, y=y, rows=rows, cols=cols)
