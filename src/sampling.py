from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FullTrainingData:
    """All candidate training pixels before balancing."""

    X: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    candidate_mask: np.ndarray
    positive_count_full: int
    negative_count_full: int


@dataclass
class BalancedTrainingSample:
    """Balanced sampled training data used for model fitting."""

    X: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    positive_count_sampled: int
    negative_count_sampled: int
    positive_count_full: int
    negative_count_full: int


def prepare_training_data(
    distance_to_built_2000: np.ndarray,
    distance_to_roads: np.ndarray,
    built_2000: np.ndarray,
    built_2010: np.ndarray,
) -> FullTrainingData:
    """Build full candidate X and Y for 2000->2010 expansion modeling."""
    candidate_mask = built_2000 == 0

    rows, cols = np.where(candidate_mask)
    X = np.column_stack(
        [
            distance_to_built_2000[rows, cols],
            distance_to_roads[rows, cols],
        ]
    ).astype(np.float32)

    y = np.where(built_2010[rows, cols] == 1, 1, 0).astype(np.uint8)

    positive_count_full = int(np.count_nonzero(y == 1))
    negative_count_full = int(np.count_nonzero(y == 0))

    return FullTrainingData(
        X=X,
        y=y,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        candidate_mask=candidate_mask,
        positive_count_full=positive_count_full,
        negative_count_full=negative_count_full,
    )


def balanced_sample(
    full_data: FullTrainingData,
    random_state: int = 42,
    negative_multiplier: int = 2,
) -> BalancedTrainingSample:
    """Use all positives and a random subset of negatives (up to 2x positives)."""
    rng = np.random.default_rng(random_state)

    pos_idx = np.where(full_data.y == 1)[0]
    neg_idx = np.where(full_data.y == 0)[0]

    if pos_idx.size == 0:
        raise ValueError("No positive samples found (Y=1).")
    if neg_idx.size == 0:
        raise ValueError("No negative samples found (Y=0).")

    n_pos = int(pos_idx.size)
    n_neg = int(min(neg_idx.size, negative_multiplier * n_pos))
    selected_neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)

    selected = np.concatenate([pos_idx, selected_neg_idx])
    rng.shuffle(selected)

    X_sampled = full_data.X[selected]
    y_sampled = full_data.y[selected]
    rows_sampled = full_data.rows[selected]
    cols_sampled = full_data.cols[selected]

    return BalancedTrainingSample(
        X=X_sampled,
        y=y_sampled,
        rows=rows_sampled,
        cols=cols_sampled,
        positive_count_sampled=n_pos,
        negative_count_sampled=n_neg,
        positive_count_full=full_data.positive_count_full,
        negative_count_full=full_data.negative_count_full,
    )


def validate_training_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Safety checks required before model training."""
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Training X must have exactly 2 columns, got shape {X.shape}.")

    if np.isnan(X).any():
        raise ValueError("Training X contains NaN values.")

    unique_y = np.unique(y)
    if not np.array_equal(unique_y, np.array([0, 1], dtype=unique_y.dtype)):
        raise ValueError(f"Training Y must be binary 0/1 and contain both classes. Got: {unique_y}")
