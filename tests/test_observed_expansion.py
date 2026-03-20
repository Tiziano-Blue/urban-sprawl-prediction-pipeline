import numpy as np
import pytest

from src.visualization import compute_observed_expansion_mask


def test_compute_observed_expansion_mask_transition_logic():
    built_2010 = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    built_2024 = np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    observed = compute_observed_expansion_mask(built_2010, built_2024)

    expected = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]
    )
    np.testing.assert_array_equal(observed, expected)


def test_compute_observed_expansion_mask_shape_mismatch_raises():
    built_2010 = np.zeros((2, 2), dtype=np.uint8)
    built_2024 = np.zeros((2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        compute_observed_expansion_mask(built_2010, built_2024)
