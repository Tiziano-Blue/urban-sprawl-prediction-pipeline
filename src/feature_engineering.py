from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _assert_binary_mask(name: str, arr: np.ndarray) -> None:
    """Ensure a raster behaves like a binary 0/1 mask."""
    unique_vals = np.unique(arr[np.isfinite(arr)])
    if unique_vals.size == 0 or not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"{name} must be binary 0/1. Found unique values: {unique_vals[:20]}"
        )


def compute_distance_from_feature(feature_mask: np.ndarray) -> np.ndarray:
    """Distance to nearest feature pixel.

    Rules:
    - feature pixels have distance 0
    - all other pixels get Euclidean pixel distance
    """
    feature_mask = feature_mask.astype(bool)
    # distance_transform_edt computes distance to nearest zero-valued element.
    # By passing ~feature_mask, feature pixels become zeros.
    distance = distance_transform_edt(~feature_mask)
    return distance.astype(np.float32)


def compute_required_distances(
    built_2000: np.ndarray,
    built_2024: np.ndarray,
    roads: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the three required distance rasters for this pipeline."""
    _assert_binary_mask("Built_2000", built_2000)
    _assert_binary_mask("Built_2024", built_2024)
    _assert_binary_mask("roads", roads)

    built_2000_mask = built_2000 == 1
    built_2024_mask = built_2024 == 1
    roads_mask = roads == 1

    return {
        "distance_to_built_2000": compute_distance_from_feature(built_2000_mask),
        "distance_to_built_2024": compute_distance_from_feature(built_2024_mask),
        "distance_to_roads": compute_distance_from_feature(roads_mask),
    }
