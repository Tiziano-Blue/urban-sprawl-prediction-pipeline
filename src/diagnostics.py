from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .io_utils import RasterLayer, valid_data_mask


@dataclass
class DiagnosticsResult:
    """Structured diagnostics output."""

    diagnostics: dict[str, Any]
    validation_passed: bool


def _unique_preview(values: np.ndarray, max_values: int = 20) -> list[Any]:
    preview = values[:max_values]
    return [float(v) if isinstance(v, np.floating) else int(v) for v in preview]


def summarize_raster(layer: RasterLayer) -> dict[str, Any]:
    """Compute core per-raster diagnostics."""
    arr = layer.array
    valid = valid_data_mask(arr, layer.nodata)
    vals = arr[valid]

    if vals.size == 0:
        min_val = None
        max_val = None
        unique_values = np.array([], dtype=float)
    else:
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        unique_values = np.unique(vals)

    unique_count = int(unique_values.size)
    preview = _unique_preview(unique_values, max_values=20) if unique_count <= 200 else None

    return {
        "path": str(layer.path),
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "nodata": None if layer.nodata is None else float(layer.nodata),
        "min": min_val,
        "max": max_val,
        "unique_count": unique_count,
        "first_20_unique_values": preview,
    }


def _is_binary_layer(layer: RasterLayer) -> bool:
    arr = layer.array
    valid = valid_data_mask(arr, layer.nodata)
    vals = np.unique(arr[valid])
    if vals.size == 0:
        return False
    return bool(np.all(np.isin(vals, [0, 1])))


def _distance_semantics(layer: RasterLayer) -> dict[str, Any]:
    arr = layer.array
    valid = valid_data_mask(arr, layer.nodata)
    vals = arr[valid]
    unique_vals = np.unique(vals)
    is_binary = bool(np.all(np.isin(unique_vals, [0, 1]))) if unique_vals.size > 0 else False
    return {
        "unique_count": int(unique_vals.size),
        "is_binary_like": is_binary,
        "looks_continuous_multi_valued": bool(unique_vals.size > 2 and not is_binary),
    }


def _evaluate_distance_alignment(distance_layer: RasterLayer, built_layer: RasterLayer) -> dict[str, Any]:
    dist_arr = distance_layer.array
    built_arr = built_layer.array

    dist_valid = valid_data_mask(dist_arr, distance_layer.nodata)
    built_valid = valid_data_mask(built_arr, built_layer.nodata)

    dist_vals = dist_arr[dist_valid]
    nonnegative = bool(np.all(dist_vals >= 0)) if dist_vals.size > 0 else False

    zero_mask = dist_valid & np.isclose(dist_arr, 0.0)
    has_zero_values = bool(np.any(zero_mask))

    built_mask = built_valid & (built_arr == 1)

    overlap = int(np.count_nonzero(zero_mask & built_mask))
    zero_count = int(np.count_nonzero(zero_mask))
    built_count = int(np.count_nonzero(built_mask))
    union_count = int(np.count_nonzero(zero_mask | built_mask))

    precision_zero_on_built = (overlap / zero_count) if zero_count > 0 else None
    recall_built_captured_by_zero = (overlap / built_count) if built_count > 0 else None
    jaccard = (overlap / union_count) if union_count > 0 else None

    alignment_good = (
        precision_zero_on_built is not None
        and recall_built_captured_by_zero is not None
        and precision_zero_on_built >= 0.8
        and recall_built_captured_by_zero >= 0.8
    )

    usable = bool(nonnegative and has_zero_values and alignment_good)

    return {
        "nonnegative": nonnegative,
        "contains_zero_values": has_zero_values,
        "zero_pixel_count": zero_count,
        "built_pixel_count": built_count,
        "zero_and_built_overlap_count": overlap,
        "precision_zero_on_built": precision_zero_on_built,
        "recall_built_captured_by_zero": recall_built_captured_by_zero,
        "jaccard_zero_vs_built": jaccard,
        "zero_distance_alignment_good": alignment_good,
        "usable": usable,
    }


def run_diagnostics(layers: dict[str, RasterLayer]) -> DiagnosticsResult:
    """Run all required diagnostics and return structured results."""
    print("[Diagnostics] Step A: Inspecting raster statistics...")

    raster_summaries = {name: summarize_raster(layer) for name, layer in layers.items()}
    for name, summary in raster_summaries.items():
        print(
            f"  - {name}: shape={summary['shape']}, dtype={summary['dtype']}, "
            f"nodata={summary['nodata']}, min={summary['min']}, max={summary['max']}, "
            f"unique_count={summary['unique_count']}"
        )
        if summary["first_20_unique_values"] is not None:
            print(f"    first_20_unique_values={summary['first_20_unique_values']}")

    print("[Diagnostics] Step B: Verifying raster compatibility...")
    shapes = {name: tuple(layer.array.shape) for name, layer in layers.items()}
    all_same_shape = len(set(shapes.values())) == 1
    if not all_same_shape:
        raise ValueError(
            "Raster shape mismatch detected. All rasters must have the same shape. "
            f"Shapes found: {shapes}"
        )

    missing_transform = [name for name, layer in layers.items() if layer.transform is None]
    missing_crs = [name for name, layer in layers.items() if layer.crs is None]
    if missing_transform or missing_crs:
        print(
            "  - Some rasters are missing transform/CRS. "
            "Proceeding by treating rasters as aligned by pixel grid only."
        )

    compatibility = {
        "all_same_shape": all_same_shape,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "missing_transform": missing_transform,
        "missing_crs": missing_crs,
        "pixel_grid_only_alignment_assumed": bool(missing_transform or missing_crs),
    }

    print("[Diagnostics] Step C: Checking semantic assumptions...")
    semantics = {
        "landuse2000_is_binary_0_1": _is_binary_layer(layers["landuse2000"]),
        "landuse2010_is_binary_0_1": _is_binary_layer(layers["landuse2010"]),
        "landuse2024_is_binary_0_1": _is_binary_layer(layers["landuse2024"]),
        "roads_is_binary_0_1": _is_binary_layer(layers["roads"]),
        "distance2000_semantics": _distance_semantics(layers["distance2000"]),
        "distance2010_semantics": _distance_semantics(layers["distance2010"]),
        "distance2024_semantics": _distance_semantics(layers["distance2024"]),
    }

    for key, value in semantics.items():
        print(f"  - {key}: {value}")

    print("[Diagnostics] Step D: Evaluating provided distance raster usability...")
    distance_checks = {
        "distance2000_vs_landuse2000": _evaluate_distance_alignment(
            layers["distance2000"], layers["landuse2000"]
        ),
        "distance2010_vs_landuse2010": _evaluate_distance_alignment(
            layers["distance2010"], layers["landuse2010"]
        ),
        "distance2024_vs_landuse2024": _evaluate_distance_alignment(
            layers["distance2024"], layers["landuse2024"]
        ),
    }
    for key, value in distance_checks.items():
        print(f"  - {key}: {value}")

    existing_distance_usable = all(v["usable"] for v in distance_checks.values())

    diagnostics = {
        "step_a_raster_inspection": raster_summaries,
        "step_b_compatibility": compatibility,
        "step_c_semantic_assumptions": semantics,
        "step_d_distance_raster_usability": {
            "checks": distance_checks,
            "all_provided_distances_look_usable": existing_distance_usable,
            "note": (
                "Pipeline will recompute all required distances explicitly and will not "
                "depend on these provided distance rasters as final model inputs."
            ),
        },
    }

    binary_checks = [
        semantics["landuse2000_is_binary_0_1"],
        semantics["landuse2010_is_binary_0_1"],
        semantics["landuse2024_is_binary_0_1"],
        semantics["roads_is_binary_0_1"],
    ]
    distance_semantic_checks = [
        semantics["distance2000_semantics"]["looks_continuous_multi_valued"],
        semantics["distance2010_semantics"]["looks_continuous_multi_valued"],
        semantics["distance2024_semantics"]["looks_continuous_multi_valued"],
    ]

    validation_passed = bool(all(binary_checks) and all(distance_semantic_checks) and all_same_shape)

    return DiagnosticsResult(diagnostics=diagnostics, validation_passed=validation_passed)
