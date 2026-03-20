from __future__ import annotations

"""LEGACY MODULE: not used by the clean tutorial pipeline entrypoint (run_pipeline.py)."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import rasterio

RasterMeta = Dict[str, object]


def load_raster(path: str | Path) -> Tuple[np.ndarray, RasterMeta]:
    """Load a single-band GeoTIFF.

    Returns:
        arr: 2D numpy array
        meta: raster metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected single-band raster at {path}, got {src.count} bands")
        arr = src.read(1)
        meta = src.meta.copy()
        meta.update(
            {
                "height": src.height,
                "width": src.width,
                "crs": src.crs,
                "transform": src.transform,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
            }
        )
    return arr, meta


def save_raster(path: str | Path, arr: np.ndarray, meta: RasterMeta, dtype: str = "float32", nodata=np.nan) -> None:
    """Save single-band raster with target dtype."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_meta = meta.copy()
    out_meta.update(dtype=dtype, count=1, nodata=nodata)

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(arr.astype(dtype), 1)


def check_alignment(metas: Iterable[RasterMeta]) -> None:
    """Ensure all rasters share shape/CRS/transform."""
    metas = list(metas)
    if not metas:
        raise ValueError("No metadata provided to check_alignment().")

    ref = metas[0]
    ref_shape = (ref["height"], ref["width"])
    ref_crs = ref["crs"]
    ref_transform = ref["transform"]

    for i, m in enumerate(metas[1:], start=1):
        shape = (m["height"], m["width"])
        if shape != ref_shape:
            raise ValueError(f"Alignment error at index {i}: shape {shape} != {ref_shape}")
        if m["crs"] != ref_crs:
            raise ValueError(f"Alignment error at index {i}: crs {m['crs']} != {ref_crs}")
        if m["transform"] != ref_transform:
            raise ValueError(f"Alignment error at index {i}: transform mismatch")


def nodata_mask(arr: np.ndarray, meta: RasterMeta) -> np.ndarray:
    """Return boolean NoData mask."""
    nd = meta.get("nodata")
    if nd is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd, float) and np.isnan(nd):
        return np.isnan(arr)
    return arr == nd


def load_stack(data_dir: str | Path, file_map: Dict[str, str]) -> tuple[Dict[str, np.ndarray], Dict[str, RasterMeta]]:
    """Load all rasters listed in file_map and validate alignment."""
    data_dir = Path(data_dir)
    arrays: Dict[str, np.ndarray] = {}
    metas: Dict[str, RasterMeta] = {}

    for key, name in file_map.items():
        arr, meta = load_raster(data_dir / name)
        arrays[key] = arr
        metas[key] = meta

    check_alignment(metas.values())
    return arrays, metas
