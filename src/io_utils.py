from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


@dataclass
class RasterLayer:
    """Container for a single-band raster and key metadata."""

    name: str
    path: Path
    array: np.ndarray
    profile: dict[str, Any]
    transform: Any
    crs: Any
    nodata: Any


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_dir(explicit_data_dir: str | None, project_root: Path) -> Path:
    """Resolve input raster folder.

    Priority:
    1) explicit CLI argument
    2) /mnt/data (requested path)
    3) sibling folder: ../New Data
    """
    if explicit_data_dir:
        data_dir = Path(explicit_data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        return data_dir

    mnt_data = Path("/mnt/data")
    if mnt_data.exists():
        return mnt_data

    fallback = (project_root.parent / "New Data").resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Could not find input rasters. Expected either /mnt/data or ../New Data."
    )


def build_input_paths(data_dir: Path, file_names: dict[str, str] | None = None) -> dict[str, Path]:
    """Build required input file map."""
    required_names = {
        "landuse2000": "landuse2000.tif",
        "landuse2010": "landuse2010.tif",
        "landuse2024": "landuse2024.tif",
        "roads": "roads.tif",
        "distance2000": "distance2000.tif",
        "distance2010": "distance2010.tif",
        "distance2024": "distance2024.tif",
    }
    if file_names:
        required_names.update(file_names)
    paths = {name: data_dir / file_name for name, file_name in required_names.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required raster file(s):\n" + "\n".join(missing))
    return paths


def load_raster(name: str, path: Path) -> RasterLayer:
    """Load one raster band and metadata."""
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    return RasterLayer(
        name=name,
        path=path,
        array=array,
        profile=profile,
        transform=transform,
        crs=crs,
        nodata=nodata,
    )


def load_rasters(paths: dict[str, Path]) -> dict[str, RasterLayer]:
    """Load all required rasters."""
    return {name: load_raster(name, path) for name, path in paths.items()}


def valid_data_mask(array: np.ndarray, nodata: Any) -> np.ndarray:
    """Mask of finite, non-nodata pixels."""
    mask = np.isfinite(array)
    if nodata is None:
        return mask
    try:
        if np.isnan(nodata):
            return mask & (~np.isnan(array))
    except TypeError:
        pass
    return mask & (array != nodata)


def save_raster(
    path: Path,
    array: np.ndarray,
    reference: RasterLayer,
    dtype: str = "float32",
    nodata: float | int | None = None,
) -> None:
    """Save single-band raster using a reference profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate")

    if nodata is not None:
        profile["nodata"] = nodata
    elif "nodata" in profile and profile["nodata"] is None:
        profile.pop("nodata", None)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def _to_json_safe(value: Any) -> Any:
    """Convert numpy/scalar/path values into JSON-safe python types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_to_json_safe)
