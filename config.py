from __future__ import annotations

"""LEGACY CONFIG MODULE: kept for backward compatibility; clean tutorial uses config.yaml + run_pipeline.py."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_FILE_NAMES: Dict[str, str] = {
    "landuse2000": "landuse2000.tif",
    "landuse2010": "landuse2010.tif",
    "landuse2024": "landuse2024.tif",
    "roads": "roads.tif",
    "distance2000": "distance2000.tif",
    "distance2010": "distance2010.tif",
    "distance2024": "distance2024.tif",
}


@dataclass
class PipelineConfig:
    """Dataclass config retained for utility scripts."""

    project_root: Path
    data_dir: Path | None = None
    outputs_dir: Path | None = None
    sample_pos: int = 10000
    sample_neg: int = 10000
    negative_distance_threshold_m: float = 10000.0
    block_size: int = 256
    random_state: int = 42
    file_names: Dict[str, str] = field(default_factory=lambda: DEFAULT_FILE_NAMES.copy())

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir) if self.data_dir else self.project_root / "data"
        self.outputs_dir = Path(self.outputs_dir) if self.outputs_dir else self.project_root / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


def default_config_dict() -> Dict[str, Any]:
    """Return default config for strict transition pipeline."""
    return {
        "project": {
            "random_state": 42,
            "output_dir": "outputs",
        },
        "data": {"real_data_dir": "data", "file_names": DEFAULT_FILE_NAMES.copy()},
        "sampling": {"sample_pos": 10000, "sample_neg": 10000},
        "model": {
            "use_gpu": False,
            "early_stopping_rounds": 20,
            "threshold": 0.5,
            "eval_metric": "auc",
            "xgb_params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "objective": "binary:logistic",
                "n_jobs": 4,
            },
        },
        "prediction": {
            "block_size": 256,
            "batch_size": 65536,
        },
    }


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config if present; otherwise return defaults."""
    cfg = default_config_dict()
    if config_path is None:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    # Shallow recursive update for known sections.
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg
