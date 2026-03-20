from __future__ import annotations

"""
Conceptual model used in this pipeline:
- Training candidate space: Built_2000 == 0.
- Training target Y: 1 if non-built(2000)->built(2010), else 0 if non-built(2000)->non-built(2010).
- Training predictors X: [distance_to_built_2000, distance_to_roads].
- Built_2000 defines candidate pixels only and is never used as a predictor.
- Prediction candidate space: Built_2024 == 0 with predictors [distance_to_built_2024, distance_to_roads].
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.diagnostics import run_diagnostics
from src.feature_engineering import compute_distance_from_feature, compute_required_distances
from src.io_utils import build_input_paths, ensure_dir, load_rasters, resolve_data_dir, save_raster, write_json
from src.modeling import (
    DEFAULT_XGB_PARAMS,
    compute_binary_metrics,
    run_spatial_group_kfold_cv,
    train_model_with_internal_validation,
)
from src.prediction import (
    TRAIN_FEATURE_ORDER,
    build_prediction_matrix,
    predict_probability_map,
    validate_feature_order,
)
from src.sampling import balanced_sample, prepare_training_data, validate_training_inputs
from src.visualization import (
    save_observed_expansion_map,
    save_final_high_risk_map,
    save_final_prediction_probability_map,
    save_final_validation_confusion_map,
)

DEFAULT_FILE_NAMES = {
    "landuse2000": "landuse2000.tif",
    "landuse2010": "landuse2010.tif",
    "landuse2024": "landuse2024.tif",
    "roads": "roads.tif",
    "distance2000": "distance2000.tif",
    "distance2010": "distance2010.tif",
    "distance2024": "distance2024.tif",
}

MAIN_IMAGE_OUTPUTS = [
    "observed_urban_expansion_2010_2024.png",
    "final_validation_confusion_map.png",
    "final_prediction_probability_map.png",
    "final_high_risk_map.png",
]

MAIN_RESULT_OUTPUTS = [
    "pred_prob_2030.tif",
    "model_xgb.json",
    "metrics_report.json",
    "final_summary.txt",
]

OBSOLETE_MAIN_OUTPUTS = [
    "training_concept.png",
    "pred_prob_future.tif",
    "metrics_train.json",
    "metrics_spatial_cv.json",
    "metrics.json",
    "probability_summary.txt",
    "training_config.json",
    "feature_importance.csv",
    "shap_importance.csv",
    "data_diagnostics.json",
]


@dataclass
class TrainingArtifacts:
    model: Any
    train_metrics: dict[str, Any]
    spatial_cv: dict[str, Any]
    sampled_train: Any


@dataclass
class ValidationArtifacts:
    validation_metrics: dict[str, Any]
    validation_threshold: float


@dataclass
class PredictionArtifacts:
    pred_prob_map: np.ndarray
    candidate_mask: np.ndarray
    high_risk_cutoff: float
    high_risk_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban expansion transition pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pipeline config file path.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Input raster folder override (defaults to config, then /mnt/data, then ../New Data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder override (defaults to config project.output_dir).",
    )
    return parser.parse_args()


def load_pipeline_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must parse to a dictionary-like object.")

    raw_project = raw.get("project", {}) or {}
    raw_data = raw.get("data", {}) or {}
    raw_sampling = raw.get("sampling", {}) or {}
    raw_model = raw.get("model", {}) or {}
    raw_validation = raw.get("validation", {}) or {}
    raw_prediction = raw.get("prediction", {}) or {}

    file_names = DEFAULT_FILE_NAMES.copy()
    file_names.update(raw_data.get("file_names", {}) or {})

    xgb_params = DEFAULT_XGB_PARAMS.copy()
    xgb_params.update(raw_model.get("xgb_params", {}) or {})

    config = {
        "project": {
            "random_state": int(raw_project.get("random_state", 42)),
            "output_dir": str(raw_project.get("output_dir", "outputs")),
        },
        "data": {
            "real_data_dir": raw_data.get("real_data_dir"),
            "file_names": file_names,
        },
        "sampling": {
            "negative_multiplier": int(raw_sampling.get("negative_multiplier", 2)),
        },
        "model": {
            "early_stopping_rounds": int(raw_model.get("early_stopping_rounds", 20)),
            "xgb_params": xgb_params,
        },
        "validation": {
            "spatial_cv_block_size": int(raw_validation.get("spatial_cv_block_size", 64)),
        },
        "prediction": {
            "high_risk_quantile": float(raw_prediction.get("high_risk_quantile", 0.9)),
        },
    }
    return config


def cleanup_obsolete_outputs(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in OBSOLETE_MAIN_OUTPUTS:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def _resolve_explicit_data_dir(
    cli_data_dir: str | None,
    cfg_data_dir: str | None,
    project_root: Path,
) -> str | None:
    raw = cli_data_dir or cfg_data_dir
    if raw is None:
        return None

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return str(p)


def run_training_stage(
    built_2000: np.ndarray,
    built_2010: np.ndarray,
    distances: dict[str, np.ndarray],
    random_state: int,
    negative_multiplier: int,
    xgb_params: dict[str, Any],
    early_stopping_rounds: int,
    spatial_cv_block_size: int,
) -> TrainingArtifacts:
    print("\n[Training Stage] 2000 -> 2010 transition model")

    full_train = prepare_training_data(
        distance_to_built_2000=distances["distance_to_built_2000"],
        distance_to_roads=distances["distance_to_roads"],
        built_2000=built_2000,
        built_2010=built_2010,
    )
    sampled_train = balanced_sample(
        full_data=full_train,
        random_state=random_state,
        negative_multiplier=negative_multiplier,
    )
    validate_training_inputs(sampled_train.X, sampled_train.y)

    train_feature_order = ["distance_to_built", "distance_to_roads"]
    validate_feature_order(train_feature_order, TRAIN_FEATURE_ORDER)

    model, train_metrics = train_model_with_internal_validation(
        sampled_train.X,
        sampled_train.y,
        random_state=random_state,
        xgb_params=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
    )

    spatial_cv = run_spatial_group_kfold_cv(
        X=sampled_train.X,
        y=sampled_train.y,
        rows=sampled_train.rows,
        cols=sampled_train.cols,
        block_size=spatial_cv_block_size,
        n_splits=5,
        random_state=random_state,
        xgb_params=xgb_params,
    )

    print(
        "[Training Stage] Sampled counts: "
        f"positive={sampled_train.positive_count_sampled}, "
        f"negative={sampled_train.negative_count_sampled}"
    )

    return TrainingArtifacts(
        model=model,
        train_metrics=train_metrics,
        spatial_cv=spatial_cv,
        sampled_train=sampled_train,
    )


def run_validation_stage(
    model: Any,
    built_2010: np.ndarray,
    built_2024: np.ndarray,
    distance_to_roads: np.ndarray,
    output_dir: Path,
    threshold: float,
) -> ValidationArtifacts:
    print("\n[Validation Stage] Apply trained model on 2010 -> 2024")

    validation_feature_order = ["distance_to_built", "distance_to_roads"]
    validate_feature_order(validation_feature_order, TRAIN_FEATURE_ORDER)

    distance_to_built_2010 = compute_distance_from_feature(built_2010 == 1)

    candidate_valid = built_2010 == 0
    valid_rows, valid_cols = np.where(candidate_valid)
    X_valid = np.column_stack(
        [
            distance_to_built_2010[valid_rows, valid_cols],
            distance_to_roads[valid_rows, valid_cols],
        ]
    ).astype(np.float32)

    if X_valid.ndim != 2 or X_valid.shape[1] != 2:
        raise ValueError(f"Validation X must have exactly 2 columns, got {X_valid.shape}.")
    if np.isnan(X_valid).any():
        raise ValueError("Validation X contains NaN values.")

    y_valid = ((built_2010[valid_rows, valid_cols] == 0) & (built_2024[valid_rows, valid_cols] == 1)).astype(
        np.uint8
    )
    valid_prob = model.predict_proba(X_valid)[:, 1].astype(np.float32)
    validation_metrics = compute_binary_metrics(y_valid, valid_prob, threshold=threshold)

    valid_prob_map = np.full(built_2010.shape, np.nan, dtype=np.float32)
    valid_prob_map[valid_rows, valid_cols] = valid_prob
    actual_change_2010_2024 = (built_2010 == 0) & (built_2024 == 1)

    save_final_validation_confusion_map(
        pred_prob_map=valid_prob_map,
        actual_change_mask=actual_change_2010_2024,
        candidate_mask=candidate_valid,
        built_mask=(built_2010 == 1),
        threshold=threshold,
        path=output_dir / "final_validation_confusion_map.png",
    )

    return ValidationArtifacts(
        validation_metrics=validation_metrics,
        validation_threshold=float(threshold),
    )


def run_prediction_stage(
    model: Any,
    built_2024: np.ndarray,
    distances: dict[str, np.ndarray],
    high_risk_quantile: float,
    output_dir: Path,
    reference_layer: Any,
) -> PredictionArtifacts:
    print("\n[Prediction Stage] Predict 2024 -> 2030 expansion probability")

    prediction_feature_order = ["distance_to_built", "distance_to_roads"]
    validate_feature_order(prediction_feature_order, TRAIN_FEATURE_ORDER)

    X_pred, pred_rows, pred_cols, candidate_pred = build_prediction_matrix(
        distance_to_built_2024=distances["distance_to_built_2024"],
        distance_to_roads=distances["distance_to_roads"],
        built_2024=built_2024,
    )

    if X_pred.ndim != 2 or X_pred.shape[1] != 2:
        raise ValueError(f"Prediction X must have exactly 2 columns, got {X_pred.shape}.")

    pred_prob = predict_probability_map(
        model=model,
        X_pred=X_pred,
        rows=pred_rows,
        cols=pred_cols,
        raster_shape=built_2024.shape,
    )

    save_raster(output_dir / "pred_prob_2030.tif", pred_prob, reference_layer, dtype="float32", nodata=np.nan)

    save_final_prediction_probability_map(
        pred_prob_map=pred_prob,
        candidate_mask=candidate_pred,
        built_mask=(built_2024 == 1),
        path=output_dir / "final_prediction_probability_map.png",
    )

    candidate_pred_values = pred_prob[candidate_pred & np.isfinite(pred_prob)]
    if candidate_pred_values.size == 0:
        raise ValueError("No candidate prediction pixels found for Built_2024 == 0.")

    high_risk_cutoff = float(np.quantile(candidate_pred_values, high_risk_quantile))
    high_risk_mask = candidate_pred & np.isfinite(pred_prob) & (pred_prob >= high_risk_cutoff)
    high_risk_count = int(np.count_nonzero(high_risk_mask))

    save_final_high_risk_map(
        high_risk_mask=high_risk_mask,
        candidate_mask=candidate_pred,
        built_mask=(built_2024 == 1),
        path=output_dir / "final_high_risk_map.png",
    )

    return PredictionArtifacts(
        pred_prob_map=pred_prob,
        candidate_mask=candidate_pred,
        high_risk_cutoff=high_risk_cutoff,
        high_risk_count=high_risk_count,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_pipeline_config((project_root / args.config).resolve())

    output_dir_name = args.output_dir or config["project"]["output_dir"]
    output_dir = ensure_dir((project_root / output_dir_name).resolve())
    intermediate_dir = ensure_dir(output_dir / "intermediate")

    explicit_data_dir = _resolve_explicit_data_dir(
        cli_data_dir=args.data_dir,
        cfg_data_dir=config["data"].get("real_data_dir"),
        project_root=project_root,
    )
    data_dir = resolve_data_dir(explicit_data_dir, project_root)

    print("=" * 80)
    print("Urban Expansion Pipeline (Sequential Stages: Train -> Validate -> Predict)")
    print("=" * 80)
    print(f"[Setup] Config: {(project_root / args.config).resolve()}")
    print(f"[Setup] Data directory: {data_dir}")
    print(f"[Setup] Output directory: {output_dir}")

    input_paths = build_input_paths(data_dir, file_names=config["data"]["file_names"])
    layers = load_rasters(input_paths)

    print("\n[Preparation] Running diagnostics...")
    diagnostics_result = run_diagnostics(layers)
    write_json(intermediate_dir / "data_diagnostics.json", diagnostics_result.diagnostics)
    if not diagnostics_result.validation_passed:
        raise ValueError("Data validation failed. Fix input rasters before running stages.")

    built_2000 = layers["landuse2000"].array
    built_2010 = layers["landuse2010"].array
    built_2024 = layers["landuse2024"].array
    roads = layers["roads"].array

    print("\n[Preparation] Building observed expansion map (2010 -> 2024)...")
    save_observed_expansion_map(
        built_base=built_2010,
        built_next=built_2024,
        path=output_dir / "observed_urban_expansion_2010_2024.png",
    )

    print("\n[Preparation] Computing required distance rasters...")
    distances = compute_required_distances(built_2000=built_2000, built_2024=built_2024, roads=roads)
    save_raster(
        intermediate_dir / "distance_to_built_2000.tif",
        distances["distance_to_built_2000"],
        layers["landuse2000"],
        dtype="float32",
    )
    save_raster(
        intermediate_dir / "distance_to_built_2024.tif",
        distances["distance_to_built_2024"],
        layers["landuse2024"],
        dtype="float32",
    )
    save_raster(
        intermediate_dir / "distance_to_roads.tif",
        distances["distance_to_roads"],
        layers["roads"],
        dtype="float32",
    )

    training = run_training_stage(
        built_2000=built_2000,
        built_2010=built_2010,
        distances=distances,
        random_state=config["project"]["random_state"],
        negative_multiplier=config["sampling"]["negative_multiplier"],
        xgb_params=config["model"]["xgb_params"],
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
        spatial_cv_block_size=config["validation"]["spatial_cv_block_size"],
    )

    model_path = output_dir / "model_xgb.json"
    training.model.save_model(model_path)

    validation_threshold = float(training.train_metrics.get("selected_threshold_f1", 0.5))
    validation = run_validation_stage(
        model=training.model,
        built_2010=built_2010,
        built_2024=built_2024,
        distance_to_roads=distances["distance_to_roads"],
        output_dir=output_dir,
        threshold=validation_threshold,
    )

    prediction = run_prediction_stage(
        model=training.model,
        built_2024=built_2024,
        distances=distances,
        high_risk_quantile=config["prediction"]["high_risk_quantile"],
        output_dir=output_dir,
        reference_layer=layers["landuse2024"],
    )

    metrics_report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline": {
            "mode": "sequential_stages",
            "stages": ["training", "validation", "prediction"],
            "feature_order": TRAIN_FEATURE_ORDER,
            "training_candidate": "Built_2000 == 0",
            "prediction_candidate": "Built_2024 == 0",
        },
        "train_metrics": {
            "roc_auc": training.train_metrics.get("roc_auc"),
            "pr_auc": training.train_metrics.get("pr_auc"),
            "precision": training.train_metrics.get("precision"),
            "recall": training.train_metrics.get("recall"),
            "f1": training.train_metrics.get("f1"),
            "confusion_matrix": training.train_metrics.get("confusion_matrix"),
        },
        "validation_metrics": {
            "roc_auc": validation.validation_metrics.get("roc_auc"),
            "pr_auc": validation.validation_metrics.get("pr_auc"),
            "precision": validation.validation_metrics.get("precision"),
            "recall": validation.validation_metrics.get("recall"),
            "f1": validation.validation_metrics.get("f1"),
            "confusion_matrix": validation.validation_metrics.get("confusion_matrix"),
        },
        "thresholds": {
            "selected_threshold_f1": validation.validation_threshold,
            "high_risk_quantile": config["prediction"]["high_risk_quantile"],
            "high_risk_probability_cutoff": prediction.high_risk_cutoff,
        },
        "spatial_cv": training.spatial_cv,
        "counts": {
            "train_positive_full": int(training.sampled_train.positive_count_full),
            "train_negative_full": int(training.sampled_train.negative_count_full),
            "train_positive_sampled": int(training.sampled_train.positive_count_sampled),
            "train_negative_sampled": int(training.sampled_train.negative_count_sampled),
            "prediction_candidate_pixels": int(np.count_nonzero(prediction.candidate_mask)),
            "prediction_high_risk_pixels": int(prediction.high_risk_count),
        },
    }
    write_json(output_dir / "metrics_report.json", metrics_report)

    summary_lines = [
        "Urban Expansion Tutorial Summary",
        f"Timestamp: {metrics_report['timestamp']}",
        "Model predicts probability of urban expansion for non-built pixels in 2024.",
        "Training stage: candidate=Built_2000==0, Y is non-built(2000)->built(2010), X=[distance_to_built_2000, distance_to_roads].",
        "Validation stage: apply trained model to 2010->2024 transition with same feature order.",
        "Prediction stage: apply trained model to 2024 candidates for 2030 probability map.",
        f"Validation ROC-AUC: {metrics_report['validation_metrics']['roc_auc']}",
        (
            "Validation Precision / Recall / F1: "
            f"{metrics_report['validation_metrics']['precision']} / "
            f"{metrics_report['validation_metrics']['recall']} / "
            f"{metrics_report['validation_metrics']['f1']}"
        ),
        f"Selected threshold: {metrics_report['thresholds']['selected_threshold_f1']}",
        f"Candidate pixels (Built_2024==0): {metrics_report['counts']['prediction_candidate_pixels']}",
        f"Predicted high-risk pixels: {metrics_report['counts']['prediction_high_risk_pixels']}",
        "Prediction outputs are probability-based (continuous values in pred_prob_2030.tif).",
    ]
    (output_dir / "final_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    removed = cleanup_obsolete_outputs(output_dir)

    print("\n[Quantitative Summary]")
    print(
        "Train     | "
        f"ROC-AUC={metrics_report['train_metrics']['roc_auc']:.4f}  "
        f"PR-AUC={metrics_report['train_metrics']['pr_auc']:.4f}  "
        f"Precision={metrics_report['train_metrics']['precision']:.4f}  "
        f"Recall={metrics_report['train_metrics']['recall']:.4f}  "
        f"F1={metrics_report['train_metrics']['f1']:.4f}"
    )
    print(
        "Validate  | "
        f"ROC-AUC={metrics_report['validation_metrics']['roc_auc']:.4f}  "
        f"PR-AUC={metrics_report['validation_metrics']['pr_auc']:.4f}  "
        f"Precision={metrics_report['validation_metrics']['precision']:.4f}  "
        f"Recall={metrics_report['validation_metrics']['recall']:.4f}  "
        f"F1={metrics_report['validation_metrics']['f1']:.4f}"
    )
    spatial_mean = metrics_report["spatial_cv"]["mean_metrics"]
    print(
        "SpatialCV | "
        f"ROC-AUC={spatial_mean['roc_auc']:.4f}  "
        f"PR-AUC={spatial_mean['pr_auc']:.4f}  "
        f"Precision={spatial_mean['precision']:.4f}  "
        f"Recall={spatial_mean['recall']:.4f}  "
        f"F1={spatial_mean['f1']:.4f}"
    )
    print(
        "Threshold | "
        f"validation={metrics_report['thresholds']['selected_threshold_f1']:.6f}  "
        f"high_risk_cutoff={metrics_report['thresholds']['high_risk_probability_cutoff']:.6f}"
    )

    if removed:
        print(f"[Cleanup] Removed obsolete main outputs: {', '.join(sorted(removed))}")

    print("[Done] Pipeline completed with sequential stages (training -> validation -> prediction).")


if __name__ == "__main__":
    main()
