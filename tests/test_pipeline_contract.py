from pathlib import Path

from run_pipeline import (
    MAIN_IMAGE_OUTPUTS,
    MAIN_RESULT_OUTPUTS,
    OBSOLETE_MAIN_OUTPUTS,
    cleanup_obsolete_outputs,
    load_pipeline_config,
)


def test_config_has_no_legacy_sampling_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
project:
  random_state: 42
  output_dir: outputs
sampling:
  negative_multiplier: 2
model:
  early_stopping_rounds: 20
  xgb_params:
    n_estimators: 200
    max_depth: 5
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_pipeline_config(cfg_path)

    assert "sample_pos" not in cfg.get("sampling", {})
    assert "sample_neg" not in cfg.get("sampling", {})


def test_cleanup_removes_obsolete_outputs(tmp_path: Path):
    for name in OBSOLETE_MAIN_OUTPUTS:
        (tmp_path / name).write_text("x", encoding="utf-8")

    removed = cleanup_obsolete_outputs(tmp_path)

    assert set(removed) == set(OBSOLETE_MAIN_OUTPUTS)
    for name in OBSOLETE_MAIN_OUTPUTS:
        assert not (tmp_path / name).exists()


def test_main_output_contract_is_minimal():
    assert set(MAIN_IMAGE_OUTPUTS) == {
        "observed_urban_expansion_2010_2024.png",
        "final_validation_confusion_map.png",
        "final_prediction_probability_map.png",
        "final_high_risk_map.png",
    }
    assert set(MAIN_RESULT_OUTPUTS) == {
        "pred_prob_2030.tif",
        "model_xgb.json",
        "metrics_report.json",
        "final_summary.txt",
    }
    assert "observed_urban_expansion_2010_2024.png" not in OBSOLETE_MAIN_OUTPUTS
