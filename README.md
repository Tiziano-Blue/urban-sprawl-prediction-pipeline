# Urban Expansion Transition Pipeline

Clean, reproducible tutorial repository for transition-based urban expansion prediction.

## What This Project Models

This is a **transition model**, not a static land classification model.

Research question:
Among pixels that were non-built in 2000, which became built by 2010?

- Candidate space (training): `Built_2000 == 0`
- Target `Y`:
  - `1`: non-built(2000) -> built(2010)
  - `0`: non-built(2000) -> non-built(2010)
- Predictors `X` only:
  - `distance_to_built_2000`
  - `distance_to_roads`

Rules enforced in the clean pipeline:
- `Built_2000` is not used as a predictor.
- `built_density` is not used.
- `pop_density` is not used.
- No future information is used during training.

## Three Conceptual Stages

The project keeps three conceptual stages and runs them sequentially in one script:

1. Training stage (2000 -> 2010)
2. Validation stage (2010 -> 2024)
3. Prediction stage (2024 -> 2030)

Main execution:

```bash
python run_pipeline.py
```

Wrapper (optional):

```bash
python run_all_pipeline.py
```

## Data

Default input location is `data/raw/` (configured in `config.yaml`).

Required rasters:
- `landuse2000.tif`
- `landuse2010.tif`
- `landuse2024.tif`
- `roads.tif`
- `distance2000.tif`
- `distance2010.tif`
- `distance2024.tif`

Data semantics:
- `landuse*` and `roads` are binary masks (0/1)
- distance rasters are continuous
- prediction probability raster is continuous

`pop_density.tif` is intentionally excluded from the clean tutorial pipeline.

## Final Main Outputs

Images:
- `outputs/final_validation_confusion_map.png`
- `outputs/final_prediction_probability_map.png`
- `outputs/final_high_risk_map.png`

Core reproducible results:
- `outputs/pred_prob_2030.tif`
- `outputs/model_xgb.json`
- `outputs/metrics_report.json`
- `outputs/final_summary.txt`

Intermediate files are stored under `outputs/intermediate/` and are not part of the main tutorial outputs.

## Repository Structure

```text
urban sprawl/
  README.md
  requirements.txt
  config.yaml
  run_pipeline.py
  run_all_pipeline.py
  .gitignore
  src/
  tests/
  data/
    raw/
    derived/
  outputs/
```

## GitHub Notes

- `.gitignore` excludes caches, temporary files, debug outputs, and intermediate outputs.
- The current input `.tif` files in `data/raw/` are small enough for normal Git tracking.
- Git LFS is not required for the current file sizes in this repository.
- If you replace inputs with larger rasters (for example >50-100 MB per file), use Git LFS.

## Legacy / Not Used by Clean Tutorial Path

These files are kept for reference only and are not used by `run_pipeline.py`:
- `src/balanced_sampling.py`
- `src/model_trainer_xgb.py`
- `src/raster_block_prediction.py`
- `src/raster_data_loader.py`
- `src/spatial_cross_validation.py`
- `config.py`
