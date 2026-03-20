# Data Layout

This tutorial expects input rasters in `data/raw/`.

Required files:
- `landuse2000.tif`
- `landuse2010.tif`
- `landuse2024.tif`
- `roads.tif`
- `distance2000.tif`
- `distance2010.tif`
- `distance2024.tif`

Notes:
- `pop_density.tif` is intentionally excluded from the clean pipeline.
- `data/derived/` is for optional regenerated artifacts and is ignored by Git.
- If your raw rasters are not committed, keep them locally and document their source/license in the project README.
