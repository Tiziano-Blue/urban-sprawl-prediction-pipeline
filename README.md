# Urban Expansion Transition Model

## 1. What this model is and is not

This project implements an **urban expansion transition model**, not a static land classification model.

It asks:

> Among pixels that were non-built in 2000, which ones became built by 2010?

So the model predicts **land conversion** from non-built to built, rather than general urban presence.

---

## 2. Input data types

The project starts from binary raster masks and derives continuous predictors from them.

### Binary (0/1) rasters

- `landuse2000.tif`
- `landuse2010.tif`
- `landuse2024.tif`
- `roads.tif`

Definitions:

- Built rasters:
  - `1` = built
  - `0` = non-built
- Roads raster:
  - `1` = road
  - `0` = non-road

### Derived continuous rasters

From these binary masks, the pipeline computes continuous distance rasters:

- `distance_to_built_2000`
- `distance_to_built_2010`
- `distance_to_built_2024`
- `distance_to_roads`

The final model output is also a **continuous probability raster**.

---

## 3. How distance-to-built is computed

The model does not use built masks directly as predictors.

Instead, it converts built-land masks into continuous distance rasters using Euclidean distance.

For each year:

- `distance_to_built_2000` is computed from `landuse2000.tif`
- `distance_to_built_2010` is computed from `landuse2010.tif`
- `distance_to_built_2024` is computed from `landuse2024.tif`

Definition:

- Built pixels → distance = `0`
- Non-built pixels → distance to nearest built pixel

This allows the model to use **proximity to existing development as a continuous variable**.

---

## 4. Candidate space

Only pixels that were **non-built in 2000** are used:

```
Built_2000 == 0
```

- Pixels already built in 2000 are excluded
- `Built_2000` is used only to define candidate space
- It is **not used as a predictor**

---

## 5. Target variable (Y)

Within the candidate space:

- `Y = 1` → non-built in 2000 → built in 2010  
- `Y = 0` → non-built in 2000 → non-built in 2010  

This means the model learns a **transition outcome**, not a static land-use label.

---

## 6. Predictor variables (X)

The model uses only two predictors:

- `X1 = distance_to_built_2000`
- `X2 = distance_to_roads`

Both are continuous variables.

Interpretation:

- `distance_to_built_2000` → proximity to existing urban areas  
- `distance_to_roads` → accessibility to infrastructure  

> Important: `Built_2000` is NOT used as a feature.

---

## 7. Why balanced sampling is used

Urban expansion is typically a **rare event**.

If all pixels were used directly:

- The dataset would be dominated by non-expansion pixels
- The model would tend to predict "no expansion" everywhere

To address this:

- All positive samples are kept
- Negative samples are randomly subsampled
- Negatives are capped at **2× the number of positives**

This improves learning without changing the definition of Y.

---

## 8. Why XGBoost is used

XGBoost is used because it can:

- Learn nonlinear relationships
- Capture interactions between predictors
- Output probabilities directly
- Avoid forcing a simple linear boundary

Even with only two predictors, it can learn patterns such as:

> High expansion probability when a pixel is both close to built-up land and close to roads

---

## 9. Three conceptual stages

### Training

Train the model on:

```
2000 → 2010 transitions
```

---

### Validation

Apply the trained model to:

```
2010 → 2024
```

Compare predictions with observed expansion.

---

### Prediction

Apply the model to:

```
2024 conditions → predict 2030 expansion
```

---

### Key idea

The model is trained once and then applied twice:

- once for validation  
- once for future prediction  

---

## 10. Data flow and model logic

Pipeline:

1. Load built-land and road rasters  
2. Confirm binary masks  
3. Compute distance rasters:
   - `distance_to_built_2000`
   - `distance_to_built_2024`
   - `distance_to_roads`  

4. Build candidate space:

```
Built_2000 == 0
```

5. Define training labels:

```
Y = 1 if built in 2010
Y = 0 if still non-built in 2010
```

6. Train model using:

```
distance_to_built_2000 + distance_to_roads
```

7. Validate using:

```
distance_to_built_2010 + distance_to_roads
```

8. Predict future expansion using:

```
distance_to_built_2024 + distance_to_roads
```

only where:

```
Built_2024 == 0
```

---

## Conceptual model

```
P(expansion) = f(distance_to_built, distance_to_roads)
```

---

## 11. How to interpret the outputs

### Validation confusion map

Shows where the model:

- correctly predicts expansion (true positive)
- overpredicts expansion (false positive)
- misses expansion (false negative)
- correctly predicts non-expansion (true negative)

Observation:

- False positives often appear near roads  
- Because road proximity is a strong signal

---

### Final prediction probability map

- Continuous surface of expansion probability
- Values range from 0 to 1

---

### Final high-risk map

- Highlights pixels with the highest predicted expansion risk  
- Typically defined as the **top 10% of probabilities**

Pattern:

- Urban edges  
- Road corridors  

---

## 12. Limitations

This is a **minimal baseline model**.

It captures a simplified proximity-driven expansion pattern, but does not include:

- zoning  
- topography  
- water bodies  
- environmental constraints  
- policy interventions  
- socioeconomic drivers  

---

## Interpretation

Outputs should be interpreted as:

> **probability-based spatial patterns**

NOT:

> **causal conclusions**
