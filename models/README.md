# Serialized Model Artifacts

This directory contains trained, production-ready LightGBM booster models serialized in text format (`.txt`). The text format is language-agnostic, portable, and natively supported across LightGBM in Python, C++, R, and Go without pickle security vulnerabilities or version locking.

---

## 1. Available Models

### `pm25_lgbm.txt`
- **Architecture**: Gradient Boosted Decision Trees (GBDT)
- **Objective**: Regression (L2 loss / MSE)
- **Features Used**: 29 base autoregressive, rolling window, cyclic temporal, and cross-station spatial features.
- **Hyperparameters**:
  - `learning_rate`: 0.03
  - `num_leaves`: 63
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `early_stopping_round`: 100
- **Test Performance (2024 Held-out)**:
  - MAE: 29.35 µg/m³
  - RMSE: 47.85 µg/m³
  - R²: 0.554

### `pm25_lgbm_tuned.txt`
- **Architecture**: Tuned Gradient Boosted Decision Trees via Optuna Bayesian Optimization (5-fold temporal expansion CV).
- **Objective**: Regression (L2 loss / MSE)
- **Features Used**: 46 expanded features (including dynamic spike/trend rates of change and diurnal ratios).
- **Optimized Hyperparameters**:
  - `learning_rate`: 0.05
  - `num_leaves`: 18
  - `min_child_samples`: 155
  - `reg_lambda`: 0.1306 (L2 regularization)
  - `reg_alpha`: 0.1225 (L1 regularization)
  - `colsample_bytree`: 0.5283
  - `subsample`: 0.8424
- **Test Performance (2024 Held-out)**:
  - MAE: 29.14 µg/m³ (29.17 on unaligned)
  - RMSE: 47.62 µg/m³
  - R²: 0.557

---

## 2. Loading and Inference Example

```python
import lightgbm as lgb
import numpy as np

# Load serialized booster
booster = lgb.Booster(model_file="models/pm25_lgbm_tuned.txt")

# Run prediction on feature vector X (shape: [num_samples, num_features])
predictions = booster.predict(X)
predictions = np.clip(predictions, 0, None)  # Enforce physical non-negativity
```
