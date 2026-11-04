# Source Code Architecture (`src/`)

This directory houses the modular, production-ready Python package implementing the Dhaka PM2.5 forecasting, anomaly detection, evaluation, and operational prediction pipelines.

---

## Module Index

### 1. `data_loader.py`
- **`load_raw_data(csv_path)`**: Loads CAMS CSV file with parsed timestamps.
- **`clean_air_quality_data(df, start_date, end_date, interp_limit, pollutants)`**:
  - Filters usable multi-year windows (2013–2024).
  - Sanitizes negative pollutant concentrations ($< 0 \rightarrow \text{NaN}$).
  - Reindexes continuous regular hourly grids for each station.
  - Imputes brief sensor outages ($\le 3$ hours) using linear time interpolation.
  - Generates boolean provenance flags (`is_imputed`).

### 2. `anomaly_detection.py`
- **`fit_anomaly_detector(df, features, contamination, n_estimators)`**: Trains Isolation Forest on multi-pollutant concentrations (`pm2_5_ug_m3`, `pm10_ug_m3`, `no2_ppb`, `co_ppm`).
- **`detect_anomalies(model, df, features)`**: Scores outlier probabilities and verifies domain consistency (such as sensor flatlines and unphysical $PM_{2.5} > PM_{10}$ inversions).

### 3. `features.py`
- **`engineer_features(df, horizons, lags, roll_windows, include_cross_station, target_col)`**:
  - Generates autoregressive lags ($1, 2, 3, 6, 12, 24, 48, 168\text{h}$).
  - Calculates rolling statistics (mean, std, min, max over 24h and 168h).
  - Computes cyclic sine/cosine harmonic temporal encodings for hour, day of week, month, and day of year.
  - Computes dynamic spike differentials and diurnal moving average ratios.
  - Pivots contemporaneous cross-station readings across CAMS-1, CAMS-2, and CAMS-3.
- **`get_feature_columns(df)`**: Filters target and metadata columns to return model-ready feature column names.

### 4. `models.py`
- **`train_lgbm_regressor(X_train, y_train, X_val, y_val, params)`**: Trains LightGBM regression booster with validation early stopping and log-target capabilities.
- **`train_quantile_regressors(X_train, y_train, X_val, y_val, quantiles)`**: Fits asymmetric pinball-loss quantile regression models for 10th, 50th, and 90th percentiles.
- **`train_high_pollution_classifier(X_train, y_train, X_val, y_val, threshold)`**: Fits dedicated binary LightGBM classifier targeting severe episodic pollution ($PM_{2.5} > 150\ \mu\text{g}/\text{m}^3$).
- **`save_lgb_model(model, filepath)` / `load_lgb_model(filepath)`**: Text-format serialization and deserialization routines.

### 5. `evaluate.py`
- **`pm25_to_aqi_category(values)`**: Bins continuous PM2.5 concentrations into official 6-tier EPA/DoE Air Quality Index categories.
- **`compute_regression_metrics(y_true, y_pred)`**: Calculates MAE, RMSE, and $R^2$.
- **`evaluate_high_pollution_detection(y_true, y_pred, threshold)`**: Computes precision, recall, F1, and confusion matrix for severe air pollution alerts.
- **`evaluate_quantiles(y_true, q10, q50, q90)`**: Measures empirical 80% coverage and prediction interval sharpness.

### 6. `predict.py`
- **`check_data_staleness(df, current_time, max_stale_hours)`**: Verifies observation freshness to prevent stale, misleading forecasts from offline stations.
- **`forecast_next_24h(booster, feature_row, feature_names)`**: Computes 24h point prediction with automatic AQI category assignment.
- **`recursive_rollout_72h(booster, history_df, station_id, feature_cols, horizons)`**: Conducts iterative autoregressive multi-step forward forecasting out to 72 hours.
