# Interactive Notebooks

This directory contains the primary research and development Jupyter notebook that executes the complete end-to-end Dhaka PM2.5 forecasting, anomaly detection, multi-horizon benchmark, and uncertainty quantification pipeline.

---

## 1. Primary Notebook: `ML_Lab_improved.ipynb`

The notebook is divided into 25 systematic steps:

1. **Step 1: Raw Data Inspection**: Examines station coverage from 2012 to 2026 across CAMS-1, CAMS-2, and CAMS-3.
2. **Step 2: Data Cleaning & Preprocessing**: Discards low-coverage 2012 data, handles negative sensor artifacts, establishes hourly timelines, and applies short-gap ($\le 3\text{h}$) linear interpolation.
3. **Step 3–4: Anomaly Detection**: Isolation Forest modeling over multi-pollutant feature sets to detect sensor dropouts, stuck values, and unphysical spikes; exports `anomalies.csv`.
4. **Step 5: Feature Engineering**: Generates autoregressive lags ($1, 2, 3, 6, 12, 24, 48, 168\text{h}$), moving window aggregations ($24\text{h}, 168\text{h}$), and cyclic temporal encodings.
5. **Step 6: Baseline Model**: Trains LightGBM v1 for 24h-ahead forecasting and evaluates on the 2024 test benchmark.
6. **Step 7: Spatial & Log Improvements**: Integrates contemporaneous cross-station features and log-transformed target modeling.
7. **Step 8: Per-Station Diagnostics & Feature Importance**: Computes individual station errors (CAMS-1, CAMS-2, CAMS-3) and visualizes feature importances.
8. **Step 9–10: Operational Forecasting & Threshold Tuning**: Implements single-timestamp prediction helper and sweeps alert thresholds on 2023 validation data.
9. **Step 11: Sanity & Robustness Checks**: Rigorously tests model reload consistency, missingness handling, extreme input stability, and temporal monotonicity.
10. **Step 12: Recursive 72h Rollout**: Iterative autoregressive forecasting simulating multi-day forward projections.
11. **Step 13–15: SARIMA & Long-term Outlook**: Aggregates data to monthly averages ($\ge 200\text{h}$ threshold), fits log-scale SARIMA, and forecasts 2025–2029 outlook.
12. **Step 16: Confusion Matrix Analysis**: Constructs 6-tier EPA/DoE AQI and binary high-pollution ($PM_{2.5} > 150$) confusion matrices.
13. **Step 17: Leakage-Free Alert Calibration**: Sweeps alert thresholds strictly on validation (2023) and evaluates once on test (2024).
14. **Step 18: Dedicated Extreme Event Classifier**: Fits LightGBM binary classifier with precision-recall curve analysis and threshold optimization.
15. **Step 19: Quantile Regression & Uncertainty Quantification**: Pins 10th, 50th, and 90th percentile regressors to yield an 80% prediction interval.
16. **Step 20: Expanded Dynamic Feature Set**: Adds rapid spike/trend differentials and diurnal moving average ratios.
17. **Step 21: Direct Multi-Horizon Modeling**: Fits dedicated LightGBM models for horizons 1h, 6h, 12h, 24h, 48h, and 72h vs persistence.
18. **Step 22: Hyperparameter Optimization**: Optuna Bayesian optimization across 5 temporal CV folds; serializes `pm25_lgbm_tuned.txt`.
19. **Step 23: Long-Term Trend Verification**: Evaluates linear trend statistical significance ($p = 0.189$).
20. **Step 24: Staleness-Aware Operational Inference**: 72-hour forecast pipeline that flags and skips offline or stale stations.
21. **Step 25: Artifact Packaging**: Compresses models, tables, and figures into `results.zip`.

---

## 2. Running Locally or on Google Colab

To run the notebook:
```bash
jupyter notebook notebooks/ML_Lab_improved.ipynb
```
Ensure the dataset `dhaka_cams_hourly_2012_2026.csv` is present in `../data/` or in the same working directory.
