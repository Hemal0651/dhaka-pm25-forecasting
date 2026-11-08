# Interactive Notebooks

This directory contains the primary research, benchmarking, and development Jupyter notebooks for the Dhaka PM2.5 forecasting, anomaly detection, multi-horizon benchmark, model interpretability, and uncertainty quantification pipeline.

---

## Available Notebooks

- **`ML_Lab_improved.ipynb`**: The complete, authoritative 36-step research notebook containing all core models and publication benchmarks.
- **`ML_Lab_improved_v2.ipynb`**: Mirrored copy reflecting the expanded version 2 release.

---

## Step-by-Step Workflow Overview (36 Systematic Sections)

### Part I: Data Ingestion & Quality Control
1. **Step 1: Raw Data Inspection**: Evaluates hourly station coverage across CAMS-1, CAMS-2, and CAMS-3 from 2012 to 2026.
2. **Step 2: Data Cleaning & Preprocessing**: Discards low-coverage 2012 data (<15%), handles negative sensor artifacts, establishes regular hourly timelines, and applies short-gap ($\le 3\text{h}$) linear interpolation with imputation flags.

### Part II: Anomaly Detection
3. **Steps 3–4: Anomaly Detection via Isolation Forest**: Multivariate outlier detection on atmospheric concentrations (`pm2_5_ug_m3`, `pm10_ug_m3`, `no2_ppb`, `co_ppm`) to detect sensor flatlines, stuck values, and unphysical spikes; exports `anomalies.csv`.

### Part III: Feature Engineering & Baseline Modeling
4. **Step 5: Feature Engineering**: Generates autoregressive lags ($1, 2, 3, 6, 12, 24, 48, 168\text{h}$), moving window aggregations ($24\text{h}, 168\text{h}$ mean/std/min/max), and harmonic cyclic temporal encodings.
5. **Step 6: Baseline Model**: Trains LightGBM v1 for 24h-ahead forecasting and evaluates against persistence on the 2024 test benchmark.
6. **Step 7: Spatial & Log Improvements**: Integrates contemporaneous cross-station features and log-transformed target modeling.
7. **Step 8: Per-Station Diagnostics & Feature Importance**: Computes individual station errors (CAMS-1, CAMS-2, CAMS-3) and visualizes feature importances.
8. **Steps 9–10: Operational Forecasting & Threshold Tuning**: Implements single-timestamp prediction helper and sweeps alert thresholds on 2023 validation data.
9. **Step 11: Sanity & Robustness Checks**: Rigorously tests model reload consistency, missingness handling, extreme input stability, and temporal monotonicity.

### Part IV: Rollout & Macro-Scale Modeling
10. **Step 12: Recursive 72h Rollout**: Iterative autoregressive forecasting simulating multi-day forward projections.
11. **Steps 13–15: SARIMA & Long-term Outlook**: Aggregates data to monthly averages ($\ge 200\text{h}$ threshold), fits log-scale SARIMA, and forecasts 2025–2029 outlook.
12. **Step 16: Confusion Matrix Analysis**: Constructs 6-tier EPA/DoE AQI and binary high-pollution ($PM_{2.5} > 150\ \mu\text{g}/\text{m}^3$) confusion matrices.
13. **Step 17: Leakage-Free Alert Calibration**: Sweeps alert thresholds strictly on validation (2023) and evaluates once on test (2024).

### Part V: Multi-Task & Uncertainty Modeling
14. **Step 18: Dedicated Extreme Event Classifier**: Fits LightGBM binary classifier with precision-recall curve analysis and validation-optimized decision thresholds.
15. **Step 19: Quantile Regression & Uncertainty Quantification**: Pins 10th, 50th, and 90th percentile pinball-loss regressors to yield an 80% prediction interval.
16. **Step 20: Expanded Dynamic Feature Set**: Adds rapid spike/trend differentials ($\Delta_{1\text{h}}, \Delta_{24\text{h}}$) and diurnal moving average ratios (46 features total).
17. **Step 21: Direct Multi-Horizon Modeling**: Fits dedicated LightGBM models for horizons 1h, 6h, 12h, 24h, 48h, and 72h vs persistence.
18. **Step 22: Hyperparameter Optimization**: Optuna Bayesian optimization across 5 temporal CV folds; serializes `pm25_lgbm_tuned.txt`.
19. **Step 23: Long-Term Trend Verification**: Evaluates linear trend statistical significance ($p = 0.189$).
20. **Step 24: Staleness-Aware Operational Inference**: 72-hour forecast pipeline that flags and skips offline or stale stations.
21. **Step 25: Artifact Packaging**: Compresses models, tables, and figures into `results.zip`.

### Part VI: Rigorous Scientific Benchmarks (Sections 26–36)
22. **Step 26: Strong Baseline Implementation**: Implements hourly climatology and seasonal-naive baselines.
23. **Step 27: Multi-Horizon Benchmarking (Table II)**: Direct multi-horizon models vs 3 baselines (Persistence, Seasonal-Naive, Climatology) across identical rows with 95% day-block bootstrap confidence intervals (`fig_horizon_strong_baselines.png`, `table_II_strong_baselines.csv`, `table_II_strong_baselines.tex`).
24. **Step 28: Competitor Model Suite**: Evaluates Ridge Regression, XGBoost, and CatBoost on the exact same features and split.
25. **Step 29: Deep Learning Sequence Baseline**: Fits a Gated Recurrent Unit (GRU) neural network over a 72-hour rolling history of PM2.5, criteria pollutants, and meteorological variables.
26. **Step 30: 24h Benchmark Comparison (Table I)**: Comprehensive comparison of all 8 models on identical test rows with 95% bootstrap confidence intervals (`table_I_benchmark.csv`, `table_I_benchmark.tex`).
27. **Step 31: Multi-Year Rolling-Origin Cross-Validation**: Validates forecasting performance across 6 consecutive test years (2019–2024), demonstrating consistent $10.6 \pm 4.2\%$ improvement over persistence (`table_rolling_origin.csv`).
28. **Step 32: Leakage & Interpolation Ablation**: Tests models strictly on genuinely measured hours (where $t-1$, $t$, and $t+24$ are 100% unimputed), confirming no synthetic interpolation bias.
29. **Step 33: SHAP Explainability & Feature Attribution**: Calculates TreeExplainer SHAP values and breaks down importance by feature family (own PM2.5 history: 66.9%, time/station: 17.0%, meteorology: 9.6%, cross-station: 4.2%, pollutants: 2.3%) (`fig_shap_summary.png`, `fig_shap_bar.png`, `table_shap_families.csv`).
30. **Step 34: Probability Calibration**: Evaluates Expected Calibration Error (ECE = 0.0071) and Brier Score (0.0797) for the extreme-event classifier (`fig_calibration.png`).
31. **Step 35: Conformalized Quantile Regression (CQR)**: Calibrates prediction intervals with conformal guarantees and evaluates coverage by station and calendar month (`table_interval_coverage_station.csv`).
32. **Step 36: Paper Table & Extra Figure Export**: Packages extra publication-ready artifacts into `results_extra.zip`.

---

## Running the Notebooks

Launch Jupyter and open either notebook:
```bash
jupyter notebook notebooks/ML_Lab_improved.ipynb
```
Ensure the dataset `dhaka_cams_hourly_2012_2026.csv` is present in `data/` or relative to the notebook working directory.
