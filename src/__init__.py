"""
Dhaka PM2.5 Air Quality Forecasting and Anomaly Detection System
===============================================================
A machine learning package for hourly particulate matter (PM2.5) forecasting,
sensor anomaly detection, multi-horizon outlook, and extreme pollution alerts
across Continuous Air Monitoring Stations (CAMS) in Dhaka, Bangladesh.
"""

__version__ = "1.0.0"
__author__ = "Hemal Fahim"

from .data_loader import load_raw_data, clean_air_quality_data
from .anomaly_detection import fit_anomaly_detector, detect_anomalies
from .features import engineer_features, get_feature_columns
from .models import (
    train_lgbm_regressor,
    train_quantile_regressors,
    train_high_pollution_classifier,
    load_lgb_model,
)
from .evaluate import (
    compute_regression_metrics,
    pm25_to_aqi_category,
    evaluate_high_pollution_detection,
    evaluate_quantiles,
)
from .predict import check_data_staleness, forecast_next_24h

__all__ = [
    "load_raw_data",
    "clean_air_quality_data",
    "fit_anomaly_detector",
    "detect_anomalies",
    "engineer_features",
    "get_feature_columns",
    "train_lgbm_regressor",
    "train_quantile_regressors",
    "train_high_pollution_classifier",
    "load_lgb_model",
    "compute_regression_metrics",
    "pm25_to_aqi_category",
    "evaluate_high_pollution_detection",
    "evaluate_quantiles",
    "check_data_staleness",
    "forecast_next_24h",
]
