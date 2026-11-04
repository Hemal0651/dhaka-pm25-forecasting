"""
Evaluation metrics, Air Quality Index (AQI) categorical mappings, and alert benchmark functions.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


AQI_CATEGORIES = [
    (0.0, 12.0, "Good"),
    (12.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, "Unhealthy"),
    (150.5, 250.4, "Very Unhealthy"),
    (250.5, 10000.0, "Hazardous"),
]


def pm25_to_aqi_category(values: pd.Series) -> pd.Series:
    """
    Categorize continuous PM2.5 concentrations into standard EPA/DoE 6-tier AQI danger levels.

    Breakpoints:
      - Good: 0.0 - 12.0 µg/m³
      - Moderate: 12.1 - 35.4 µg/m³
      - Unhealthy for Sensitive Groups: 35.5 - 55.4 µg/m³
      - Unhealthy: 55.5 - 150.4 µg/m³
      - Very Unhealthy: 150.5 - 250.4 µg/m³
      - Hazardous: > 250.4 µg/m³
    """
    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    bins = [-np.inf, 12.0, 35.4, 55.4, 150.4, 250.4, np.inf]
    return pd.cut(values, bins=bins, labels=labels)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard continuous forecasting accuracy metrics: MAE, RMSE, and R2.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]

    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
    }


def evaluate_high_pollution_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 150.0,
) -> Dict[str, Any]:
    """
    Evaluate binary alerting performance for severe pollution episodes (PM2.5 > threshold).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt_bin = (np.asarray(y_true)[mask] > threshold).astype(int)
    yp_bin = (np.asarray(y_pred)[mask] > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(yt_bin, yp_bin, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(yt_bin, yp_bin, average="binary", zero_division=0)

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "recall": round(float(r), 4),
        "precision": round(float(p), 4),
        "f1": round(float(f1), 4),
        "total_severe_hours": int(tp + fn),
        "catch_rate_pct": round(float(r) * 100, 2),
    }


def evaluate_quantiles(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate empirical coverage and dispersion for an 80% quantile prediction interval.
    """
    mask = np.isfinite(y_true) & np.isfinite(q10) & np.isfinite(q90)
    yt = np.asarray(y_true)[mask]
    l = np.asarray(q10)[mask]
    u = np.asarray(q90)[mask]

    # Enforce non-crossing sorting
    lower = np.minimum(l, u)
    upper = np.maximum(l, u)

    covered = (yt >= lower) & (yt <= upper)
    coverage = float(np.mean(covered))
    avg_width = float(np.mean(upper - lower))
    median_mae = float(mean_absolute_error(yt, np.asarray(q50)[mask]))

    return {
        "empirical_coverage_80pct": round(coverage * 100, 2),
        "mean_interval_width": round(avg_width, 2),
        "median_mae": round(median_mae, 2),
    }
