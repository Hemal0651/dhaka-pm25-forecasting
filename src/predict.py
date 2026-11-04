"""
Inference routines: 24h forecasts, autoregressive 72h rollouts, and sensor staleness detection.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from .evaluate import pm25_to_aqi_category


def check_data_staleness(
    df: pd.DataFrame,
    current_time: Optional[pd.Timestamp] = None,
    max_stale_hours: int = 72,
) -> Dict[str, Any]:
    """
    Check whether a station has recent observations before issuing operational forecasts.

    Args:
        df: Station history DataFrame with 'timestamp' and 'pm2_5_ug_m3'.
        current_time: Reference system timestamp (defaults to max timestamp in df).
        max_stale_hours: Maximum allowable observation gap before flagging as offline/stale.

    Returns:
        Dictionary with staleness boolean, hours since last valid reading, and last timestamp.
    """
    valid = df.dropna(subset=["pm2_5_ug_m3"]).sort_values("timestamp")
    if len(valid) == 0:
        return {
            "is_stale": True,
            "hours_stale": np.inf,
            "last_valid_time": None,
            "reason": "No valid observations found in history.",
        }

    last_valid_time = valid["timestamp"].iloc[-1]
    if current_time is None:
        current_time = df["timestamp"].max()

    gap_hours = (current_time - last_valid_time).total_seconds() / 3600.0
    is_stale = gap_hours > max_stale_hours

    return {
        "is_stale": is_stale,
        "hours_stale": round(gap_hours, 1),
        "last_valid_time": last_valid_time,
        "reason": f"Station observation age is {gap_hours:.1f}h (> {max_stale_hours}h limit)" if is_stale else "Active",
    }


def forecast_next_24h(
    booster: lgb.Booster,
    feature_row: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Generate a 24-hour ahead single-step point prediction.

    Args:
        booster: Loaded LightGBM Booster object.
        feature_row: 1-row DataFrame containing all expected features.
        feature_names: Ordered list of feature names used during model training.

    Returns:
        Dictionary containing predicted continuous PM2.5 (µg/m³) and its AQI health classification.
    """
    X = feature_row[feature_names].values
    pred = float(booster.predict(X)[0])
    pred = max(0.0, pred)  # Physical floor at 0

    aqi_class = str(pm25_to_aqi_category(pd.Series([pred])).iloc[0])

    return {
        "forecast_pm25_ug_m3": round(pred, 2),
        "aqi_category": aqi_class,
    }


def recursive_rollout_72h(
    booster: lgb.Booster,
    history_df: pd.DataFrame,
    station_id: str,
    feature_cols: List[str],
    horizons: int = 72,
) -> pd.DataFrame:
    """
    Produce iterative autoregressive forecasts out to 72 hours, feeding step predictions back as lags.

    Args:
        booster: Trained 1-step or short-horizon LightGBM Booster.
        history_df: Historical records for this station (must span >= 168 hours of continuous history).
        station_id: CAMS station identifier.
        feature_cols: Required model features.
        horizons: Number of forward hours to predict.

    Returns:
        DataFrame containing future timestamps, step index, predicted PM2.5, and AQI category.
    """
    recent = history_df[history_df["station_id"] == station_id].sort_values("timestamp").copy()
    if len(recent) < 168:
        raise ValueError(f"Station {station_id} history must have at least 168 hours for lag computation.")

    # Check staleness
    staleness = check_data_staleness(recent)
    if staleness["is_stale"]:
        raise RuntimeError(f"Cannot generate recursive forecast for {station_id}: {staleness['reason']}")

    forecasts = []
    current_time = recent["timestamp"].iloc[-1]
    series_pm25 = list(recent["pm2_5_ug_m3"].values)

    for step in range(1, horizons + 1):
        target_time = current_time + pd.Timedelta(hours=step)

        # Build feature vector dynamically from historical + predicted series
        h = target_time.hour
        dow = target_time.dayofweek
        mon = target_time.month
        doy = target_time.dayofyear

        feat_dict = {
            "hour_sin": np.sin(2 * np.pi * h / 24.0),
            "hour_cos": np.cos(2 * np.pi * h / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "month_sin": np.sin(2 * np.pi * (mon - 1) / 12.0),
            "month_cos": np.cos(2 * np.pi * (mon - 1) / 12.0),
            "doy_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
            "doy_cos": np.cos(2 * np.pi * (doy - 1) / 365.25),
        }

        # Lags from the tail of series_pm25
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            feat_dict[f"pm25_lag{lag}"] = series_pm25[-lag] if len(series_pm25) >= lag else np.nan

        # Rolling statistics
        feat_dict["pm25_roll24"] = float(np.nanmean(series_pm25[-24:]))
        feat_dict["pm25_roll168"] = float(np.nanmean(series_pm25[-168:]))

        row_df = pd.DataFrame([feat_dict])
        available_cols = [c for c in feature_cols if c in row_df.columns]
        X = row_df[available_cols].values

        pred = float(booster.predict(X)[0])
        pred = max(0.0, pred)
        series_pm25.append(pred)

        forecasts.append({
            "step_hour": step,
            "forecast_timestamp": target_time,
            "predicted_pm25": round(pred, 2),
            "aqi_category": str(pm25_to_aqi_category(pd.Series([pred])).iloc[0]),
        })

    return pd.DataFrame(forecasts)
