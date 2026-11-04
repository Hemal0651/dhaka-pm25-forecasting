"""
Feature engineering pipelines for spatial and temporal PM2.5 forecasting.
"""

from typing import List, Tuple, Sequence
import pandas as pd
import numpy as np


STATION_CODE_MAP = {
    "CAMS-1": 0,
    "CAMS-2": 1,
    "CAMS-3": 2,
}


def engineer_features(
    df: pd.DataFrame,
    horizons: Sequence[int] = (24,),
    lags: Sequence[int] = (1, 2, 3, 6, 12, 24, 48, 168),
    roll_windows: Sequence[int] = (24, 168),
    include_cross_station: bool = True,
    target_col: str = "pm2_5_ug_m3",
) -> pd.DataFrame:
    """
    Construct multi-lag, rolling statistics, cyclic temporal, and cross-station spatial features.

    Args:
        df: Input cleaned DataFrame with 'timestamp', 'station_id', and target column.
        horizons: Target forecast horizon hours (e.g. (24,) for 24h-ahead).
        lags: Autoregressive lag steps in hours.
        roll_windows: Rolling moving average and dispersion windows in hours.
        include_cross_station: Whether to pivot contemporaneous readings across network stations.
        target_col: Target air quality metric name.

    Returns:
        DataFrame enriched with predictive features and target columns.
    """
    data = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True).copy()
    g = data.groupby("station_id")[target_col]

    # 1. Autoregressive Lags
    for lag in lags:
        data[f"pm25_lag{lag}"] = g.shift(lag)

    # 2. Rolling Window Aggregations
    for w in roll_windows:
        min_p = max(4, w // 4)
        data[f"pm25_roll{w}"] = g.transform(lambda s: s.rolling(w, min_periods=min_p).mean())
        data[f"pm25_std{w}"] = g.transform(lambda s: s.rolling(w, min_periods=min_p).std())

    # Short-window diurnal min/max
    data["pm25_min24"] = g.transform(lambda s: s.rolling(24, min_periods=6).min())
    data["pm25_max24"] = g.transform(lambda s: s.rolling(24, min_periods=6).max())

    # 3. Dynamic Spike & Trend Features
    data["pm25_diff1"] = data[target_col] - data.get("pm25_lag1", g.shift(1))
    data["pm25_diff24"] = data[target_col] - data.get("pm25_lag24", g.shift(24))
    if "pm25_roll24" in data.columns:
        data["pm25_ratio_roll24"] = data[target_col] / (data["pm25_roll24"] + 1.0)

    # 4. Harmonic Cyclic Temporal Encodings
    dt = data["timestamp"].dt
    data["hour"] = dt.hour
    data["dow"] = dt.dayofweek
    data["month"] = dt.month
    data["doy"] = dt.dayofyear

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["dow_sin"] = np.sin(2 * np.pi * data["dow"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["dow"] / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)
    data["doy_sin"] = np.sin(2 * np.pi * (data["doy"] - 1) / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * (data["doy"] - 1) / 365.25)

    # Station categorical code
    data["station_code"] = data["station_id"].map(lambda s: STATION_CODE_MAP.get(s, -1))

    # 5. Cross-Station Spatial Network Features
    if include_cross_station and "station_id" in data.columns:
        pivot_stns = data.pivot(index="timestamp", columns="station_id", values=target_col)
        pivot_cols = {}
        for col in pivot_stns.columns:
            pivot_cols[col] = f"pm25_{col}"
        pivot_stns = pivot_stns.rename(columns=pivot_cols).reset_index()
        data = data.merge(pivot_stns, on="timestamp", how="left")

    # 6. Future Targets for Multi-Horizon Evaluation
    for h in horizons:
        col_name = "target" if len(horizons) == 1 and h == 24 else f"target_{h}"
        data[col_name] = g.shift(-h)

    return data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Filter and return the model input feature list from a feature-engineered DataFrame.
    """
    exclude = {
        "timestamp",
        "station_id",
        "station_name",
        "is_imputed",
        "anomaly",
        "anomaly_score",
        "physical_inversion",
        "vertical_wind_speed_ms",
        "year",
    }
    feature_cols = [
        col for col in df.columns
        if not col.startswith("target") and col not in exclude
    ]
    return feature_cols
