"""
Data loading and quality assurance routines for Dhaka CAMS air quality observations.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


POLLUTANT_COLUMNS = [
    "pm2_5_ug_m3",
    "pm10_ug_m3",
    "so2_ppb",
    "no2_ppb",
    "no_ppb",
    "nox_ppb",
    "co_ppm",
    "o3_ppb",
]

METEOROLOGICAL_COLUMNS = [
    "temp_c",
    "rh_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "barometric_pressure_hpa",
    "solar_rad_wm2",
    "rain_mm",
]


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw continuous hourly air quality observations from CSV.

    Args:
        csv_path: Absolute or relative path to raw/processed CSV.

    Returns:
        DataFrame with parsed datetime timestamps.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df


def clean_air_quality_data(
    df: pd.DataFrame,
    start_date: str = "2013-01-01",
    end_date: str = "2025-01-01",
    interp_limit: int = 3,
    pollutants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply quality assurance, negative filtering, timeline reindexing, and short-gap interpolation.

    Steps:
    1. Filter time period to high-coverage years (default: 2013-2024).
    2. Drop columns with excessive missingness (e.g. vertical_wind_speed_ms).
    3. Treat negative pollutant readings as sensor artifacts and convert to NaN.
    4. Ensure regular hourly timeline per station (fill missing timestamps).
    5. Linearly interpolate short sensor dropouts (gap <= interp_limit hours).
    6. Maintain a boolean indicator flag (`is_imputed`) for data lineage.

    Args:
        df: Raw DataFrame with 'timestamp', 'station_id', and pollutant measurements.
        start_date: Start of valid study window (inclusive string 'YYYY-MM-DD').
        end_date: End of valid study window (exclusive string 'YYYY-MM-DD').
        interp_limit: Maximum consecutive hours to impute via linear interpolation.
        pollutants: List of pollutant column names to sanitize.

    Returns:
        Cleaned, regularly spaced hourly DataFrame.
    """
    df = df.copy()
    if pollutants is None:
        pollutants = POLLUTANT_COLUMNS

    # Filter temporal window
    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)].copy()

    # Drop columns with high sparsity if present
    cols_to_drop = [c for c in ["vertical_wind_speed_ms", "year"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Sanitize unphysical negative concentrations
    for col in pollutants:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Build regular hourly timeline per station
    stations = df["station_id"].unique()
    full_idx = pd.date_range(start_date, pd.to_datetime(end_date) - pd.Timedelta(hours=1), freq="1h")

    cleaned_stations = []
    for stn in stations:
        sub = df[df["station_id"] == stn].drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        sub = sub.reindex(full_idx)
        sub["station_id"] = stn

        # Retain station_name if present
        if "station_name" in sub.columns and sub["station_name"].dropna().shape[0] > 0:
            stn_name = sub["station_name"].dropna().iloc[0]
            sub["station_name"] = stn_name

        # Linear interpolation for short outages on primary pollutants
        if "pm2_5_ug_m3" in sub.columns:
            was_missing = sub["pm2_5_ug_m3"].isna()
            sub["pm2_5_ug_m3"] = sub["pm2_5_ug_m3"].interpolate(method="time", limit=interp_limit)
            sub["is_imputed"] = was_missing & sub["pm2_5_ug_m3"].notna()
        else:
            sub["is_imputed"] = False

        cleaned_stations.append(sub.reset_index().rename(columns={"index": "timestamp"}))

    clean_df = pd.concat(cleaned_stations, ignore_index=True)
    clean_df = clean_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return clean_df
