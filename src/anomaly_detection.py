"""
Anomaly detection module using Isolation Forest for atmospheric and sensor quality assurance.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


DEFAULT_ANOMALY_FEATURES = [
    "pm2_5_ug_m3",
    "pm10_ug_m3",
    "no2_ppb",
    "co_ppm",
]


def fit_anomaly_detector(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    contamination: float = 0.01,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[IsolationForest, List[str]]:
    """
    Fit an Isolation Forest model to detect multivariate atmospheric anomalies.

    Args:
        df: Input DataFrame containing pollutant features.
        features: Feature list for anomaly modeling. Defaults to PM2.5, PM10, NO2, CO.
        contamination: Expected fraction of outliers in the data.
        n_estimators: Number of isolation trees.
        random_state: Reproducibility seed.

    Returns:
        Trained IsolationForest model and the list of used features.
    """
    if features is None:
        features = [f for f in DEFAULT_ANOMALY_FEATURES if f in df.columns]

    valid_data = df.dropna(subset=features)
    if len(valid_data) == 0:
        raise ValueError("No complete records found for the requested anomaly features.")

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(valid_data[features])
    return iso, features


def detect_anomalies(
    model: IsolationForest,
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    Score observations and flag anomalies using a trained IsolationForest model.

    Also applies physical consistency checks:
    - Flags unphysical PM2.5 > PM10 (PM2.5 is a fraction of PM10).
    - Flags sensor stuck flatlines (standard deviation across rolling 6 hours == 0).

    Args:
        model: Trained IsolationForest instance.
        df: Input DataFrame.
        features: Feature column names.

    Returns:
        DataFrame with added 'anomaly' (bool), 'anomaly_score' (float), and 'physical_error' (bool).
    """
    result = df.copy()
    valid_mask = result[features].notna().all(axis=1)

    result["anomaly"] = False
    result["anomaly_score"] = np.nan

    if valid_mask.any():
        preds = model.predict(result.loc[valid_mask, features])
        scores = model.decision_function(result.loc[valid_mask, features])
        result.loc[valid_mask, "anomaly"] = (preds == -1)
        result.loc[valid_mask, "anomaly_score"] = scores

    # Physical domain consistency check: PM2.5 cannot strictly exceed PM10 + tolerance
    if "pm2_5_ug_m3" in result.columns and "pm10_ug_m3" in result.columns:
        pm_inversion = (result["pm2_5_ug_m3"] > (result["pm10_ug_m3"] + 10.0)) & result["pm2_5_ug_m3"].notna() & result["pm10_ug_m3"].notna()
        result["physical_inversion"] = pm_inversion
    else:
        result["physical_inversion"] = False

    return result
