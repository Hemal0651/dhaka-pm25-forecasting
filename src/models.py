"""
LightGBM training, quantile estimation, classification, and persistence pipelines.
"""

from typing import Dict, Any, Optional, Sequence
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None



DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

TUNED_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 18,
    "min_child_samples": 155,
    "reg_lambda": 0.1306,
    "reg_alpha": 0.1225,
    "colsample_bytree": 0.5283,
    "subsample": 0.8424,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _require_lgb():
    if lgb is None:
        raise ImportError(
            "LightGBM is required for modeling functionality. "
            "Please install dependencies via: pip install -r requirements.txt"
        )


def train_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
    use_log_target: bool = False,
):
    """
    Train a LightGBM regressor with early stopping on validation loss.
    """
    _require_lgb()
    p = (DEFAULT_LGBM_PARAMS if params is None else params).copy()

    y_tr = np.log1p(y_train) if use_log_target else y_train
    y_va = np.log1p(y_val) if use_log_target else y_val

    model = lgb.LGBMRegressor(**p)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(0),
    ]

    model.fit(
        X_train,
        y_tr,
        eval_set=[(X_val, y_va)],
        callbacks=callbacks,
    )
    return model


def train_quantile_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
):
    """
    Train pinball loss quantile regression models for uncertainty quantification (e.g. 80% intervals).
    """
    _require_lgb()
    base_params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        base_params.update(params)

    models = {}
    for q in quantiles:
        p = base_params.copy()
        p["objective"] = "quantile"
        p["alpha"] = q

        m = lgb.LGBMRegressor(**p)
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ]
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
        models[q] = m

    return models


def train_high_pollution_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 150.0,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
):
    """
    Train a binary LightGBM classifier specifically targeting severe/hazardous episodes (PM2.5 > threshold).
    """
    _require_lgb()
    p = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        p.update(params)

    ytr_bin = (y_train > threshold).astype(int)
    yva_bin = (y_val > threshold).astype(int)

    clf = lgb.LGBMClassifier(**p)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(0),
    ]
    clf.fit(X_train, ytr_bin, eval_set=[(X_val, yva_bin)], callbacks=callbacks)
    return clf


def save_lgb_model(model: Any, filepath: str) -> None:
    """
    Export LightGBM Booster text representation.
    """
    _require_lgb()
    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(filepath)


def load_lgb_model(filepath: str):
    """
    Load LightGBM Booster from text representation.
    """
    _require_lgb()
    return lgb.Booster(model_file=filepath)
