"""
preprocessing.py — Feature engineering and train/test preparation.

Three data products are produced here, corresponding to the three experimental levels:
  1. RF Level 1 dataset  : [precip only]                         -> turbidity (+6 hr)
  2. RF Level 2 dataset  : [precip + soil_moisture + rolling]    -> turbidity (+6 hr)
  3. LSTM Level 3 dataset: sliding-window sequences of
       (precip, soil_moisture) — same features as Level 2.
       The LSTM's advantage comes from its temporal memory architecture alone,
       not from additional features. This keeps the experiment controlled:
         L1 -> L2: one change (add soil moisture)
         L2 -> L3: one change (architecture, same features)

Constants (all levels):
  - 80/20 train/test split (RF), 64/16/20 (LSTM — 20% test preserved)
  - 6-hour forecasting window (FORECAST_HORIZON)
  - Hazard threshold: 25 NTU
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ==============================================================================
#  Rolling / lag feature engineering
# ==============================================================================

def add_rolling_precip(df: pd.DataFrame,
                        precip_col: str = "precipitation_mm_hr",
                        windows: list[int] | None = None) -> pd.DataFrame:
    """
    Add cumulative precipitation over multiple look-back windows.
    These features help Random Forest learn antecedent wetness without memory.

    Windows are capped at 12 hours. The LSTM's 36-hour input window gives it
    access to precip history from t-35 to t, including a range no RF rolling
    window reaches — providing LSTM a genuine temporal-memory advantage.
    """
    if windows is None:
        windows = [3, 6, 12]
    df = df.copy()
    for w in windows:
        df[f"precip_roll{w}hr"] = (
            df[precip_col].rolling(w, min_periods=1).sum()
        )
    return df


def add_soil_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a physics-motivated interaction term: rolling_precip × soil_moisture.
    Captures the 'amplifier' effect — wet soils make rainfall more damaging.
    Uses the 12-hour window (the longest RF rolling window available).
    """
    df = df.copy()
    df["precip_x_sm"] = df["precip_roll12hr"] * df["soil_moisture"]
    return df


# ==============================================================================
#  Random Forest data preparation
# ==============================================================================

def prepare_rf_level1(df: pd.DataFrame,
                       forecast_horizon: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """
    Level 1 (Baseline RF): only current precipitation as input.

    Predicts turbidity `forecast_horizon` hours ahead.
    The 6-hour forecasting window is a constant across all levels.
    """
    df = df.dropna(subset=["precipitation_mm_hr", "turbidity"]).copy()
    df["turbidity_future"] = df["turbidity"].shift(-forecast_horizon)
    df = df.dropna(subset=["turbidity_future"])
    X = df[["precipitation_mm_hr"]]
    y = df["turbidity_future"]
    return X, y


def prepare_rf_level2(df: pd.DataFrame,
                       forecast_horizon: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """
    Level 2 (RF + Soil Moisture): precipitation + rolling windows + soil moisture.

    The target is shifted forward by forecast_horizon so the model learns to
    predict future turbidity from current conditions (early-warning window).
    Adding soil moisture gives the RF Antecedent Moisture Condition (AMC) awareness —
    the fix for its blindness to pre-storm soil saturation.
    """
    df = add_rolling_precip(df)
    df = add_soil_interaction(df)
    df = df.copy()
    df["turbidity_future"] = df["turbidity"].shift(-forecast_horizon)
    df = df.dropna()

    feature_cols = [
        "precipitation_mm_hr",
        "precip_roll3hr",
        "precip_roll6hr",
        "precip_roll12hr",
        "soil_moisture",
        "precip_x_sm",
    ]
    X = df[feature_cols]
    y = df["turbidity_future"]
    return X, y


# ==============================================================================
#  LSTM data preparation (Level 3)
# ==============================================================================

def prepare_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "turbidity",
    window_size: int = 48,
    forecast_horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Create sliding-window input sequences for the LSTM.

    Uses the same features as Level 2 (precipitation + soil_moisture).
    The LSTM's advantage over Level 2 RF comes solely from its temporal
    memory architecture, keeping the experiment controlled.

    X shape: (samples, window_size, n_features)
    y shape: (samples,)

    Returns:
        X_seq, y_seq — numpy arrays ready for model.fit()
        scaler_X, scaler_y — fitted scalers (needed to invert predictions)
    """
    df = df[feature_cols + [target_col]].dropna().copy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_raw = df[feature_cols].values
    y_raw = df[target_col].values

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    X_seqs, y_seqs = [], []
    for i in range(window_size, len(X_scaled) - forecast_horizon + 1):
        X_seqs.append(X_scaled[i - window_size: i])
        y_seqs.append(y_scaled[i + forecast_horizon - 1])

    return np.array(X_seqs), np.array(y_seqs), scaler_X, scaler_y


# ==============================================================================
#  Train / Test splits (temporal — never shuffle time series!)
# ==============================================================================

def rf_train_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_frac: float = 0.20,
) -> tuple:
    """
    80/20 temporal split for Random Forest (no validation tier needed for RF).

    Returns: X_train, X_test, y_train, y_test
    """
    n      = len(X)
    i_test = int(n * (1 - test_frac))

    def _slice(arr, a, b):
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            return arr.iloc[a:b]
        return arr[a:b]

    return (
        _slice(X, 0,      i_test),
        _slice(X, i_test, n),
        _slice(y, 0,      i_test),
        _slice(y, i_test, n),
    )


def lstm_temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.64,
    val_frac:   float = 0.16,
) -> tuple:
    """
    64/16/20 temporal split for LSTM training.

    The 20% test set is the same final slice as the 80/20 RF split,
    making all levels directly comparable on the same test period.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    n       = len(X)
    i_val   = int(n * train_frac)
    i_test  = int(n * (train_frac + val_frac))

    return (
        X[:i_val],        X[i_val:i_test],  X[i_test:],
        y[:i_val],        y[i_val:i_test],  y[i_test:],
    )
