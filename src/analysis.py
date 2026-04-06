"""
analysis.py — Statistical analysis functions.

Provides:
  compute_ccf()         — Cross-Correlation Function between precipitation and turbidity
  find_optimal_lag()    — Peak lag from CCF (background evidence for the Hydrological Buffer Effect)
  classify_storms()     — Assign Dry / Light / Moderate / Heavy labels to each hour
  storm_intensity_metrics() — R² and RMSE broken down by storm intensity (robustness sub-analysis)
  hazard_latency()      — Average hours between a real hazard spike and model alert
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==============================================================================
#  Cross-Correlation Function (CCF)
# ==============================================================================

def compute_ccf(
    precip:    pd.Series | np.ndarray,
    turbidity: pd.Series | np.ndarray,
    max_lag:   int = config.MAX_LAG_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the normalized cross-correlation between precipitation and turbidity
    for lags 0 … max_lag hours.

    A high correlation at lag k means turbidity peaks k hours after rainfall.

    Returns:
        lags          — integer array [0, 1, … max_lag]
        correlations  — Pearson-r values at each lag
    """
    x = np.asarray(precip,    dtype=float)
    y = np.asarray(turbidity, dtype=float)

    # Z-score normalization
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)

    # Full cross-correlation (scipy uses the signal convention)
    full_corr = signal.correlate(y, x, mode="full")
    lags_full = signal.correlation_lags(len(y), len(x), mode="full")

    # Select only non-negative lags (turbidity follows rain, not precedes it)
    mask  = (lags_full >= 0) & (lags_full <= max_lag)
    lags  = lags_full[mask]
    corrs = full_corr[mask] / len(x)   # normalize to Pearson-like coefficient

    return lags, corrs


def find_optimal_lag(
    precip:    pd.Series | np.ndarray,
    turbidity: pd.Series | np.ndarray,
    max_lag:   int = config.MAX_LAG_HOURS,
) -> int:
    """
    Return the lag (hours) with the highest cross-correlation.

    Used as background evidence to quantify the Hydrological Buffer Effect.
    The difference between the IL lag and CA lag IS the measured Sponge Effect.
    (This is background evidence, not an experimental variable.)
    """
    lags, corrs = compute_ccf(precip, turbidity, max_lag)
    return int(lags[np.argmax(corrs)])


# ==============================================================================
#  Storm classification
# ==============================================================================

def classify_storms(
    precip: pd.Series,
    classes: dict | None = None,
) -> pd.Series:
    """
    Assign a storm-intensity label to each hourly observation.

    Labels: 'none', 'light', 'moderate', 'heavy'

    Uses config.STORM_CLASSES by default.
    """
    if classes is None:
        classes = config.STORM_CLASSES

    labels = pd.Series("none", index=precip.index, dtype="object")
    for name, (lo, hi) in classes.items():
        mask = (precip >= lo) & (precip < hi)
        labels[mask] = name
    return labels


# ==============================================================================
#  Stress-test breakdown
# ==============================================================================

def storm_intensity_metrics(
    y_true:  np.ndarray | pd.Series,
    y_pred:  np.ndarray | pd.Series,
    precip:  pd.Series,
    threshold: float = None,
) -> pd.DataFrame:
    """
    Compute R², RMSE, TPR, and FAR separately for each storm intensity class.

    This is the robustness sub-analysis — storm intensity is a significant variable
    that is identified, tracked, and controlled for (not an independent variable).
    Frame as: "To assess model robustness across precipitation conditions, results
    were further analyzed by storm intensity classification."

    Returns a DataFrame indexed by storm class with columns
    [r2, rmse, tpr, far, n_samples, n_hazard].
    """
    from sklearn.metrics import r2_score, mean_squared_error
    if threshold is None:
        threshold = config.TURBIDITY_HAZARD_NTU

    storm_labels = classify_storms(precip)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = []
    for cls in ["dry", "light", "moderate", "heavy"]:
        mask = (storm_labels == cls).values
        idx = np.where(mask)[0]
        idx = idx[idx < len(y_true)]

        if len(idx) < 5:
            rows.append({"storm_class": cls, "r2": np.nan, "rmse": np.nan,
                         "tpr": np.nan, "far": np.nan,
                         "n_samples": 0, "n_hazard": 0})
            continue

        yt = y_true[idx]
        yp = y_pred[idx]

        # Regression metrics
        r2   = round(r2_score(yt, yp), 4)
        rmse = round(np.sqrt(mean_squared_error(yt, yp)), 3)

        # Safety metrics within this storm class
        actual_haz = yt >= threshold
        pred_haz   = yp >= threshold
        tp = np.sum(actual_haz & pred_haz)
        fn = np.sum(actual_haz & ~pred_haz)
        fp = np.sum(~actual_haz & pred_haz)
        tpr = round(float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan, 4)
        far = round(float(fp / (fp + tp)) if (fp + tp) > 0 else np.nan, 4)

        rows.append({
            "storm_class": cls,
            "r2":          r2,
            "rmse":        rmse,
            "tpr":         tpr,
            "far":         far,
            "n_samples":   len(idx),
            "n_hazard":    int(tp + fn),
        })

    return pd.DataFrame(rows).set_index("storm_class")


# ==============================================================================
#  Hazard latency
# ==============================================================================

def hazard_latency(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    threshold: float = config.TURBIDITY_HAZARD_NTU,
) -> float:
    """
    Estimate the average lead-time (hours) between a real hazard event being
    detected by the model and the same event occurring in the true record.

    A positive latency means the model alerts BEFORE the real spike (good!).
    Negative means it alerts after (late warning).

    Uses a simple peak-matching strategy:
      For each true hazard peak, find the closest predicted-hazard crossing.
    """
    actual    = np.asarray(y_true) >= threshold
    predicted = np.asarray(y_pred) >= threshold

    # Find rising-edge event times
    true_events  = np.where(np.diff(actual.astype(int))    == 1)[0]
    pred_events  = np.where(np.diff(predicted.astype(int)) == 1)[0]

    if len(true_events) == 0 or len(pred_events) == 0:
        return np.nan

    latencies = []
    for te in true_events:
        diffs = pred_events.astype(int) - int(te)
        # Favour predicted events that come BEFORE the true event (early warning)
        if np.any(diffs <= 0):
            best = diffs[diffs <= 0].max()   # least-negative = closest before
        else:
            best = diffs.min()               # earliest after if no prior match
        latencies.append(-best)              # sign convention: + = early

    return float(np.mean(latencies))
