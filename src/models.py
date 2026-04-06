"""
models.py — Machine-learning models for The Sponge Effect.

Provides:
  RandomForestModel  — sklearn RandomForestRegressor with a scikit-learn-style API
  LSTMModel          — Keras LSTM for multi-step time-series forecasting
  compute_metrics()  — R², RMSE, MAE and hazard-detection metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==============================================================================
#  Shared metric helper
# ==============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     threshold: float = config.TURBIDITY_HAZARD_NTU) -> dict:
    """
    Compute regression accuracy + hazard-detection metrics.

    Hazard event = turbidity exceeds `threshold` NTU.

    Returns dict with keys:
      r2, rmse, mae, tpr, far, n_hazard_events, n_false_alarms, latency_hrs
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # Hazard flags
    actual_hazard    = y_true >= threshold
    predicted_hazard = y_pred >= threshold

    tp = np.sum(actual_hazard  & predicted_hazard)
    fn = np.sum(actual_hazard  & ~predicted_hazard)
    fp = np.sum(~actual_hazard & predicted_hazard)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return {
        "r2":                round(float(r2),   4),
        "rmse":              round(float(rmse), 3),
        "mae":               round(float(mae),  3),
        "tpr":               round(float(tpr),  4),
        "far":               round(float(far),  4),
        "n_hazard_events":   int(tp + fn),
        "n_false_alarms":    int(fp),
    }


# ==============================================================================
#  Random Forest
# ==============================================================================

class RandomForestModel:
    """
    Thin wrapper around sklearn RandomForestRegressor.

    Usage:
        model = RandomForestModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = model.score(X_test, y_test)
    """

    def __init__(self,
                 n_estimators: int = 150,
                 max_depth:    int = 12,
                 random_state: int = config.RANDOM_SEED):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def fit(self, X_train, y_train):
        if hasattr(X_train, "columns"):
            self.feature_names = list(X_train.columns)
        self.rf.fit(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        return self.rf.predict(X)

    def score(self, X, y_true,
               threshold: float = config.TURBIDITY_HAZARD_NTU) -> dict:
        return compute_metrics(y_true, self.predict(X), threshold)

    @property
    def feature_importances(self) -> pd.Series:
        imp = pd.Series(self.rf.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False)


# ==============================================================================
#  LSTM
# ==============================================================================

class LSTMModel:
    """
    Stacked Keras LSTM for turbidity time-series forecasting.

    Architecture:
        Input   (window_size, n_features)
        LSTM    64 units  -> return sequences
        Dropout 20%
        LSTM    32 units
        Dense   16 ReLU
        Dense   1  (turbidity, scaled)

    The model works in *scaled* space; call predict_rescaled() to get real NTU.
    """

    def __init__(self,
                 window_size:      int = config.WINDOW_SIZE,
                 n_features:       int = 2,
                 epochs:           int = config.LSTM_EPOCHS,
                 batch_size:       int = config.LSTM_BATCH,
                 patience:         int = config.LSTM_PATIENCE):
        self.window_size = window_size
        self.n_features  = n_features
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience
        self.model       = None
        self.history     = None
        self.scaler_y    = None   # set during train() call

    def _build(self):
        """Construct the Keras model graph."""
        import tensorflow as tf
        from tensorflow import keras
        tf.random.set_seed(config.RANDOM_SEED)

        inp = keras.Input(shape=(self.window_size, self.n_features))
        x   = keras.layers.LSTM(64, return_sequences=True)(inp)
        x   = keras.layers.Dropout(0.20)(x)
        x   = keras.layers.LSTM(32)(x)
        x   = keras.layers.Dense(16, activation="relu")(x)
        out = keras.layers.Dense(1)(x)

        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss="mse")
        return model

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val:   np.ndarray,
              y_val:   np.ndarray,
              scaler_y=None,
              verbose:  int = 1):
        """
        Train the LSTM.  Stores training history for later plotting.
        scaler_y is kept so predict_rescaled() can invert the normalization.
        """
        from tensorflow import keras

        self.scaler_y = scaler_y
        self.n_features = X_train.shape[-1]
        self.model = self._build()

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in *scaled* space."""
        return self.model.predict(X, verbose=0).flatten()

    def predict_rescaled(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in original NTU units."""
        if self.scaler_y is None:
            raise RuntimeError("No scaler_y stored.  Call train() with scaler_y.")
        y_scaled = self.predict(X).reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled).flatten()

    def score(self, X: np.ndarray, y_true_scaled: np.ndarray,
               threshold: float = config.TURBIDITY_HAZARD_NTU) -> dict:
        """
        Evaluate on scaled test data.
        Rescales both prediction and ground truth for interpretable metrics.
        """
        if self.scaler_y is None:
            raise RuntimeError("scaler_y not set; call train() first.")
        y_pred_ntu = self.predict_rescaled(X)
        y_true_ntu = self.scaler_y.inverse_transform(
            y_true_scaled.reshape(-1, 1)).flatten()
        return compute_metrics(y_true_ntu, y_pred_ntu, threshold)
