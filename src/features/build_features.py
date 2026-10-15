"""Covariates use published reference observation t; target is the next observation."""
import numpy as np
from src.config import FEATURES


def build_features(prices):
    frame = prices.copy()
    if not {"date", "close"} <= set(frame):raise ValueError("Need date and close")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:raise ValueError("Prices must have unique increasing dates")
    if not np.isfinite(frame.close.to_numpy(dtype=float)).all() or (frame.close<=0).any():raise ValueError("Prices must be finite positive values")
    close = frame.close
    frame["return_1"] = np.log(close).diff()
    frame["momentum_5"] = np.log(close).diff(5)
    frame["momentum_20"] = np.log(close).diff(20)
    frame["volatility_20"] = frame.return_1.rolling(20).std()
    frame["drawdown_60"] = close / close.rolling(60).max() - 1
    frame["target"] = frame.return_1.shift(-1)
    frame["target_date"] = frame.date.shift(-1)
    # Keep last feature row for live inference, even though its target is unknown.
    return frame.dropna(subset=FEATURES).reset_index(drop=True)
