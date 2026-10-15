"""All covariates are available at close t; target is close t+1 return."""
import numpy as np
from src.config import FEATURES


def build_features(prices):
    frame = prices.copy()
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
