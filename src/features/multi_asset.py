"""Causal features; delayed label t+1 -> t+2; peer features lag one observation."""
import numpy as np
import pandas as pd
from src.data.multi_asset import ASSETS, validate_panel

OWN_FEATURES = ["return_1", "momentum_3", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
                "volatility_5", "volatility_20", "volatility_60", "ewma_volatility", "volatility_ratio",
                "downside_20", "drawdown_60", "zscore_20", "rsi_14", "autocorrelation_20", "skew_20", "gap_days"]
PEER_FEATURES = ["peer_return_lag1", "peer_momentum_lag1", "peer_volatility_lag1", "relative_momentum"]
FEATURES = OWN_FEATURES + PEER_FEATURES


def build_panel_features(panel, assets=ASSETS):
    panel = validate_panel(panel, assets)
    log = np.log(panel[list(assets)])
    returns = log.diff()
    result = {}
    for asset in assets:
        r, close = returns[asset], panel[asset]
        frame = pd.DataFrame({"date": panel.date, "close": close, "return_1": r})
        for period in [3, 5, 10, 20, 60]:
            frame[f"momentum_{period}"] = log[asset].diff(period)
        for period in [5, 20, 60]:
            frame[f"volatility_{period}"] = r.rolling(period).std()
        frame["ewma_volatility"] = r.ewm(span=20, adjust=False, min_periods=20).std()
        frame["volatility_ratio"] = frame.volatility_5 / frame.volatility_20.clip(lower=1e-8)
        frame["downside_20"] = np.sqrt(r.clip(upper=0).pow(2).rolling(20).mean())
        frame["drawdown_60"] = close / close.rolling(60).max() - 1
        frame["zscore_20"] = (log[asset] - log[asset].rolling(20).mean()) / log[asset].rolling(20).std().clip(lower=1e-8)
        up, down = r.clip(lower=0).rolling(14).mean(), -r.clip(upper=0).rolling(14).mean()
        frame["rsi_14"] = up / (up + down).clip(lower=1e-8)
        frame["autocorrelation_20"] = r.rolling(20).corr(r.shift(1)).fillna(0)
        frame["skew_20"] = r.rolling(20).skew().fillna(0)
        frame["gap_days"] = panel.date.diff().dt.total_seconds() / 86400
        peers = [other for other in assets if other != asset]
        frame["peer_return_lag1"] = returns[peers].mean(axis=1).shift(1)
        frame["peer_momentum_lag1"] = log[peers].diff(5).mean(axis=1).shift(1)
        frame["peer_volatility_lag1"] = returns[peers].rolling(20).std().mean(axis=1).shift(1)
        frame["relative_momentum"] = frame.momentum_20 - log[peers].diff(20).mean(axis=1).shift(1)
        frame["execution_date"] = panel.date.shift(-1)
        frame["target_date"] = panel.date.shift(-2)
        frame["target"] = r.shift(-2)
        result[asset] = frame.dropna(subset=FEATURES).reset_index(drop=True)
    return result
