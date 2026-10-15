"""Advanced asset-specific HMM-gated models, preserving legacy checkpoints."""
import numpy as np
from xgboost import XGBRegressor
from src.features.multi_asset import FEATURES
from src.models.hmm import GaussianHMM

REGIME_FEATURES = ["return_1", "volatility_20"]


def fit_asset(frame, tree, iterations=20, fitted_hmm=None):
    if frame.empty or not frame.date.is_monotonic_increasing or frame.date.duplicated().any():
        raise ValueError("Asset training must be chronological and nonempty")
    if not np.isfinite(frame[FEATURES + ["target"]].to_numpy(float)).all():
        raise ValueError("Training requires finite causal features and matured labels")
    if not (frame.target_date > frame.execution_date).all() or not (frame.execution_date > frame.date).all():
        raise ValueError("Delayed label must follow execution, which must follow decision")
    hmm = fitted_hmm or GaussianHMM(states=2, iterations=iterations).fit(frame[REGIME_FEATURES].to_numpy())
    posterior = hmm.filter(frame[REGIME_FEATURES].to_numpy())
    # Forward regime forecast at execution time; never heldout smoothing.
    execution_probabilities = posterior @ hmm.transition
    x, y = frame[FEATURES].to_numpy(), frame.target.to_numpy()
    def learner(seed):
        return XGBRegressor(n_estimators=150, max_depth=tree["max_depth"], reg_lambda=tree["reg_lambda"],
                            learning_rate=.03, subsample=.8, colsample_bytree=.9,
                            random_state=seed, n_jobs=2, objective="reg:squarederror")
    global_model = learner(42).fit(x, y)
    experts = [learner(42 + k).fit(x, y, sample_weight=np.maximum(execution_probabilities[:, k], .001)) for k in range(2)]
    return {"feature_version": "multi_asset_v1", "features": FEATURES, "hmm": hmm,
            "global": global_model, "experts": experts, "last_posterior": posterior[-1],
            "last_feature_date": str(frame.date.iloc[-1]), "last_target_date": str(frame.target_date.max())}


def predict_asset(bundle, rows, previous=None):
    if bundle.get("feature_version") != "multi_asset_v1" or list(bundle["features"]) != FEATURES:
        raise ValueError("Unsupported advanced feature contract")
    if rows.empty:
        return np.empty(0), np.empty(0), np.empty((0, 2)), np.empty(0)
    if not rows.date.is_monotonic_increasing or rows.date.duplicated().any():
        raise ValueError("Forecast rows must have unique increasing dates")
    import pandas as pd
    if rows.date.min() <= pd.Timestamp(bundle["last_feature_date"]):
        raise ValueError("Forecast cannot replay the fitted feature history")
    posterior = bundle["hmm"].filter(rows[REGIME_FEATURES].to_numpy(), bundle["last_posterior"] if previous is None else previous)
    future = posterior @ bundle["hmm"].transition
    x = rows[FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Forecast features must be finite")
    global_prediction = bundle["global"].predict(x)
    gated = (np.column_stack([model.predict(x) for model in bundle["experts"]]) * future).sum(axis=1)
    state_vol = np.maximum(0., bundle["hmm"].means[:, 1] * bundle["hmm"].scale[1] + bundle["hmm"].center[1])
    return global_prediction, gated, future, future @ state_vol
