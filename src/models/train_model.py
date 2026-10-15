"""Expanding chronological folds with purged next-day targets and expert gating."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.config import FEATURES, SEED
from src.data.make_dataset import load_prices
from src.features.build_features import build_features
from src.models.hmm import GaussianHMM

REGIME_FEATURES = ["return_1", "volatility_20"]


def learner(seed=SEED, tree_settings=None):
    tree_settings=tree_settings or {"max_depth":2,"reg_lambda":10.}
    return XGBRegressor(n_estimators=100, max_depth=tree_settings["max_depth"], learning_rate=.03,
                        subsample=.8, colsample_bytree=1, reg_lambda=tree_settings["reg_lambda"],
                        random_state=seed, n_jobs=2, objective="reg:squarederror")


def fit_bundle(train, states=3, strength=1., tree_settings=None, hmm_iterations=60, fitted_hmm=None, context_prices=None):
    if not 0<=strength<=1:raise ValueError("Strength must be in [0,1]")
    if train.empty or not train.date.is_monotonic_increasing or train.date.duplicated().any():raise ValueError("Training dates must be unique and increasing")
    if not np.isfinite(train[FEATURES+["target"]].to_numpy()).all():raise ValueError("Training features/targets must be finite")
    if not np.isfinite(train.close).all() or (train.close<=0).any():raise ValueError("Training reference prices must be finite and positive")
    if "target_date" not in train or not (train.target_date>train.date).all():raise ValueError("Every training label must occur after its feature date")
    hmm = fitted_hmm or GaussianHMM(states=states,iterations=hmm_iterations).fit(train[REGIME_FEATURES].to_numpy())
    posterior = hmm.filter(train[REGIME_FEATURES].to_numpy())
    x, y = train[FEATURES].to_numpy(), train.target.to_numpy()
    baseline = learner(tree_settings=tree_settings).fit(x, y)
    experts = []
    for k in range(hmm.states):
        # Soft sample weighting changes feature relationships for each expert.
        experts.append(learner(SEED+k,tree_settings).fit(x, y, sample_weight=np.maximum(posterior[:, k], .001)))
    context=train[["date","close"]].copy() if context_prices is None else context_prices.loc[context_prices.date<=train.target_date.max(),["date","close"]].copy()
    if context_prices is None and "target_date" in train and not pd.isna(train.target_date.iloc[-1]):
        context=pd.concat([context,pd.DataFrame({"date":[train.target_date.iloc[-1]],"close":[float(train.close.iloc[-1]*np.exp(train.target.iloc[-1]))]})],ignore_index=True)
    context=context.drop_duplicates("date").tail(64).reset_index(drop=True)
    return {"hmm": hmm, "experts": experts, "baseline": baseline,"strength":strength,"tree_settings":tree_settings,"price_context":context,
            "last_posterior": posterior[-1], "last_date": str(train.date.iloc[-1]),
            "last_target_date":str(train.target_date.max()),"last_observed_date":str(context.date.max()),
            "features": FEATURES, "regime_features": REGIME_FEATURES}


def forecast(bundle, frame, previous=None):
    if frame.empty:return np.empty(0),np.empty((0,bundle["hmm"].states))
    if not frame.date.is_monotonic_increasing or frame.date.duplicated().any() or frame.date.iloc[0]<=pd.Timestamp(bundle["last_date"]):raise ValueError("Forecast rows must be unique, chronological and after the checkpoint cutoff")
    if not np.isfinite(frame[FEATURES].to_numpy()).all():raise ValueError("Forecast features must be finite")
    posterior = bundle["hmm"].filter(frame[REGIME_FEATURES].to_numpy(),
                                     bundle["last_posterior"] if previous is None else previous)
    experts = np.column_stack([model.predict(frame[FEATURES].to_numpy()) for model in bundle["experts"]])
    gated=(experts * posterior).sum(axis=1)
    strength=bundle.get("strength",1.)
    global_prediction=bundle["baseline"].predict(frame[FEATURES].to_numpy())
    return strength*gated+(1-strength)*global_prediction, posterior


def returns_summary(prediction, actual, cost_bps=2):
    prediction=np.asarray(prediction,dtype=float);actual=np.asarray(actual,dtype=float)
    if prediction.ndim!=1 or actual.shape!=prediction.shape or len(actual)<2 or not np.isfinite(prediction).all() or not np.isfinite(actual).all():raise ValueError("Need matching finite prediction/target vectors with >=2 rows")
    if not np.isfinite(cost_bps) or cost_bps<0:raise ValueError("Cost must be nonnegative and finite")
    active=prediction!=0
    position = np.sign(prediction)
    # Hypothetical reference-level diagnostic only; averages are not executable.
    turnover = np.abs(np.diff(position, prepend=0))
    net = position * actual - turnover * cost_bps / 10000
    equity = np.exp(np.cumsum(net))
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    return {"rmse": float(mean_squared_error(actual, prediction)**.5),
            "direction_accuracy": float(np.mean((prediction[active] > 0) == (actual[active] > 0))) if active.any() else None,
            "prediction_coverage":float(active.mean()),
            "annualized_sharpe": float(np.mean(net)/max(np.std(net, ddof=1), 1e-12)*np.sqrt(252)),
            "net_log_return": float(net.sum()), "max_drawdown": float(np.min(equity/peak-1)),
            "turnover": float(turnover.sum()), "cost_bps_per_unit_turnover": cost_bps}


def evaluate(prices, output=Path("models/run"), folds=3):
    if folds < 1:
        raise ValueError("folds must be positive")
    frame = build_features(prices).dropna(subset=["target", "target_date"]).reset_index(drop=True)
    if len(frame) < 300:
        raise ValueError("Need >=300 feature/target observations")
    start = len(frame)//2
    bounds = np.linspace(start, len(frame), folds+1, dtype=int)
    if np.min(np.diff(bounds))<2:raise ValueError("Each evaluation fold needs at least2observations")
    reports, prediction_frames = [], []
    for fold, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        test = frame.iloc[lo:hi]
        # Strict embargo: training target must precede the first evaluation feature time.
        train = frame.iloc[:lo]
        train = train.loc[train.target_date < test.date.iloc[0]]
        bundle = fit_bundle(train)
        # Advance through purged feature-only rows; their labels are never used.
        bridge = frame.loc[(frame.date > train.date.iloc[-1]) & (frame.date < test.date.iloc[0])]
        previous = bundle["last_posterior"]
        if len(bridge):
            previous = bundle["hmm"].filter(bridge[REGIME_FEATURES].to_numpy(), previous)[-1]
        predicted, posterior = forecast(bundle, test, previous)
        base = bundle["baseline"].predict(test[FEATURES].to_numpy())
        actual = test.target.to_numpy()
        reports.append({"fold": fold, "train_rows": len(train), "test_rows": len(test),
                        "train_last_target": str(train.target_date.max()), "test_first": str(test.date.min()),
                        "test_last": str(test.date.max()), "regime": returns_summary(predicted, actual),
                        "global_xgboost": returns_summary(base, actual),
                        "zero_forecast_rmse": float(np.sqrt(np.mean(actual**2))),
                        "buy_hold_log_return": float(actual.sum()),
                        "hmm_iterations": len(bundle["hmm"].log_likelihoods)})
        result = pd.DataFrame({"date": test.date, "target": actual, "regime_prediction": predicted,
                               "baseline_prediction": base, "state": posterior.argmax(axis=1)})
        prediction_frames.append(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final_bundle = fit_bundle(frame)
    joblib.dump(final_bundle, output / "bundle.joblib")
    combined = pd.concat(prediction_frames, ignore_index=True)
    combined.to_csv(output / "predictions.csv", index=False)
    summary = {"seed": SEED, "data_rows": len(prices), "feature_rows": len(frame),
               "features": FEATURES, "folds": reports,
               "overall_regime": returns_summary(combined.regime_prediction.to_numpy(), combined.target.to_numpy()),
               "overall_baseline": returns_summary(combined.baseline_prediction.to_numpy(), combined.target.to_numpy()),
               "evaluation": "Expanding chronological out-of-sample; HMM and XGBoost refit per fold; causal filter",
               "execution_assumption": "Daily reference-price friction diagnostic; reference rate is not executable; delay/slippage sensitivity required",
               "limitations": ["Bank of Canada daily reference rates are not executable quotes; financing is omitted",
                               "Regimes are latent state indices, not externally validated economic labels",
                               "No profitability claim; compare global and zero-forecast baselines"]}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/prices.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/run"))
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    summary = evaluate(load_prices(args.data), args.output, args.folds)
    summary["source_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
