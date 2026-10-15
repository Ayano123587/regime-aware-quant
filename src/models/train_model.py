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


def learner(seed=SEED):
    return XGBRegressor(n_estimators=100, max_depth=2, learning_rate=.03,
                        subsample=.8, colsample_bytree=1, reg_lambda=10,
                        random_state=seed, n_jobs=2, objective="reg:squarederror")


def fit_bundle(train):
    hmm = GaussianHMM().fit(train[REGIME_FEATURES].to_numpy())
    posterior = hmm.filter(train[REGIME_FEATURES].to_numpy())
    x, y = train[FEATURES].to_numpy(), train.target.to_numpy()
    baseline = learner().fit(x, y)
    experts = []
    for k in range(hmm.states):
        # Soft sample weighting changes feature relationships for each expert.
        experts.append(learner(SEED+k).fit(x, y, sample_weight=np.maximum(posterior[:, k], .001)))
    return {"hmm": hmm, "experts": experts, "baseline": baseline,
            "last_posterior": posterior[-1], "last_date": str(train.date.iloc[-1]),
            "features": FEATURES, "regime_features": REGIME_FEATURES}


def forecast(bundle, frame, previous=None):
    posterior = bundle["hmm"].filter(frame[REGIME_FEATURES].to_numpy(),
                                     bundle["last_posterior"] if previous is None else previous)
    experts = np.column_stack([model.predict(frame[FEATURES].to_numpy()) for model in bundle["experts"]])
    return (experts * posterior).sum(axis=1), posterior


def returns_summary(prediction, actual, cost_bps=2):
    position = np.sign(prediction)
    # Position chosen at close t earns t->t+1 return. Initial entry has turnover.
    turnover = np.abs(np.diff(position, prepend=0))
    net = position * actual - turnover * cost_bps / 10000
    equity = np.exp(np.cumsum(net))
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    return {"rmse": float(mean_squared_error(actual, prediction)**.5),
            "direction_accuracy": float(np.mean((prediction > 0) == (actual > 0))),
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
