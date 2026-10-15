"""Prespecified multi-asset development, one new holdout, and research inference."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
from src.data.multi_asset import ASSETS, fetch_snapshot, sha, validate_panel
from src.features.multi_asset import FEATURES, build_panel_features
from src.models.multi_asset import REGIME_FEATURES, fit_asset, predict_asset
from src.models.hmm import GaussianHMM
from src.models.portfolio import RiskPolicy, simulate
from src.models.drift import DelayedResidualQueue

DEFAULT_PLAN = Path("docs/multi-asset-plan.json")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def plan_read(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["assets"] != ASSETS or plan["execution_delay_observations"] != 1 or plan["states"] != 2:
        raise ValueError("This version implements only the frozen three-asset delayed contract")
    fixed = {"trees": 150, "covariance_window": 60, "covariance_diagonal_shrinkage": .25,
             "maximum_calendar_gap_days": 7, "seed": 42}
    if any(plan[key] != value for key, value in fixed.items()):
        raise ValueError("Plan changes an unsupported fixed implementation parameter")
    return plan


def feature_split(frame, start, end):
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    validation = frame.loc[(frame.date >= lo) & (frame.target_date <= hi)].copy()
    if validation.empty:
        raise ValueError("Empty validation cohort")
    train = frame.loc[frame.target_date < validation.date.min()].copy()
    bridge = frame.loc[(frame.date > train.date.max()) & (frame.date < validation.date.min())]
    if train.target_date.max() >= validation.date.min():
        raise ValueError("Two-observation label purge failed")
    return train, validation, bridge


def predict_with_bridge(bundle, bridge, cohort):
    rows = pd.concat([bridge, cohort], ignore_index=True)
    result = predict_asset(bundle, rows)
    return tuple(value[len(bridge):] for value in result)


def prediction_rows(asset, cohort, global_pred, gated, probabilities, state_vol, strength):
    frame = cohort[["date", "execution_date", "target_date", "target", "volatility_20"]].copy()
    frame["asset"] = asset
    frame["prediction"] = strength * gated + (1 - strength) * global_pred
    frame["global_prediction"] = global_pred
    frame["regime_probability_0"] = probabilities[:, 0]
    frame["regime_probability_1"] = probabilities[:, 1]
    frame["predicted_state_volatility"] = state_vol
    return frame


def signal_panel(predictions):
    frame = predictions.pivot(index="date", columns="asset", values="prediction").reindex(columns=ASSETS)
    if frame.isna().any().any():
        raise ValueError("Forecast cohorts do not align across all assets")
    return frame.reset_index()


def forecast_scores(predictions):
    result = {}
    for asset, frame in predictions.groupby("asset", sort=False):
        actual = frame.target.to_numpy()
        probability = frame[["regime_probability_0", "regime_probability_1"]].to_numpy()
        result[asset] = {"rows": len(frame), "rmse": float(np.sqrt(np.mean((actual - frame.prediction.to_numpy()) ** 2))),
                         "global_rmse": float(np.sqrt(np.mean((actual - frame.global_prediction.to_numpy()) ** 2))),
                         "zero_rmse": float(np.sqrt(np.mean(actual ** 2))),
                         "mean_execution_state_probability": probability.mean(axis=0).tolist(),
                         "mean_regime_entropy": float(-(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1).mean()),
                         "mean_predicted_state_daily_volatility": float(frame.predicted_state_volatility.mean())}
    result["pooled"] = {"rows": len(predictions), "rmse": float(np.sqrt(np.mean((predictions.target - predictions.prediction) ** 2))),
                        "global_rmse": float(np.sqrt(np.mean((predictions.target - predictions.global_prediction) ** 2))),
                        "zero_rmse": float(np.sqrt(np.mean(predictions.target ** 2)))}
    return result


def risk_from_spec(spec, plan):
    return RiskPolicy(**spec, gross_cap=plan["gross_cap"], per_asset_cap=plan["per_asset_cap"])


def mature_diagnostics(panel, predictions):
    queue = DelayedResidualQueue(ASSETS)
    forecasts = {(row.date, row.asset): row for row in predictions.itertuples()}
    returns = np.log(panel[ASSETS]).diff()
    alerts = defaultdict(int)
    for j, date in enumerate(panel.date):
        realized = {} if j == 0 else returns.iloc[j].to_dict()
        for observation in queue.arrive(j, realized):
            alerts[observation["asset"]] += int(observation["alert"])
        for asset in ASSETS:
            if (date, asset) in forecasts:
                row = forecasts[(date, asset)]
                queue.enqueue(j, asset, row.prediction, row.volatility_20)
    return {"labels_arrive_after_observations": 2, "adaptive_trades_or_refitting": False,
            "assets": {asset: {"matured_labels": queue.monitors[asset].count,
                               "alerts": alerts[asset], "mean_standardized_residual": queue.monitors[asset].mean} for asset in ASSETS},
            "pending_unmatured": len(queue.pending), "interpretation": "Residual change alarms are diagnostic, not verified market regime changes"}


def development(output="models/multi-asset-v1", plan_path=DEFAULT_PLAN):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("Advanced selection is already frozen; use a fresh reproduction directory")
    plan = plan_read(plan_path)
    started = time.perf_counter()
    panel, provenance = fetch_snapshot("data/external/multi-asset-v1/development", plan["development_start"], plan["development_end"])
    frames = build_panel_features(panel)
    candidates, specs, fold_details = defaultdict(list), {}, []
    local = Path("data/interim/multi-asset-v1")
    local.mkdir(parents=True, exist_ok=True)
    for fold, (start, end) in enumerate(plan["validation_windows"]):
        details = {"fold": fold, "start": start, "end": end, "assets": {}}
        for asset in ASSETS:
            train, validation, bridge = feature_split(frames[asset], start, end)
            details["assets"][asset] = {"training_rows": len(train), "validation_rows": len(validation),
                                        "last_training_label": str(train.target_date.max()), "first_validation_decision": str(validation.date.min()), "bridge_rows": len(bridge)}
            hmm = GaussianHMM(states=2, iterations=plan["hmm_iterations"]).fit(train[REGIME_FEATURES].to_numpy())
            for tree in plan["tree_settings"]:
                bundle = fit_asset(train, tree, fitted_hmm=hmm)
                global_pred, gated, probability, state_vol = predict_with_bridge(bundle, bridge, validation)
                for strength in plan["gating_strengths"]:
                    spec = {"tree": tree, "strength": strength}
                    key = json.dumps(spec, sort_keys=True)
                    specs[key] = spec
                    rows = prediction_rows(asset, validation, global_pred, gated, probability, state_vol, strength)
                    rows["fold"] = fold
                    candidates[key].append(rows)
            print(json.dumps({"fold": fold, "asset": asset, "training_rows": len(train), "validation_rows": len(validation), "elapsed_seconds": time.perf_counter() - started}), flush=True)
            with (output / "development-progress.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"fold": fold, "asset": asset, "training_rows": len(train), "validation_rows": len(validation), "elapsed_seconds": time.perf_counter() - started}) + "\n")
            if time.perf_counter() - started > plan["cpu_budget_minutes"] * 60:
                raise RuntimeError("Prespecified first-pass budget exhausted; preserve completed work")
        fold_details.append(details)
    runs = [{"spec": specs[key], "rmse": forecast_scores(pd.concat(parts))["pooled"]["rmse"]} for key, parts in candidates.items()]
    winner = min(runs, key=lambda row: (row["rmse"], row["spec"]["strength"], row["spec"]["tree"]["max_depth"]))
    best_key = json.dumps(winner["spec"], sort_keys=True)
    predictions = pd.concat(candidates[best_key], ignore_index=True)
    predictions.to_csv(local / "development-predictions.csv", index=False, lineterminator="\n")
    risk_runs = []
    for spec in plan["risk_policies"]:
        policy = risk_from_spec(spec, plan)
        folds = []
        for fold in range(len(plan["validation_windows"])):
            rows = predictions.loc[predictions.fold == fold]
            history = panel.loc[panel.date <= rows.target_date.max()]
            _, report = simulate(history, signal_panel(rows), policy, start_date=rows.date.min())
            report["utility"] = report["annualized_arithmetic_return"] - 3 * report["annualized_volatility"] ** 2
            folds.append(report)
        risk_runs.append({"policy": asdict(policy), "mean_validation_utility": float(np.mean([r["utility"] for r in folds])), "folds": folds})
    risk_winner = max(risk_runs, key=lambda row: (row["mean_validation_utility"], -row["policy"]["annual_volatility"]))
    selection = {"plan_sha256": sha(plan_path), "historical_provenance": provenance, "features": FEATURES,
                 "model_runs": runs, "selected_model": winner, "selected_risk_policy": risk_winner["policy"],
                 "risk_runs": risk_runs, "folds": fold_details, "validation_forecasts": forecast_scores(predictions),
                 "new2026values_used": False, "runtime_seconds": time.perf_counter() - started,
                 "disclosure": "Retrospective development only; all model/risk policies fixed before new three-asset 2026 retrieval; no profitable strategy is guaranteed"}
    write(output / "selection.json", selection)
    write("docs/multi-asset-development-provenance.json", provenance)
    print(json.dumps({"selected_model": winner, "selected_risk_policy": risk_winner["policy"], "runtime_seconds": selection["runtime_seconds"]}, indent=2), flush=True)
    return selection


def final_test(output="models/multi-asset-v1", plan_path=DEFAULT_PLAN):
    output = Path(output)
    if (output / "evaluation.json").exists():
        raise ValueError("Final multi-asset holdout has already been released; never tune on it")
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    plan = plan_read(plan_path)
    if sha(plan_path) != selection["plan_sha256"] or selection["new2026values_used"]:
        raise ValueError("Frozen development plan/selection state is invalid")
    prior, provenance = fetch_snapshot("data/external/multi-asset-v1/development", plan["development_start"], plan["development_end"])
    if provenance["sha256"] != selection["historical_provenance"]["sha256"]:
        raise ValueError("Historical snapshot changed after selection")
    # The previously unobserved EUR/GBP/AUD 2026 snapshot is requested only here.
    new, new_provenance = fetch_snapshot("data/external/multi-asset-v1/holdout", plan["test_start"], plan["test_end"])
    panel = validate_panel(pd.concat([prior, new], ignore_index=True))
    historical_frames = build_panel_features(prior)
    frames = build_panel_features(panel)
    spec = selection["selected_model"]["spec"]
    bundles, parts = {}, []
    for asset in ASSETS:
        train = historical_frames[asset].dropna(subset=["target", "target_date"])
        test = frames[asset].loc[(frames[asset].date >= new.date.min()) & frames[asset].target.notna()].copy()
        if train.target_date.max() >= test.date.min():
            raise ValueError("Final delayed-label purge failed")
        bundle = fit_asset(train, spec["tree"], iterations=plan["hmm_iterations"])
        bridge = frames[asset].loc[(frames[asset].date > train.date.max()) & (frames[asset].date < test.date.min())]
        parts.append(prediction_rows(asset, test, *predict_with_bridge(bundle, bridge, test), spec["strength"]))
        bundles[asset] = bundle
    predictions = pd.concat(parts, ignore_index=True)
    policy = RiskPolicy(**selection["selected_risk_policy"])
    sensitivity = []
    ledger = None
    for cost in plan["cost_sensitivity_bps"]:
        current, report = simulate(panel, signal_panel(predictions), policy, cost_bps=cost, start_date=new.date.min())
        sensitivity.append(report)
        if cost == 2:
            ledger = current
    local = Path("data/interim/multi-asset-v1")
    local.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(local / "final-predictions.csv", index=False, lineterminator="\n")
    ledger.to_csv(local / "final-reference-ledger.csv", index=False, lineterminator="\n")
    checkpoint = {"version": "multi_asset_v1", "assets": ASSETS, "bundles": bundles, "strength": spec["strength"],
                  "risk_policy": asdict(policy), "price_context": prior.copy(),
                  "selection_sha256": sha(output / "selection.json"), "feature_contract": FEATURES}
    joblib.dump(checkpoint, output / "bundle.joblib")
    report = {"study": plan["study"], "assets": ASSETS, "plan_sha256": sha(plan_path),
              "selection_sha256": checkpoint["selection_sha256"], "selected_model": spec,
              "selected_risk_policy": asdict(policy), "new_provenance": new_provenance,
              "test_decisions_first": str(predictions.date.min()), "test_decisions_last": str(predictions.date.max()),
              "last_matured_label": str(predictions.target_date.max()), "forecast_scores": forecast_scores(predictions),
              "checkpoint_context_rows": len(prior), "recursive_feature_context": "Full historical feature-only price context preserves exact EWM initialization",
              "cost_turnover_sensitivity": sensitivity, "delayed_residual_monitor": mature_diagnostics(panel, predictions),
              "local_prediction_sha256": sha(local / "final-predictions.csv"), "local_ledger_sha256": sha(local / "final-reference-ledger.csv"),
              "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]}},
              "limitations": ["Indicative averages are non-executable; no brokerage actions or investment recommendation",
                              "One-observation delayed reference accounting is still not actual after-publication execution",
                              "No historical vintages, financing, executable spreads, intraday limits or market impact",
                              "Peer features lag one common observation; synchronous release/calendar assumptions remain",
                              "All pairs share CAD exposure; one partial year, one seed, no proven economic regime labels",
                              "USD/CAD2026 previously observed and explicitly excluded from new-test claims",
                              "Residual alarms do not prove drift and never adapt the heldout model or positions"]}
    write(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def forecast_panel(checkpoint, supplied):
    if checkpoint.get("version") != "multi_asset_v1" or checkpoint["assets"] != ASSETS or set(checkpoint["bundles"]) != set(ASSETS):
        raise ValueError("Unsupported advanced checkpoint; legacy checkpoints use regime-predict")
    supplied = validate_panel(supplied)
    context = checkpoint["price_context"]
    if supplied.date.min() < context.date.min():
        raise ValueError("Earlier history changes recursive feature initialization; explicit refit required")
    overlaps = supplied.loc[supplied.date.isin(context.date)].merge(context, on="date", suffixes=("_new", "_old"))
    for asset in ASSETS:
        if not np.allclose(overlaps[asset + "_new"], overlaps[asset + "_old"], rtol=0, atol=1e-12):
            raise ValueError("Input revises saved price context; explicit refit required")
    history = validate_panel(pd.concat([context, supplied]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True))
    frames = build_panel_features(history)
    parts = []
    for asset, bundle in checkpoint["bundles"].items():
        rows = frames[asset].loc[frames[asset].date > pd.Timestamp(bundle["last_feature_date"])]
        if rows.empty:
            continue
        if (rows.date.iloc[0] - pd.Timestamp(bundle["last_feature_date"])).days > 7:
            raise ValueError("Missing bridge observations; supply complete post-cutoff history")
        output = prediction_rows(asset, rows, *predict_asset(bundle, rows), checkpoint["strength"])
        parts.append(output.loc[output.date.isin(supplied.date)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["search", "test", "report", "forecast"])
    parser.add_argument("--output", default="models/multi-asset-v1")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--panel", help="Strict date,FXEURCAD,FXGBPCAD,FXAUDCAD CSV for forecast")
    parser.add_argument("--predictions", default="data/interim/multi-asset-v1/new-forecasts.csv")
    args = parser.parse_args()
    if args.action == "search":
        development(args.output, args.plan)
    elif args.action == "test":
        final_test(args.output, args.plan)
    elif args.action == "report":
        print((Path(args.output) / "evaluation.json").read_text(encoding="utf-8"))
    else:
        if not args.panel:
            parser.error("forecast requires --panel")
        checkpoint = joblib.load(Path(args.output) / "bundle.joblib")
        forecasts = forecast_panel(checkpoint, pd.read_csv(args.panel))
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(path, index=False, lineterminator="\n")
        print(json.dumps({"forecast_rows": len(forecasts), "path": str(path), "reference_only": True}))


if __name__ == "__main__":
    main()
