"""Replay authenticated saved fold models and a fixed causal abstention policy."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from src.models.interaction_study import ASSETS, FOLD_ROWS, YEARS, sha, split_records
from src.models.regime_interactions import BASE_FEATURES, number, predict
from src.models.volatility_gate import check_freeze, load_inputs, gate_predictions

PARENT_REPORT_SHA256 = (
    "50a33eb2711a1d016225cc14eece5553edce59c4fc527aecc5bb19a194be4bf8"
)
THRESHOLD = 1.2
MULTIPLIER = 2


def predict_records(model, records):
    """The retained estimator receives only fixed features, then causal routing."""
    raw = predict(model, records, False)
    return gate_predictions(
        raw, [number(r["volatility_ratio"]) for r in records], THRESHOLD, MULTIPLIER
    )


def checkpoint_contract(report, checkpoints):
    expected = {
        f"linear_10000-{year}-{asset}.joblib" for year in YEARS for asset in ASSETS
    }
    hashes = report.get("checkpoint_sha256", {})
    if not isinstance(hashes, dict) or not expected <= hashes.keys():
        raise ValueError("All twelve authenticated retained fold checkpoints required")
    for name in sorted(expected):
        if sha(Path(checkpoints) / name) != hashes[name]:
            raise ValueError("Retained learned checkpoint bytes changed")
    return {name: hashes[name] for name in sorted(expected)}


def validate_estimator(model):
    # Pickle loading is restricted to hash-authenticated artifacts from our own run.
    steps = getattr(model, "named_steps", {})
    if set(steps) != {"standardscaler", "ridge"}:
        raise ValueError("Retained scaler/ridge pipeline required")
    ridge = steps["ridge"]
    if (
        ridge.alpha != 10000
        or ridge.fit_intercept is not False
        or model.n_features_in_ != len(BASE_FEATURES)
    ):
        raise ValueError("Retained ridge hyperparameters or feature width changed")


def run(snapshot, predictions, protocol, freeze, parent_report, checkpoints, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh parity output required")
    frozen = check_freeze(freeze, protocol, predictions)
    plan, data, ratios = load_inputs(snapshot, predictions, protocol)
    # The report itself must be the actual audited cloud report, not a caller supplied checkpoint registry.
    expected_report = Path(freeze).parent / "saved-model-parity-plan.json"
    contract = json.loads(expected_report.read_text(encoding="utf8"))
    if (
        contract.get("schema") != "saved_gate_parity_v1"
        or contract.get("parent_report_sha256") != PARENT_REPORT_SHA256
        or sha(parent_report) != PARENT_REPORT_SHA256
    ):
        raise ValueError("Authenticated parent report identity required")
    if (
        contract.get("freeze_sha256") != sha(freeze)
        or contract.get("threshold") != THRESHOLD
        or contract.get("multiplier") != MULTIPLIER
    ):
        raise ValueError("Frozen selected inference policy changed")
    report = json.loads(Path(parent_report).read_text(encoding="utf8"))
    hashes = checkpoint_contract(report, checkpoints)
    # Imports and unpickling occur only after every source/input/checkpoint guard.
    import joblib
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import FEATURES, build_panel_features

    if FEATURES != BASE_FEATURES:
        raise ValueError("Production feature order changed")
    started = time.monotonic()
    frames = build_panel_features(parse_panel(Path(snapshot).read_bytes()))
    records = {}
    for asset, frame in frames.items():
        frame = frame.copy()
        for key in ("date", "execution_date", "target_date"):
            frame[key] = frame[key].map(
                lambda value: None if pd.isna(value) else value.strftime("%Y-%m-%d")
            )
        records[asset] = frame.to_dict("records")
    original = {(r["date"], r["asset"]): r for r in data["linear_10000"]}
    seen = set()
    raw_max = gated_max = ratio_max = 0.0
    folds = []
    with threadpool_limits(limits=2):
        for year, expected_count in zip(YEARS, FOLD_ROWS):
            count = 0
            for asset in ASSETS:
                _, rows = split_records(records[asset], year)
                model = joblib.load(
                    Path(checkpoints) / f"linear_10000-{year}-{asset}.joblib"
                )
                validate_estimator(model)
                raw = predict(model, rows, False)
                gated = predict_records(model, rows)
                for row, before, after in zip(rows, raw, gated):
                    key = (row["date"], asset)
                    if key in seen or key not in original:
                        raise ValueError("Duplicate or unexpected saved inference row")
                    seen.add(key)
                    reference = original[key]
                    if abs(row["target"] - reference["target"]) > 1e-14:
                        raise ValueError("Reconstructed target changed")
                    ratio = ratios[key]
                    raw_max = max(raw_max, abs(before - reference["prediction"]))
                    ratio_max = max(ratio_max, abs(row["volatility_ratio"] - ratio))
                    expected = (
                        MULTIPLIER * reference["prediction"]
                        if ratio > THRESHOLD
                        else 0.0
                    )
                    gated_max = max(gated_max, abs(after - expected))
                    count += 1
            if count != expected_count:
                raise ValueError("Complete frozen fold inference required")
            folds.append({"year": year, "rows": count})
    if len(seen) != 2961 or raw_max > 1e-14 or gated_max > 1e-14 or ratio_max > 1e-10:
        raise ValueError("Saved learned model/feature/gating parity failed")
    result = {
        "schema": "saved_gate_parity_result_v1",
        "rows": len(seen),
        "folds": folds,
        "raw_prediction_max_abs_difference": raw_max,
        "gated_prediction_max_abs_difference": gated_max,
        "causal_ratio_max_abs_difference": ratio_max,
        "checkpoint_sha256": hashes,
        "source_sha256": frozen["source_sha256"],
        "parent_report_sha256": sha(parent_report),
        "freeze_sha256": sha(freeze),
        "parity_plan_sha256": sha(expected_report),
        "inference_source_sha256": sha(__file__),
        "threshold": THRESHOLD,
        "multiplier": MULTIPLIER,
        "weights_refitted": False,
        "accessed_2025_or_2026": False,
        "production_changed": False,
        "seconds": time.monotonic() - started,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                    "threadpoolctl",
                ]
            },
        },
    }
    output.mkdir(parents=True)
    (output / "parity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "snapshot",
        "predictions",
        "protocol",
        "freeze",
        "parent-report",
        "checkpoints",
        "output",
    ]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.snapshot,
            args.predictions,
            args.protocol,
            args.freeze,
            args.parent_report,
            args.checkpoints,
            args.output,
        )
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, "Saved inference refused: " + str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
