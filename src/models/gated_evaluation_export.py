"""Export all frozen retrospective forecasts for independent numerical auditing.

This is a deterministic replay of an authenticated evaluation, with no fitting,
selection, changed interval policy or additional evaluation cohort.
"""

import csv
import importlib.metadata
import json
import math
from pathlib import Path
import platform

from src.models import gated_evaluation as evaluation

RESULT_SHA256 = "4d51d5ad63d69eb1cb06b01669b85b7bcddcf360e3bb37ebf9d63300621316e6"
EXECUTION_SHA256 = "f7d3f34edd4910b950cb15536ef653ac287903f5706c315ca514bdf1f41f326b"
COLUMNS = [
    "asset",
    "date",
    "execution_date",
    "target_date",
    "target",
    "selected",
    "ungated_same_mean",
    "zero",
    "selected_lower",
    "selected_upper",
    "true_legacy",
    "true_legacy_lower",
    "true_legacy_upper",
]


def authenticate_result(result_path, execution_path, release_sha256):
    if (
        evaluation.sha(result_path) != RESULT_SHA256
        or evaluation.sha(execution_path) != EXECUTION_SHA256
    ):
        raise ValueError("Exact completed evaluation and execution receipts required")
    result, execution = evaluation.read(result_path), evaluation.read(execution_path)
    if (
        execution.get("result_sha256") != RESULT_SHA256
        or execution.get("release_sha256") != release_sha256
        or result.get("release_sha256") != release_sha256
        or result.get("rows") != 582
        or result.get("weights_refitted") is not False
        or result.get("policy_reselected") is not False
        or execution.get("weights_refitted") is not False
        or execution.get("policy_reselected") is not False
        or not result.get("true_legacy_comparator")
    ):
        raise ValueError("Completed unchanged full-cohort evaluation required")
    return result, execution


def check_runtime(expected):
    actual = {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "threadpoolctl",
            )
        },
    }
    if actual != expected:
        raise ValueError(
            "Exact successful compatibility runtime required before model load"
        )
    return actual


def require_metrics(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Completed aggregate schema differs")
        for name in expected:
            require_metrics(actual[name], expected[name])
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError("Completed aggregate row counts differ")
    elif not math.isclose(
        evaluation.number(actual),
        evaluation.number(expected),
        rel_tol=1e-12,
        abs_tol=1e-14,
    ):
        raise ValueError("Replay metrics differ from completed evaluation")


def verify_rows(rows, ids, original):
    """Recompute every comparator and proper interval score before publishing CSV."""
    if (
        len(rows) != 582
        or [
            (r["asset"], r["date"], r["execution_date"], r["target_date"]) for r in rows
        ]
        != ids
        or len(set(ids)) != 582
    ):
        raise ValueError("All 582 ordered unique decisions required")
    if any(
        set(row) != set(COLUMNS) or evaluation.number(row["zero"]) != 0 for row in rows
    ):
        raise ValueError("Complete immutable prediction columns required")
    target = [evaluation.number(r["target"]) for r in rows]
    metrics = evaluation.evaluate_rows(
        ids,
        target,
        [r["selected"] for r in rows],
        [r["ungated_same_mean"] for r in rows],
        [r["selected_lower"] for r in rows],
        [r["selected_upper"] for r in rows],
        False,
    )
    require_metrics(metrics, original["metrics"])
    legacy = original["true_legacy_comparator"]
    for asset in [None] + evaluation.ASSETS:
        selected = [r for r in rows if asset is None or r["asset"] == asset]
        y = [r["target"] for r in selected]
        actual = {
            "regression": evaluation.regression(
                y, [r["true_legacy"] for r in selected]
            ),
            "intervals": evaluation.intervals(
                y,
                [r["true_legacy_lower"] for r in selected],
                [r["true_legacy_upper"] for r in selected],
            ),
        }
        expected = legacy if asset is None else legacy["per_asset"][asset]
        for key in actual:
            require_metrics(actual[key], expected[key])
    return metrics


def check_postflight(originals, production, policy, bundle, states):
    if (
        any(evaluation.sha(path) != digest for path, digest in originals.items())
        or production.model_state_fingerprints(bundle["models"]) != states
    ):
        raise ValueError("Immutable input or model changed during replay")
    production.check_policy(policy)


def run(
    snapshot,
    provenance,
    calibrated,
    policy,
    release,
    release_sha256,
    frozen_inputs,
    frozen_inputs_sha256,
    original_result,
    original_execution,
    output,
    *,
    legacy,
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh immutable forecast export required")
    original, execution = authenticate_result(
        original_result, original_execution, release_sha256
    )
    if evaluation.sha(frozen_inputs) != frozen_inputs_sha256:
        raise ValueError("Frozen inputs changed")
    inputs = evaluation.read(frozen_inputs)
    runtime = check_runtime(inputs["runtime"])
    if execution["runtime"] != runtime:
        raise ValueError("Replay runtime differs from completed evaluation")
    plan, receipt = evaluation.authorize(
        snapshot, provenance, calibrated, policy, release, release_sha256, legacy
    )
    if (
        plan["legacy_comparator"] != original["true_legacy_comparator"]["contract"]
        or plan["legacy_comparator"] != inputs["legacy_contract"]
    ):
        raise ValueError("Original legacy comparator changed")
    originals = {
        str(Path(p).resolve()): evaluation.sha(p)
        for p in (
            snapshot,
            provenance,
            policy,
            release,
            frozen_inputs,
            original_result,
            original_execution,
            Path(calibrated) / "receipt.json",
            Path(calibrated) / "bundle.joblib",
            legacy["proof"],
            *legacy["artifacts"].values(),
        )
    }
    observations = evaluation.read(snapshot)["observations"]
    ids, pending, counts = evaluation.source_cohort(observations)
    if (
        len(ids) != 582
        or len(pending) != 6
        or evaluation.canonical(ids) != plan["expected_decision_ids_sha256"]
        or evaluation.canonical(pending) != plan["expected_unmatured_ids_sha256"]
        or counts != plan["expected_counts"]
        or evaluation.calibration_prefix_sha256(observations)
        != receipt["calibration_source_sha256"]
    ):
        raise ValueError("Complete original publication cohort required")
    # Heavy dependencies and deserialization start only after all receipt,
    # runtime, source, policy and original-result identities are authenticated.
    from src.models import gated_forecaster as production
    from src.models.regime_interactions import predict
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import build_panel_features

    production.check_policy(policy)
    bundle, loaded_receipt = production.load_bundle(calibrated)
    if (
        loaded_receipt != receipt
        or production.model_state_fingerprints(bundle["models"])
        != plan["mean_model_state_sha256"]
        or bundle.get("policy_sha256") != plan["mean_policy_sha256"]
        or bundle.get("source_sha256") != plan["production_source_sha256"]
        or any(
            bundle.get("quantiles", {}).get(asset)
            != receipt["calibration"][asset]["normalized_quantile"]
            for asset in evaluation.ASSETS
        )
    ):
        raise ValueError("Frozen calibrated mean state changed")
    panel = parse_panel(json.dumps({"observations": observations}).encode())
    frames = build_panel_features(panel)
    forecasts = production.forecast(bundle, panel)
    selected = {
        (r["asset"], r["date"].strftime("%Y-%m-%d")): r
        for r in forecasts.to_dict("records")
    }
    if len(selected) != len(forecasts):
        raise ValueError("Duplicate selected forecast identities")
    legacy_rows = legacy["callback"](panel.copy(), list(ids), dict(legacy["artifacts"]))
    if [
        (r["asset"], r["date"], r["execution_date"], r["target_date"])
        for r in legacy_rows
    ] != ids:
        raise ValueError("Original legacy prediction cohort changed")
    legacy_map = {(r["asset"], r["date"]): r for r in legacy_rows}
    source = {r["d"]: r for r in observations}
    rows = []
    for asset in evaluation.ASSETS:
        wanted = [key for key in ids if key[0] == asset]
        dates = {key[1] for key in wanted}
        records = [
            r for r in production.frame_records(frames[asset]) if r["date"] in dates
        ]
        if [
            (asset, r["date"], r["execution_date"], r["target_date"]) for r in records
        ] != wanted:
            raise ValueError("Original feature cohort changed")
        ungated = predict(bundle["models"][asset], records, False)
        if len(ungated) != len(wanted):
            raise ValueError("Ungated replay cohort truncated")
        for key, record, raw in zip(wanted, records, ungated):
            saved = selected[(asset, key[1])]
            old = legacy_map[(asset, key[1])]
            target = math.log(evaluation.number(source[key[3]][asset]["v"])) - math.log(
                evaluation.number(source[key[2]][asset]["v"])
            )
            if abs(target - evaluation.number(record["target"])) > 1e-14:
                raise ValueError("Original delayed source target changed")
            rows.append(
                dict(
                    zip(
                        COLUMNS,
                        (
                            asset,
                            key[1],
                            key[2],
                            key[3],
                            target,
                            evaluation.number(saved["prediction"]),
                            evaluation.number(raw),
                            0.0,
                            evaluation.number(saved["lower"]),
                            evaluation.number(saved["upper"]),
                            evaluation.number(old["prediction"]),
                            evaluation.number(old["lower"]),
                            evaluation.number(old["upper"]),
                        ),
                    )
                )
            )
    verify_rows(rows, ids, original)
    check_postflight(
        originals, production, policy, bundle, plan["mean_model_state_sha256"]
    )
    output.mkdir(parents=True)
    csv_path = output / "predictions.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        restored = list(csv.DictReader(stream))
    for row in restored:
        for name in COLUMNS[4:]:
            row[name] = evaluation.number(row[name])
    verify_rows(restored, ids, original)
    result = {
        "schema": "gated_evaluation_forecast_export_v1",
        "rows": len(rows),
        "release_sha256": release_sha256,
        "original_result_sha256": RESULT_SHA256,
        "original_execution_sha256": EXECUTION_SHA256,
        "exporter_source_sha256": evaluation.sha(__file__),
        "predictions_sha256": evaluation.sha(csv_path),
        "runtime": runtime,
        "decision_ids_sha256": evaluation.canonical(ids),
        "calibrated_bundle_sha256": plan["calibrated_bundle_sha256"],
        "legacy_contract": plan["legacy_comparator"],
        "aggregate_matches_completed_evaluation": True,
        "csv_roundtrip_metrics_match": True,
        "weights_refitted": False,
        "policy_reselected": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
