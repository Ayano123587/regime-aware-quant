"""Fixed causal volatility routing of frozen chronological ridge predictions."""

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import statistics

from src.models.interaction_study import (
    ASSETS,
    DERIVED_SHA256,
    FOLD_ROWS,
    YEARS,
    metrics,
    month_interval,
)
from src.models.regime_interactions import number

CANDIDATES = [(1.0, 1), (1.0, 2), (1.2, 1), (1.2, 2)]
CONTROL_NAMES = [
    "zero",
    "linear_1000",
    "linear_10000",
    "linear_30000",
    "unconditional_2",
]

BASE_SOURCE_COMMIT = "0ef2b3234bfa9357c585b7ef2c59ab8389ea0477"
PARENT_AUDIT_SHA256 = "20129a76fce8c994b03a2e3fc52eafe608629c24e7a7b823bb34c9cba1e3db8e"
SOURCE_FILES = (
    "src/models/volatility_gate.py",
    "src/models/interaction_study.py",
    "src/models/regime_interactions.py",
    "src/features/multi_asset.py",
    "src/data/multi_asset.py",
    "docs/interaction-plan.json",
)
PREDICTION_PROVENANCE = {
    "canonical": "aikyatansinha/causal-volatility-interaction-ridge-study",
    "version": 1,
    "source_commit": BASE_SOURCE_COMMIT,
    "manifest_sha256": "f33a468215d87f03aeb9231f01b24ba95be9a265312c666ddff8596ac7e5b32c",
    "runner_sha256": "21f2a6577e413dc56ebc2ce6a4d5d4016cde10f27a288afefe48febaecc39539",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def policy_name(threshold, multiplier):
    return f"gate_{threshold:.1f}_{multiplier}"


def gate_predictions(predictions, ratios, threshold, multiplier):
    """No targets or future information enter this prediction transform."""
    if (
        type(threshold) not in (int, float)
        or threshold not in (1.0, 1.2)
        or type(multiplier) is not int
        or multiplier not in (1, 2)
    ):
        raise ValueError("Declared fixed threshold and multiplier required")
    if len(predictions) != len(ratios):
        raise ValueError("Aligned predictions and causal ratios required")
    values = []
    for prediction, ratio in zip(predictions, ratios):
        prediction, ratio = number(prediction), number(ratio)
        if ratio < 0:
            raise ValueError("Nonnegative volatility ratio required")
        values.append(number(multiplier * prediction) if ratio > threshold else 0.0)
    return values


def causal_ratios(observations):
    """Rolling sample standard deviations from current and prior source levels."""
    days = [date.fromisoformat(row["d"]) for row in observations]
    if not days or days != sorted(set(days)):
        raise ValueError("Unique increasing source dates required")
    result = {}
    for asset in ASSETS:
        levels = []
        for row in observations:
            value = row[asset]["v"]
            if isinstance(value, bool):
                raise ValueError("Positive finite source levels required")
            value = number(float(value))
            if value <= 0:
                raise ValueError("Positive finite source levels required")
            levels.append(math.log(value))
        returns = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
        for i in range(20, len(levels)):
            short = statistics.stdev(returns[i - 5 : i])
            long = statistics.stdev(returns[i - 20 : i])
            result[(observations[i]["d"], asset)] = short / max(long, 1e-8)
    return result


def check_plan(path):
    plan = read(path)
    expected = {
        "schema": "volatility_gate_plan_v1",
        "parent_audit_sha256": PARENT_AUDIT_SHA256,
        "prediction_provenance": PREDICTION_PROVENANCE,
        "source_sha256": DERIVED_SHA256,
        "assets": ASSETS,
        "validation_years": YEARS,
        "expected_fold_rows": FOLD_ROWS,
        "candidates": [[t, m] for t, m in CANDIDATES],
        "control_names": CONTROL_NAMES,
        "minimum_relative_rmse_gain": 0.001,
        "minimum_year_wins": 3,
        "maximum_asset_relative_rmse_regression": 0.001,
        "ratio": {
            "short_observations": 5,
            "long_observations": 20,
            "ddof": 1,
            "denominator_floor": 1e-8,
            "causal": True,
        },
        "gate_comparison": "strict_greater_than",
        "maximum_source_date": "2024-12-31",
        "maximum_target_maturity": "2024-12-31",
        "bootstrap_draws": 5000,
        "bootstrap_seed": 20261016,
        "bootstrap_selection_gate": False,
        "tie_order": "declared_candidate_order",
        "refit_weights": False,
    }
    if not isinstance(plan, dict) or json.dumps(
        {k: plan.get(k) for k in expected}, sort_keys=True, allow_nan=False
    ) != json.dumps(expected, sort_keys=True):
        raise ValueError("Changed frozen volatility-routing contract")
    if set(plan.get("prediction_sha256", {})) != {
        name + "-private-predictions.json"
        for name in CONTROL_NAMES
        if name != "unconditional_2"
    }:
        raise ValueError("Missing authenticated prediction input identities")
    return plan


def check_freeze(freeze, protocol, predictions):
    """Reject incomplete or changed identities before any candidate calculation."""
    frozen = read(freeze)
    plan = check_plan(protocol)
    root = Path(__file__).resolve().parents[2]
    if (
        not isinstance(frozen, dict)
        or frozen.get("schema") != "volatility_gate_freeze_v1"
        or frozen.get("base_source_commit") != BASE_SOURCE_COMMIT
        or frozen.get("candidate_scoring_performed") is not False
        or frozen.get("protocol_sha256") != sha(protocol)
        or frozen.get("parent_audit_sha256") != PARENT_AUDIT_SHA256
        or frozen.get("snapshot_sha256") != DERIVED_SHA256
        or frozen.get("prediction_sha256") != plan["prediction_sha256"]
    ):
        raise ValueError("Pre-scoring provenance freeze changed")
    sources = frozen.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES):
        raise ValueError("Exact nonempty scientific source registry required")
    if any(sha(root / path) != digest for path, digest in sources.items()):
        raise ValueError("Pre-scoring scientific source bytes changed")
    audit_path = Path(freeze).parent / "parent-prediction-audit.json"
    if sha(audit_path) != PARENT_AUDIT_SHA256:
        raise ValueError("Parent prediction audit bytes changed")
    audit = read(audit_path)
    if (
        audit.get("source_commit") != BASE_SOURCE_COMMIT
        or audit.get("canonical") != PREDICTION_PROVENANCE["canonical"]
        or audit.get("per_policy_unique_complete_validation_rows") != sum(FOLD_ROWS)
        or audit.get("all_seven_prediction_files_audited") is not True
        or audit.get("same_ids_targets_across_all_policies") is not True
        or audit.get("no_2025_2026_input") is not True
        or any(
            audit.get("prediction_file_sha256", {}).get(name) != digest
            for name, digest in plan["prediction_sha256"].items()
        )
    ):
        raise ValueError("Parent prediction audit identity or coverage changed")
    for name, digest in plan["prediction_sha256"].items():
        if sha(Path(predictions) / name) != digest:
            raise ValueError("Frozen learned prediction bytes changed")
    return frozen


def load_inputs(snapshot, predictions, protocol):
    plan = check_plan(protocol)
    if sha(snapshot) != DERIVED_SHA256:
        raise ValueError("Frozen pre-2025 source changed")
    observations = read(snapshot)["observations"]
    if len(observations) != 1995 or any(
        row["d"] > "2024-12-31" for row in observations
    ):
        raise ValueError("Complete pre-2025 source required")
    expected = {
        (row["d"], asset)
        for i, row in enumerate(observations)
        if i >= 60
        and i + 2 < len(observations)
        and int(row["d"][:4]) in YEARS
        and observations[i + 2]["d"] <= row["d"][:4] + "-12-31"
        for asset in ASSETS
    }
    if len(expected) != sum(FOLD_ROWS):
        raise ValueError("Frozen full validation source cohort changed")
    positions = {row["d"]: i for i, row in enumerate(observations)}
    data = {}
    baseline_ids = None
    for name in CONTROL_NAMES:
        if name == "unconditional_2":
            continue
        filename = name + "-private-predictions.json"
        path = Path(predictions) / filename
        if sha(path) != plan["prediction_sha256"][filename]:
            raise ValueError("Frozen learned prediction bytes changed")
        rows = read(path)
        ids = [(row["date"], row["asset"]) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("Missing, duplicate or changed full prediction cohort")
        paired = [(row["date"], row["asset"], number(row["target"])) for row in rows]
        if baseline_ids is None:
            baseline_ids = paired
        elif paired != baseline_ids:
            raise ValueError("Predictions do not share exact targets/order")
        for row in rows:
            i, asset = positions[row["date"]], row["asset"]
            target = math.log(float(observations[i + 2][asset]["v"])) - math.log(
                float(observations[i + 1][asset]["v"])
            )
            if abs(target - number(row["target"])) > 1e-14:
                raise ValueError("Source-derived delayed target changed")
            number(row["prediction"])
        if [
            sum(int(row["date"][:4]) == year for row in rows) for year in YEARS
        ] != FOLD_ROWS:
            raise ValueError("Chronological cohort counts changed")
        data[name] = rows
    ratios = causal_ratios(observations)
    if not expected <= ratios.keys():
        raise ValueError("Missing causal decision-date ratio")
    return plan, data, ratios


def summarize(name, rows):
    return {
        "name": name,
        "validation": metrics(rows),
        "folds": [
            {"year": year, **metrics([r for r in rows if int(r["date"][:4]) == year])}
            for year in YEARS
        ],
        "assets": {a: metrics([r for r in rows if r["asset"] == a]) for a in ASSETS},
    }


def choose_policy(runs):
    names = CONTROL_NAMES + [policy_name(t, m) for t, m in CANDIDATES]
    if len(runs) != len(names) or {r["name"] for r in runs} != set(names):
        raise ValueError("Complete controls and exact four candidates required")
    lookup = {r["name"]: r for r in runs}
    for row in runs:
        if (
            [r["year"] for r in row["folds"]] != YEARS
            or [r["n"] for r in row["folds"]] != FOLD_ROWS
            or row["validation"]["n"] != sum(FOLD_ROWS)
            or set(row["assets"]) != set(ASSETS)
            or any(row["assets"][a]["n"] != 987 for a in ASSETS)
        ):
            raise ValueError("All years/assets/questions must remain in selection")
        for value in (
            [row["validation"]["rmse"]]
            + [r["rmse"] for r in row["folds"]]
            + [r["rmse"] for r in row["assets"].values()]
        ):
            if number(value) < 0:
                raise ValueError("Negative RMSE")
    retained = lookup["linear_10000"]
    reference = min(
        (lookup[name] for name in CONTROL_NAMES), key=lambda r: r["validation"]["rmse"]
    )
    diagnostics = []
    for threshold, multiplier in CANDIDATES:
        name = policy_name(threshold, multiplier)
        candidate = lookup[name]
        same_scale = lookup["linear_10000" if multiplier == 1 else "unconditional_2"]
        retained_wins = sum(
            c["rmse"] < b["rmse"] for c, b in zip(candidate["folds"], retained["folds"])
        )
        same_scale_wins = sum(
            c["rmse"] < b["rmse"]
            for c, b in zip(candidate["folds"], same_scale["folds"])
        )
        gates = {
            "minimum_pooled_gain": candidate["validation"]["rmse"]
            <= reference["validation"]["rmse"] * 0.999,
            "retained_year_support": retained_wins >= 3,
            "same_scale_year_support": same_scale_wins >= 3,
            "asset_regression_limit": all(
                candidate["assets"][a]["rmse"] <= retained["assets"][a]["rmse"] * 1.001
                for a in ASSETS
            ),
        }
        diagnostics.append(
            {
                "name": name,
                "threshold": threshold,
                "multiplier": multiplier,
                "gates": gates,
                "eligible": all(gates.values()),
                "retained_year_wins": retained_wins,
                "same_scale_year_wins": same_scale_wins,
            }
        )
    eligible = [lookup[row["name"]] for row in diagnostics if row["eligible"]]
    winner = (
        min(eligible, key=lambda r: (r["validation"]["rmse"], names.index(r["name"])))
        if eligible
        else retained
    )
    return {
        "selected": winner["name"],
        "promoted": bool(eligible),
        "strongest_control": reference["name"],
        "candidates": diagnostics,
    }


def evaluate(snapshot, predictions, protocol, freeze, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh policy comparison output required")
    frozen = check_freeze(freeze, protocol, predictions)
    plan, data, ratios = load_inputs(snapshot, predictions, protocol)
    retained = data["linear_10000"]
    data["unconditional_2"] = [
        {**row, "prediction": number(2 * row["prediction"])} for row in retained
    ]
    row_ratios = [ratios[(row["date"], row["asset"])] for row in retained]
    for threshold, multiplier in CANDIDATES:
        values = gate_predictions(
            [row["prediction"] for row in retained], row_ratios, threshold, multiplier
        )
        data[policy_name(threshold, multiplier)] = [
            {**row, "prediction": value} for row, value in zip(retained, values)
        ]
    runs = [summarize(name, rows) for name, rows in data.items()]
    decision = choose_policy(runs)
    report = {
        "runs": runs,
        "decision": decision,
        "source_sha256": DERIVED_SHA256,
        "prediction_sha256": plan["prediction_sha256"],
        "protocol_sha256": sha(protocol),
        "pre_scoring_freeze_sha256": sha(freeze),
        "source_sha256_files": frozen["source_sha256"],
        "month_intervals": {
            policy_name(t, m): month_interval(retained, data[policy_name(t, m)])
            for t, m in CANDIDATES
        },
        "validation_rows": sum(FOLD_ROWS),
        "weights_refitted": False,
        "accessed_2025_or_2026": False,
        "deployment_changed": False,
        "inference_parity_required_before_promotion": True,
        "interpretation": "Repeated validation-development selection informed by observed residual diagnostics; no independent confirmation",
    }
    output.mkdir(parents=True)
    for name, rows in data.items():
        (output / (name + "-private-predictions.json")).write_text(
            json.dumps(rows), encoding="utf8"
        )
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "predictions", "protocol", "freeze", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(
            args.snapshot, args.predictions, args.protocol, args.freeze, args.output
        )
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, "Volatility comparison refused: " + str(error) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
