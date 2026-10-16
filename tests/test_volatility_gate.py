import copy
from datetime import date, timedelta
import json
import math
import statistics

import pytest

from src.models.volatility_gate import (
    ASSETS,
    CANDIDATES,
    CONTROL_NAMES,
    FOLD_ROWS,
    YEARS,
    causal_ratios,
    check_plan,
    choose_policy,
    gate_predictions,
    policy_name,
)


def test_gate_is_strict_and_preserves_zero_abstentions_and_alignment():
    assert gate_predictions([0.1, -0.2, 0.3], [1.19, 1.2, 1.21], 1.2, 2) == [
        0.0,
        0.0,
        0.6,
    ]
    with pytest.raises(ValueError, match="Aligned"):
        gate_predictions([0.1], [], 1.2, 1)


@pytest.mark.parametrize(
    "values,ratios,threshold,multiplier",
    [
        ([True], [1.3], 1.2, 1),
        ([0.1], [float("nan")], 1.2, 1),
        ([0.1], [-1.0], 1.2, 1),
        ([0.1], [1.3], True, 1),
        ([0.1], [1.3], 1.2, True),
        ([0.1], [1.3], 1.1, 1),
        ([0.1], [1.3], 1.2, 3),
    ],
)
def test_invalid_or_undeclared_policy_inputs_fail(
    values, ratios, threshold, multiplier
):
    with pytest.raises(ValueError):
        gate_predictions(values, ratios, threshold, multiplier)


def observations(count):
    return [
        {
            "d": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            **{
                a: {"v": str(math.exp(0.001 * i + 0.002 * math.sin(i)))} for a in ASSETS
            },
        }
        for i in range(count)
    ]


def test_causal_ratio_sample_ddof_and_future_prefix_invariance():
    source = observations(25)
    ratios = causal_ratios(source)
    log = [math.log(float(row[ASSETS[0]]["v"])) for row in source]
    returns = [log[i] - log[i - 1] for i in range(1, len(log))]
    assert ratios[(source[20]["d"], ASSETS[0])] == pytest.approx(
        statistics.stdev(returns[15:20]) / statistics.stdev(returns[:20])
    )
    extended = causal_ratios(observations(30))
    assert all(value == extended[key] for key, value in ratios.items())
    with pytest.raises(ValueError, match="dates"):
        causal_ratios(source + source[:1])
    source[0][ASSETS[0]]["v"] = "0"
    with pytest.raises(ValueError, match="Positive"):
        causal_ratios(source)


def grid():
    result = []
    for name in CONTROL_NAMES + [policy_name(t, m) for t, m in CANDIDATES]:
        value = 0.98 if name.startswith("gate_") else 0.99
        result.append(
            {
                "name": name,
                "validation": {"n": 2961, "rmse": value},
                "folds": [
                    {"year": y, "n": n, "rmse": value} for y, n in zip(YEARS, FOLD_ROWS)
                ],
                "assets": {a: {"n": 987, "rmse": value} for a in ASSETS},
            }
        )
    return result


@pytest.mark.parametrize(
    "defect", ["pooled", "retained_years", "same_scale_years", "asset"]
)
def test_each_frozen_gate_can_retain_baseline(defect):
    runs = grid()
    candidate = next(r for r in runs if r["name"] == "gate_1.0_2")
    for row in runs:
        if row["name"].startswith("gate_") and row is not candidate:
            row["validation"]["rmse"] = 1.0
    if defect == "pooled":
        candidate["validation"]["rmse"] = 0.9895
    elif defect == "asset":
        candidate["assets"][ASSETS[0]]["rmse"] = 0.992
    else:
        control = "linear_10000" if defect == "retained_years" else "unconditional_2"
        for row in next(r for r in runs if r["name"] == control)["folds"][:2]:
            row["rmse"] = 0.97
    decision = choose_policy(runs)
    assert not decision["promoted"] and decision["selected"] == "linear_10000"


def test_complete_all_asset_cohort_and_declared_ties_are_required():
    assert choose_policy(grid())["selected"] == "gate_1.0_1"
    for defect in ("missing", "cohort", "asset", "nonfinite"):
        runs = copy.deepcopy(grid())
        if defect == "missing":
            runs.pop()
        elif defect == "cohort":
            runs[0]["folds"][0]["n"] -= 1
        elif defect == "asset":
            runs[0]["assets"].pop(ASSETS[0])
        else:
            runs[0]["validation"]["rmse"] = float("nan")
        with pytest.raises(ValueError):
            choose_policy(runs)


def test_mutable_contract_cannot_change_source_gate_or_bootstrap(tmp_path):
    plan = check_plan("docs/volatility-gate-plan.json")
    for key, value in [
        ("candidates", [[1.1, 3]]),
        ("maximum_target_maturity", "2025-12-31"),
        ("maximum_asset_relative_rmse_regression", 0.1),
        ("bootstrap_selection_gate", True),
    ]:
        changed = {**plan, key: value}
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="contract"):
            check_plan(path)


def frozen_fixture(tmp_path, monkeypatch):
    import src.models.volatility_gate as module
    from pathlib import Path

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    plan = json.loads(Path("docs/volatility-gate-plan.json").read_text())
    for name in plan["prediction_sha256"]:
        path = predictions / name
        path.write_text("[]")
        plan["prediction_sha256"][name] = module.sha(path)
    audit = {
        "source_commit": module.BASE_SOURCE_COMMIT,
        "canonical": module.PREDICTION_PROVENANCE["canonical"],
        "per_policy_unique_complete_validation_rows": 2961,
        "all_seven_prediction_files_audited": True,
        "same_ids_targets_across_all_policies": True,
        "no_2025_2026_input": True,
        "prediction_file_sha256": plan["prediction_sha256"],
    }
    audit_path = tmp_path / "parent-prediction-audit.json"
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(module, "PARENT_AUDIT_SHA256", module.sha(audit_path))
    plan["parent_audit_sha256"] = module.PARENT_AUDIT_SHA256
    protocol = tmp_path / "plan.json"
    protocol.write_text(json.dumps(plan))
    freeze = tmp_path / "freeze.json"
    frozen = {
        "schema": "volatility_gate_freeze_v1",
        "base_source_commit": module.BASE_SOURCE_COMMIT,
        "candidate_scoring_performed": False,
        "protocol_sha256": module.sha(protocol),
        "parent_audit_sha256": module.PARENT_AUDIT_SHA256,
        "snapshot_sha256": module.DERIVED_SHA256,
        "prediction_sha256": plan["prediction_sha256"],
        "source_sha256": {name: module.sha(name) for name in module.SOURCE_FILES},
    }
    freeze.write_text(json.dumps(frozen))
    return module, freeze, protocol, predictions, frozen


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "missing",
        "extra",
        "changed",
        "schema",
        "commit",
        "scored",
        "prediction",
        "audit",
    ],
)
def test_freeze_rejects_tamper_before_scoring_or_partial_output(
    tmp_path, monkeypatch, defect
):
    module, freeze, protocol, predictions, frozen = frozen_fixture(
        tmp_path, monkeypatch
    )
    assert module.check_freeze(freeze, protocol, predictions) == frozen
    if defect == "empty":
        frozen["source_sha256"] = {}
    elif defect == "missing":
        frozen["source_sha256"].pop(module.SOURCE_FILES[0])
    elif defect == "extra":
        frozen["source_sha256"]["unexpected.py"] = "0" * 64
    elif defect == "changed":
        frozen["source_sha256"][module.SOURCE_FILES[0]] = "0" * 64
    elif defect == "schema":
        frozen["schema"] = "other"
    elif defect == "commit":
        frozen["base_source_commit"] = "0" * 40
    elif defect == "scored":
        frozen["candidate_scoring_performed"] = True
    elif defect == "prediction":
        next(predictions.iterdir()).write_text("[{}]")
    else:
        (tmp_path / "parent-prediction-audit.json").write_text("{}")
    freeze.write_text(json.dumps(frozen))

    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate scoring or input parsing must not begin")

    monkeypatch.setattr(module, "load_inputs", forbidden)
    monkeypatch.setattr(module, "gate_predictions", forbidden)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        module.evaluate(
            tmp_path / "absent-snapshot", predictions, protocol, freeze, output
        )
    assert not output.exists()
