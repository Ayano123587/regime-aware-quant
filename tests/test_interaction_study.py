import copy
import json

import pytest

from src.models.interaction_study import (
    ALPHAS,
    FOLD_ROWS,
    YEARS,
    check_plan,
    choose,
    month_interval,
    run,
    split_records,
)
from src.prepare_interaction_kernel import development_observations


def record(decision, execution, maturity, target=0.1):
    return {
        "date": decision,
        "execution_date": execution,
        "target_date": maturity,
        "target": target,
    }


def test_fold_purges_labels_across_boundary_and_excludes_2025_maturity():
    rows = [
        record("2020-12-28", "2020-12-29", "2020-12-30"),
        record("2020-12-31", "2021-01-01", "2021-01-02"),
        record("2021-01-01", "2021-01-02", "2021-01-03"),
        record("2024-12-30", "2024-12-31", "2025-01-01"),
        record("2024-12-31", None, None, target=None),
    ]
    train, validation = split_records(rows, 2021)
    assert [r["date"] for r in train] == ["2020-12-28"]
    assert [r["date"] for r in validation] == ["2021-01-01"]
    with pytest.raises(ValueError, match="Post-2024"):
        split_records(rows + [record("2025-01-01", None, None)], 2021)


def grid():
    runs = []
    for name, value in (
        [("zero", 1.0)]
        + [(f"linear_{a}", 0.99) for a in ALPHAS]
        + [(f"interaction_{a}", 0.98) for a in ALPHAS]
    ):
        alpha = None if name == "zero" else int(name.split("_")[1])
        runs.append(
            {
                "name": name,
                "alpha": alpha,
                "validation": {"n": sum(FOLD_ROWS), "rmse": value},
                "folds": [
                    {"year": y, "n": n, "rmse": value} for y, n in zip(YEARS, FOLD_ROWS)
                ],
            }
        )
    return runs


@pytest.mark.parametrize(
    "defect", ["gain", "retained_years", "matched_years", "stronger_control"]
)
def test_each_frozen_promotion_gate_blocks_a_candidate(defect):
    runs = grid()
    for r in runs:
        if r["name"].startswith("interaction_") and r["name"] != "interaction_1000":
            r["validation"]["rmse"] = 1.1
    candidate = next(r for r in runs if r["name"] == "interaction_1000")
    if defect == "gain":
        candidate["validation"]["rmse"] = 0.9895
    elif defect == "stronger_control":
        next(r for r in runs if r["name"] == "linear_30000")["validation"][
            "rmse"
        ] = 0.97
    else:
        comparator = "linear_10000" if defect == "retained_years" else "linear_1000"
        for row in next(r for r in runs if r["name"] == comparator)["folds"][:2]:
            row["rmse"] = 0.97
    decision = choose(runs)
    assert not decision["promoted"] and decision["selected"] == "linear_10000"


def test_complete_grid_count_tie_order_and_nonfinite_guards():
    assert choose(grid())["selected"] == "interaction_1000"
    for malformed in (grid()[:-1], copy.deepcopy(grid())):
        if len(malformed) == 7:
            malformed[0]["folds"][0]["n"] -= 1
        with pytest.raises(ValueError, match="grid|cohort"):
            choose(malformed)
    malformed = grid()
    malformed[0]["validation"]["rmse"] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        choose(malformed)


def test_month_bootstrap_is_paired_and_not_a_promotion_gate():
    baseline = [
        {
            "date": f"2021-{month:02d}-01",
            "asset": "fixture",
            "target": 0.0,
            "prediction": 2.0,
        }
        for month in range(1, 5)
    ]
    candidate = [{**r, "prediction": 1.0} for r in baseline]
    interval = month_interval(baseline, candidate, draws=100)
    assert interval["low"] == interval["high"] == 1.0 and interval["months"] == 4
    candidate[0]["asset"] = "changed"
    with pytest.raises(ValueError, match="paired"):
        month_interval(baseline, candidate)


def test_missing_or_changed_contract_refuses_before_heavy_import_or_fit(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="contract"):
        check_plan(path)
    with pytest.raises(ValueError, match="contract"):
        run(tmp_path / "absent", tmp_path / "absent", path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_private_source_derivation_excludes_every_post_2024_observation():
    rows = [{"d": "2024-12-31"}, {"d": "2025-01-01"}, {"d": "2026-01-01"}]
    assert development_observations(json.dumps({"observations": rows})) == rows[:1]
    with pytest.raises(ValueError, match="dates"):
        development_observations(json.dumps({"observations": [{"d": "2016-12-31"}]}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("minimum_relative_rmse_gain", 0.0),
        ("minimum_year_wins", 2),
        ("bootstrap_draws", 100),
        ("bootstrap_selection_gate", True),
        ("maximum_target_maturity", "2025-12-31"),
        ("fit_intercept", 0),
        ("gate", {"ratio_clip": [0.1, 10.0]}),
    ],
)
def test_every_hardcoded_gate_or_source_parameter_is_bound(tmp_path, field, value):
    plan = check_plan("docs/interaction-plan.json")
    plan[field] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="contract"):
        check_plan(path)
