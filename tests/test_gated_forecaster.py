"""Pure software contracts; no historical future labels or host neural/ML fits."""

import copy
import json
from types import SimpleNamespace
import pytest
from src.models import gated_forecaster as production
from src.models.regime_interactions import BASE_FEATURES
from src.models.interaction_study import ASSETS


def row(decision="2024-12-27", execution="2024-12-30", maturity="2024-12-31"):
    return {
        **{k: 0.0 for k in BASE_FEATURES},
        "date": decision,
        "execution_date": execution,
        "target_date": maturity,
        "target": 0.001,
    }


def test_matured_training_drops_last_two_unknown_labels():
    rows = [row(), row("2024-12-30", "2024-12-31", None), row("2024-12-31", None, None)]
    assert production.matured_training_rows(rows) == rows[:1]


@pytest.mark.parametrize(
    "defect",
    [
        "future_decision",
        "future_target",
        "ordering",
        "duplicate",
        "nonfinite",
        "no_support",
    ],
)
def test_training_boundary_fails_closed(defect):
    rows = [row()]
    if defect == "future_decision":
        rows[0]["date"] = "2025-01-01"
    elif defect == "future_target":
        rows[0]["target_date"] = "2025-01-02"
    elif defect == "ordering":
        rows[0]["execution_date"] = rows[0]["date"]
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "nonfinite":
        rows[0][BASE_FEATURES[0]] = float("nan")
    else:
        rows = []
    with pytest.raises(ValueError):
        production.matured_training_rows(rows)


def test_fixed_order_statistic_support_and_invalid_residuals():
    assert production.quantile90(list(range(10))) == 9
    assert production.quantile90(list(range(19))) == 17
    for values in [[], [0.0] * 8, [-1.0] + [0.0] * 9, [True] * 10, [float("inf")] * 10]:
        with pytest.raises(ValueError):
            production.quantile90(values)


def bundle():
    model = SimpleNamespace(
        named_steps={
            "standardscaler": object(),
            "ridge": SimpleNamespace(alpha=10000, fit_intercept=False),
        },
        n_features_in_=22,
    )
    return {
        "version": production.VERSION,
        "assets": ASSETS,
        "features": BASE_FEATURES,
        "mean_source_sha256": production.DERIVED_SHA256,
        "threshold": 1.2,
        "multiplier": 2,
        "models": {a: model for a in ASSETS},
        "quantiles": None,
    }


@pytest.mark.parametrize(
    "defect",
    [
        "version",
        "threshold",
        "multiplier",
        "source",
        "feature_order",
        "asset",
        "quantile",
    ],
)
def test_bundle_rejects_mutated_inference_policy(defect):
    value = bundle()
    production.validate_bundle(value)
    if defect == "version":
        value["version"] = "other"
    elif defect == "threshold":
        value["threshold"] = 1.0
    elif defect == "multiplier":
        value["multiplier"] = True
    elif defect == "source":
        value["mean_source_sha256"] = "0" * 64
    elif defect == "feature_order":
        value["features"] = list(reversed(BASE_FEATURES))
    elif defect == "asset":
        value["models"].pop(ASSETS[0])
    else:
        value["quantiles"] = {a: -1.0 for a in ASSETS}
    with pytest.raises(ValueError):
        production.validate_bundle(value)


def test_checkpoint_hash_guard_precedes_pickle_import(tmp_path):
    (tmp_path / "bundle.joblib").write_bytes(b"not-a-pickle")
    (tmp_path / "receipt.json").write_text(
        json.dumps({"version": production.VERSION, "bundle_sha256": "0" * 64})
    )
    with pytest.raises(ValueError, match="hash"):
        production.load_bundle(tmp_path)


def test_existing_final_artifact_refuses_before_source_or_fit(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("No input or estimator access")

    monkeypatch.setattr(production, "check_policy", forbidden)
    with pytest.raises(ValueError, match="immutable"):
        production.train(None, None, tmp_path)
    monkeypatch.setattr(production, "load_bundle", forbidden)
    with pytest.raises(ValueError, match="immutable"):
        production.calibrate(None, None, None, tmp_path)


def test_calibration_requires_frozen_mean_and_no_search_before_source_parse(tmp_path):
    snapshot = tmp_path / "source"
    snapshot.write_text("fixture")
    receipt = {"bundle_sha256": "a" * 64, "policy_sha256": "b" * 64}
    plan = {
        "schema": "causal_gate_calibration_v1",
        "decision_year": 2025,
        "execution_year": 2025,
        "maximum_target_maturity": "2025-12-31",
        "alpha": 0.1,
        "scale_floor": 1e-5,
        "mean_refit": False,
        "interval_policy_search": False,
        "mean_bundle_sha256": "a" * 64,
        "mean_policy_sha256": "b" * 64,
        "source_sha256": production.sha(snapshot),
    }
    provenance = {
        "schema": "causal_gate_calibration_source_v1",
        "parent_snapshot_sha256": "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e",
        "derived_source_sha256": production.sha(snapshot),
        "mean_prefix_sha256": production.DERIVED_SHA256,
        "maximum_source_date": "2025-12-31",
        "expected_decision_ids_sha256": "e" * 64,
        "expected_counts": {a: 250 for a in ASSETS},
    }
    plan.update(
        {
            "expected_decision_ids_sha256": "e" * 64,
            "expected_counts": {a: 250 for a in ASSETS},
        }
    )
    production.check_calibration_release(plan, receipt, snapshot, provenance)
    for key, value in [
        ("mean_refit", True),
        ("interval_policy_search", True),
        ("maximum_target_maturity", "2026-01-01"),
        ("mean_bundle_sha256", "c" * 64),
        ("source_sha256", "d" * 64),
    ]:
        with pytest.raises(ValueError):
            production.check_calibration_release(
                {**plan, key: value}, receipt, snapshot, provenance
            )


def test_every_matured_source_decision_and_label_is_checked():
    from datetime import date, timedelta
    import math

    observations = [
        {
            "d": str(date(2024, 1, 1) + timedelta(days=i)),
            ASSETS[0]: {"v": str(1 + i * 0.001)},
        }
        for i in range(65)
    ]
    rows = [
        row(observations[i]["d"], observations[i + 1]["d"], observations[i + 2]["d"])
        for i in range(60, 63)
    ]
    for i, r in enumerate(rows, start=60):
        r["target"] = math.log(float(observations[i + 2][ASSETS[0]]["v"])) - math.log(
            float(observations[i + 1][ASSETS[0]]["v"])
        )
    production.validate_training_cohort(rows, observations, ASSETS[0])
    with pytest.raises(ValueError, match="coverage"):
        production.validate_training_cohort(rows[:-1], observations, ASSETS[0])
    bad = copy.deepcopy(rows)
    bad[0]["target"] += 0.001
    with pytest.raises(ValueError, match="target"):
        production.validate_training_cohort(bad, observations, ASSETS[0])
    bad = copy.deepcopy(rows)
    bad[0]["execution_date"] = bad[0]["target_date"]
    with pytest.raises(ValueError, match="maturity"):
        production.validate_training_cohort(bad, observations, ASSETS[0])


@pytest.mark.parametrize(
    "defect",
    [
        "empty_sources",
        "changed_source",
        "boolean_alpha",
        "changed_gate",
        "changed_parity",
    ],
)
def test_production_policy_requires_reviewed_recipe_and_source(tmp_path, defect):
    plan = production.check_policy("docs/gated-forecast-policy.json")
    if defect == "empty_sources":
        plan["production_source_sha256"] = {}
    elif defect == "changed_source":
        plan["production_source_sha256"]["src/models/gated_forecaster.py"] = "0" * 64
    elif defect == "boolean_alpha":
        plan["alpha"] = True
    elif defect == "changed_gate":
        plan["gate_threshold"] = 1.0
    else:
        plan["saved_fold_parity_sha256"] = "0" * 64
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        production.check_policy(path)


def learned_fixture():
    models = bundle()["models"]
    for model in models.values():
        scaler = model.named_steps["standardscaler"] = SimpleNamespace(
            with_mean=True,
            with_std=True,
            mean_=[0.0] * 22,
            scale_=[1.0] * 22,
            var_=[1.0] * 22,
            n_samples_seen_=1933,
            n_features_in_=22,
        )
        ridge = model.named_steps["ridge"]
        ridge.coef_ = [0.01] * 22
        ridge.intercept_ = 0.0
        ridge.n_features_in_ = 22
    return models


def test_canonical_learned_state_rejects_coefficient_and_scaler_changes():
    models = learned_fixture()
    original = production.model_state_fingerprints(models)
    assert production.model_state_fingerprints(copy.deepcopy(models)) == original
    models[ASSETS[0]].named_steps["ridge"].coef_[0] += 0.001
    assert production.model_state_fingerprints(models) != original
    models = learned_fixture()
    models[ASSETS[0]].named_steps["standardscaler"].mean_[0] += 0.001
    assert production.model_state_fingerprints(models) != original


def test_calibration_save_rejects_changed_inherited_state_without_output(
    tmp_path, monkeypatch
):
    import sys

    def forbidden(*args, **kwargs):
        raise AssertionError("No pickle or fit allowed")

    monkeypatch.setitem(
        sys.modules, "joblib", SimpleNamespace(dump=forbidden, load=forbidden)
    )
    value = bundle()
    value["models"] = learned_fixture()
    fingerprints = production.model_state_fingerprints(value["models"])
    value["models"][ASSETS[0]].named_steps["ridge"].coef_[0] += 0.001
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Inherited"):
        production.save_bundle(value, output, {"mean_model_state_sha256": fingerprints})
    assert not output.exists()


def test_complete_source_calibration_ids_detect_missing_interior_publication():
    observations = [
        {"d": d}
        for d in [
            "2024-12-31",
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-12-31",
        ]
    ]
    ids, digest, counts = production.calibration_source_cohort(observations)
    assert counts == {a: 3 for a in ASSETS}
    assert all(
        row[1].startswith("2025-")
        and row[2].startswith("2025-")
        and row[3].startswith("2025-")
        for row in ids
    )
    changed = observations[:2] + observations[3:]
    assert production.calibration_source_cohort(changed)[1] != digest
    with pytest.raises(ValueError):
        production.calibration_source_cohort(observations + [{"d": "2026-01-02"}])


def test_final_fit_packager_preserves_existing_source_and_dirty_guard(
    tmp_path, monkeypatch
):
    from src import prepare_gated_final as staging

    with pytest.raises(ValueError, match="Fresh"):
        staging.prepare(tmp_path, "owner/inputs")
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *args, **kwargs: "?? changed.py"
    )
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="Commit"):
        staging.prepare(output, "owner/inputs")
    assert not output.exists()


def test_mean_canonicalization_is_key_order_invariant_and_past_only():
    rows = [
        {"d": "2024-12-30", "FXEURCAD": {"v": "1.25"}, "FXGBPCAD": {"v": "1.5"}},
        {"d": "2024-12-31", "FXGBPCAD": {"v": "1.6"}, "FXEURCAD": {"v": "1.3"}},
    ]
    permuted = [dict(reversed(list(row.items()))) for row in rows]
    assert production.canonical_mean_bytes(rows) == production.canonical_mean_bytes(
        permuted
    )
    changed = copy.deepcopy(rows)
    changed[0]["FXEURCAD"]["v"] = "1.2500"
    assert production.canonical_mean_bytes(rows) != production.canonical_mean_bytes(
        changed
    )
    with pytest.raises(ValueError, match="pre2025"):
        production.canonical_mean_bytes(rows + [{"d": "2025-01-02"}])
