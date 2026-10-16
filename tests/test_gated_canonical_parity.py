"""Software guard fixtures; no benchmark data, model loads, or fitting."""

import csv
from datetime import date, timedelta
import pytest
from src.models import gated_canonical_parity as p
from src import prepare_gated_canonical_parity as stage


def test_release_hash_precedes_all_other_reads(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.write_text("not JSON")
    monkeypatch.setattr(p, "read", lambda *_: pytest.fail("parsed before SHA gate"))
    with pytest.raises(ValueError, match="Reviewed release SHA"):
        p.authorize(*(["missing"] * 9), release, "wrong")


def test_date_cohort_uses_no_values(monkeypatch):
    days = [(date(2026, 1, 2) + timedelta(days=i)).isoformat() for i in range(195)]
    days.append("2026-10-14")

    class DateOnly(dict):
        def __getitem__(self, key):
            assert key == "d", "Outcome/value read while deriving IDs"
            return super().__getitem__(key)

    monkeypatch.setattr(
        p,
        "REFERENCE_IDS_SHA",
        p.canonical(
            [
                (a, d, days[i + 1], days[i + 2])
                for a in p.ASSETS
                for i, d in enumerate(days[:-2])
            ]
        ),
    )
    full, mature, pending = p.source_ids([DateOnly(d=d) for d in days])
    assert (len(full), len(mature), len(pending)) == (588, 582, 6)
    with pytest.raises(ValueError, match="196"):
        p.source_ids([DateOnly(d=d) for d in days[:-1]])


def write_reference(path, columns=p.COLUMNS, value="0.1"):
    row = dict(
        zip(
            p.COLUMNS,
            ["FXEURCAD", "2026-01-02", "2026-01-05", "2026-01-06", value, "0", "1"],
        )
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({k: row.get(k, "label") for k in columns})
    return [tuple(row[k] for k in p.COLUMNS[:4])]


@pytest.mark.parametrize("extra", ["target", "accuracy", "legacy"])
def test_reference_refuses_extra_columns(tmp_path, extra):
    path = tmp_path / "reference.csv"
    ids = write_reference(path, p.COLUMNS + [extra])
    with pytest.raises(ValueError, match="seven-column"):
        p.reference_rows(path, ids)


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "2"])
def test_reference_rejects_nonfinite_or_unordered(tmp_path, value):
    path = tmp_path / "reference.csv"
    ids = write_reference(path, value=value)
    with pytest.raises(ValueError):
        p.reference_rows(path, ids)


def test_compare_checks_each_interval_and_identity(tmp_path):
    path = tmp_path / "reference.csv"
    ids = write_reference(path)
    rows = p.reference_rows(path, ids)
    assert p.compare(rows, rows) == dict.fromkeys(p.COLUMNS[4:], 0)
    for key in p.COLUMNS[4:]:
        altered = [dict(rows[0], **{key: rows[0][key] + 1e-8})]
        with pytest.raises(ValueError, match="changed selected"):
            p.compare(altered, rows)
    with pytest.raises(ValueError, match="cohort"):
        p.compare([dict(rows[0], asset="wrong")], rows)


def test_dirty_stage_refuses_private_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(stage.subprocess, "check_output", lambda *a, **k: " M source")
    monkeypatch.setattr(
        p, "sha", lambda *_: pytest.fail("private read before clean Git")
    )
    with pytest.raises(ValueError, match="Commit reviewed"):
        stage.prepare(*(["missing"] * 10), tmp_path / "fresh", "unused")


@pytest.fixture
def runner_payload(tmp_path, monkeypatch):
    import json
    from src.cloud import gated_canonical_parity_runner as runner

    rows = []
    for i, name in enumerate(sorted(runner.SOURCE_PATHS)):
        raw = (
            "EXPECTED_MANIFEST_SHA256 = None\n"
            if name.endswith("gated_canonical_parity_runner.py")
            else "fixture\n"
        ).encode()
        target = tmp_path / f"payload-{i}"
        target.write_bytes(raw)
        rows.append(dict(path=name, file=target.name, sha256=p.sha(target)))
    manifest = tmp_path / "manifest.json"
    value = dict(
        schema="canonical_gated_parity_source_v1",
        dispatchable=True,
        target_labels_packaged=False,
        original_model_artifacts_packaged=False,
        runtime=dict(python=runner.PYTHON, **runner.RUNTIME),
        files=rows,
    )
    manifest.write_text(json.dumps(value))
    digest = p.sha(manifest)
    monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", digest)
    rendered = tmp_path / "runner.py"
    rendered.write_bytes(("EXPECTED_MANIFEST_SHA256 = " + repr(digest) + "\n").encode())
    return runner, manifest, rendered, tmp_path / "workspace", value


def test_runner_restores_only_authenticated_bytes(runner_payload):
    runner, manifest, rendered, output, value = runner_payload
    assert runner.restore_source([manifest], output, rendered) == value
    assert len([x for x in output.rglob("*") if x.is_file()]) == 24


def test_runner_rejects_render_change_before_copy(runner_payload):
    runner, manifest, rendered, output, _ = runner_payload
    rendered.write_text("altered")
    with pytest.raises(ValueError, match="Rendered"):
        runner.restore_source([manifest], output, rendered)
    assert not output.exists()


def test_runner_rejects_ambiguous_manifest_before_copy(runner_payload):
    runner, manifest, rendered, output, _ = runner_payload
    with pytest.raises(ValueError, match="One exact"):
        runner.restore_source([manifest, manifest], output, rendered)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["duplicate", "runtime", "unsafe", "payload"])
def test_runner_rejects_payload_contracts(runner_payload, monkeypatch, mutation):
    import json

    runner, manifest, rendered, output, value = runner_payload
    if mutation == "duplicate":
        value["files"][-1] = value["files"][0]
    elif mutation == "runtime":
        value["runtime"]["python"] = "0.0"
    elif mutation == "unsafe":
        value["files"][0]["file"] = "../escape"
    else:
        (manifest.parent / value["files"][0]["file"]).write_text("tampered")
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", p.sha(manifest))
    with pytest.raises(ValueError):
        runner.restore_source([manifest], output, rendered)
    assert not output.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_finite_guard(value):
    with pytest.raises(ValueError):
        p.finite(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 246),
        ("normalized_quantile", float("nan")),
        ("normalized_quantile", -1),
        ("last_target_maturity", "2026-01-01"),
    ],
)
def test_calibration_complete_before_future_parse(field, value):
    item = dict(
        rows=247,
        first_decision="2025-01-02",
        last_decision="2025-12-29",
        last_target_maturity="2025-12-31",
        normalized_quantile=1.5,
    )
    calibration = {a: dict(item) for a in p.ASSETS}
    p.check_calibration(calibration)
    calibration[p.ASSETS[0]][field] = value
    with pytest.raises(ValueError):
        p.check_calibration(calibration)


@pytest.mark.parametrize("tamper", [None, "pending", "nan", "point"])
def test_saved_forecast_roundtrip(tmp_path, tamper):
    path = tmp_path / "saved.csv"
    mature = dict(
        zip(
            p.COLUMNS, ["FXEURCAD", "2026-01-02", "2026-01-05", "2026-01-06", 0.1, 0, 1]
        )
    )
    pending = dict(
        mature, date="2026-01-05", execution_date="2026-01-06", target_date=None
    )
    rows = [dict(mature), pending]
    ids = [tuple(r[k] for k in p.COLUMNS[:4]) for r in rows]
    if tamper == "pending":
        pending["target_date"] = "None"
    if tamper == "nan":
        rows[0]["selected"] = float("nan")
    if tamper == "point":
        rows[0]["selected"] = 0.2
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=p.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if tamper:
        with pytest.raises(ValueError):
            p.check_saved_forecasts(path, ids, [mature])
    else:
        assert p.check_saved_forecasts(path, ids, [mature]) == dict.fromkeys(
            p.COLUMNS[4:], 0
        )


@pytest.mark.parametrize("tamper", [None, "state", "quantile"])
def test_postflight_checks_actual_states_and_quantiles(tamper):
    from types import SimpleNamespace

    states = {a: "fingerprint" for a in p.ASSETS}
    bundle = dict(models="fixture", quantiles={a: 1.5 for a in p.ASSETS})
    receipt = dict(calibration={a: dict(normalized_quantile=1.5) for a in p.ASSETS})
    backend = SimpleNamespace(
        model_state_fingerprints=lambda _: {} if tamper == "state" else states
    )
    if tamper == "quantile":
        bundle["quantiles"][p.ASSETS[0]] = 2
    if tamper:
        with pytest.raises(ValueError):
            p.check_loaded_state(bundle, receipt, backend, states)
    else:
        p.check_loaded_state(bundle, receipt, backend, states)


@pytest.mark.parametrize("tamper", [None, "freeze", "quantile", "state", "labels"])
def test_authorization_metadata_precedes_future_parse(tmp_path, monkeypatch, tamper):
    from src.models import gated_forecaster as production
    from src import gated_workflow as workflow

    states = workflow.REFERENCE_STATES
    plan = dict(
        schema="canonical_gated_forecast_parity_v1",
        policy_sha256=p.POLICY_SHA,
        mean_source_sha256=p.MEAN_SOURCE_SHA,
        calibration_source_sha256=p.CALIBRATION_SOURCE_SHA,
        snapshot_sha256=p.SOURCE_SHA,
        reference_sha256=p.REFERENCE_SHA,
        reference_decision_ids_sha256=p.REFERENCE_IDS_SHA,
        expected_forecast_rows=588,
        expected_comparison_rows=582,
        expected_pending_rows=6,
        fit_allowed=False,
        accuracy_allowed=False,
        policy_selection_allowed=False,
        artifact_freeze_reviewed=True,
        reference_receipt_sha256="refreceipt",
        parity_code_sha256="code",
        production_source_sha256={},
        **p.FIXED,
    )
    calibration = {
        a: dict(
            rows=247,
            first_decision="2025-01-02",
            last_decision="2025-12-29",
            last_target_maturity="2025-12-31",
            normalized_quantile=1.5,
        )
        for a in p.ASSETS
    }
    mr = dict(
        stage="mean_frozen",
        policy_sha256=p.POLICY_SHA,
        mean_source_sha256=p.MEAN_SOURCE_SHA,
        training={
            a: dict(
                rows=1933,
                first_decision="2017-03-29",
                last_decision="2024-12-27",
                last_target_maturity="2024-12-31",
            )
            for a in p.ASSETS
        },
        calibration_accessed=False,
        test_accessed=False,
        mean_model_state_sha256=states,
    )
    cr = dict(
        stage="calibrated_frozen",
        policy_sha256=p.POLICY_SHA,
        calibration_source_sha256=p.CALIBRATION_SOURCE_SHA,
        calibration=calibration,
        mean_model_state_sha256=states,
        mean_refitted=False,
        test_accessed=False,
        calibration_decision_ids_sha256="caf7c827abd65d7ffe689bb174c2f37afaa6adc10f5d9f70d5146bee80f71ac3",
    )
    mf = dict(
        policy_sha256=p.POLICY_SHA,
        mean_source_sha256=p.MEAN_SOURCE_SHA,
        mean_receipt_sha256=p.FIXED["mean_receipt_sha256"],
        mean_bundle_sha256=p.FIXED["mean_bundle_sha256"],
        calibration_started=False,
        reported_accuracy_reused=False,
        test_accessed=False,
        workflow_sha256="workflow",
        production_code_sha256="production",
        mean_model_state_sha256=states,
    )
    cf = dict(
        receipt_sha256=p.FIXED["calibrated_receipt_sha256"],
        bundle_sha256=p.FIXED["calibrated_bundle_sha256"],
        mean_freeze_sha256=p.FIXED["mean_freeze_sha256"],
    )
    cold = dict(
        schema="public_gated_canonical_coldstart_integration_v1",
        mean_source_sha256=p.MEAN_SOURCE_SHA,
        actual_fresh_mean_fits=3,
        matured_training_rows=5799,
        calibration_rows=741,
        pre2026_mean_forecast_rows=747,
        full747_representation_max_abs_difference=0,
        reference_numerical_state_equal=True,
        test_accessed=False,
        accuracy_evaluated=False,
        original_model_artifacts_mounted=False,
        mean_model_state_sha256=states,
        calibration=calibration,
    )
    old = dict(
        schema="frozen_selected_forecast_parity_reference_v1",
        reference_sha256=p.REFERENCE_SHA,
        original_prediction_export_sha256=p.OLD_EXPORT_SHA,
        original_result_sha256=p.OLD_RESULT_SHA,
        decision_ids_sha256=p.REFERENCE_IDS_SHA,
        rows=582,
        columns=p.COLUMNS,
        targets_present=False,
        evaluation_metrics_computed=False,
        policy_selection_authorized=False,
    )
    if tamper == "freeze":
        mf["calibration_started"] = True
    if tamper == "quantile":
        calibration[p.ASSETS[0]]["normalized_quantile"] = float("nan")
    if tamper == "state":
        cr["mean_model_state_sha256"] = {}
    if tamper == "labels":
        old["targets_present"] = True
    documents = {
        "release": plan,
        "refreceipt": old,
        "cold": cold,
        "mf": mf,
        "cf": cf,
        "mean/receipt.json": mr,
        "model/receipt.json": cr,
    }
    hashes = {
        "release": "releasehash",
        "future": p.SOURCE_SHA,
        "reference": p.REFERENCE_SHA,
        "refreceipt": "refreceipt",
        "policy": p.POLICY_SHA,
        "cold": p.FIXED["coldstart_proof_sha256"],
        "mf": p.FIXED["mean_freeze_sha256"],
        "cf": p.FIXED["calibration_freeze_sha256"],
        "mean/receipt.json": p.FIXED["mean_receipt_sha256"],
        "mean/bundle.joblib": p.FIXED["mean_bundle_sha256"],
        "model/receipt.json": p.FIXED["calibrated_receipt_sha256"],
        "model/bundle.joblib": p.FIXED["calibrated_bundle_sha256"],
        str(p.__file__): "code",
        str(workflow.__file__): "workflow",
        str(production.__file__): "production",
    }
    monkeypatch.setattr(
        p,
        "sha",
        lambda path: hashes[
            (
                str(path).replace("\\", "/")
                if str(path).replace("\\", "/") in hashes
                else str(path)
            )
        ],
    )
    monkeypatch.setattr(p, "read", lambda path: documents[str(path).replace("\\", "/")])
    monkeypatch.setattr(
        production, "check_policy", lambda _: {"production_source_sha256": {}}
    )
    monkeypatch.setattr(
        production,
        "load_bundle",
        lambda *_: pytest.fail("unpickle before authorization"),
    )
    args = (
        "future",
        "reference",
        "refreceipt",
        "model",
        "mean",
        "policy",
        "cold",
        "mf",
        "cf",
        "release",
        "releasehash",
    )
    if tamper:
        with pytest.raises(ValueError):
            p.authorize(*args)
    else:
        assert p.authorize(*args) == (plan, cr)
