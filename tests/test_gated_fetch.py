"""Source acquisition and selected CLI software fixtures; no network or model fits."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.data import gated_fetch as fetch
from src.models.gated_forecaster import canonical_mean_bytes


@pytest.fixture
def past(tmp_path, monkeypatch):
    rows = [
        {"d": "2017-01-03", **{a: {"v": "1.25"} for a in fetch.ASSETS}},
        {"d": "2024-12-31", **{a: {"v": "1.50"} for a in fetch.ASSETS}},
    ]
    raw = json.dumps({"observations": rows}).encode()
    digest = hashlib.sha256(canonical_mean_bytes(rows)).hexdigest()
    monkeypatch.setattr(fetch, "MEAN_ROWS", 2)
    monkeypatch.setattr(fetch, "MEAN_SHA256", digest)
    calls = []

    def download(start, end):
        calls.append((start, end))
        return raw, "https://official.invalid/software-fixture"

    monkeypatch.setattr(fetch, "download", download)
    return SimpleNamespace(
        rows=rows, raw=raw, digest=digest, calls=calls, output=tmp_path / "mean.json"
    )


def test_mean_fetch_is_past_only_and_publishes_atomic_verified_source(past):
    record = fetch.mean(past.output)
    assert past.calls == [("2017-01-01", "2024-12-31")]
    assert past.output.read_bytes() == canonical_mean_bytes(past.rows)
    assert (
        record["published_sha256"] == past.digest
        and record["provider_revision_fallback"] is False
    )
    assert (
        json.loads(past.output.with_name("mean.json.provenance.json").read_bytes())
        == record
    )


def test_source_key_permutation_preserves_values_and_canonical_bytes(past, monkeypatch):
    reversed_rows = [dict(reversed(list(r.items()))) for r in past.rows]
    assert reversed_rows == past.rows and canonical_mean_bytes(
        reversed_rows
    ) == canonical_mean_bytes(past.rows)
    monkeypatch.setattr(
        fetch,
        "download",
        lambda *a: (json.dumps({"observations": reversed_rows}).encode(), "fixture"),
    )
    assert fetch.mean(past.output)["published_sha256"] == past.digest


@pytest.mark.parametrize("defect", ["value", "missing_asset", "date", "count"])
def test_changed_source_fails_without_output_or_fallback(past, monkeypatch, defect):
    rows = copy.deepcopy(past.rows)
    if defect == "value":
        rows[0][fetch.ASSETS[0]]["v"] = "1.251"
    elif defect == "missing_asset":
        del rows[0][fetch.ASSETS[0]]
    elif defect == "date":
        rows[0]["d"] = "2017-01-04"
    else:
        rows.pop()
    monkeypatch.setattr(
        fetch,
        "download",
        lambda *a: (json.dumps({"observations": rows}).encode(), "fixture"),
    )
    with pytest.raises((ValueError, KeyError)):
        fetch.mean(past.output)
    assert (
        not past.output.exists()
        and not past.output.with_name("mean.json.provenance.json").exists()
    )


def test_existing_output_refuses_network(past, monkeypatch):
    past.output.write_bytes(b"preserve")
    monkeypatch.setattr(fetch, "download", lambda *a: pytest.fail("No network"))
    with pytest.raises(ValueError, match="Fresh"):
        fetch.mean(past.output)
    assert past.output.read_bytes() == b"preserve"


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "mean").mkdir(parents=True)
    (run / "mean-source.json").write_bytes(b"past-only software fixture")
    monkeypatch.setattr(fetch, "MEAN_SHA256", fetch.sha(run / "mean-source.json"))
    backend = SimpleNamespace(DERIVED_SHA256=fetch.MEAN_SHA256, sha=fetch.sha)
    monkeypatch.setattr(fetch.workflow, "production", lambda _: backend)
    (run / "mean/bundle.joblib").write_bytes(b"own fixture never deserialized")
    training = {
        a: {
            "rows": 1933,
            "first_decision": "2017-03-29",
            "last_decision": "2024-12-27",
            "last_target_maturity": "2024-12-31",
        }
        for a in fetch.ASSETS
    }
    receipt = {
        "stage": "mean_frozen",
        "policy_sha256": fetch.workflow.POLICY_SHA256,
        "mean_source_sha256": fetch.MEAN_SHA256,
        "bundle_sha256": fetch.sha(run / "mean/bundle.joblib"),
        "mean_model_state_sha256": fetch.workflow.REFERENCE_STATES.copy(),
        "training": training,
        "calibration_accessed": False,
        "test_accessed": False,
    }
    receipt_path = run / "mean/receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    freeze = {
        "schema": "fresh_gated_mean_freeze_v1",
        "policy_sha256": fetch.workflow.POLICY_SHA256,
        "parent_snapshot_sha256": fetch.PARENT_SHA256,
        "mean_source_sha256": fetch.MEAN_SHA256,
        "production_code_sha256": backend.sha("src/models/gated_forecaster.py"),
        "workflow_sha256": fetch.sha(fetch.workflow.__file__),
        "calibration_started": False,
        "test_accessed": False,
        "reported_accuracy_reused": False,
        "mean_receipt_sha256": fetch.sha(receipt_path),
        "mean_bundle_sha256": receipt["bundle_sha256"],
        "mean_model_state_sha256": receipt["mean_model_state_sha256"],
        "reference_numerical_state_equal": True,
    }
    path = run / "mean-freeze.json"
    path.write_text(json.dumps(freeze))
    return SimpleNamespace(
        run=run,
        freeze=freeze,
        path=path,
        receipt=receipt,
        receipt_path=receipt_path,
        policy=tmp_path / "policy",
        output=tmp_path / "parent.json",
    )


@pytest.mark.parametrize(
    "field",
    [
        "parent_snapshot_sha256",
        "mean_source_sha256",
        "production_code_sha256",
        "workflow_sha256",
        "policy_sha256",
        "calibration_started",
        "test_accessed",
        "reported_accuracy_reused",
        "mean_bundle_sha256",
        "reference_numerical_state_equal",
    ],
)
def test_bad_semantic_freeze_refuses_all_network(frozen, monkeypatch, field):
    frozen.freeze[field] = "tampered"
    frozen.path.write_text(json.dumps(frozen.freeze))
    monkeypatch.setattr(
        fetch, "download", lambda *a: pytest.fail("No 2025 network before own freeze")
    )
    with pytest.raises(ValueError):
        fetch.calibration(
            frozen.run, frozen.policy, fetch.sha(frozen.path), frozen.output
        )
    assert not frozen.output.exists()


def test_bad_freeze_digest_and_output_alias_refuse_network(frozen, monkeypatch):
    monkeypatch.setattr(fetch, "download", lambda *a: pytest.fail("No network"))
    with pytest.raises(ValueError, match="digest"):
        fetch.calibration(frozen.run, frozen.policy, "0" * 64, frozen.output)
    with pytest.raises(ValueError, match="Fresh"):
        fetch.calibration(
            frozen.run,
            frozen.policy,
            fetch.sha(frozen.path),
            frozen.run / "new-parent.json",
        )


def test_different_state_is_explicit_and_never_claims_reference(frozen):
    states = {a: "0" * 64 for a in fetch.ASSETS}
    frozen.receipt["mean_model_state_sha256"] = states
    frozen.receipt_path.write_text(json.dumps(frozen.receipt))
    frozen.freeze.update(
        mean_model_state_sha256=states,
        mean_receipt_sha256=fetch.sha(frozen.receipt_path),
        reference_numerical_state_equal=False,
    )
    frozen.path.write_text(json.dumps(frozen.freeze))
    with pytest.raises(ValueError, match="different-state"):
        fetch.own_mean_freeze(frozen.run, frozen.policy, fetch.sha(frozen.path))
    assert (
        fetch.own_mean_freeze(frozen.run, frozen.policy, fetch.sha(frozen.path), True)[
            "reference_numerical_state_equal"
        ]
        is False
    )


def test_calibration_provider_revision_fails_after_valid_own_freeze(
    frozen, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        fetch,
        "download",
        lambda *a: (calls.append(a) or (b"changed provider", "fixture")),
    )
    with pytest.raises(ValueError, match="Provider revision"):
        fetch.calibration(
            frozen.run, frozen.policy, fetch.sha(frozen.path), frozen.output
        )
    assert calls == [("2017-01-01", "2025-12-31")] and not frozen.output.exists()


def test_root_selected_cli_delegates_direct_snapshot_without_old_weights(
    monkeypatch, tmp_path, capsys
):
    import train as entry

    calls = []
    monkeypatch.setattr(
        fetch.workflow,
        "train",
        lambda *args: calls.append(args) or {"software_fixture": True},
    )
    entry.main(
        [
            "selected",
            "--snapshot",
            str(tmp_path / "past.json"),
            "--output",
            str(tmp_path / "fresh"),
        ]
    )
    assert calls == [
        (
            tmp_path / "past.json",
            "docs/gated-forecast-policy.json",
            tmp_path / "fresh",
            False,
        )
    ]
    with pytest.raises(SystemExit):
        entry.main(["selected", "--output", str(tmp_path / "other")])
