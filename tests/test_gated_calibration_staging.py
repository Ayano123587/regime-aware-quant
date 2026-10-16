"""Calibration source derivation fixtures; no future historical labels used."""

import json
import hashlib
import pytest
from src import prepare_gated_calibration as staging


def source_fixture(tmp_path, monkeypatch):
    rows = [
        {"d": d, **{a: {"v": "1.0"} for a in staging.production.ASSETS}}
        for d in [
            "2017-01-03",
            "2024-12-31",
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-12-31",
        ]
    ]
    # Deliberately unusable future values: cutoff must never parse them.
    rows.append(
        {
            "d": "2026-01-02",
            **{a: {"v": {"must_not_be_used": True}} for a in staging.production.ASSETS},
        }
    )
    p = tmp_path / "parent.json"
    p.write_text(json.dumps({"observations": rows}))
    monkeypatch.setattr(staging, "PARENT_SHA256", staging.production.sha(p))
    prefix = staging.production.canonical_mean_bytes(rows[:2])
    monkeypatch.setattr(
        staging.production, "DERIVED_SHA256", hashlib.sha256(prefix).hexdigest()
    )
    return p, rows


def test_cutoff_complete_cohort_excludes_future_values_and_last_two_labels(
    tmp_path, monkeypatch
):
    parent, rows = source_fixture(tmp_path, monkeypatch)
    raw, receipt = staging.cutoff_source(parent)
    assert json.loads(raw)["observations"] == rows[:-1]
    assert receipt["source_rows"] == 6 and receipt["excluded_after_cutoff"] == 1
    assert receipt["expected_counts"] == {a: 2 for a in staging.production.ASSETS}
    assert receipt["unmatured_2025_decisions_per_asset"] == 2
    assert receipt["post2025_values_inspected"] is False
    assert receipt["target_values_computed_during_derivation"] is False


def test_parent_and_mean_prefix_tamper_rejected(tmp_path, monkeypatch):
    parent, rows = source_fixture(tmp_path, monkeypatch)
    parent.write_text("{}")
    with pytest.raises(ValueError, match="original"):
        staging.cutoff_source(parent)
    parent.write_text(json.dumps({"observations": rows}))
    monkeypatch.setattr(staging.production, "DERIVED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="prefix"):
        staging.cutoff_source(parent)


def test_stager_requires_actual_frozen_mean_before_parent_read(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt"
    receipt.write_text("{}")

    def forbidden(*args):
        raise AssertionError("Future source cannot be touched")

    monkeypatch.setattr(staging.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(staging, "cutoff_source", forbidden)
    with pytest.raises(ValueError, match="mean receipt"):
        staging.prepare(
            tmp_path / "source", receipt, tmp_path / "stage", "owner/private"
        )


def test_dirty_source_fails_before_private_read_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *args, **kwargs: "?? uncommitted.py"
    )

    def forbidden(*args):
        raise AssertionError("No private source/receipt parse allowed")

    monkeypatch.setattr(staging.production, "sha", forbidden)
    output = tmp_path / "stage"
    with pytest.raises(ValueError, match="Commit"):
        staging.prepare(tmp_path / "parent", tmp_path / "mean", output, "owner/private")
    assert not output.exists()


def test_cloud_policy_checked_before_loading_and_rejects_tamper():
    from types import SimpleNamespace
    from src.cloud.gated_calibration_runner import check_frozen_policy

    calls = []
    backend = SimpleNamespace(
        sha=lambda _: "actual",
        check_policy=lambda p: calls.append(p) or {"verified": True},
    )
    with pytest.raises(ValueError, match="policy bytes"):
        check_frozen_policy(backend, {"mean_policy_sha256": "wrong"}, "fixture")
    assert calls == []
    assert check_frozen_policy(
        backend, {"mean_policy_sha256": "actual"}, "fixture"
    ) == {"verified": True}
    assert calls == ["fixture"]

    def refused(_):
        raise ValueError("Exact frozen production source bytes required")

    backend.check_policy = refused
    with pytest.raises(ValueError, match="source bytes"):
        check_frozen_policy(backend, {"mean_policy_sha256": "actual"}, "fixture")
