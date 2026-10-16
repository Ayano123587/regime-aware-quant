"""Pure cold-start transport/runtime guards; no arrays, fits or private data."""

import pytest
from src import prepare_gated_canonical_coldstart as staging
from src.cloud import gated_canonical_coldstart_runner as runner


def test_dirty_source_precedes_private_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *a, **k: "?? pending"
    )
    monkeypatch.setattr(staging, "sha", lambda _: pytest.fail("No private read"))
    with pytest.raises(ValueError, match="Commit"):
        staging.prepare(
            "snapshot",
            "parent",
            "calibration",
            "reference-proof",
            "reference-csv",
            tmp_path / "fresh",
            "owner/private",
        )
    assert not (tmp_path / "fresh").exists()


def test_existing_output_preserved(tmp_path, monkeypatch):
    output = tmp_path / "stage"
    output.mkdir()
    (output / "keep").write_text("immutable")
    monkeypatch.setattr(
        staging.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("No private/Git reads"),
    )
    with pytest.raises(ValueError, match="Fresh"):
        staging.prepare(
            "snapshot",
            "parent",
            "calibration",
            "reference-proof",
            "reference-csv",
            output,
            "owner/private",
        )
    assert (output / "keep").read_text() == "immutable"


def test_wrong_fixed_input_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(staging, "sha", lambda _: "wrong")
    with pytest.raises(ValueError, match="fixed training/calibration"):
        staging.prepare(
            "snapshot",
            "parent",
            "calibration",
            "reference-proof",
            "reference-csv",
            tmp_path / "fresh",
            "owner/private",
        )
    assert not (tmp_path / "fresh").exists()


def test_runtime_guard_rejects_package_or_python_mismatch(monkeypatch):
    monkeypatch.setattr(runner.platform, "python_version", lambda: runner.PYTHON)
    monkeypatch.setattr(
        runner.importlib.metadata, "version", lambda n: runner.RUNTIME[n]
    )
    assert runner.verify_runtime() == {"python": runner.PYTHON, **runner.RUNTIME}
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda n: "different")
    with pytest.raises(ValueError, match="numerical runtime"):
        runner.verify_runtime()
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.14.0")
    with pytest.raises(ValueError, match="numerical runtime"):
        runner.verify_runtime()


def test_python_mismatch_before_installation_or_private_source(monkeypatch):
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.14.0")
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("No installation or fit")
    )
    with pytest.raises(ValueError, match="Python runtime"):
        runner.main()


@pytest.fixture
def payload(tmp_path, monkeypatch):
    import json, hashlib
    from pathlib import Path

    root = tmp_path / "payload"
    root.mkdir()
    rows = []
    for name in sorted(runner.SOURCE_PATHS):
        raw = (
            Path(runner.__file__).read_bytes()
            if name == "src/cloud/gated_canonical_coldstart_runner.py"
            else b"explicit software fixture"
        )
        flat = name.replace("/", "__")
        (root / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "public_gated_canonical_coldstart_source_v1",
        "dispatchable": True,
        "evaluation_data_packaged": False,
        "original_model_artifacts_packaged": False,
        "runtime": {"python": runner.PYTHON, **runner.RUNTIME},
        "files": rows,
    }
    path = root / "source-manifest.json"
    rendered = tmp_path / "rendered.py"

    def seal():
        path.write_text(json.dumps(manifest))
        digest = runner.sha(path)
        monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", digest)
        template = (
            root / "src__cloud__gated_canonical_coldstart_runner.py"
        ).read_bytes()
        rendered.write_bytes(
            template.replace(
                b"EXPECTED_MANIFEST_SHA256" + b" = None",
                ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode(),
            )
        )

    seal()
    return root, manifest, path, rendered, tmp_path / "workspace", seal


def test_raw_template_has_exactly_one_stage_token():
    from pathlib import Path

    assert (
        Path(runner.__file__)
        .read_bytes()
        .count(b"EXPECTED_MANIFEST_SHA256" + b" = None")
        == 1
    )


def test_authenticated_restore_copies_complete_set(payload):
    root, manifest, path, rendered, workspace, seal = payload
    assert runner.restore_source([path], workspace, rendered) == manifest
    assert all(
        (workspace / row["path"]).read_bytes() == (root / row["file"]).read_bytes()
        for row in manifest["files"]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "rendered",
        "template",
        "manifest",
        "ambiguous",
        "missing",
        "duplicate",
        "unsafe",
        "runtime",
        "schema",
    ],
)
def test_changed_transport_rejected_before_copy(payload, defect):
    import hashlib, shutil

    root, manifest, path, rendered, workspace, seal = payload
    candidates = [path]
    if defect == "rendered":
        rendered.write_bytes(rendered.read_bytes() + b"\nchanged")
    elif defect == "template":
        row = next(
            r
            for r in manifest["files"]
            if r["path"] == "src/cloud/gated_canonical_coldstart_runner.py"
        )
        template = root / row["file"]
        template.write_bytes(template.read_bytes() + b"\nchanged")
        row["sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()
        old = rendered.read_bytes()
        seal()
        rendered.write_bytes(old)
    elif defect == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
    elif defect == "ambiguous":
        other = root / "other-manifest.json"
        shutil.copyfile(path, other)
        candidates.append(other)
    elif defect == "missing":
        manifest["files"].pop()
        seal()
    elif defect == "duplicate":
        manifest["files"].append(manifest["files"][0].copy())
        seal()
    elif defect == "unsafe":
        manifest["files"][0]["file"] = "../escape"
        seal()
    elif defect == "runtime":
        manifest["runtime"]["pandas"] = "different"
        seal()
    else:
        manifest["schema"] = "different"
        seal()
    with pytest.raises(ValueError):
        runner.restore_source(candidates, workspace, rendered)
    assert not workspace.exists()


def test_runtime_failure_precedes_manifest_parsing(monkeypatch):
    monkeypatch.setattr(runner.platform, "python_version", lambda: runner.PYTHON)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)

    def refused():
        raise ValueError("runtime mismatch")

    monkeypatch.setattr(runner, "verify_runtime", refused)
    monkeypatch.setattr(
        runner,
        "Path",
        lambda *a: pytest.fail("No manifest/input parse before runtime check"),
    )
    with pytest.raises(ValueError, match="runtime mismatch"):
        runner.main()


def test_forecast_arguments_are_accepted_by_real_public_parser(monkeypatch, capsys):
    import sys
    from src import gated_workflow
    from src.models import gated_predict

    captured = []
    monkeypatch.setattr(
        gated_predict,
        "predict",
        lambda **args: captured.append(args) or {"software_fixture": True},
    )
    arguments = runner.forecast_arguments(
        "model", "source", "policy", "a" * 64, "b" * 64, "output"
    )
    monkeypatch.setattr(sys, "argv", ["gated_workflow", *arguments])
    gated_workflow.main()
    assert captured == [
        {
            "model": "model",
            "snapshot": "source",
            "policy": "policy",
            "policy_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "output": "output",
        }
    ]


def test_actual_reference_comparison_checks_all_ids_and_values():
    ids = {("FIXTURE", "2025-01-02"), ("FIXTURE", "2025-01-03")}
    rows = [
        {"asset": a, "date": day, "prediction": str(i * 0.1)}
        for i, (a, day) in enumerate(sorted(ids))
    ]
    assert runner.compare_reference_forecasts(rows, list(reversed(rows)), ids) == 0
    import copy

    changed = copy.deepcopy(rows)
    changed[0]["prediction"] = "0.001"
    with pytest.raises(ValueError, match="changed actual point"):
        runner.compare_reference_forecasts(changed, rows, ids)
    with pytest.raises(ValueError, match="complete"):
        runner.compare_reference_forecasts(rows[:-1], rows, ids)
    with pytest.raises(ValueError, match="Unique"):
        runner.compare_reference_forecasts(rows + rows[:1], rows, ids)
    changed[0]["prediction"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        runner.compare_reference_forecasts(changed, rows, ids)


def test_prior_2025_statistics_cannot_be_parsed_before_own_mean_freeze(
    tmp_path, monkeypatch
):
    import json

    mean = tmp_path / "mean.json"
    mean.write_bytes(b"own mean fixture")
    bundle = tmp_path / "bundle"
    bundle.write_bytes(b"own weights fixture not deserialized")
    forecasts = tmp_path / "forecasts.csv"
    forecasts.write_bytes(b"private point fixture")
    states = {"FIXTURE": "state"}
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "mean_model_state_sha256": states,
                "pre2026_mean_forecast_rows": 747,
                "forecast_provenance": {"predictions_sha256": runner.sha(forecasts)},
                "calibration": {
                    "future_statistics": "must remain unread before own freeze"
                },
            }
        )
    )
    monkeypatch.setattr(runner, "PRIOR_PROOF_SHA256", runner.sha(prior))
    monkeypatch.setattr(runner, "PRIOR_FORECAST_SHA256", runner.sha(forecasts))
    freeze = {
        "schema": "fresh_gated_mean_freeze_v1",
        "mean_receipt_sha256": runner.sha(mean),
        "mean_bundle_sha256": runner.sha(bundle),
        "mean_model_state_sha256": states,
        "reference_numerical_state_equal": True,
        "calibration_started": False,
        "test_accessed": False,
    }
    original = runner.json.loads
    calls = []

    def guarded(raw):
        calls.append("prior parsed")
        return original(raw)

    monkeypatch.setattr(runner.json, "loads", guarded)
    with pytest.raises(ValueError, match="before prior2025"):
        runner.prior_after_mean_freeze(
            prior,
            forecasts,
            {**freeze, "mean_receipt_sha256": "wrong"},
            mean,
            bundle,
            states,
        )
    assert calls == []
    actual = runner.prior_after_mean_freeze(
        prior, forecasts, freeze, mean, bundle, states
    )
    assert calls == ["prior parsed"] and actual["calibration"][
        "future_statistics"
    ].startswith("must remain")
