"""Transport software fixtures, without real source values or model loads."""

import hashlib
import json
from pathlib import Path

import pytest

from src import prepare_gated_evaluation_export as stage
from src.cloud import gated_evaluation_export_runner as cloud


def test_dirty_stage_refuses_before_private_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage.subprocess, "check_output", lambda *a, **k: " M source.py"
    )
    monkeypatch.setattr(
        stage.export,
        "authenticate_result",
        lambda *a: pytest.fail("private read before clean guard"),
    )
    with pytest.raises(ValueError, match="Commit reviewed"):
        stage.prepare(
            "never-read",
            "never-read",
            "never-read",
            tmp_path / "output",
            "owner/dataset",
            "owner/kernel",
        )
    assert not (tmp_path / "output").exists()


def test_runner_has_one_renderable_manifest_assignment():
    assert (
        Path(cloud.__file__).read_text().count("EXPECTED_MANIFEST_SHA256 = None") == 1
    )


@pytest.mark.parametrize("digest", [None, "", "0" * 63])
def test_missing_manifest_pin_never_copies(tmp_path, digest):
    with pytest.raises(ValueError, match="manifest pin"):
        cloud.restore(tmp_path, tmp_path / "workspace", "never-open", digest)
    assert not (tmp_path / "workspace").exists()


def transport_fixture(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    root = Path(stage.__file__).resolve().parents[1]
    names = sorted(stage.evaluation.PRODUCTION_FILES) + [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/gated_evaluation.py",
        "src/models/gated_legacy.py",
        "src/prepare_gated_evaluation.py",
        "src/cloud/gated_evaluation_runner.py",
        "docs/gated-forecast-policy.json",
        "src/models/gated_evaluation_export.py",
        "src/cloud/gated_evaluation_export_runner.py",
        "src/prepare_gated_evaluation_export.py",
    ]
    names += [
        "private/" + n
        for n in (
            "source.json",
            "provenance.json",
            "release.json",
            "frozen-inputs.json",
            "calibrated-receipt.json",
            "legacy-proof.json",
            "legacy-bundle.joblib",
            "legacy-selection.json",
            "original-result.json",
            "original-execution.json",
        )
    ]
    rows = []
    for name in names:
        raw = (
            (root / name).read_bytes()
            if not name.startswith("private/")
            else b"software fixture; never deserialized"
        )
        flat = name.replace("/", "__")
        (inputs / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "gated_evaluation_export_transport_v1",
        "dispatchable": True,
        "weights_refitted": False,
        "policy_reselected": False,
        "files": rows,
    }
    path = inputs / "source-manifest.json"
    path.write_text(json.dumps(manifest))
    digest = cloud.sha(path)
    program = tmp_path / "run.py"
    program.write_bytes(
        Path(cloud.__file__)
        .read_bytes()
        .replace(
            b"EXPECTED_MANIFEST_SHA256 = None",
            ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode(),
        )
    )
    return inputs, program, digest


def test_exact29_payload_transport_restores_without_import_or_load(tmp_path):
    inputs, program, digest = transport_fixture(tmp_path)
    result = cloud.restore(inputs, tmp_path / "workspace", program, digest)
    assert len(result["files"]) == 29
    assert (
        tmp_path / "workspace/private/legacy-bundle.joblib"
    ).read_bytes() == b"software fixture; never deserialized"


@pytest.mark.parametrize(
    "defect", ["payload", "manifest", "runner", "duplicate", "escape", "missing"]
)
def test_changed_transport_never_creates_workspace(tmp_path, defect):
    inputs, program, digest = transport_fixture(tmp_path)
    if defect == "payload":
        (inputs / "private__original-result.json").write_bytes(b"changed")
    elif defect == "runner":
        program.write_bytes(program.read_bytes() + b"\n# changed")
    else:
        path = inputs / "source-manifest.json"
        manifest = json.loads(path.read_bytes())
        if defect == "manifest":
            manifest["policy_reselected"] = True
        elif defect == "duplicate":
            manifest["files"].append(manifest["files"][0])
        elif defect == "escape":
            manifest["files"][0]["path"] = "../escape"
        else:
            manifest["files"].pop()
        path.write_text(json.dumps(manifest))
        digest = cloud.sha(path)
    with pytest.raises(ValueError):
        cloud.restore(inputs, tmp_path / "workspace", program, digest)
    assert not (tmp_path / "workspace").exists()
