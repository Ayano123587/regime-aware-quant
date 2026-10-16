"""Pure cold-start transport/runtime guards; no arrays, fits or private data."""

import pytest
from src import prepare_gated_coldstart as staging
from src.cloud import gated_coldstart_runner as runner


def test_dirty_source_precedes_private_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *a, **k: "?? pending"
    )
    monkeypatch.setattr(staging, "sha", lambda _: pytest.fail("No private read"))
    with pytest.raises(ValueError, match="Commit"):
        staging.prepare(
            "snapshot", "parent", "calibration", tmp_path / "fresh", "owner/private"
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
        staging.prepare("snapshot", "parent", "calibration", output, "owner/private")
    assert (output / "keep").read_text() == "immutable"


def test_wrong_fixed_input_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(staging, "sha", lambda _: "wrong")
    with pytest.raises(ValueError, match="fixed training/calibration"):
        staging.prepare(
            "snapshot", "parent", "calibration", tmp_path / "fresh", "owner/private"
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
            if name == "src/cloud/gated_coldstart_runner.py"
            else b"explicit software fixture"
        )
        flat = name.replace("/", "__")
        (root / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "public_gated_coldstart_source_v1",
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
        template = (root / "src__cloud__gated_coldstart_runner.py").read_bytes()
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
            if r["path"] == "src/cloud/gated_coldstart_runner.py"
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
