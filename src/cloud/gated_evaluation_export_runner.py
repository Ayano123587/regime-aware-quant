"""Replay frozen forecasts and publish a small, auditable complete prediction CSV."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(inputs, workspace, program, expected=None):
    expected = EXPECTED_MANIFEST_SHA256 if expected is None else expected
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Exact rendered export manifest pin required")
    manifests = [
        p for p in Path(inputs).rglob("source-manifest.json") if sha(p) == expected
    ]
    if len(manifests) != 1:
        raise ValueError("One exact export source manifest required")
    path = manifests[0]
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema") != "gated_evaluation_export_transport_v1"
        or manifest.get("dispatchable") is not True
        or manifest.get("weights_refitted") is not False
        or manifest.get("policy_reselected") is not False
    ):
        raise ValueError("Fixed artifact-only export transport required")
    sources = {
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/gated_evaluation.py",
        "src/models/gated_legacy.py",
        "src/prepare_gated_evaluation.py",
        "src/cloud/gated_evaluation_runner.py",
        "docs/gated-forecast-policy.json",
        "src/models/gated_forecaster.py",
        "src/models/volatility_gate_inference.py",
        "src/models/regime_interactions.py",
        "src/models/interaction_study.py",
        "src/models/volatility_gate.py",
        "src/data/multi_asset.py",
        "src/features/multi_asset.py",
        "src/models/gated_evaluation_export.py",
        "src/cloud/gated_evaluation_export_runner.py",
        "src/prepare_gated_evaluation_export.py",
    }
    private = {
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
    }
    allowed = sources | {"private/" + name for name in private}
    files, flats = {}, set()
    for row in manifest["files"]:
        name, flat = row["path"], row["file"]
        relative = PurePosixPath(name)
        if (
            name not in allowed
            or name in files
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or Path(flat).name != flat
            or "\\" in flat
            or flat in flats
        ):
            raise ValueError("Unsafe or duplicate export payload")
        raw = (path.parent / flat).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Frozen export payload changed")
        files[name] = raw
        flats.add(flat)
    if set(files) != allowed:
        raise ValueError("Complete exclusive export payload required")
    token = "EXPECTED_" + "MANIFEST_SHA256 = None"
    template = files["src/cloud/gated_evaluation_export_runner.py"].decode()
    if (
        template.count(token) != 1
        or Path(program).read_bytes()
        != template.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(expected)
        ).encode()
    ):
        raise ValueError("Executed export runner differs from frozen template")
    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError("Fresh export workspace required")
    workspace.mkdir(parents=True)
    for name, raw in files.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return manifest


def main(
    inputs=Path("/kaggle/input"), working=Path("/kaggle/working"), program=__file__
):
    root = Path(working) / "forecast-export-source"
    manifest = restore(inputs, root, program)
    frozen = json.loads((root / "private/frozen-inputs.json").read_bytes())
    if sha(root / "private/frozen-inputs.json") != manifest["frozen_inputs_sha256"]:
        raise ValueError("Frozen export inputs changed")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *[
                name + "==" + version
                for name, version in frozen["runtime"].items()
                if name != "python"
            ],
        ],
        check=True,
    )
    os.chdir(root)
    sys.path.insert(0, str(root))
    from src.models import gated_evaluation_export as export, gated_legacy
    from src.prepare_gated_evaluation import check_frozen_inputs
    from src.cloud.gated_evaluation_runner import locate, check_mounted_legacy_proof

    export.check_runtime(frozen["runtime"])
    export.authenticate_result(
        "private/original-result.json",
        "private/original-execution.json",
        manifest["release_sha256"],
    )
    check_frozen_inputs(
        "private/frozen-inputs.json",
        manifest["frozen_inputs_sha256"],
        "private/calibrated-receipt.json",
        "private/legacy-proof.json",
    )
    check_mounted_legacy_proof(
        inputs, "private/legacy-proof.json", frozen["legacy_proof_sha256"]
    )
    bundle = locate(inputs, "bundle.joblib", frozen["calibrated_bundle_sha256"])
    legacy = {
        "callback": gated_legacy.forecast,
        "source": gated_legacy.__file__,
        "proof": Path("private/legacy-proof.json"),
        "artifacts": {
            "bundle": Path("private/legacy-bundle.joblib"),
            "selection": Path("private/legacy-selection.json"),
        },
    }
    result = export.run(
        "private/source.json",
        "private/provenance.json",
        bundle.parent,
        "docs/gated-forecast-policy.json",
        "private/release.json",
        manifest["release_sha256"],
        "private/frozen-inputs.json",
        manifest["frozen_inputs_sha256"],
        "private/original-result.json",
        "private/original-execution.json",
        Path(working) / "forecast-export",
        legacy=legacy,
    )
    execution = {
        "schema": "gated_evaluation_export_execution_v1",
        "source_commit": manifest["source_commit"],
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(program),
        "receipt_sha256": sha(Path(working) / "forecast-export/receipt.json"),
        "predictions_sha256": result["predictions_sha256"],
        "rows": result["rows"],
        "weights_refitted": False,
        "policy_reselected": False,
    }
    (Path(working) / "forecast-export-execution.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
