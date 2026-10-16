"""Prove complete selected forecast parity without target labels or scoring."""

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None
RUNTIME = {
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.6.1",
    "joblib": "1.4.2",
    "threadpoolctl": "3.6.0",
}
PYTHON = "3.13.15"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime():
    actual = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in RUNTIME},
    }
    if actual != {"python": PYTHON, **RUNTIME}:
        raise ValueError("Exact original mean-training numerical runtime required")
    return actual


SOURCE_PATHS = {
    "src/models/gated_forecaster.py",
    "src/models/volatility_gate_inference.py",
    "src/models/regime_interactions.py",
    "src/models/interaction_study.py",
    "src/models/volatility_gate.py",
    "src/data/multi_asset.py",
    "src/features/multi_asset.py",
    "src/gated_workflow.py",
    "src/models/gated_canonical_parity.py",
    "src/cloud/gated_canonical_parity_runner.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "docs/gated-forecast-policy.json",
    "private/observed-source.json",
    "private/reference.csv",
    "private/reference-receipt.json",
    "private/cold-proof.json",
    "private/mean-freeze.json",
    "private/calibration-freeze.json",
    "private/mean-receipt.json",
    "private/calibrated-receipt.json",
    "private/release.json",
}


def restore_source(candidates, workspace, rendered):
    matches = [p for p in candidates if sha(p) == EXPECTED_MANIFEST_SHA256]
    if len(matches) != 1:
        raise ValueError("One exact private cold-start manifest required")
    source = matches[0]
    manifest = json.loads(source.read_bytes())
    if (
        manifest.get("schema") != "canonical_gated_parity_source_v1"
        or manifest.get("dispatchable") is not True
        or manifest.get("target_labels_packaged") is not False
        or manifest.get("original_model_artifacts_packaged") is not False
        or manifest.get("runtime") != {"python": PYTHON, **RUNTIME}
    ):
        raise ValueError("Exact reviewed cold-start schema/runtime required")
    files = manifest.get("files", [])
    if len(files) != len(SOURCE_PATHS) or {r["path"] for r in files} != SOURCE_PATHS:
        raise ValueError("Complete unique cold-start source set required")
    contents = {}
    flat_seen = set()
    for row in files:
        relative = PurePosixPath(row["path"])
        flat = row["file"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in row["path"]
            or ":" in row["path"]
            or flat in flat_seen
            or Path(flat).name != flat
            or "\\" in flat
        ):
            raise ValueError("Unsafe or duplicate cold-start payload path")
        flat_seen.add(flat)
        raw = (source.parent / flat).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Bound cold-start bytes changed")
        contents[row["path"]] = raw
    template = contents["src/cloud/gated_canonical_parity_runner.py"]
    token = b"EXPECTED_MANIFEST_SHA256" + b" = None"
    if template.count(token) != 1 or rendered.read_bytes() != template.replace(
        token, ("EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256)).encode()
    ):
        raise ValueError("Rendered runner differs from authenticated raw template")
    workspace.mkdir()
    for name, raw in contents.items():
        destination = workspace.joinpath(*PurePosixPath(name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return manifest


def main():
    if platform.python_version() != PYTHON:
        raise ValueError("Exact selected-model Python runtime required")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *[n + "==" + v for n, v in RUNTIME.items()],
        ],
        check=True,
    )
    environment = verify_runtime()
    workspace = Path("/kaggle/working/canonical-parity-source")
    manifest = restore_source(
        Path("/kaggle/input").rglob("source-manifest.json"), workspace, Path(__file__)
    )
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src.models import gated_canonical_parity as parity

    if sha("private/release.json") != manifest["release_sha256"]:
        raise ValueError("Reviewed actual migration release changed")
    models = []
    for receipt_sha in [
        parity.FIXED["mean_receipt_sha256"],
        parity.FIXED["calibrated_receipt_sha256"],
    ]:
        matches = [
            p
            for p in Path("/kaggle/input").rglob("receipt.json")
            if sha(p) == receipt_sha
        ]
        if len(matches) != 1:
            raise ValueError("One exact new canonical model receipt required")
        models.append(matches[0].parent)
    result = parity.run(
        "private/observed-source.json",
        "private/reference.csv",
        "private/reference-receipt.json",
        models[1],
        models[0],
        "docs/gated-forecast-policy.json",
        "private/cold-proof.json",
        "private/mean-freeze.json",
        "private/calibration-freeze.json",
        "private/release.json",
        manifest["release_sha256"],
        "/kaggle/working/canonical-parity",
    )
    proof = {
        "source_commit": manifest["source_commit"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(__file__),
        "files": manifest["files"],
        "environment": environment,
        "parity_sha256": sha("/kaggle/working/canonical-parity/parity.json"),
        "parity": result,
        "model_receipt_sha256": [sha(p / "receipt.json") for p in models],
        "accuracy_evaluated": False,
        "weights_refitted": False,
    }
    Path("/kaggle/working/canonical-parity-execution.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(proof, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
