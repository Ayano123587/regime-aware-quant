"""Complete fixed 2025 calibration; inherited mean weights never refitted."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path, PurePosixPath
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def check_frozen_policy(production, release, path):
    """Authenticate policy bytes and all seven source files before model loading."""
    if production.sha(path) != release["mean_policy_sha256"]:
        raise ValueError("Frozen mean policy bytes changed")
    return production.check_policy(path)


def main():
    started = time.monotonic()
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "numpy==2.2.6",
            "pandas==2.2.3",
            "scipy==1.15.3",
            "scikit-learn==1.6.1",
            "joblib==1.4.2",
            "threadpoolctl==3.6.0",
        ],
        check=True,
    )
    manifests = [
        p
        for p in Path("/kaggle/input").rglob("source-manifest.json")
        if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    ]
    if len(manifests) != 1:
        raise ValueError("Expected one immutable private source dataset")
    source = manifests[0]
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Mounted source manifest changed")
    manifest = json.loads(source.read_text(encoding="utf8"))
    if manifest.get("dispatchable") is not True:
        raise ValueError(
            "Preview or uncommitted calibration payload is not dispatchable"
        )
    workspace = Path("/kaggle/working/calibration-source")
    workspace.mkdir()
    seen = set()
    for row in manifest["files"]:
        relative = PurePosixPath(row["path"])
        flat = row["file"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in row["path"]
            or ":" in row["path"]
            or relative in seen
            or Path(flat).name != flat
            or "\\" in flat
        ):
            raise ValueError("Unsafe or duplicate mounted source path")
        seen.add(relative)
        raw = (source.parent / flat).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Mounted source file changed")
        destination = workspace / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src.models import gated_forecaster as production
    from src.models.volatility_gate import sha

    release = json.loads(
        Path("private/calibration-release.json").read_text(encoding="utf8")
    )
    check_frozen_policy(production, release, "docs/gated-forecast-policy.json")
    candidates = [
        p
        for p in Path("/kaggle/input").rglob("receipt.json")
        if sha(p) == release["mean_receipt_sha256"]
    ]
    if len(candidates) != 1:
        raise ValueError("Exactly one frozen original mean receipt required")
    model = candidates[0].parent
    if sha(model / "bundle.joblib") != release["mean_bundle_sha256"]:
        raise ValueError("Actual original mean checkpoint bytes changed")
    receipt = json.loads(candidates[0].read_text(encoding="utf8"))
    if (
        receipt["mean_model_state_sha256"] != release["mean_model_state_sha256"]
        or receipt["policy_sha256"] != release["mean_policy_sha256"]
    ):
        raise ValueError("Frozen actual mean recipe/state changed")
    if sha("src/models/gated_forecaster.py") != release["calibration_code_sha256"]:
        raise ValueError("Frozen calibration source changed")
    result = production.calibrate(
        model,
        "private/calibration-source.json",
        "private/calibration-release.json",
        "/kaggle/working/calibrated-gated-mean",
    )
    loaded, saved_receipt = production.load_bundle(
        "/kaggle/working/calibrated-gated-mean"
    )
    if (
        loaded["quantiles"] is None
        or production.model_state_fingerprints(loaded["models"])
        != release["mean_model_state_sha256"]
    ):
        raise ValueError("Calibrated artifact altered inherited learned mean")
    import importlib.metadata

    environment = {
        "python": platform.python_version(),
        **{
            n: importlib.metadata.version(n)
            for n in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "threadpoolctl",
            ]
        },
    }
    proof = {
        "source_commit": manifest["source_commit"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(__file__),
        "files": manifest["files"],
        "environment": environment,
        "seconds": time.monotonic() - started,
        "calibrated_receipt": saved_receipt,
        "original_mean_receipt_sha256": sha(candidates[0]),
        "original_mean_bundle_sha256": sha(model / "bundle.joblib"),
        "inherited_mean_state_equal": True,
        "mean_refitted": False,
        "test_accessed": False,
    }
    Path("/kaggle/working/calibration-source-receipt.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == "__main__":
    main()
