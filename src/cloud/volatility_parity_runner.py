"""Hash-verified saved-model inference only; no fit or future observations."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path, PurePosixPath
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


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
    workspace = Path("/kaggle/working/parity-source")
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
    from src.models.volatility_gate_inference import run, PARENT_REPORT_SHA256

    reports = [
        path
        for path in Path("/kaggle/input").rglob("results.json")
        if hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_REPORT_SHA256
    ]
    if len(reports) != 1:
        raise ValueError(
            "Exactly one authenticated original cloud model report required"
        )
    parent = reports[0]
    result = run(
        "private/development-through2024.json",
        "private/predictions",
        "docs/volatility-gate-plan.json",
        "private/frozen-design.json",
        parent,
        parent.parent,
        "/kaggle/working/saved-gate-parity",
    )
    Path("/kaggle/working/parity-source-receipt.json").write_text(
        json.dumps(
            {
                "source_commit": manifest["source_commit"],
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "inference_source_sha256": result["inference_source_sha256"],
                "files": manifest["files"],
                "environment": result["environment"],
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "total_seconds": time.monotonic() - started,
                "rows": result["rows"],
                "weights_refitted": False,
                "accessed_2025_or_2026": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
