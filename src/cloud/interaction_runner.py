"""Private free CPU runner with readable mounted source and exact file hashes."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def main():
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
    manifests = list(Path("/kaggle/input").rglob("source-manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one immutable private source dataset")
    source = manifests[0]
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Mounted source manifest changed")
    manifest = json.loads(source.read_text(encoding="utf8"))
    workspace = Path("/kaggle/working/interaction-source")
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
    from src.models.interaction_study import run

    result = run(
        "private/development-through2024.json",
        "private/provenance.json",
        "docs/interaction-plan.json",
        "/kaggle/working/interaction-results",
    )
    Path("/kaggle/working/source-receipt.json").write_text(
        json.dumps(
            {
                "source_commit": manifest["source_commit"],
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "validation_rows": result["validation_rows"],
                "test_accessed": False,
                "calibration_accessed": False,
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
