"""Three fixed final mean fits, immutable checkpoint proof; no future source."""

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
    workspace = Path("/kaggle/working/final-mean-source")
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
    from src.data.multi_asset import parse_panel
    from src.models.volatility_gate import sha

    for name, digest in [
        ("selection.json", production.SELECTION_SHA256),
        ("saved-fold-parity.json", production.PARITY_SHA256),
        ("frozen-design.json", production.FREEZE_SHA256),
    ]:
        if sha(Path("private") / name) != digest:
            raise ValueError("Approved production promotion proof changed")
    result = production.train(
        "private/development-through2024.json",
        "docs/gated-forecast-policy.json",
        "/kaggle/working/final-gated-mean",
    )
    saved, receipt = production.load_bundle("/kaggle/working/final-gated-mean")
    # Historical tail mechanics only: copied context, actual pre-2025 observations,
    # no accuracy score and no mutation to the final saved artifact.
    panel = parse_panel(Path("private/development-through2024.json").read_bytes())
    probe = {**saved, "context": panel.iloc[:-5].copy()}
    before = production.forecast(probe, panel.iloc[-8:].copy())
    loaded, _ = production.load_bundle("/kaggle/working/final-gated-mean")
    after = production.forecast(
        {**loaded, "context": panel.iloc[:-5].copy()}, panel.iloc[-8:].copy()
    )
    import pandas as pd

    pd.testing.assert_frame_equal(before, after)
    if len(before) != 15:
        raise ValueError("Complete historical forecast mechanics required")
    revised = panel.iloc[-8:].copy()
    revised.loc[revised.index[0], production.ASSETS[0]] *= 1.001
    try:
        production.forecast(probe, revised)
    except ValueError as error:
        if "revise" not in str(error):
            raise
    else:
        raise ValueError("Saved context revision was not rejected")
    import importlib.metadata

    environment = {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in [
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
        "model_receipt": receipt,
        "historical_tail_forecast_mechanics_rows": len(before),
        "forecast_save_reload_exact": True,
        "context_revision_rejected": True,
        "historical_mechanics_are_not_accuracy_evaluation": True,
        "calibration_accessed": False,
        "test_accessed": False,
    }
    Path("/kaggle/working/final-mean-source-receipt.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == "__main__":
    main()
