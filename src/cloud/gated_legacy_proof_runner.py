"""Authenticate original legacy numerical state and calibration; no evaluation dataset or scoring."""

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
            "numpy==2.4.2",
            "pandas==3.0.0",
            "scipy==1.17.0",
            "scikit-learn==1.8.0",
            "joblib==1.5.2",
            "threadpoolctl==3.6.0",
        ],
        check=True,
    )
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    candidates = [
        p
        for p in Path("/kaggle/input").rglob("source-manifest.json")
        if sha(p) == EXPECTED_MANIFEST_SHA256
    ]
    if len(candidates) != 1:
        raise ValueError("One exact private proof manifest required")
    source = candidates[0]
    manifest = json.loads(source.read_bytes())
    if manifest.get("dispatchable") is not True:
        raise ValueError("Reviewed clean source required")
    workspace = Path("/kaggle/working/legacy-proof-source")
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
            raise ValueError("Unsafe proof payload path")
        seen.add(relative)
        raw = (source.parent / flat).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Proof payload bytes changed")
        destination = workspace.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src.models import gated_forecaster as production
    from src.models import gated_legacy as legacy

    if (
        sha("docs/gated-forecast-policy.json")
        != "a60394eb9df2d8078adba99cb30043edf9ff695393f4ef8ecb3a4406e55a43a5"
    ):
        raise ValueError("Frozen policy changed")
    production.check_policy("docs/gated-forecast-policy.json")
    mean = json.loads(Path("private/calibrated-receipt.json").read_bytes())
    if (
        sha("private/calibrated-receipt.json")
        != "9b7efb90cfd1a90d087d9e0a810ef98b2daef87ba0f11784565a16c157955462"
    ):
        raise ValueError("Actual frozen calibrated receipt changed")
    checkpoint, proof = legacy.authenticate(
        {
            "bundle": "private/legacy-bundle.joblib",
            "selection": "private/legacy-selection.json",
        }
    )
    # Verify original context boundary without forecasts or any evaluation source.
    from src.data.multi_asset import validate_panel

    context = validate_panel(checkpoint["context"])
    if context.date.max().strftime("%Y-%m-%d") != "2025-12-31":
        raise ValueError("Original pre2026 context required")
    # Actual saved historical context, not a fabricated future forecast or score.
    import numpy as np
    from src.features.multi_asset import build_panel_features, FEATURES

    frames = build_panel_features(context)
    reloaded, reloaded_proof = legacy.authenticate(
        {
            "bundle": "private/legacy-bundle.joblib",
            "selection": "private/legacy-selection.json",
        }
    )
    if reloaded_proof != proof:
        raise ValueError("Original legacy numerical state changed on reload")
    mechanics = []
    for asset in legacy.ASSETS:
        frame = frames[asset]
        rows = frame.loc[frame.date.dt.year == 2025].tail(5)
        if len(rows) != 5:
            raise ValueError("Five actual 2025 historical feature rows required")
        old = checkpoint["bundles"][asset]
        values, lower, upper = legacy.predict_rows(old, rows)
        again = legacy.predict_rows(reloaded["bundles"][asset], rows)
        direct = old["model"].predict(rows[FEATURES])
        width = (
            np.maximum(rows.ewma_volatility.to_numpy(), 1e-5)
            * old["normalized_quantile"]
        )
        reference = (direct, direct - width, direct + width)
        formula_delta = max(
            float(np.max(np.abs(x - y)))
            for x, y in zip((values, lower, upper), reference)
        )
        reload_delta = max(
            float(np.max(np.abs(x - y))) for x, y in zip((values, lower, upper), again)
        )
        if formula_delta > 1e-14 or reload_delta > 1e-14:
            raise ValueError("Actual legacy row formula or reload replay differs")
        mechanics.append(
            {
                "asset": asset,
                "rows": len(rows),
                "first_decision": rows.date.min().strftime("%Y-%m-%d"),
                "last_decision": rows.date.max().strftime("%Y-%m-%d"),
                "direct_formula_max_delta": formula_delta,
                "reload_max_delta": reload_delta,
            }
        )
    proof["historical_row_mechanics"] = mechanics
    proof["accuracy_evaluated"] = False
    proof["historical_rows_are_future_forecasts"] = False
    proof["comparison_to_frozen_mean"] = {
        a: proof["model_state_sha256"][a] == mean["mean_model_state_sha256"][a]
        for a in legacy.ASSETS
    }
    proof["frozen_mean_model_state_sha256"] = mean["mean_model_state_sha256"]
    from src.models.gated_runtime_proof import verify

    proof["cross_runtime_current_mean_calibration"] = verify(
        "private/calibration-source.json", "/kaggle/input"
    )
    import importlib.metadata
    import platform

    proof["environment"] = {
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
    proof["runtime_interpretation"] = (
        "Prospective compatibility environment; complete historical creation environment is unverified"
    )
    proof["test_accessed"] = False
    proof["source_commit"] = manifest["source_commit"]
    proof["manifest_sha256"] = EXPECTED_MANIFEST_SHA256
    proof["runner_sha256"] = sha(__file__)
    proof["files"] = manifest["files"]
    Path("/kaggle/working/authenticated-legacy-proof.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(proof, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
