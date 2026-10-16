"""Exercise the public fresh-artifact workflow without evaluation data or scores."""

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
    "src/models/gated_predict.py",
    "src/prepare_gated_calibration.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "docs/gated-forecast-policy.json",
    "src/cloud/gated_coldstart_runner.py",
    "private/mean-source.json",
    "private/development-parent.json",
    "private/calibration-source.json",
}


def restore_source(candidates, workspace, rendered):
    matches = [p for p in candidates if sha(p) == EXPECTED_MANIFEST_SHA256]
    if len(matches) != 1:
        raise ValueError("One exact private cold-start manifest required")
    source = matches[0]
    manifest = json.loads(source.read_bytes())
    if (
        manifest.get("schema") != "public_gated_coldstart_source_v1"
        or manifest.get("dispatchable") is not True
        or manifest.get("evaluation_data_packaged") is not False
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
    template = contents["src/cloud/gated_coldstart_runner.py"]
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


def command(*args):
    subprocess.run(
        [sys.executable, "-m", "src.gated_workflow", *map(str, args)], check=True
    )


def main():
    if platform.python_version() != PYTHON:
        raise ValueError("Exact original Python runtime required before installation")
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
    workspace = Path("/kaggle/working/coldstart-source")
    manifest = restore_source(
        Path("/kaggle/input").rglob("source-manifest.json"), workspace, Path(__file__)
    )
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src import gated_workflow as workflow

    # Raw input authentication does not parse the 2025 parent before mean freeze.
    required = {
        "private/mean-source.json": "21b11ed0fb6a6d8348b39a4553763a68a9a37a4e81aa7474cb6b843c9660bacc",
        "private/development-parent.json": workflow.PARENT_SHA256,
        "private/calibration-source.json": "6f0439d2d3f323ad7aecd21803eb3a27d9a8ac042e277f46db2701cac0058c36",
        "docs/gated-forecast-policy.json": workflow.POLICY_SHA256,
    }
    if any(sha(path) != digest for path, digest in required.items()):
        raise ValueError("Frozen policy/input bytes changed")
    workflow.production("docs/gated-forecast-policy.json")
    policy = "docs/gated-forecast-policy.json"
    run = Path("/kaggle/working/fresh-own-run")
    calibration = Path("/kaggle/working/fresh-own-calibration")
    command(
        "train",
        "--snapshot",
        "private/mean-source.json",
        "--policy",
        policy,
        "--output",
        run,
    )
    freeze = json.loads((run / "mean-freeze.json").read_bytes())
    mean = json.loads((run / "mean/receipt.json").read_bytes())
    if (
        freeze["reference_numerical_state_equal"] is not True
        or freeze["mean_model_state_sha256"] != workflow.REFERENCE_STATES
        or freeze["mean_receipt_sha256"] != sha(run / "mean/receipt.json")
        or freeze["mean_bundle_sha256"] != sha(run / "mean/bundle.joblib")
    ):
        raise ValueError(
            "Actual fresh own mean freeze required before any 2025 operation"
        )
    import math

    expected_training = {
        "rows": 1933,
        "first_decision": "2017-03-29",
        "last_decision": "2024-12-27",
        "last_target_maturity": "2024-12-31",
    }
    delta = mean.get("final_model_replay_max_abs_difference")
    if (
        mean.get("training")
        != {a: expected_training for a in workflow.REFERENCE_STATES}
        or sum(r["rows"] for r in mean["training"].values()) != 5799
        or type(delta) not in (int, float)
        or not math.isfinite(delta)
        or not 0 <= delta <= 1e-14
    ):
        raise ValueError(
            "Actual complete5799 fitted mean and save/reload replay required"
        )
    forecasts = Path("/kaggle/working/fresh-mean-2025-forecasts.csv")
    command(
        "forecast",
        "--model",
        run / "mean",
        "--snapshot",
        "private/calibration-source.json",
        "--policy",
        policy,
        "--policy-sha256",
        workflow.POLICY_SHA256,
        "--receipt-sha256",
        sha(run / "mean/receipt.json"),
        "--output",
        forecasts,
    )
    with forecasts.open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["asset"], r["date"][:10]) for r in rows]
    observations = json.loads(Path("private/calibration-source.json").read_bytes())[
        "observations"
    ]
    expected_forecast_ids = {
        (a, r["d"])
        for a in workflow.REFERENCE_STATES
        for r in observations
        if r["d"].startswith("2025-")
    }
    if set(keys) != expected_forecast_ids:
        raise ValueError("Exact complete source-derived pre2026 forecast IDs required")
    if (
        len(rows) != 747
        or len(set(keys)) != 747
        or any(
            len([k for k in keys if k[0] == a]) != 249
            for a in workflow.REFERENCE_STATES
        )
        or min(k[1] for k in keys) != "2025-01-02"
        or max(k[1] for k in keys) != "2025-12-31"
    ):
        raise ValueError("Complete fresh pre2026 mean forecast cohort required")
    forecast_proof = json.loads(
        forecasts.with_name(forecasts.name + ".provenance.json").read_bytes()
    )
    if forecast_proof["receipt_sha256"] != sha(
        run / "mean/receipt.json"
    ) or forecast_proof["predictions_sha256"] != sha(forecasts):
        raise ValueError("Own forecast provenance changed")
    command(
        "calibrate",
        "--run",
        run,
        "--parent",
        "private/development-parent.json",
        "--policy",
        policy,
        "--freeze-sha256",
        sha(run / "mean-freeze.json"),
        "--output",
        calibration,
    )
    receipt = json.loads((calibration / "calibrated/receipt.json").read_bytes())
    release = json.loads((calibration / "calibration-release.json").read_bytes())
    if (
        receipt["mean_model_state_sha256"] != mean["mean_model_state_sha256"]
        or release["mean_bundle_sha256"] != mean["bundle_sha256"]
        or release["mean_receipt_sha256"] != sha(run / "mean/receipt.json")
        or any(r["rows"] != 247 for r in receipt["calibration"].values())
    ):
        raise ValueError("Actual full741 own calibration or inherited state changed")
    if set(receipt.get("calibration", {})) != set(workflow.REFERENCE_STATES):
        raise ValueError("Complete three-asset calibration required")
    if any(
        type(r.get("normalized_quantile")) not in (int, float)
        or not math.isfinite(r["normalized_quantile"])
        or r["normalized_quantile"] < 0
        for r in receipt["calibration"].values()
    ):
        raise ValueError("Finite nonnegative actual calibration quantiles required")
    reference_quantiles = {
        "FXEURCAD": 1.6174206110971754,
        "FXGBPCAD": 1.804818398669295,
        "FXAUDCAD": 1.5819973403259258,
    }
    if (
        receipt.get("calibration_source_sha256")
        != required["private/calibration-source.json"]
        or receipt.get("calibration_decision_ids_sha256")
        != "caf7c827abd65d7ffe689bb174c2f37afaa6adc10f5d9f70d5146bee80f71ac3"
        or receipt.get("calibration_release_sha256")
        != sha(calibration / "calibration-release.json")
        or any(
            abs(receipt["calibration"][a]["normalized_quantile"] - q) > 1e-12
            for a, q in reference_quantiles.items()
        )
    ):
        raise ValueError(
            "Fresh complete calibration source/IDs/own release or fixed quantiles differ"
        )
    if receipt["bundle_sha256"] != sha(calibration / "calibrated/bundle.joblib"):
        raise ValueError("Fresh calibrated checkpoint bytes changed")
    proof = {
        "schema": "public_gated_coldstart_integration_v1",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(__file__),
        "files": manifest["files"],
        "environment": environment,
        "actual_fresh_mean_fits": 3,
        "matured_training_rows": sum(x["rows"] for x in mean["training"].values()),
        "calibration_rows": 741,
        "pre2026_mean_forecast_rows": 747,
        "mean_freeze_sha256": sha(run / "mean-freeze.json"),
        "mean_receipt_sha256": sha(run / "mean/receipt.json"),
        "mean_bundle_sha256": sha(run / "mean/bundle.joblib"),
        "calibrated_receipt_sha256": sha(calibration / "calibrated/receipt.json"),
        "calibrated_bundle_sha256": sha(calibration / "calibrated/bundle.joblib"),
        "mean_model_state_sha256": mean["mean_model_state_sha256"],
        "reference_numerical_state_equal": True,
        "mean_final_reload_max_delta": mean["final_model_replay_max_abs_difference"],
        "calibration": receipt["calibration"],
        "forecast_provenance": forecast_proof,
        "reported_accuracy_reused": False,
        "accuracy_evaluated": False,
        "test_accessed": False,
        "parent_parsed_only_after_mean_freeze": True,
        "original_model_artifacts_mounted": False,
    }
    Path("/kaggle/working/public-gated-coldstart-proof.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(proof, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
