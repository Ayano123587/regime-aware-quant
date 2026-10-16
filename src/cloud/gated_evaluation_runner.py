"""Evaluate frozen forecasts on the complete retrospective publication cohort."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def restore(inputs, workspace, program, expected=None):
    """Authenticate the reviewed manifest and every byte before copy or import."""
    expected = EXPECTED_MANIFEST_SHA256 if expected is None else expected
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Rendered exact manifest pin required")
    matches = [
        p for p in Path(inputs).rglob("source-manifest.json") if sha(p) == expected
    ]
    if len(matches) != 1:
        raise ValueError("One exact evaluation source manifest required")
    path = matches[0]
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema") != "gated_evaluation_transport_v1"
        or manifest.get("dispatchable") is not True
        or manifest.get("weights_refitted") is not False
    ):
        raise ValueError("Frozen evaluation transport required")
    private = {
        "source.json",
        "provenance.json",
        "release.json",
        "frozen-inputs.json",
        "calibrated-receipt.json",
        "legacy-proof.json",
        "legacy-bundle.joblib",
        "legacy-selection.json",
    }
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
    }
    allowed = sources | {"private/" + n for n in private}
    files = {}
    flats = set()
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
            raise ValueError("Unsafe or duplicate evaluation payload path")
        files[name] = (path.parent / flat).read_bytes()
        flats.add(flat)
        if hashlib.sha256(files[name]).hexdigest() != row["sha256"]:
            raise ValueError("Evaluation payload changed")
    if set(files) != allowed:
        raise ValueError("Complete exclusive evaluation payload required")
    template = files["src/cloud/gated_evaluation_runner.py"].decode()
    token = "EXPECTED_" + "MANIFEST_SHA256 = None"
    rendered = template.replace(
        token,
        "EXPECTED_MANIFEST_SHA256 = " + repr(expected),
    ).encode()
    if Path(program).read_bytes() != rendered:
        raise ValueError("Executed runner differs from rendered frozen template")
    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError("Fresh evaluation workspace required")
    workspace.mkdir(parents=True)
    for name, raw in files.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return manifest


def check_runtime(expected):
    actual = {
        "python": platform.python_version(),
        **{
            n: importlib.metadata.version(n)
            for n in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "threadpoolctl",
            )
        },
    }
    if actual != expected:
        raise ValueError(
            "Successful compatibility runtime differs before model loading"
        )
    return actual


def locate(inputs, filename, digest):
    matches = [p for p in Path(inputs).rglob(filename) if sha(p) == digest]
    if len(matches) != 1:
        raise ValueError("One unique immutable artifact required")
    return matches[0]


def check_mounted_legacy_proof(inputs, staged_proof, digest):
    """Require the actual completed backend proof, not an unused metadata mount."""
    mounted = locate(inputs, "authenticated-legacy-proof.json", digest)
    if (
        sha(staged_proof) != digest
        or mounted.read_bytes() != Path(staged_proof).read_bytes()
    ):
        raise ValueError("Mounted and frozen legacy proof bytes differ")
    return mounted


def main(
    inputs=Path("/kaggle/input"), working=Path("/kaggle/working"), program=__file__
):
    workspace = Path(working) / "evaluation-source"
    manifest = restore(inputs, workspace, program)
    contract = json.loads((workspace / "private/frozen-inputs.json").read_bytes())
    if (
        sha(workspace / "private/frozen-inputs.json")
        != manifest["frozen_inputs_sha256"]
    ):
        raise ValueError("Frozen inputs changed")
    expected_runtime = contract["runtime"]
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *[n + "==" + v for n, v in expected_runtime.items() if n != "python"],
        ],
        check=True,
    )
    actual_runtime = check_runtime(expected_runtime)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src.prepare_gated_evaluation import check_frozen_inputs
    from src.models import gated_evaluation as evaluation, gated_legacy

    check_frozen_inputs(
        "private/frozen-inputs.json",
        manifest["frozen_inputs_sha256"],
        "private/calibrated-receipt.json",
        "private/legacy-proof.json",
    )
    check_mounted_legacy_proof(
        inputs, "private/legacy-proof.json", contract["legacy_proof_sha256"]
    )
    bundle = locate(inputs, "bundle.joblib", contract["calibrated_bundle_sha256"])
    receipt = bundle.parent / "receipt.json"
    if sha(receipt) != contract["calibrated_receipt_sha256"]:
        raise ValueError("Calibrated mounted receipt differs")
    legacy_contract = contract["legacy_contract"]
    artifacts = {
        "bundle": Path("private/legacy-bundle.joblib"),
        "selection": Path("private/legacy-selection.json"),
    }
    if {k: sha(v) for k, v in artifacts.items()} != legacy_contract["artifact_sha256"]:
        raise ValueError("Original legacy inference artifacts changed")
    legacy = {
        "callback": gated_legacy.forecast,
        "source": gated_legacy.__file__,
        "artifacts": artifacts,
        "proof": Path("private/legacy-proof.json"),
    }
    # Runtime, current production source, both model identities and legacy proof
    # have been authenticated; evaluator rechecks before any deserialization.
    result = evaluation.run(
        "private/source.json",
        "private/provenance.json",
        bundle.parent,
        "docs/gated-forecast-policy.json",
        "private/release.json",
        manifest["release_sha256"],
        Path(working) / "evaluation",
        legacy=legacy,
    )
    execution = {
        "schema": "gated_evaluation_execution_v1",
        "source_commit": manifest["source_commit"],
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(program),
        "runtime": actual_runtime,
        "release_sha256": manifest["release_sha256"],
        "result_sha256": sha(Path(working) / "evaluation/evaluation.json"),
        "rows": result["rows"],
        "weights_refitted": False,
        "policy_reselected": False,
        "retrospective_observed_2026": True,
    }
    (Path(working) / "evaluation-execution.json").write_text(
        json.dumps(execution, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
