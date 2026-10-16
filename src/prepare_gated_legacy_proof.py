"""Stage an authenticated legacy proof without an evaluation dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from src.models import gated_legacy as legacy

RECEIPT_SHA256 = "9b7efb90cfd1a90d087d9e0a810ef98b2daef87ba0f11784565a16c157955462"
POLICY_SHA256 = "a60394eb9df2d8078adba99cb30043edf9ff695393f4ef8ecb3a4406e55a43a5"


def prepare(bundle, selection, receipt, output, dataset, calibration_source=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh legacy proof staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed source before private artifact reads")
    artifacts = {
        "private/legacy-bundle.joblib": (bundle, legacy.BUNDLE_SHA256),
        "private/legacy-selection.json": (selection, legacy.SELECTION_SHA256),
        "private/calibrated-receipt.json": (receipt, RECEIPT_SHA256),
    }
    artifacts["private/calibration-source.json"] = (
        calibration_source,
        "6f0439d2d3f323ad7aecd21803eb3a27d9a8ac042e277f46db2701cac0058c36",
    )
    # Hash bytes only; no host deserialization, feature arrays or predictions.
    for path, expected in artifacts.values():
        if legacy.sha(path) != expected:
            raise ValueError("Root-reviewed original artifact required")
    policy_path = Path("docs/gated-forecast-policy.json")
    if legacy.sha(policy_path) != POLICY_SHA256:
        raise ValueError("Frozen policy required")
    from src.models import gated_forecaster as production

    policy = production.check_policy(policy_path)
    names = list(policy["production_source_sha256"]) + [
        "src/models/gated_legacy.py",
        "src/models/gated_runtime_proof.py",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        str(policy_path).replace("\\", "/"),
    ]
    contents = {name: Path(name).read_bytes() for name in names}
    contents.update(
        {name: Path(path).read_bytes() for name, (path, _) in artifacts.items()}
    )
    # Recheck copied identities against the fixed reviewed digests.
    if any(
        hashlib.sha256(contents[name]).hexdigest() != expected
        for name, (_, expected) in artifacts.items()
    ):
        raise ValueError("Private original artifact changed during staging")
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    rows = []
    for name, raw in contents.items():
        flat = name.replace("/", "__")
        (upload / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "gated_legacy_proof_source_v1",
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "test_data_packaged": False,
    }
    manifest_path = upload / "source-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    runner = Path("src/cloud/gated_legacy_proof_runner.py").read_text(encoding="utf8")
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if runner.count(token) != 1:
        raise ValueError("One explicit runner manifest binding required")
    (output / "run.py").write_text(
        runner.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(legacy.sha(manifest_path))
        ),
        encoding="utf8",
        newline="\n",
    )
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Authenticated Original Legacy Proof Inputs",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
        newline="\n",
    )
    metadata = {
        "id": dataset.split("/")[0] + "/authenticated-original-legacy-proof-v2",
        "title": "Authenticated Original Legacy Proof V2",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [
            "aikyatansinha/fixed-causal-gate-final-mean-training/1",
            "aikyatansinha/frozen-gated-mean-complete-calibration/1",
        ],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    return {
        "source_commit": manifest["source_commit"],
        "manifest_sha256": legacy.sha(manifest_path),
        "runner_sha256": legacy.sha(output / "run.py"),
        "payloads": len(rows),
        "uploaded": False,
        "dispatched": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "bundle",
        "selection",
        "receipt",
        "output",
        "dataset",
        "calibration-source",
    ]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
