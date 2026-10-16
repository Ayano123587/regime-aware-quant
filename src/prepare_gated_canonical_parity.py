"""Stage a reviewed, targetless canonical forecast parity release."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from src.models import gated_canonical_parity as parity


def prepare(
    snapshot,
    reference,
    reference_receipt,
    cold_proof,
    mean_freeze,
    calibration_freeze,
    mean_receipt,
    calibrated_receipt,
    release,
    release_sha256,
    output,
    dataset,
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh parity staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed parity source before private input reads")
    if parity.sha(release) != release_sha256:
        raise ValueError("Root-reviewed actual artifact release SHA required")
    plan = parity.release_contract(parity.read(release))
    values = {
        "private/observed-source.json": (snapshot, "snapshot_sha256"),
        "private/reference.csv": (reference, "reference_sha256"),
        "private/reference-receipt.json": (
            reference_receipt,
            "reference_receipt_sha256",
        ),
        "private/cold-proof.json": (cold_proof, "coldstart_proof_sha256"),
        "private/mean-freeze.json": (mean_freeze, "mean_freeze_sha256"),
        "private/calibration-freeze.json": (
            calibration_freeze,
            "calibration_freeze_sha256",
        ),
        "private/mean-receipt.json": (mean_receipt, "mean_receipt_sha256"),
        "private/calibrated-receipt.json": (
            calibrated_receipt,
            "calibrated_receipt_sha256",
        ),
    }
    if any(parity.sha(p) != plan[k] for p, k in values.values()):
        raise ValueError("Actual frozen input/reference/artifact metadata differs")
    from src.models import gated_forecaster as production

    policy = production.check_policy("docs/gated-forecast-policy.json")
    if (
        parity.sha("docs/gated-forecast-policy.json") != parity.POLICY_SHA
        or policy["production_source_sha256"] != plan.get("production_source_sha256")
        or parity.sha(parity.__file__) != plan.get("parity_code_sha256")
    ):
        raise ValueError("Current parity source/policy differs from reviewed release")
    from src.cloud.gated_canonical_parity_runner import SOURCE_PATHS, RUNTIME, PYTHON

    names = sorted(n for n in SOURCE_PATHS if not n.startswith("private/"))
    contents = {n: Path(n).read_bytes() for n in names}
    contents.update({n: Path(p).read_bytes() for n, (p, _) in values.items()})
    contents["private/release.json"] = Path(release).read_bytes()
    if set(contents) != SOURCE_PATHS:
        raise ValueError("Exact parity source registry required")
    if any(
        hashlib.sha256(contents[n]).hexdigest() != plan[k]
        for n, (_, k) in values.items()
    ):
        raise ValueError("Private input changed during staging")
    template = Path("src/cloud/gated_canonical_parity_runner.py").read_text(
        encoding="utf8"
    )
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if template.count(token) != 1:
        raise ValueError("One manifest runner binding required")
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
        "schema": "canonical_gated_parity_source_v1",
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": {"python": PYTHON, **RUNTIME},
        "target_labels_packaged": False,
        "original_model_artifacts_packaged": False,
        "release_sha256": release_sha256,
        "files": rows,
    }
    path = upload / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    (output / "run.py").write_text(
        template.replace(token, "EXPECTED_MANIFEST_SHA256 = " + repr(parity.sha(path))),
        encoding="utf8",
        newline="\n",
    )
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Canonical Selected Forecast Parity Inputs",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
        newline="\n",
    )
    metadata = {
        "id": dataset.split("/")[0] + "/canonical-selected-forecast-parity",
        "title": "Canonical Selected Forecast Parity",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [
            "aikyatansinha/public-gated-canonical-coldstart-integration/1"
        ],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    return {
        "manifest_sha256": parity.sha(path),
        "runner_sha256": parity.sha(output / "run.py"),
        "payloads": len(rows),
        "release_sha256": release_sha256,
        "uploaded": False,
        "dispatched": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in [
        "snapshot",
        "reference",
        "reference-receipt",
        "cold-proof",
        "mean-freeze",
        "calibration-freeze",
        "mean-receipt",
        "calibrated-receipt",
        "release",
        "release-sha256",
        "output",
        "dataset",
    ]:
        p.add_argument("--" + n, required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
