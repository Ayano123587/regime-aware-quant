"""Derive complete 2025 calibration source only after mean-artifact freeze."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from src.models import gated_forecaster as production

PARENT_SHA256 = "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e"
MEAN_BUNDLE_SHA256 = "bc918cc4f3940fb56d25b9710c6a7b9c9201160d5b30db9459946927dbfb4be0"
MEAN_RECEIPT_SHA256 = "1f92dbb7594f47a289c5a3df09a2b47c49fb41035620686b55825362e938ac86"


def cutoff_source(parent):
    """Only date fields are examined on excluded observations."""
    if production.sha(parent) != PARENT_SHA256:
        raise ValueError("Pinned original development snapshot required")
    content = json.loads(Path(parent).read_bytes())
    original = content["observations"]
    observations = [row for row in original if row["d"] <= "2025-12-31"]
    if (
        not observations
        or observations[0]["d"] != "2017-01-03"
        or observations[-1]["d"] != "2025-12-31"
    ):
        raise ValueError("Complete source cutoff publication coverage required")
    raw = (
        json.dumps({"observations": observations}, separators=(",", ":")) + "\n"
    ).encode()
    prefix = [row for row in observations if row["d"] <= "2024-12-31"]
    prefix_raw = production.canonical_mean_bytes(prefix)
    if hashlib.sha256(prefix_raw).hexdigest() != production.DERIVED_SHA256:
        raise ValueError("Frozen mean training prefix revised")
    ids, ids_sha, counts = production.calibration_source_cohort(observations)
    provenance = {
        "schema": "causal_gate_calibration_source_v1",
        "parent_snapshot_sha256": PARENT_SHA256,
        "derived_source_sha256": hashlib.sha256(raw).hexdigest(),
        "mean_prefix_sha256": production.DERIVED_SHA256,
        "maximum_source_date": "2025-12-31",
        "source_rows": len(observations),
        "parent_observation_rows": len(original),
        "excluded_after_cutoff": len(original) - len(observations),
        "expected_decision_ids_sha256": ids_sha,
        "expected_counts": counts,
        "first_decision": ids[0][1],
        "last_decision": ids[-1][1],
        "last_execution": ids[-1][2],
        "last_target_maturity": ids[-1][3],
        "unmatured_2025_decisions_per_asset": sum(
            r["d"].startswith("2025-") for r in observations
        )
        - counts[production.ASSETS[0]],
        "post2025_values_inspected": False,
        "target_values_computed_during_derivation": False,
    }
    return raw, provenance


def prepare(parent, mean_receipt, output, dataset):
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh calibration staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError(
            "Commit reviewed calibration source before private input parsing"
        )
    if production.sha(mean_receipt) != MEAN_RECEIPT_SHA256:
        raise ValueError("Root-reviewed frozen mean receipt required")
    receipt = json.loads(Path(mean_receipt).read_text(encoding="utf8"))
    if (
        receipt.get("stage") != "mean_frozen"
        or receipt.get("bundle_sha256") != MEAN_BUNDLE_SHA256
    ):
        raise ValueError("Root-reviewed actual final mean artifact required")
    raw, provenance = cutoff_source(parent)
    policy = production.check_policy("docs/gated-forecast-policy.json")
    provenance_raw = (json.dumps(provenance, indent=2) + "\n").encode()
    release = {
        "schema": "causal_gate_calibration_v1",
        "decision_year": 2025,
        "execution_year": 2025,
        "maximum_target_maturity": "2025-12-31",
        "alpha": 0.1,
        "scale_floor": 1e-5,
        "mean_refit": False,
        "interval_policy_search": False,
        "mean_bundle_sha256": MEAN_BUNDLE_SHA256,
        "mean_receipt_sha256": MEAN_RECEIPT_SHA256,
        "mean_policy_sha256": receipt["policy_sha256"],
        "mean_model_state_sha256": receipt["mean_model_state_sha256"],
        "calibration_code_sha256": production.sha("src/models/gated_forecaster.py"),
        "source_sha256": provenance["derived_source_sha256"],
        "source_provenance_sha256": hashlib.sha256(provenance_raw).hexdigest(),
        "expected_decision_ids_sha256": provenance["expected_decision_ids_sha256"],
        "expected_counts": provenance["expected_counts"],
        "mean_kernel": "aikyatansinha/fixed-causal-gate-final-mean-training/1",
        "test_accessed": False,
    }
    names = list(policy["production_source_sha256"]) + [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "docs/gated-forecast-policy.json",
    ]
    contents = {n: Path(n).read_bytes() for n in names}
    contents["private/calibration-source.json"] = raw
    contents["private/calibration-source-provenance.json"] = provenance_raw
    contents["private/calibration-release.json"] = (
        json.dumps(release, indent=2) + "\n"
    ).encode()
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    rows = []
    for name, content in contents.items():
        flat = name.replace("/", "__")
        (upload / flat).write_bytes(content)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {
        "schema": "gated_calibration_source_v1",
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "mean_refit": False,
        "test_observations_packaged": False,
    }
    path = upload / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    runner = (
        Path("src/cloud/gated_calibration_runner.py")
        .read_text(encoding="utf8")
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(production.sha(path)),
        )
    )
    (output / "run.py").write_text(runner, encoding="utf8", newline="\n")
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Fixed gated mean full calibration inputs",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    metadata = {
        "id": dataset.split("/")[0] + "/portfolio-quant-calibration-v1",
        "title": "Frozen Gated Mean Complete Calibration",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [release["mean_kernel"]],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8"
    )
    return {
        "manifest_sha256": production.sha(path),
        "runner_sha256": production.sha(output / "run.py"),
        "release_sha256": hashlib.sha256(
            contents["private/calibration-release.json"]
        ).hexdigest(),
        "provenance": provenance,
        "uploaded": False,
        "dispatched": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["parent", "mean-receipt", "output", "dataset"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.parent, args.mean_receipt, args.output, args.dataset), indent=2
        )
    )


if __name__ == "__main__":
    main()
