"""Stage private fixed final training source; no upload or future-data access."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from src.models import gated_forecaster as production


def prepare(output, dataset):
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh final fit staging path required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed production source before staging")
    policy = production.check_policy("docs/gated-forecast-policy.json")
    if policy.get("stage_A_runner_sha256") != production.sha(
        "src/cloud/gated_final_train_runner.py"
    ) or policy.get("stage_A_packager_sha256") != production.sha(__file__):
        raise ValueError("Reviewed final training transport changed")
    files = list(policy["production_source_sha256"]) + [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "docs/gated-forecast-policy.json",
    ]
    contents = {name: Path(name).read_bytes() for name in files}
    contents["private/development-through2024.json"] = Path(
        "data/interim/interaction-study-v5/dataset/private__development-through2024.json"
    ).read_bytes()
    proofs = {
        "selection.json": (
            "data/interim/volatility-gate-v6/results/results.json",
            production.SELECTION_SHA256,
        ),
        "saved-fold-parity.json": (
            "data/interim/volatility-gate-parity-v1-reviewed/output/saved-gate-parity/parity.json",
            production.PARITY_SHA256,
        ),
        "frozen-design.json": (
            "data/interim/volatility-gate-v6/frozen-design.json",
            production.FREEZE_SHA256,
        ),
    }
    for name, (path, digest) in proofs.items():
        if production.sha(path) != digest:
            raise ValueError("Approved promotion evidence changed")
        contents["private/" + name] = Path(path).read_bytes()
    if (
        hashlib.sha256(contents["private/development-through2024.json"]).hexdigest()
        != production.DERIVED_SHA256
    ):
        raise ValueError("Frozen final mean source changed")
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
        "schema": "final_gated_mean_source_v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "future_observations_packaged": False,
        "fixed_fits": 3,
    }
    path = upload / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    runner = (
        Path("src/cloud/gated_final_train_runner.py")
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
                "title": "Fixed causal gate final training inputs",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    metadata = {
        "id": dataset.split("/")[0] + "/portfolio-quant-fixed-final-mean-v1",
        "title": "Fixed Causal Gate Final Mean Training",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8"
    )
    return {
        "manifest_sha256": production.sha(path),
        "runner_sha256": production.sha(output / "run.py"),
        "files": len(rows),
        "uploaded": False,
        "dispatched": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.dataset), indent=2))


if __name__ == "__main__":
    main()
