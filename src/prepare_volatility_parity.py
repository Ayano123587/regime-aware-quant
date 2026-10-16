"""Stage readable saved-model parity source; never upload or dispatch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from src.models.volatility_gate import SOURCE_FILES, check_freeze, sha
from src.models.volatility_gate_inference import PARENT_REPORT_SHA256


def prepare(study, output, dataset):
    study, output = Path(study), Path(output)
    if output.exists():
        raise ValueError("Fresh private staging path required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit the reviewed readiness source before staging")
    base = Path("data/interim/interaction-study-v5")
    predictions = base / "results/interaction-results"
    freeze = study / "frozen-design.json"
    frozen = check_freeze(freeze, "docs/volatility-gate-plan.json", predictions)
    plan_path = study / "saved-model-parity-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if (
        plan["parent_report_sha256"] != PARENT_REPORT_SHA256
        or plan["freeze_sha256"] != sha(freeze)
        or plan["inference_source_sha256"]
        != sha("src/models/volatility_gate_inference.py")
        or plan["runner_template_sha256"]
        != sha("src/cloud/volatility_parity_runner.py")
    ):
        raise ValueError("Reviewed parity plan/source changed")
    names = list(SOURCE_FILES) + [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/volatility_gate_inference.py",
        "docs/volatility-gate-plan.json",
    ]
    contents = {name: Path(name).read_bytes() for name in names}
    contents["private/development-through2024.json"] = (
        base / "dataset/private__development-through2024.json"
    ).read_bytes()
    for name in frozen["prediction_sha256"]:
        contents["private/predictions/" + name] = (predictions / name).read_bytes()
    for name in [
        "frozen-design.json",
        "parent-prediction-audit.json",
        "saved-model-parity-plan.json",
    ]:
        contents["private/" + name] = (study / name).read_bytes()
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
        "schema": "saved_parity_source_v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "weights_refitted": False,
        "future_observations_packaged": False,
    }
    path = upload / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    runner = (
        Path("src/cloud/volatility_parity_runner.py")
        .read_text(encoding="utf8")
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(path)),
        )
    )
    (output / "run.py").write_text(runner, encoding="utf8", newline="\n")
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Saved causal ridge inference parity inputs",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    metadata = {
        "id": dataset.split("/")[0] + "/portfolio-quant-saved-gate-parity-v1",
        "title": "Saved Ridge Causal Gate Inference Parity",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": ["aikyatansinha/causal-volatility-interaction-ridge-study/1"],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8"
    )
    return {
        "manifest_sha256": sha(path),
        "runner_sha256": sha(output / "run.py"),
        "files": len(rows),
        "source_commit": manifest["source_commit"],
        "uploaded": False,
        "dispatched": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "output", "dataset"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.study, args.output, args.dataset), indent=2))


if __name__ == "__main__":
    main()
