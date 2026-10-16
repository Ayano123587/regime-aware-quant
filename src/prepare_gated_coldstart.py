"""Stage the real public-workflow cold-start integration with fixed private data."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

INPUTS = {
    "private/mean-source.json": "21b11ed0fb6a6d8348b39a4553763a68a9a37a4e81aa7474cb6b843c9660bacc",
    "private/development-parent.json": "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e",
    "private/calibration-source.json": "6f0439d2d3f323ad7aecd21803eb3a27d9a8ac042e277f46db2701cac0058c36",
}
POLICY_SHA256 = "a60394eb9df2d8078adba99cb30043edf9ff695393f4ef8ecb3a4406e55a43a5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(snapshot, parent, calibration_source, output, dataset):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh cold-start stage required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed source before private input reads")
    private = dict(zip(INPUTS, [snapshot, parent, calibration_source]))
    if any(sha(private[n]) != digest for n, digest in INPUTS.items()):
        raise ValueError("Exact fixed training/calibration inputs required")
    policy_path = Path("docs/gated-forecast-policy.json")
    if sha(policy_path) != POLICY_SHA256:
        raise ValueError("Frozen selected policy required")
    policy = json.loads(policy_path.read_bytes())
    if any(
        sha(n) != digest for n, digest in policy["production_source_sha256"].items()
    ):
        raise ValueError("Frozen production source changed")
    names = list(policy["production_source_sha256"]) + [
        "src/gated_workflow.py",
        "src/cloud/gated_coldstart_runner.py",
        "src/models/gated_predict.py",
        "src/prepare_gated_calibration.py",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "docs/gated-forecast-policy.json",
    ]
    contents = {n: Path(n).read_bytes() for n in names}
    contents.update({n: Path(p).read_bytes() for n, p in private.items()})
    if any(
        hashlib.sha256(contents[n]).hexdigest() != digest
        for n, digest in INPUTS.items()
    ):
        raise ValueError("Private input changed during staging")
    runner = Path("src/cloud/gated_coldstart_runner.py").read_text(encoding="utf8")
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if runner.count(token) != 1:
        raise ValueError("One runner manifest binding required")
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
    from src.cloud.gated_coldstart_runner import RUNTIME, PYTHON, SOURCE_PATHS

    if set(contents) != SOURCE_PATHS:
        raise ValueError("Complete cold-start source registry required")
    manifest = {
        "schema": "public_gated_coldstart_source_v1",
        "runtime": {"python": PYTHON, **RUNTIME},
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "evaluation_data_packaged": False,
        "original_model_artifacts_packaged": False,
    }
    path = upload / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    (output / "run.py").write_text(
        runner.replace(token, "EXPECTED_MANIFEST_SHA256 = " + repr(sha(path))),
        encoding="utf8",
        newline="\n",
    )
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Public Gated Coldstart Inputs",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
        newline="\n",
    )
    metadata = {
        "id": dataset.split("/")[0] + "/public-gated-coldstart-integration",
        "title": "Public Gated Coldstart Integration",
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
        json.dumps(metadata, indent=2) + "\n", encoding="utf8", newline="\n"
    )
    return {
        "source_commit": manifest["source_commit"],
        "manifest_sha256": sha(path),
        "runner_sha256": sha(output / "run.py"),
        "payloads": len(rows),
        "uploaded": False,
        "dispatched": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["snapshot", "parent", "calibration-source", "output", "dataset"]:
        parser.add_argument("--" + name, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
