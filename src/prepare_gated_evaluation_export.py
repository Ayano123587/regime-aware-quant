"""Stage an artifact-only replay of the authenticated complete evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from src.models import gated_evaluation_export as export
from src.models import gated_evaluation as evaluation

C_MANIFEST = "ad17003611f80661e81c66286a0570005e2705c33ebf99efce19cd8bc1cb10ae"
FROZEN_INPUTS = "0fcbdf079b48c2d55debbc385be9ed41bf5aeea26926ea5a9ed49d77017541b9"
RELEASE = "f7dc924aff48b7cc196bce8b6b84f6ff25d666fef64c6e9c1dc1f9bef34c550d"


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def prepare(
    previous_stage, original_result, original_execution, output, dataset, kernel
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh export staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed source before private artifact reads")
    for name in (dataset, kernel):
        if not isinstance(name, str) or not re.fullmatch(r"[\w-]+/[\w-]+", name):
            raise ValueError("Exact owner/slug required")
    export.authenticate_result(original_result, original_execution, RELEASE)
    previous = Path(previous_stage) / "dataset"
    if evaluation.sha(previous / "source-manifest.json") != C_MANIFEST:
        raise ValueError("Exact completed evaluation source payload required")
    manifest = evaluation.read(previous / "source-manifest.json")
    files = {}
    for row in manifest["files"]:
        raw = (previous / row["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Original frozen evaluation payload changed")
        files[row["path"]] = raw
    if (
        hashlib.sha256(files["private/frozen-inputs.json"]).hexdigest() != FROZEN_INPUTS
        or hashlib.sha256(files["private/release.json"]).hexdigest() != RELEASE
    ):
        raise ValueError("Actual frozen inputs or release changed")
    frozen = json.loads(files["private/frozen-inputs.json"])
    for name in (
        "src/models/gated_evaluation_export.py",
        "src/cloud/gated_evaluation_export_runner.py",
        "src/prepare_gated_evaluation_export.py",
    ):
        files[name] = Path(name).read_bytes()
    files["private/original-result.json"] = Path(original_result).read_bytes()
    files["private/original-execution.json"] = Path(original_execution).read_bytes()
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    rows = []
    for name, raw in files.items():
        flat = name.replace("/", "__")
        (upload / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    identity = {
        "schema": "gated_evaluation_export_transport_v1",
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "frozen_inputs_sha256": FROZEN_INPUTS,
        "release_sha256": RELEASE,
        "original_result_sha256": export.RESULT_SHA256,
        "original_execution_sha256": export.EXECUTION_SHA256,
        "evaluation_source_manifest_sha256": C_MANIFEST,
        "weights_refitted": False,
        "policy_reselected": False,
    }
    path = upload / "source-manifest.json"
    path.write_bytes(encoded(identity))
    runner = files["src/cloud/gated_evaluation_export_runner.py"].decode()
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if runner.count(token) != 1:
        raise ValueError("One rendered manifest assignment required")
    (output / "run.py").write_text(
        runner.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(evaluation.sha(path))
        ),
        encoding="utf-8",
        newline="\n",
    )
    (upload / "dataset-metadata.json").write_bytes(
        encoded(
            {
                "id": dataset,
                "title": dataset.split("/")[1].replace("-", " "),
                "licenses": [{"name": "other"}],
            }
        )
    )
    metadata = {
        "id": kernel,
        "title": kernel.split("/")[1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [frozen["calibrated_kernel"], frozen["legacy_kernel"]],
    }
    (output / "kernel-metadata.json").write_bytes(encoded(metadata))
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "previous-stage",
        "original-result",
        "original-execution",
        "output",
        "dataset",
        "kernel",
    ):
        parser.add_argument("--" + name, required=True)
    prepare(**vars(parser.parse_args()))
