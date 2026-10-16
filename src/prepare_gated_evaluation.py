"""Prepare a complete retrospective evaluation from frozen artifact contracts."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from src.models import gated_evaluation as evaluation

PARENTS = {
    "development": "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e",
    "observed2026": "c64e0805e00a265135f122bf1df9c74091918e2f8a3567e821cbf3c96935b7e4",
}
PREFIX = "6f0439d2d3f323ad7aecd21803eb3a27d9a8ac042e277f46db2701cac0058c36"
B_RECEIPT = "9b7efb90cfd1a90d087d9e0a810ef98b2daef87ba0f11784565a16c157955462"
B_BUNDLE = "87993d065d29b0e0e10ba71771f8fe3875e92d2a2361f0cb113350cb4818c4a5"
POLICY = "a60394eb9df2d8078adba99cb30043edf9ff695393f4ef8ecb3a4406e55a43a5"


def raw_json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def check_compatibility(actual, receipt):
    """Require measured replay, complete historical mechanics and frozen quantiles."""
    states = receipt.get("mean_model_state_sha256")
    compatibility = actual.get("cross_runtime_current_mean_calibration", {})
    if (
        not isinstance(states, dict)
        or set(states) != set(evaluation.ASSETS)
        or actual.get("frozen_mean_model_state_sha256") != states
        or actual.get("historical_rows_are_future_forecasts") is not False
        or compatibility.get("decision_ids_sha256")
        != receipt.get("calibration_decision_ids_sha256")
        or not receipt.get("calibration_decision_ids_sha256")
        or compatibility.get("accuracy_evaluated") is not False
    ):
        raise ValueError(
            "Complete frozen-state, calibration IDs and retrospective proof required"
        )
    historical = actual.get("historical_row_mechanics", [])
    if [r.get("asset") for r in historical] != evaluation.ASSETS or any(
        r.get("rows") != 5
        or not 0 <= evaluation.number(r.get("direct_formula_max_delta")) <= 1e-14
        or not 0 <= evaluation.number(r.get("reload_max_delta")) <= 1e-14
        for r in historical
    ):
        raise ValueError("All fifteen historical formula and reload checks required")
    per_asset = compatibility.get("per_asset", [])
    if [r.get("asset") for r in per_asset] != evaluation.ASSETS:
        raise ValueError("All three compatibility assets required")
    for row in per_asset:
        quantile = evaluation.number(
            receipt["calibration"][row["asset"]]["normalized_quantile"]
        )
        frozen = evaluation.number(row.get("frozen_quantile"))
        reproduced = evaluation.number(row.get("reproduced_quantile"))
        delta = evaluation.number(row.get("quantile_abs_delta"))
        if (
            row.get("rows") != 247
            or not 0 <= evaluation.number(row.get("direct_formula_max_delta")) <= 1e-14
            or frozen != quantile
            or not 0 <= delta <= 1e-12
            or delta != abs(reproduced - frozen)
        ):
            raise ValueError(
                "Measured compatibility prediction or frozen quantile differs"
            )


def check_frozen_inputs(path, digest, receipt, proof):
    """The successful compatibility proof is mandatory before source reads."""
    if (
        not isinstance(digest, str)
        or not re.fullmatch("[0-9a-f]{64}", digest)
        or evaluation.sha(path) != digest
    ):
        raise ValueError("Exact frozen-inputs SHA256 required")
    contract = evaluation.read(path)
    required = {
        "schema",
        "calibrated_receipt_sha256",
        "calibrated_bundle_sha256",
        "legacy_proof_sha256",
        "legacy_contract",
        "runtime",
        "production_source_sha256",
        "evaluator_source_sha256",
        "mean_policy_sha256",
        "calibrated_kernel",
        "legacy_kernel",
    }
    if (
        set(contract) != required
        or contract["schema"] != "gated_evaluation_frozen_inputs_v1"
    ):
        raise ValueError("Complete frozen artifact contract required")
    if (
        contract["calibrated_receipt_sha256"] != B_RECEIPT
        or contract["calibrated_bundle_sha256"] != B_BUNDLE
        or contract["mean_policy_sha256"] != POLICY
    ):
        raise ValueError("Frozen calibrated artifact identity differs")
    for name in ("calibrated_kernel", "legacy_kernel"):
        if not isinstance(contract[name], str) or not re.fullmatch(
            r"[\w-]+/[\w-]+/[1-9][0-9]*", contract[name]
        ):
            raise ValueError("Explicit artifact kernel version required")
    if (
        evaluation.sha(receipt) != B_RECEIPT
        or evaluation.sha(proof) != contract["legacy_proof_sha256"]
    ):
        raise ValueError(
            "Actual calibrated receipt and successful legacy proof required"
        )
    legacy = contract["legacy_contract"]
    if (
        not isinstance(legacy, dict)
        or legacy.get("proof_sha256") != contract["legacy_proof_sha256"]
    ):
        raise ValueError("Mandatory authenticated legacy comparator required")
    from src.models import gated_legacy

    expected_artifacts = {
        "bundle": gated_legacy.BUNDLE_SHA256,
        "selection": gated_legacy.SELECTION_SHA256,
    }
    actual = evaluation.read(proof)
    if (
        actual.get("schema") != "authenticated_legacy_forecast_v1"
        or actual.get("weights_refitted") is not False
        or actual.get("test_accessed") is not False
        or actual.get("accuracy_evaluated") is not False
    ):
        raise ValueError("Successful test-free original legacy proof required")
    if (
        actual.get("artifact_sha256") != expected_artifacts
        or legacy.get("artifact_sha256") != expected_artifacts
        or actual.get("model_state_sha256") != legacy.get("model_state_sha256")
        or actual.get("calibration_identity") != legacy.get("calibration_identity")
        or not actual.get("model_state_sha256")
        or not actual.get("calibration_identity")
    ):
        raise ValueError("Original legacy model/calibration contract differs")
    if actual.get("environment") != contract["runtime"] or set(contract["runtime"]) != {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
    }:
        raise ValueError("Actual compatibility runtime required")
    compatibility = actual.get("cross_runtime_current_mean_calibration", {})
    br = evaluation.read(receipt)
    if (
        compatibility.get("b_receipt_sha256") != B_RECEIPT
        or compatibility.get("b_bundle_sha256") != B_BUNDLE
        or compatibility.get("source_sha256") != PREFIX
        or compatibility.get("rows") != 741
        or compatibility.get("mean_state_exact") is not True
        or compatibility.get("mean_refitted") is not False
        or compatibility.get("test_accessed") is not False
    ):
        raise ValueError("Complete current-model compatibility proof required")
    check_compatibility(actual, br)
    policy_path = Path("docs/gated-forecast-policy.json")
    if (
        evaluation.sha(policy_path) != POLICY
        or evaluation.sha(evaluation.__file__) != contract["evaluator_source_sha256"]
    ):
        raise ValueError("Frozen policy/evaluator changed")
    from src.models import gated_forecaster

    policy = gated_forecaster.check_policy(policy_path)
    if policy["production_source_sha256"] != contract["production_source_sha256"]:
        raise ValueError("Frozen production source changed")
    if (
        legacy.get("callback_source_sha256") != evaluation.sha(gated_legacy.__file__)
        or legacy.get("callback_qualified_name") != "src.models.gated_legacy.forecast"
    ):
        raise ValueError("Authenticated legacy callback required")
    if (
        br.get("stage") != "calibrated_frozen"
        or br.get("bundle_sha256") != B_BUNDLE
        or br.get("calibration_source_sha256") != PREFIX
        or br.get("mean_refitted") is not False
        or br.get("test_accessed") is not False
    ):
        raise ValueError("Frozen calibrated receipt differs")
    return contract, br


def derive_source(development, observed):
    paths = {"development": development, "observed2026": observed}
    if {k: evaluation.sha(p) for k, p in paths.items()} != PARENTS:
        raise ValueError("Both original source parents required")
    parts = {k: evaluation.read(p)["observations"] for k, p in paths.items()}
    if {k: len(v) for k, v in parts.items()} != {
        "development": 2244,
        "observed2026": 196,
    }:
        raise ValueError("Full parent publication counts required")
    observations = parts["development"] + parts["observed2026"]
    days = [r["d"] for r in observations]
    if (
        days != sorted(set(days))
        or parts["development"][0]["d"] != "2017-01-03"
        or parts["development"][-1]["d"] != "2025-12-31"
        or parts["observed2026"][0]["d"] != "2026-01-02"
        or days[-1] != "2026-10-14"
    ):
        raise ValueError("Complete chronological source parents required")
    if evaluation.calibration_prefix_sha256(observations) != PREFIX:
        raise ValueError("Frozen calibration prefix revised")
    # IDs depend only on publication dates: no targets, feature fitting or scoring.
    ids, pending = [], []
    future = parts["observed2026"]
    for asset in evaluation.ASSETS:
        for i, row in enumerate(future):
            execution = future[i + 1]["d"] if i + 1 < len(future) else None
            maturity = future[i + 2]["d"] if i + 2 < len(future) else None
            (ids if maturity else pending).append(
                (asset, row["d"], execution, maturity)
            )
    raw = (
        json.dumps({"observations": observations}, separators=(",", ":")) + "\n"
    ).encode()
    provenance = {
        "schema": "causal_gate_evaluation_source_v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_parent_sha256": PARENTS,
        "source_parent_counts": {k: len(v) for k, v in parts.items()},
        "source_parent_ranges": {
            k: {"first": v[0]["d"], "last": v[-1]["d"]} for k, v in parts.items()
        },
        "calibration_prefix_sha256": PREFIX,
        "expected_decision_ids_sha256": evaluation.canonical(ids),
        "expected_unmatured_ids_sha256": evaluation.canonical(pending),
        "expected_counts": {a: 194 for a in evaluation.ASSETS},
        "expected_ranges": evaluation.cohort_ranges(ids),
        "source_rows": len(observations),
        "maximum_source_date": days[-1],
    }
    return raw, provenance


def prepare(
    development_parent,
    observed_parent,
    calibrated_receipt,
    legacy_proof,
    frozen_inputs,
    frozen_inputs_sha256,
    output,
    dataset,
    kernel,
    legacy_bundle,
    legacy_selection,
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh evaluation staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit source before reading private evaluation inputs")
    contract, receipt = check_frozen_inputs(
        frozen_inputs, frozen_inputs_sha256, calibrated_receipt, legacy_proof
    )
    legacy_paths = {"bundle": legacy_bundle, "selection": legacy_selection}
    if {k: evaluation.sha(v) for k, v in legacy_paths.items()} != contract[
        "legacy_contract"
    ]["artifact_sha256"]:
        raise ValueError("Original legacy artifacts changed before source parsing")
    for identity in (dataset, kernel):
        if not isinstance(identity, str) or not re.fullmatch(
            r"[\w-]+/[\w-]+", identity
        ):
            raise ValueError("Explicit owner/slug required")
    raw, provenance = derive_source(development_parent, observed_parent)
    proof_raw = raw_json(provenance)
    release = {
        "schema": evaluation.SCHEMA,
        "decision_year": 2026,
        "execution_year": 2026,
        "target_year": 2026,
        "interval_alpha": 0.1,
        "weights_refitted": False,
        "policy_reselected": False,
        "retrospective_observed_2026": True,
        "comparators": ["selected", "ungated_same_mean", "zero"],
        **{k: v for k, v in provenance.items() if k != "schema"},
        "source_provenance_sha256": hashlib.sha256(proof_raw).hexdigest(),
        **{
            k: contract[k]
            for k in (
                "calibrated_bundle_sha256",
                "calibrated_receipt_sha256",
                "mean_policy_sha256",
                "production_source_sha256",
                "evaluator_source_sha256",
            )
        },
        **{
            k: receipt[k]
            for k in (
                "mean_model_state_sha256",
                "calibration_release_sha256",
                "calibration_source_sha256",
                "calibration_decision_ids_sha256",
            )
        },
        "calibration_source_provenance_sha256": receipt["source_provenance_sha256"],
        "matched_width_diagnostics": False,
        "legacy_comparator": contract["legacy_contract"],
    }
    names = sorted(evaluation.PRODUCTION_FILES) + [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/gated_evaluation.py",
        "src/models/gated_legacy.py",
        "src/prepare_gated_evaluation.py",
        "src/cloud/gated_evaluation_runner.py",
        "docs/gated-forecast-policy.json",
    ]
    files = {n: Path(n).read_bytes() for n in names}
    files.update(
        {
            "private/source.json": raw,
            "private/provenance.json": proof_raw,
            "private/release.json": raw_json(release),
            "private/frozen-inputs.json": Path(frozen_inputs).read_bytes(),
            "private/calibrated-receipt.json": Path(calibrated_receipt).read_bytes(),
            "private/legacy-proof.json": Path(legacy_proof).read_bytes(),
        }
    )
    files.update(
        {
            "private/legacy-bundle.joblib": Path(legacy_bundle).read_bytes(),
            "private/legacy-selection.json": Path(legacy_selection).read_bytes(),
        }
    )
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    rows = []
    for name, content in files.items():
        flat = name.replace("/", "__")
        (upload / flat).write_bytes(content)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {
        "schema": "gated_evaluation_transport_v1",
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": rows,
        "frozen_inputs_sha256": frozen_inputs_sha256,
        "release_sha256": hashlib.sha256(raw_json(release)).hexdigest(),
        "weights_refitted": False,
    }
    path = upload / "source-manifest.json"
    path.write_bytes(raw_json(manifest))
    runner = files["src/cloud/gated_evaluation_runner.py"].decode()
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if runner.count(token) != 1:
        raise ValueError("One explicit transport manifest pin required")
    (output / "run.py").write_text(
        runner.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(evaluation.sha(path))
        ),
        encoding="utf-8",
        newline="\n",
    )
    (upload / "dataset-metadata.json").write_bytes(
        raw_json(
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
        "kernel_sources": [contract["calibrated_kernel"], contract["legacy_kernel"]],
    }
    (output / "kernel-metadata.json").write_bytes(raw_json(metadata))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "development-parent",
        "observed-parent",
        "calibrated-receipt",
        "legacy-proof",
        "frozen-inputs",
        "frozen-inputs-sha256",
        "output",
        "dataset",
        "kernel",
        "legacy-bundle",
        "legacy-selection",
    ):
        parser.add_argument("--" + name, required=True)
    prepare(**vars(parser.parse_args()))
