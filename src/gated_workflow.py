"""Train and calibrate fresh own-run artifacts under the fixed research policy."""

import argparse
import hashlib
import json
from pathlib import Path

POLICY_SHA256 = "0f511814a9cd0e4f97626bdd8deafbe3d290b25b7829e210eeed6495c4339bd0"
PARENT_SHA256 = "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e"
REFERENCE_STATES = {
    "FXEURCAD": "a8acfaf6cd1dd1c271da81f3d0fabcb75d86a198b1785fe4f34f1921ebd555cf",
    "FXGBPCAD": "5282ee193242cd4b17beea43e081b599bb66573f7f93121b2be910d89e39ab39",
    "FXAUDCAD": "c13e6ab342dff479ca40f89bcabfc625299a4f4779147adeb3d599cb1d65499e",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fresh(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Fresh immutable output required")
    return path


def production(policy):
    if sha(policy) != POLICY_SHA256:
        raise ValueError("Reviewed fixed policy required")
    from src.models import gated_forecaster as backend

    backend.check_policy(policy)
    return backend


def check_mean_receipt(receipt, backend):
    expected = {
        "rows": 1933,
        "first_decision": "2017-03-29",
        "last_decision": "2024-12-27",
        "last_target_maturity": "2024-12-31",
    }
    if (
        receipt.get("stage") != "mean_frozen"
        or receipt.get("policy_sha256") != POLICY_SHA256
        or receipt.get("mean_source_sha256") != backend.DERIVED_SHA256
        or receipt.get("training") != {a: expected for a in REFERENCE_STATES}
        or receipt.get("calibration_accessed") is not False
        or receipt.get("test_accessed") is not False
    ):
        raise ValueError("Own complete pre2025 mean recipe/cohort receipt required")


def train(snapshot, policy, output, allow_different_state=False):
    output = fresh(output)
    backend = production(policy)
    # This entrypoint never receives or parses a parent containing 2025 values.
    raw = Path(snapshot).read_bytes()
    if hashlib.sha256(raw).hexdigest() != backend.DERIVED_SHA256:
        raise ValueError("Frozen pre2025 snapshot required")
    output.mkdir(parents=True)
    source = output / "mean-source.json"
    source.write_bytes(raw)
    receipt = backend.train(source, policy, output / "mean")
    actual = json.loads((output / "mean/receipt.json").read_bytes())
    if (
        actual != receipt
        or receipt.get("stage") != "mean_frozen"
        or receipt.get("bundle_sha256") != sha(output / "mean/bundle.joblib")
        or receipt.get("policy_sha256") != POLICY_SHA256
    ):
        raise ValueError("Fresh actual mean artifact identity differs")
    check_mean_receipt(receipt, backend)
    matches = receipt["mean_model_state_sha256"] == REFERENCE_STATES
    if not matches and not allow_different_state:
        raise ValueError(
            "Fresh numerical state differs from reference; no reference-score claim permitted"
        )
    freeze = {
        "schema": "fresh_gated_mean_freeze_v1",
        "policy_sha256": POLICY_SHA256,
        "parent_snapshot_sha256": PARENT_SHA256,
        "production_code_sha256": backend.sha("src/models/gated_forecaster.py"),
        "workflow_sha256": sha(__file__),
        "mean_source_sha256": sha(source),
        "mean_receipt_sha256": sha(output / "mean/receipt.json"),
        "mean_bundle_sha256": receipt["bundle_sha256"],
        "mean_model_state_sha256": receipt["mean_model_state_sha256"],
        "reference_numerical_state_equal": matches,
        "reported_accuracy_reused": False,
        "calibration_started": False,
        "test_accessed": False,
    }
    write(output / "mean-freeze.json", freeze)
    return freeze


def calibrate(run, parent, policy, freeze_sha256, output):
    output = fresh(output)
    backend = production(policy)
    run = Path(run)
    freeze_path = run / "mean-freeze.json"
    if sha(freeze_path) != freeze_sha256:
        raise ValueError("Explicit reviewed own mean freeze digest required")
    freeze = json.loads(freeze_path.read_bytes())
    receipt_path = run / "mean/receipt.json"
    if (
        freeze.get("schema") != "fresh_gated_mean_freeze_v1"
        or freeze.get("policy_sha256") != POLICY_SHA256
        or freeze.get("parent_snapshot_sha256") != PARENT_SHA256
        or freeze.get("mean_source_sha256") != backend.DERIVED_SHA256
        or sha(run / "mean-source.json") != backend.DERIVED_SHA256
        or freeze.get("production_code_sha256")
        != backend.sha("src/models/gated_forecaster.py")
        or freeze.get("workflow_sha256") != sha(__file__)
        or freeze.get("calibration_started") is not False
        or freeze.get("test_accessed") is not False
        or freeze.get("reported_accuracy_reused") is not False
        or sha(receipt_path) != freeze.get("mean_receipt_sha256")
        or sha(run / "mean/bundle.joblib") != freeze.get("mean_bundle_sha256")
    ):
        raise ValueError("Own frozen mean identities changed")
    receipt = json.loads(receipt_path.read_bytes())
    if (
        receipt.get("stage") != "mean_frozen"
        or receipt.get("mean_model_state_sha256")
        != freeze.get("mean_model_state_sha256")
        or receipt.get("bundle_sha256") != freeze.get("mean_bundle_sha256")
        or freeze.get("reference_numerical_state_equal")
        is not (receipt.get("mean_model_state_sha256") == REFERENCE_STATES)
    ):
        raise ValueError("Own frozen inherited mean state changed")
    check_mean_receipt(receipt, backend)
    # Complete calibration source is derived only after the own mean freeze check.
    from src.prepare_gated_calibration import cutoff_source

    raw, provenance = cutoff_source(parent)
    output.mkdir(parents=True)
    source = output / "calibration-source.json"
    source.write_bytes(raw)
    provenance_path = output / "calibration-source-provenance.json"
    write(provenance_path, provenance)
    release = {
        "schema": "causal_gate_calibration_v1",
        "decision_year": 2025,
        "execution_year": 2025,
        "maximum_target_maturity": "2025-12-31",
        "alpha": 0.1,
        "scale_floor": 1e-5,
        "mean_refit": False,
        "interval_policy_search": False,
        "mean_bundle_sha256": freeze["mean_bundle_sha256"],
        "mean_receipt_sha256": freeze["mean_receipt_sha256"],
        "mean_policy_sha256": POLICY_SHA256,
        "mean_model_state_sha256": freeze["mean_model_state_sha256"],
        "calibration_code_sha256": backend.sha("src/models/gated_forecaster.py"),
        "source_sha256": sha(source),
        "source_provenance_sha256": sha(provenance_path),
        "expected_decision_ids_sha256": provenance["expected_decision_ids_sha256"],
        "expected_counts": provenance["expected_counts"],
        "test_accessed": False,
    }
    release_path = output / "calibration-release.json"
    write(release_path, release)
    result = backend.calibrate(
        run / "mean", source, release_path, output / "calibrated"
    )
    actual = json.loads((output / "calibrated/receipt.json").read_bytes())
    if (
        result != actual
        or actual.get("stage") != "calibrated_frozen"
        or actual.get("mean_model_state_sha256") != freeze["mean_model_state_sha256"]
        or actual.get("bundle_sha256") != sha(output / "calibrated/bundle.joblib")
    ):
        raise ValueError("Fresh calibrated artifact or inherited state differs")
    summary = {
        "schema": "fresh_gated_calibration_freeze_v1",
        "mean_freeze_sha256": freeze_sha256,
        "receipt_sha256": sha(output / "calibrated/receipt.json"),
        "bundle_sha256": actual["bundle_sha256"],
        "policy_sha256": POLICY_SHA256,
        "calibration_release_sha256": sha(release_path),
        "reference_numerical_state_equal": freeze["reference_numerical_state_equal"],
        "reported_accuracy_reused": False,
        "test_accessed": False,
    }
    write(output / "calibration-freeze.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    for name in ["snapshot", "policy", "output"]:
        t.add_argument("--" + name, required=True)
    t.add_argument("--allow-different-state", action="store_true")
    c = sub.add_parser("calibrate")
    for name in ["run", "parent", "policy", "freeze-sha256", "output"]:
        c.add_argument("--" + name, required=True)
    f = sub.add_parser("forecast")
    for name in [
        "model",
        "snapshot",
        "policy",
        "policy-sha256",
        "receipt-sha256",
        "output",
    ]:
        f.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "forecast":
        from src.models.gated_predict import predict

        result = predict(**args)
    else:
        result = {"train": train, "calibrate": calibrate}[action](**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
