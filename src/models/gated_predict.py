"""Forecast from an authenticated own-run gated checkpoint and frozen policy."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _production():
    from src.models import gated_forecaster

    return gated_forecaster


def _parse_panel(content):
    from src.data.multi_asset import parse_panel

    return parse_panel(content)


def predict(model, snapshot, policy, policy_sha256, receipt_sha256, output):
    """Verify identities before deserialization; write fresh forecasts and provenance.

    Digests must come from the reviewed own-run artifact receipt. Hash checks
    establish artifact consistency; they do not make arbitrary pickle files safe.
    """
    model, snapshot, policy, output = map(Path, (model, snapshot, policy, output))
    receipt_path = model / "receipt.json"
    bundle_path = model / "bundle.joblib"
    provenance_path = output.with_name(output.name + ".provenance.json")
    protected = [snapshot, policy, receipt_path, bundle_path, Path(__file__)]
    for destination in (output, provenance_path):
        resolved = destination.resolve()
        if (
            destination.exists()
            or destination.is_symlink()
            or resolved in [path.resolve() for path in protected]
            or resolved.is_relative_to(model.resolve())
        ):
            raise ValueError("Fresh output outside the immutable model required")
    if sha(policy) != policy_sha256 or sha(receipt_path) != receipt_sha256:
        raise ValueError("Reviewed policy and receipt digests required")
    production = _production()
    frozen = production.check_policy(policy)
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if (
        receipt.get("stage") not in {"mean_frozen", "calibrated_frozen"}
        or receipt.get("policy_sha256") != policy_sha256
        or receipt.get("bundle_sha256") != sha(bundle_path)
    ):
        raise ValueError("Checkpoint receipt does not match the frozen policy")
    before = {str(path.resolve()): sha(path) for path in protected}
    bundle, loaded_receipt = production.load_bundle(model)
    if (
        loaded_receipt != receipt
        or bundle.get("policy_sha256") != policy_sha256
        or bundle.get("source_sha256") != frozen["production_source_sha256"]
        or (bundle.get("quantiles") is None) != (receipt["stage"] == "mean_frozen")
    ):
        raise ValueError("Loaded checkpoint policy, source or calibration changed")
    result = production.forecast(bundle, _parse_panel(snapshot.read_bytes()))
    production.check_policy(policy)
    if any(sha(path) != before[str(path.resolve())] for path in protected):
        raise ValueError("Authenticated inputs changed during inference")
    csv = result.to_csv(index=False, lineterminator="\n")
    provenance = {
        "schema": "authenticated_gated_forecast_v1",
        "rows": len(result),
        "stage": receipt["stage"],
        "policy_sha256": policy_sha256,
        "receipt_sha256": receipt_sha256,
        "bundle_sha256": receipt["bundle_sha256"],
        "input_sha256": before[str(snapshot.resolve())],
        "wrapper_sha256": before[str(Path(__file__).resolve())],
        "production_source_sha256": frozen["production_source_sha256"],
        "mean_model_state_sha256": receipt["mean_model_state_sha256"],
        "predictions_sha256": hashlib.sha256(csv.encode("utf8")).hexdigest(),
        "accuracy_evaluated": False,
    }
    serialized = json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects immutable artifacts and concurrent writers.
    with provenance_path.open("x", encoding="utf8", newline="") as handle:
        handle.write(serialized)
    with output.open("x", encoding="utf8", newline="") as handle:
        handle.write(csv)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "snapshot",
        "policy",
        "policy-sha256",
        "receipt-sha256",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(predict(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
