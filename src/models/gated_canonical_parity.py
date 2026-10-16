"""Compare frozen selected forecasts after canonical input migration, without labels."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ASSETS = ["FXEURCAD", "FXGBPCAD", "FXAUDCAD"]
COLUMNS = [
    "asset",
    "date",
    "execution_date",
    "target_date",
    "selected",
    "selected_lower",
    "selected_upper",
]
REFERENCE_SHA = "5741866852de214bf3310cd1b1a4314b1f29dcbaf31b39070cb5e496fd94814e"
REFERENCE_IDS_SHA = "f172833348eaff0f96599cf578535a4d86a917d67f9f165063a43770a4ea3f55"
OLD_EXPORT_SHA = "ae7445deb3f74da97052b94cdb8e9fd90a15b4d0a72b42bd666a20b31ee3143e"
OLD_RESULT_SHA = "4d51d5ad63d69eb1cb06b01669b85b7bcddcf360e3bb37ebf9d63300621316e6"
SOURCE_SHA = "c64e0805e00a265135f122bf1df9c74091918e2f8a3567e821cbf3c96935b7e4"
POLICY_SHA = "0f511814a9cd0e4f97626bdd8deafbe3d290b25b7829e210eeed6495c4339bd0"
MEAN_SOURCE_SHA = "cff4c3aa15c528f308de6b7916b875ea8d60fef4c501f1bf52815f67f9e5f85e"
CALIBRATION_SOURCE_SHA = (
    "6f0439d2d3f323ad7aecd21803eb3a27d9a8ac042e277f46db2701cac0058c36"
)
FIXED = {
    "coldstart_proof_sha256": "28ef518d9166b28edb8533941985c10cfef596faf77045218f01883f4cdf4dc4",
    "mean_receipt_sha256": "d1401dae76b532d0e05ada83c529df57e32436ba53cbdae70eacd56f9d65e3a3",
    "mean_bundle_sha256": "5a4d6575d86f7e42e300c7e9421346884d27b2b4a6495bb7ba8401f687224bc8",
    "mean_freeze_sha256": "51acb3b56802b57a4b2a3b365e0de25b0b4cf8a11245438a977a18df2cccde3e",
    "calibrated_receipt_sha256": "5506bc773682b8868f4e4934ff3eaf189deea8bb9aeb270525c7c8f7d17b5875",
    "calibrated_bundle_sha256": "ca28597f9ff5c6023ed6c945bc23cc5434cc47d3817c68c11826f02f04aca60e",
    "calibration_freeze_sha256": "ffdeaf3f952ee6cf32b61d664db1a9d016dd9396f38eb3dbdbb35c1df7c29bc6",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool):
        raise ValueError("Numeric values cannot be booleans")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Finite forecast values required")
    return value


def release_contract(plan):
    expected = {
        "schema": "canonical_gated_forecast_parity_v1",
        "policy_sha256": POLICY_SHA,
        "mean_source_sha256": MEAN_SOURCE_SHA,
        "calibration_source_sha256": CALIBRATION_SOURCE_SHA,
        "snapshot_sha256": SOURCE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "reference_decision_ids_sha256": REFERENCE_IDS_SHA,
        "expected_forecast_rows": 588,
        "expected_comparison_rows": 582,
        "expected_pending_rows": 6,
        "fit_allowed": False,
        "accuracy_allowed": False,
        "policy_selection_allowed": False,
        "artifact_freeze_reviewed": True,
        **FIXED,
    }
    if any(
        type(plan.get(k)) is not type(v) or plan.get(k) != v
        for k, v in expected.items()
    ):
        raise ValueError(
            "Explicit reviewed new-artifact targetless parity release required"
        )
    return plan


def source_ids(observations):
    days = [r["d"] for r in observations]
    if (
        len(days) != 196
        or days != sorted(set(days))
        or days[0] != "2026-01-02"
        or days[-1] != "2026-10-14"
        or any(not d.startswith("2026-") for d in days)
    ):
        raise ValueError("Exact complete observed196-publication source required")
    full = [
        (
            a,
            day,
            days[i + 1] if i + 1 < len(days) else None,
            days[i + 2] if i + 2 < len(days) else None,
        )
        for a in ASSETS
        for i, day in enumerate(days)
    ]
    mature = [r for r in full if r[3] is not None]
    pending = [r for r in full if r[3] is None]
    if (
        len(full) != 588
        or len(mature) != 582
        or len(pending) != 6
        or canonical(mature) != REFERENCE_IDS_SHA
    ):
        raise ValueError("Complete date-only forecast IDs differ from frozen reference")
    return full, mature, pending


def reference_rows(path, ids):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Exact targetless seven-column reference required")
        rows = list(reader)
    actual = [tuple(r[k] for k in COLUMNS[:4]) for r in rows]
    if actual != ids or len(set(actual)) != len(ids):
        raise ValueError("Exact full582 reference ID order required")
    for row in rows:
        for k in COLUMNS[4:]:
            row[k] = finite(row[k])
        if not row["selected_lower"] <= row["selected"] <= row["selected_upper"]:
            raise ValueError("Ordered original intervals required")
    return rows


def compare(rows, reference):
    keys = COLUMNS[:4]
    if [tuple(r[k] for k in keys) for r in rows] != [
        tuple(r[k] for k in keys) for r in reference
    ]:
        raise ValueError("Full selected parity cohort differs")
    maximum = {
        k: max(abs(finite(a[k]) - finite(b[k])) for a, b in zip(rows, reference))
        for k in COLUMNS[4:]
    }
    if any(v > 1e-14 for v in maximum.values()):
        raise ValueError("Canonical input changed selected point or interval forecasts")
    return maximum


def check_calibration(calibration):
    if set(calibration) != set(ASSETS) or any(
        item.get("rows") != 247
        or item.get("first_decision") != "2025-01-02"
        or item.get("last_decision") != "2025-12-29"
        or item.get("last_target_maturity") != "2025-12-31"
        or finite(item.get("normalized_quantile")) < 0
        for item in calibration.values()
    ):
        raise ValueError(
            "Complete finite full741 calibration required before future parsing"
        )


def check_saved_forecasts(path, ids, reference):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Exact saved seven-column forecasts required")
        rows = list(reader)
    for row in rows:
        for key in ["execution_date", "target_date"]:
            row[key] = row[key] or None
        for key in COLUMNS[4:]:
            row[key] = finite(row[key])
        if not row["selected_lower"] <= row["selected"] <= row["selected_upper"]:
            raise ValueError("Saved intervals must remain ordered")
    if [tuple(r[k] for k in COLUMNS[:4]) for r in rows] != ids:
        raise ValueError("Saved full588 IDs or pending semantics changed")
    return compare([r for r in rows if r["target_date"] is not None], reference)


def check_loaded_state(bundle, receipt, production, states):
    if production.model_state_fingerprints(bundle["models"]) != states or any(
        bundle["quantiles"].get(a) != receipt["calibration"][a]["normalized_quantile"]
        for a in ASSETS
    ):
        raise ValueError("Postflight learned states or calibrated quantiles changed")


def authorize(
    snapshot,
    reference,
    reference_receipt,
    model,
    mean_model,
    policy,
    cold_proof,
    mean_freeze,
    calibration_freeze,
    release,
    release_sha256,
):
    if sha(release) != release_sha256:
        raise ValueError(
            "Reviewed release SHA required before any future parse or model loading"
        )
    plan = release_contract(read(release))
    paths = {
        "snapshot_sha256": snapshot,
        "reference_sha256": reference,
        "reference_receipt_sha256": reference_receipt,
        "policy_sha256": policy,
        "coldstart_proof_sha256": cold_proof,
        "mean_freeze_sha256": mean_freeze,
        "calibration_freeze_sha256": calibration_freeze,
        "mean_receipt_sha256": Path(mean_model) / "receipt.json",
        "mean_bundle_sha256": Path(mean_model) / "bundle.joblib",
        "calibrated_receipt_sha256": Path(model) / "receipt.json",
        "calibrated_bundle_sha256": Path(model) / "bundle.joblib",
        "parity_code_sha256": __file__,
    }
    if any(sha(p) != plan.get(k) for k, p in paths.items()):
        raise ValueError("Frozen source/reference/artifact/code identity changed")
    from src.models import gated_forecaster as production
    from src import gated_workflow

    frozen = production.check_policy(policy)
    if frozen["production_source_sha256"] != plan.get("production_source_sha256"):
        raise ValueError("Current production source registry differs")
    old = read(reference_receipt)
    if (
        old.get("schema") != "frozen_selected_forecast_parity_reference_v1"
        or old.get("reference_sha256") != REFERENCE_SHA
        or old.get("original_prediction_export_sha256") != OLD_EXPORT_SHA
        or old.get("original_result_sha256") != OLD_RESULT_SHA
        or old.get("decision_ids_sha256") != REFERENCE_IDS_SHA
        or old.get("rows") != 582
        or old.get("columns") != COLUMNS
        or old.get("targets_present") is not False
        or old.get("evaluation_metrics_computed") is not False
        or old.get("policy_selection_authorized") is not False
    ):
        raise ValueError(
            "Authenticated targetless original reference provenance required"
        )
    cold = read(cold_proof)
    mf = read(mean_freeze)
    cf = read(calibration_freeze)
    mr = read(Path(mean_model) / "receipt.json")
    cr = read(Path(model) / "receipt.json")
    gated_workflow.check_mean_receipt(mr, production)
    check_calibration(cr.get("calibration", {}))
    states = gated_workflow.REFERENCE_STATES
    if (
        cold.get("schema") != "public_gated_canonical_coldstart_integration_v1"
        or cold.get("mean_source_sha256") != MEAN_SOURCE_SHA
        or cold.get("actual_fresh_mean_fits") != 3
        or cold.get("matured_training_rows") != 5799
        or cold.get("calibration_rows") != 741
        or cold.get("pre2026_mean_forecast_rows") != 747
        or finite(cold.get("full747_representation_max_abs_difference")) > 1e-14
        or cold.get("reference_numerical_state_equal") is not True
        or cold.get("test_accessed") is not False
        or cold.get("accuracy_evaluated") is not False
        or cold.get("original_model_artifacts_mounted") is not False
        or any(x.get("mean_model_state_sha256") != states for x in [cold, mf, mr, cr])
    ):
        raise ValueError(
            "Actual complete canonical cold-start freeze and unchanged states required"
        )
    if (
        mf.get("policy_sha256") != POLICY_SHA
        or mf.get("workflow_sha256") != sha(gated_workflow.__file__)
        or mf.get("production_code_sha256") != sha(production.__file__)
        or mf.get("reported_accuracy_reused") is not False
        or mf.get("test_accessed") is not False
        or mf.get("mean_source_sha256") != MEAN_SOURCE_SHA
        or mf.get("mean_receipt_sha256") != FIXED["mean_receipt_sha256"]
        or mf.get("mean_bundle_sha256") != FIXED["mean_bundle_sha256"]
        or mf.get("calibration_started") is not False
        or cf.get("receipt_sha256") != FIXED["calibrated_receipt_sha256"]
        or cf.get("bundle_sha256") != FIXED["calibrated_bundle_sha256"]
        or cf.get("mean_freeze_sha256") != FIXED["mean_freeze_sha256"]
        or cr.get("stage") != "calibrated_frozen"
        or cr.get("policy_sha256") != POLICY_SHA
        or cr.get("calibration_source_sha256") != CALIBRATION_SOURCE_SHA
        or cr.get("calibration_decision_ids_sha256")
        != "caf7c827abd65d7ffe689bb174c2f37afaa6adc10f5d9f70d5146bee80f71ac3"
        or cr.get("mean_refitted") is not False
        or cr.get("test_accessed") is not False
        or cr.get("calibration") != cold.get("calibration")
    ):
        raise ValueError("Actual own mean/calibration freeze inheritance changed")
    return plan, cr


def run(
    snapshot,
    reference,
    reference_receipt,
    model,
    mean_model,
    policy,
    cold_proof,
    mean_freeze,
    calibration_freeze,
    release,
    release_sha256,
    output,
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh immutable parity output required")
    plan, receipt = authorize(
        snapshot,
        reference,
        reference_receipt,
        model,
        mean_model,
        policy,
        cold_proof,
        mean_freeze,
        calibration_freeze,
        release,
        release_sha256,
    )
    # Explicit root-reviewed artifact release precedes all future parsing and unpickling.
    observations = read(snapshot)["observations"]
    full, ids, pending = source_ids(observations)
    original = reference_rows(reference, ids)
    from src.models import gated_forecaster as production
    from src.data.multi_asset import parse_panel

    bundle, loaded = production.load_bundle(model)
    if (
        loaded != receipt
        or bundle.get("policy_sha256") != POLICY_SHA
        or bundle.get("source_sha256") != plan["production_source_sha256"]
        or bundle.get("mean_source_sha256") != MEAN_SOURCE_SHA
        or bundle["context"].date.max().strftime("%Y-%m-%d") != "2025-12-31"
    ):
        raise ValueError("Loaded canonical calibrated model identity changed")
    if any(
        bundle["quantiles"][a] != receipt["calibration"][a]["normalized_quantile"]
        for a in ASSETS
    ):
        raise ValueError("Frozen calibrated quantiles changed")
    panel = parse_panel(Path(snapshot).read_bytes())

    def forecast_rows(own):
        frame = production.forecast(own, panel.copy())
        lookup = {}
        for r in frame.to_dict("records"):
            key = (r["asset"], r["date"].strftime("%Y-%m-%d"))
            if key in lookup:
                raise ValueError("Duplicate actual forecast ID")
            lookup[key] = r
        if set(lookup) != {(a, d) for a, d, _, _ in full}:
            raise ValueError("All588 forecast decisions required")
        result = []
        for a, d, e, t in full:
            r = lookup[(a, d)]
            lo, point, hi = [finite(r[k]) for k in ["lower", "prediction", "upper"]]
            if not lo <= point <= hi:
                raise ValueError("Ordered finite actual intervals required")
            result.append(dict(zip(COLUMNS, [a, d, e, t, point, lo, hi])))
        return result

    actual = forecast_rows(bundle)
    mature = [r for r in actual if r["target_date"] is not None]
    maximum = compare(mature, original)
    again, again_receipt = production.load_bundle(model)
    if again_receipt != receipt:
        raise ValueError("Reloaded artifact receipt changed")
    replay = forecast_rows(again)
    reload_delta = compare(actual, replay)
    # Reauthenticate every immutable input and the current scientific code after execution.
    authorize(
        snapshot,
        reference,
        reference_receipt,
        model,
        mean_model,
        policy,
        cold_proof,
        mean_freeze,
        calibration_freeze,
        release,
        release_sha256,
    )
    from src import gated_workflow

    check_loaded_state(bundle, receipt, production, gated_workflow.REFERENCE_STATES)
    check_loaded_state(again, receipt, production, gated_workflow.REFERENCE_STATES)
    output.mkdir(parents=True)
    path = output / "forecasts.csv"
    with path.open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(actual)
    saved_maximum = check_saved_forecasts(path, full, original)
    proof = {
        "schema": "canonical_selected_forecast_parity_v1",
        "release_sha256": release_sha256,
        "policy_sha256": POLICY_SHA,
        "source_sha256": SOURCE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "reference_original_result_sha256": OLD_RESULT_SHA,
        "comparison_rows": len(mature),
        "forecast_rows": len(actual),
        "pending_rows": len(pending),
        "decision_ids_sha256": canonical(ids),
        "max_abs_difference": maximum,
        "reload_max_abs_difference": reload_delta,
        "saved_forecast_max_abs_difference": saved_maximum,
        "forecasts_sha256": sha(path),
        "calibrated_bundle_sha256": FIXED["calibrated_bundle_sha256"],
        "coldstart_proof_sha256": FIXED["coldstart_proof_sha256"],
        "parity_code_sha256": sha(__file__),
        "weights_refitted": False,
        "policy_reselected": False,
        "accuracy_evaluated": False,
        "targets_used_for_comparison": False,
        "score_carryover_authorized": False,
    }
    (output / "parity.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in [
        "snapshot",
        "reference",
        "reference-receipt",
        "model",
        "mean-model",
        "policy",
        "cold-proof",
        "mean-freeze",
        "calibration-freeze",
        "release",
        "release-sha256",
        "output",
    ]:
        p.add_argument("--" + n, required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
