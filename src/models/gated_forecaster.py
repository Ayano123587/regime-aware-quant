"""Fixed causal ridge forecasts with frozen volatility abstention and intervals."""

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from src.models.interaction_study import ASSETS
from src.models.regime_interactions import BASE_FEATURES, fit, number
from src.models.volatility_gate import sha
from src.models.volatility_gate_inference import predict_records, validate_estimator

VERSION = "causal_gate_forecast_v1"
MEAN_END = "2024-12-31"
DERIVED_SHA256 = "cff4c3aa15c528f308de6b7916b875ea8d60fef4c501f1bf52815f67f9e5f85e"
MEAN_SERIALIZATION = {
    "container": "observations",
    "sort_keys": True,
    "separators": [",", ":"],
    "ensure_ascii": True,
    "allow_nan": False,
    "terminal_newline": "LF",
}


def canonical_mean_bytes(observations):
    """Stable past-only representation; retain list order and exact source values."""
    days = [row["d"] for row in observations]
    if not days or days != sorted(set(days)) or days[-1] > MEAN_END:
        raise ValueError("Unique chronological pre2025 mean publications required")
    return (
        json.dumps(
            {"observations": observations},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf8")


SELECTION_SHA256 = "af2adb17f4c11a30e8db26f48b383a922c61762da7e36212b563f136a5bce7e1"
PARITY_SHA256 = "1bb28c92e2c0cc0422a15f99745b9dc0dd99e78317453c11e027b95088561032"
FREEZE_SHA256 = "bc7c90f3dbac0e08f7d46edf25f1bfe3952c9a06691f25e176ad7f5944ae427c"


def matured_training_rows(records):
    """Every training label must already have matured inside the mean period."""
    result = []
    last = None
    for row in records:
        decision = date.fromisoformat(row["date"])
        if last is not None and decision <= last:
            raise ValueError("Unique chronological training decisions required")
        last = decision
        if row["date"] > MEAN_END:
            raise ValueError("Post-2024 mean training observation forbidden")
        if row["target_date"] is None:
            continue
        execution = date.fromisoformat(row["execution_date"])
        maturity = date.fromisoformat(row["target_date"])
        if not decision < execution < maturity:
            raise ValueError("Delayed label chronology required")
        if row["target_date"] > MEAN_END:
            raise ValueError("Future target cannot enter final mean training")
        number(row["target"])
        for name in BASE_FEATURES:
            number(row[name])
        result.append(row)
    if not result:
        raise ValueError("Matured training support required")
    return result


def validate_training_cohort(rows, observations, asset):
    """Audit feature warmup and every execution/maturity/target against source."""
    expected = observations[60:-2]
    if [r["date"] for r in rows] != [r["d"] for r in expected]:
        raise ValueError("Exact all-source matured feature coverage required")
    for i, row in enumerate(rows, start=60):
        if (
            row["execution_date"] != observations[i + 1]["d"]
            or row["target_date"] != observations[i + 2]["d"]
        ):
            raise ValueError("Final training execution or maturity identity changed")
        target = math.log(float(observations[i + 2][asset]["v"])) - math.log(
            float(observations[i + 1][asset]["v"])
        )
        if abs(number(row["target"]) - target) > 1e-14:
            raise ValueError("Final training source-derived target changed")
    return rows


def quantile90(values):
    """Fixed finite-sample order statistic; no interpolation or width selection."""
    if not values:
        raise ValueError("Nonempty interval calibration support required")
    values = sorted(number(value) for value in values)
    if values[0] < 0:
        raise ValueError("Nonnegative calibration residuals required")
    rank = math.ceil((len(values) + 1) * 0.9)
    if rank > len(values):
        raise ValueError("Insufficient interval calibration support")
    return values[rank - 1]


def check_policy(path):
    policy = json.loads(Path(path).read_text(encoding="utf8"))
    expected = {
        "schema": VERSION,
        "assets": ASSETS,
        "features": BASE_FEATURES,
        "training_source_sha256": DERIVED_SHA256,
        "mean_source_serialization": MEAN_SERIALIZATION,
        "final_mean_target_end": MEAN_END,
        "training_rows_per_asset": 1933,
        "alpha": 10000,
        "fit_intercept": False,
        "scaler": "StandardScaler, fitted per asset on matured training rows only",
        "gate_threshold": 1.2,
        "gate_multiplier": 2,
        "gate_comparison": "strict_greater_than",
        "selection_sha256": SELECTION_SHA256,
        "saved_fold_parity_sha256": PARITY_SHA256,
        "development_freeze_sha256": FREEZE_SHA256,
        "interval_alpha": 0.1,
        "interval_scale_floor": 1e-5,
        "calibration_decision_year": 2025,
        "calibration_maximum_target_maturity": "2025-12-31",
        "test_interpretation": "Already observed 2026 snapshot; retrospective, no policy selection",
    }
    if not isinstance(policy, dict) or json.dumps(
        {k: policy.get(k) for k in expected}, sort_keys=True, allow_nan=False
    ) != json.dumps(expected, sort_keys=True):
        raise ValueError("Frozen production policy changed")
    keys = {
        "src/models/gated_forecaster.py",
        "src/models/volatility_gate_inference.py",
        "src/models/regime_interactions.py",
        "src/models/interaction_study.py",
        "src/models/volatility_gate.py",
        "src/data/multi_asset.py",
        "src/features/multi_asset.py",
    }
    sources = policy.get("production_source_sha256")
    root = Path(__file__).resolve().parents[2]
    if (
        not isinstance(sources, dict)
        or set(sources) != keys
        or any(sha(root / p) != v for p, v in sources.items())
    ):
        raise ValueError("Exact frozen production source bytes required")
    return policy


def validate_bundle(bundle):
    if (
        not isinstance(bundle, dict)
        or bundle.get("version") != VERSION
        or bundle.get("assets") != ASSETS
        or bundle.get("features") != BASE_FEATURES
        or bundle.get("mean_source_sha256") != DERIVED_SHA256
        or bundle.get("threshold") != 1.2
        or type(bundle.get("multiplier")) is not int
        or bundle["multiplier"] != 2
        or set(bundle.get("models", {})) != set(ASSETS)
    ):
        raise ValueError("Frozen gated model bundle required")
    for model in bundle["models"].values():
        validate_estimator(model)
    quantiles = bundle.get("quantiles")
    if quantiles is not None:
        if not isinstance(quantiles, dict) or set(quantiles) != set(ASSETS):
            raise ValueError("Complete per-asset calibration required")
        for value in quantiles.values():
            if number(value) < 0:
                raise ValueError("Nonnegative interval quantile required")
    return bundle


def model_state_fingerprints(models):
    """Canonical learned numerical state, independent of pickle container bytes."""
    if not isinstance(models, dict) or set(models) != set(ASSETS):
        raise ValueError("Three inherited mean estimators required")

    def finite(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        return number(value)

    result = {}
    for asset, model in models.items():
        validate_estimator(model)
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        state = {
            "features": BASE_FEATURES,
            "alpha": number(ridge.alpha),
            "fit_intercept": ridge.fit_intercept,
            "scaler_class": type(scaler).__module__ + "." + type(scaler).__qualname__,
            "ridge_class": type(ridge).__module__ + "." + type(ridge).__qualname__,
            "scaler_options": {
                "with_mean": scaler.with_mean,
                "with_std": scaler.with_std,
            },
            "scaler": {
                k: finite(getattr(scaler, k))
                for k in [
                    "mean_",
                    "scale_",
                    "var_",
                    "n_samples_seen_",
                    "n_features_in_",
                ]
            },
            "ridge": {
                k: finite(getattr(ridge, k))
                for k in ["coef_", "intercept_", "n_features_in_"]
            },
        }
        result[asset] = hashlib.sha256(
            json.dumps(
                state, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    return result


def calibration_source_cohort(observations):
    """Complete eligible IDs from source chronology, independently of features."""
    days = [r["d"] for r in observations]
    if days != sorted(set(days)) or not days or days[-1] > "2025-12-31":
        raise ValueError("Unique pre-2026 calibration publications required")
    ids = [
        (asset, days[i], days[i + 1], days[i + 2])
        for asset in ASSETS
        for i in range(len(days) - 2)
        if days[i].startswith("2025-")
        and days[i + 1].startswith("2025-")
        and days[i + 2].startswith("2025-")
    ]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return ids, digest, {asset: sum(row[0] == asset for row in ids) for asset in ASSETS}


def frame_records(frame):
    import pandas as pd

    frame = frame.copy()
    for key in ["date", "execution_date", "target_date"]:
        frame[key] = frame[key].map(
            lambda value: None if pd.isna(value) else value.strftime("%Y-%m-%d")
        )
    return frame.to_dict("records")


def save_bundle(bundle, output, receipt):
    import joblib

    output = Path(output)
    if output.exists():
        raise ValueError("Preserve immutable model output")
    before = model_state_fingerprints(bundle["models"])
    if (
        "mean_model_state_sha256" in receipt
        and receipt["mean_model_state_sha256"] != before
    ):
        raise ValueError("Inherited mean model state changed before save")
    output.mkdir(parents=True)
    joblib.dump(bundle, output / "bundle.joblib")
    loaded = validate_bundle(joblib.load(output / "bundle.joblib"))
    after = model_state_fingerprints(loaded["models"])
    if before != after:
        raise ValueError("Mean model state changed after checkpoint reload")
    receipt = {
        **receipt,
        "mean_model_state_sha256": after,
        "bundle_sha256": sha(output / "bundle.joblib"),
        "version": VERSION,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return loaded, receipt


def load_bundle(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf8"))
    if receipt.get("version") != VERSION or receipt.get("bundle_sha256") != sha(
        directory / "bundle.joblib"
    ):
        raise ValueError("Saved own-run checkpoint hash changed")
    import joblib

    bundle = validate_bundle(joblib.load(directory / "bundle.joblib"))
    if model_state_fingerprints(bundle["models"]) != receipt.get(
        "mean_model_state_sha256"
    ):
        raise ValueError("Saved mean numerical state fingerprint changed")
    return bundle, receipt


def train(snapshot, policy_path, output):
    if Path(output).exists():
        raise ValueError("Preserve immutable model output")
    policy = check_policy(policy_path)
    if sha(snapshot) != DERIVED_SHA256:
        raise ValueError("Only frozen pre-2025 training source permitted")
    source = Path(snapshot).read_bytes()
    observations = json.loads(source)["observations"]
    if (
        len(observations) != 1995
        or observations[-1]["d"] != MEAN_END
        or any(r["d"] > MEAN_END for r in observations)
    ):
        raise ValueError("Complete frozen mean source required")
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import build_panel_features
    from threadpoolctl import threadpool_limits
    import joblib
    import tempfile

    panel = parse_panel(source)
    frames = build_panel_features(panel)
    rows = {a: matured_training_rows(frame_records(frames[a])) for a in ASSETS}
    for asset in ASSETS:
        validate_training_cohort(rows[asset], observations, asset)
    if any(len(v) != 1933 for v in rows.values()):
        raise ValueError("Exact final matured training cohort required")
    with threadpool_limits(limits=2):
        models = {a: fit(rows[a], 10000, False) for a in ASSETS}
    bundle = {
        "version": VERSION,
        "assets": ASSETS,
        "features": BASE_FEATURES,
        "mean_source_sha256": DERIVED_SHA256,
        "threshold": 1.2,
        "multiplier": 2,
        "models": models,
        "quantiles": None,
        "context": panel,
        "policy_sha256": sha(policy_path),
        "source_sha256": policy["production_source_sha256"],
    }
    # Prove actual final checkpoint serialization before creating the release output.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(bundle, path)
        loaded = validate_bundle(joblib.load(path))
        delta = max(
            abs(x - y)
            for a in ASSETS
            for x, y in zip(
                predict_records(models[a], rows[a]),
                predict_records(loaded["models"][a], rows[a]),
            )
        )
        if delta > 1e-14:
            raise ValueError("Final model checkpoint replay changed")
    receipt = {
        "stage": "mean_frozen",
        "policy_sha256": sha(policy_path),
        "mean_source_sha256": DERIVED_SHA256,
        "training": {
            a: {
                "rows": len(rows[a]),
                "first_decision": rows[a][0]["date"],
                "last_decision": rows[a][-1]["date"],
                "last_target_maturity": max(r["target_date"] for r in rows[a]),
            }
            for a in ASSETS
        },
        "final_model_replay_max_abs_difference": delta,
        "calibration_accessed": False,
        "test_accessed": False,
    }
    return save_bundle(bundle, output, receipt)[1]


def forecast(bundle, supplied):
    """Current observations only; supplied future labels are never read."""
    validate_bundle(bundle)
    import pandas as pd
    import numpy as np
    from src.data.multi_asset import validate_panel
    from src.features.multi_asset import build_panel_features

    supplied = validate_panel(supplied)
    context = validate_panel(bundle["context"])
    overlap = supplied.merge(context, on="date", suffixes=("_new", "_old"))
    if any(
        not np.allclose(overlap[a + "_new"], overlap[a + "_old"], atol=1e-12, rtol=0)
        for a in ASSETS
    ):
        raise ValueError("Supplied observations revise frozen context")
    panel = validate_panel(
        pd.concat([context, supplied])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    frames = build_panel_features(panel)
    output = []
    for asset in ASSETS:
        rows = frames[asset].loc[
            frames[asset].date.isin(supplied.date)
            & (frames[asset].date > context.date.max())
        ]
        if rows.empty:
            continue
        values = predict_records(bundle["models"][asset], frame_records(rows))
        part = pd.DataFrame({"date": rows.date, "asset": asset, "prediction": values})
        if bundle["quantiles"] is not None:
            widths = (
                np.maximum(rows.ewma_volatility.to_numpy(), 1e-5)
                * bundle["quantiles"][asset]
            )
            if not np.isfinite(widths).all():
                raise ValueError("Finite interval widths required")
            part["half_width"] = widths
            part["lower"] = part.prediction - widths
            part["upper"] = part.prediction + widths
        output.append(part)
    return (
        pd.concat(output, ignore_index=True)
        if output
        else pd.DataFrame(columns=["date", "asset", "prediction"])
    )


def check_calibration_release(plan, mean_receipt, snapshot, provenance):
    expected = {
        "schema": "causal_gate_calibration_v1",
        "decision_year": 2025,
        "execution_year": 2025,
        "maximum_target_maturity": "2025-12-31",
        "alpha": 0.1,
        "scale_floor": 1e-5,
        "mean_refit": False,
        "interval_policy_search": False,
    }
    if (
        not isinstance(plan, dict)
        or json.dumps(
            {k: plan.get(k) for k in expected}, sort_keys=True, allow_nan=False
        )
        != json.dumps(expected, sort_keys=True)
        or plan.get("mean_bundle_sha256") != mean_receipt["bundle_sha256"]
        or plan.get("mean_policy_sha256") != mean_receipt["policy_sha256"]
        or plan.get("source_sha256") != sha(snapshot)
    ):
        raise ValueError("Frozen post-mean calibration release required")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != "causal_gate_calibration_source_v1"
        or provenance.get("parent_snapshot_sha256")
        != "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e"
        or provenance.get("derived_source_sha256") != plan["source_sha256"]
        or provenance.get("mean_prefix_sha256") != DERIVED_SHA256
        or provenance.get("maximum_source_date") != "2025-12-31"
        or provenance.get("expected_decision_ids_sha256")
        != plan.get("expected_decision_ids_sha256")
        or provenance.get("expected_counts") != plan.get("expected_counts")
        or set(provenance.get("expected_counts", {})) != set(ASSETS)
        or any(
            type(n) is not int or n <= 0 for n in provenance["expected_counts"].values()
        )
    ):
        raise ValueError(
            "Complete independently derived calibration source provenance required"
        )
    return plan


def calibrate(model_directory, snapshot, release_path, output):
    """Separate explicit release after mean bytes are frozen; no model fitting."""
    if Path(output).exists():
        raise ValueError("Preserve immutable calibration output")
    bundle, mean_receipt = load_bundle(model_directory)
    if mean_receipt.get("stage") != "mean_frozen" or bundle["quantiles"] is not None:
        raise ValueError("Uncalibrated frozen mean checkpoint required")
    plan = json.loads(Path(release_path).read_text(encoding="utf8"))
    provenance_path = Path(release_path).parent / "calibration-source-provenance.json"
    if plan.get("source_provenance_sha256") != sha(provenance_path):
        raise ValueError("Frozen calibration provenance bytes changed")
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    check_calibration_release(plan, mean_receipt, snapshot, provenance)
    inherited = model_state_fingerprints(bundle["models"])
    if mean_receipt.get("mean_model_state_sha256") != inherited:
        raise ValueError("Frozen inherited mean coefficients/scaler state changed")
    content = Path(snapshot).read_bytes()
    observations = json.loads(content)["observations"]
    if any(row["d"] > "2025-12-31" for row in observations):
        raise ValueError("2026 observations cannot enter calibration")
    ids, cohort_sha, expected_counts = calibration_source_cohort(observations)
    prefix = [r for r in observations if r["d"] <= MEAN_END]
    prefix_sha = hashlib.sha256(canonical_mean_bytes(prefix)).hexdigest()
    if (
        prefix_sha != DERIVED_SHA256
        or cohort_sha != plan["expected_decision_ids_sha256"]
        or expected_counts != plan["expected_counts"]
        or len(observations) != provenance.get("source_rows")
    ):
        raise ValueError("Exact complete calibration source cohort changed")
    from src.data.multi_asset import parse_panel, validate_panel
    from src.features.multi_asset import build_panel_features
    import numpy as np
    import pandas as pd

    supplied = parse_panel(content)
    # forecast checks every overlapping saved context observation for revision.
    predicted = forecast(bundle, supplied)
    context = validate_panel(bundle["context"])
    panel = validate_panel(
        pd.concat([context, supplied])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if panel.date.max().strftime("%Y-%m-%d") != "2025-12-31":
        raise ValueError("Complete frozen 2025 publication history required")
    frames = build_panel_features(panel)
    quantiles = {}
    counts = {}
    actual_ids = []
    for asset in ASSETS:
        rows = frames[asset]
        rows = rows.loc[
            (rows.date.dt.year == 2025)
            & (rows.execution_date.dt.year == 2025)
            & (rows.target_date.dt.year == 2025)
            & rows.target.notna()
        ]
        joined = rows.merge(
            predicted.loc[predicted.asset == asset, ["date", "prediction"]],
            on="date",
            validate="one_to_one",
        )
        if len(joined) != len(rows) or not len(rows):
            raise ValueError("All matured 2025 calibration decisions required")
        actual_ids.extend(
            (asset, r["date"], r["execution_date"], r["target_date"])
            for r in frame_records(rows)
        )
        if len(rows) != expected_counts[asset]:
            raise ValueError(
                "Complete source-derived calibration decision coverage required"
            )
        residuals = (
            abs(joined.target.to_numpy() - joined.prediction.to_numpy())
            / np.maximum(joined.ewma_volatility.to_numpy(), 1e-5)
        ).tolist()
        quantiles[asset] = quantile90(residuals)
        counts[asset] = {
            "rows": len(rows),
            "first_decision": rows.date.min().strftime("%Y-%m-%d"),
            "last_decision": rows.date.max().strftime("%Y-%m-%d"),
            "last_target_maturity": rows.target_date.max().strftime("%Y-%m-%d"),
            "normalized_quantile": quantiles[asset],
        }
    if actual_ids != ids:
        raise ValueError("Exact calibration decision IDs changed")
    if model_state_fingerprints(bundle["models"]) != inherited:
        raise ValueError("Calibration changed mean coefficients or scaler state")
    calibrated = {**bundle, "quantiles": quantiles, "context": panel}
    receipt = {
        "stage": "calibrated_frozen",
        "mean_model_state_sha256": inherited,
        "calibration_decision_ids_sha256": cohort_sha,
        "source_provenance_sha256": sha(provenance_path),
        "mean_bundle_sha256": mean_receipt["bundle_sha256"],
        "policy_sha256": mean_receipt["policy_sha256"],
        "calibration_release_sha256": sha(release_path),
        "calibration_source_sha256": sha(snapshot),
        "calibration": counts,
        "mean_refitted": False,
        "test_accessed": False,
        "coverage_interpretation": "Descriptive under serial dependence; no exchangeability guarantee",
    }
    return save_bundle(calibrated, output, receipt)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "calibrate", "forecast"])
    parser.add_argument("--snapshot")
    parser.add_argument("--policy")
    parser.add_argument("--model")
    parser.add_argument("--release")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "train":
        result = train(args.snapshot, args.policy, args.output)
    elif args.action == "calibrate":
        result = calibrate(args.model, args.snapshot, args.release, args.output)
    else:
        from src.data.multi_asset import parse_panel

        model, _ = load_bundle(args.model)
        result = forecast(model, parse_panel(Path(args.snapshot).read_bytes()))
        path = Path(args.output)
        if path.exists():
            raise ValueError("Fresh forecast output required")
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False, lineterminator="\n")
        result = {"rows": len(result), "path": str(path)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
