"""Release-gated retrospective evaluation; no fitting, selection or source fetch."""

import argparse, hashlib, inspect, json, math, re
from datetime import date
from pathlib import Path

ASSETS = ["FXEURCAD", "FXGBPCAD", "FXAUDCAD"]
SCHEMA = "causal_gate_evaluation_v1"
PRODUCTION_FILES = {
    "src/models/gated_forecaster.py",
    "src/models/volatility_gate_inference.py",
    "src/models/regime_interactions.py",
    "src/models/interaction_study.py",
    "src/models/volatility_gate.py",
    "src/data/multi_asset.py",
    "src/features/multi_asset.py",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def number(value):
    if isinstance(value, bool):
        raise ValueError("Finite numerical value required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Finite numerical value required")
    return value


def source_cohort(observations):
    """Chronology-derived IDs, independent of feature rows and model predictions."""
    days = []
    for row in observations:
        d = row["d"]
        if (
            not isinstance(d, str)
            or date.fromisoformat(d).isoformat() != d
            or days
            and (
                d <= days[-1]
                or (date.fromisoformat(d) - date.fromisoformat(days[-1])).days > 7
            )
        ):
            raise ValueError("Unique complete publication chronology required")
        for asset in ASSETS:
            if number(row[asset]["v"]) <= 0:
                raise ValueError("Positive shared source prices required")
        days.append(d)
    if (
        not days
        or not days[-1].startswith("2026-")
        or any(d > "2026-12-31" for d in days)
    ):
        raise ValueError("Frozen observed2026 source required")
    ids = []
    pending = []
    for asset in ASSETS:
        for i, d in enumerate(days):
            if not d.startswith("2026-"):
                continue
            execution = days[i + 1] if i + 1 < len(days) else None
            maturity = days[i + 2] if i + 2 < len(days) else None
            if (
                execution is not None
                and maturity is not None
                and execution.startswith("2026-")
                and maturity.startswith("2026-")
            ):
                ids.append((asset, d, execution, maturity))
            else:
                pending.append((asset, d, execution, maturity))
    if not ids or any(sum(row[0] == a for row in pending) != 2 for a in ASSETS):
        raise ValueError("Two unmatured final decisions per asset required")
    return ids, pending, {a: sum(r[0] == a for r in ids) for a in ASSETS}


def calibration_prefix_sha256(observations):
    prefix = [row for row in observations if row["d"] <= "2025-12-31"]
    raw = (json.dumps({"observations": prefix}, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def cohort_ranges(ids):
    return {
        a: {
            "first_decision": min(r[1] for r in ids if r[0] == a),
            "last_decision": max(r[1] for r in ids if r[0] == a),
            "last_execution": max(r[2] for r in ids if r[0] == a),
            "last_target_maturity": max(r[3] for r in ids if r[0] == a),
        }
        for a in ASSETS
    }


def authorize(
    snapshot, provenance, calibrated, policy, release, release_sha256, legacy=None
):
    """Authenticate release before parsing future source or loading any checkpoint."""
    if (
        not isinstance(release_sha256, str)
        or len(release_sha256) != 64
        or sha(release) != release_sha256
    ):
        raise ValueError("Explicit exact evaluation release SHA256 required")
    plan = read(release)
    fixed = {
        "schema": SCHEMA,
        "decision_year": 2026,
        "execution_year": 2026,
        "target_year": 2026,
        "interval_alpha": 0.1,
        "weights_refitted": False,
        "policy_reselected": False,
        "retrospective_observed_2026": True,
        "comparators": ["selected", "ungated_same_mean", "zero"],
    }
    if any(type(plan.get(k)) is not type(v) or plan[k] != v for k, v in fixed.items()):
        raise ValueError("Frozen retrospective evaluation policy required")
    bindings = {
        "source_sha256": snapshot,
        "source_provenance_sha256": provenance,
        "calibrated_bundle_sha256": Path(calibrated) / "bundle.joblib",
        "calibrated_receipt_sha256": Path(calibrated) / "receipt.json",
        "mean_policy_sha256": policy,
        "evaluator_source_sha256": __file__,
    }
    if any(plan.get(k) != sha(path) for k, path in bindings.items()):
        raise ValueError("Evaluation release artifact/source identity changed")
    sources = plan.get("production_source_sha256")
    root = Path(__file__).resolve().parents[2]
    if (
        not isinstance(sources, dict)
        or set(sources) != PRODUCTION_FILES
        or any(sha(root / name) != digest for name, digest in sources.items())
    ):
        raise ValueError("Frozen production source identity changed")
    receipt = read(Path(calibrated) / "receipt.json")
    mean_policy = read(policy)
    proof = read(provenance)
    if (
        receipt.get("stage") != "calibrated_frozen"
        or receipt.get("mean_refitted") is not False
        or receipt.get("test_accessed") is not False
        or receipt.get("bundle_sha256") != plan["calibrated_bundle_sha256"]
        or receipt.get("policy_sha256") != plan["mean_policy_sha256"]
        or mean_policy.get("production_source_sha256") != sources
    ):
        raise ValueError("Authenticated calibrated checkpoint required")
    for key in (
        "mean_model_state_sha256",
        "calibration_release_sha256",
        "calibration_source_sha256",
        "calibration_decision_ids_sha256",
    ):
        if plan.get(key) != receipt.get(key) or not receipt.get(key):
            raise ValueError("Inherited mean/calibration identity changed")
    if plan.get("calibration_source_provenance_sha256") != receipt.get(
        "source_provenance_sha256"
    ):
        raise ValueError("Calibration provenance changed")
    expected = {
        "schema": "causal_gate_evaluation_source_v1",
        "source_sha256": plan["source_sha256"],
        "source_parent_sha256": plan.get("source_parent_sha256"),
        "source_parent_counts": plan.get("source_parent_counts"),
        "source_parent_ranges": plan.get("source_parent_ranges"),
        "calibration_prefix_sha256": receipt["calibration_source_sha256"],
        "expected_decision_ids_sha256": plan.get("expected_decision_ids_sha256"),
        "expected_unmatured_ids_sha256": plan.get("expected_unmatured_ids_sha256"),
        "expected_counts": plan.get("expected_counts"),
        "expected_ranges": plan.get("expected_ranges"),
        "source_rows": plan.get("source_rows"),
        "maximum_source_date": plan.get("maximum_source_date"),
    }
    if (
        proof != expected
        or not isinstance(plan.get("expected_counts"), dict)
        or set(plan["expected_counts"]) != set(ASSETS)
        or any(type(n) is not int or n <= 0 for n in plan["expected_counts"].values())
        or type(plan.get("matched_width_diagnostics")) is not bool
        or not isinstance(plan.get("expected_ranges"), dict)
        or set(plan["expected_ranges"]) != set(ASSETS)
    ):
        raise ValueError("Complete independent source provenance required")
    parents = plan.get("source_parent_sha256")
    parent_counts = plan.get("source_parent_counts")
    parent_ranges = plan.get("source_parent_ranges")
    if (
        "legacy_comparator" not in plan
        or not isinstance(parents, dict)
        or set(parents) != {"development", "observed2026"}
        or any(
            not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
            for v in parents.values()
        )
        or not isinstance(parent_counts, dict)
        or set(parent_counts) != set(parents)
        or any(type(n) is not int or n <= 0 for n in parent_counts.values())
        or not isinstance(parent_ranges, dict)
        or set(parent_ranges) != set(parents)
    ):
        raise ValueError("Explicit two-source provenance and legacy policy required")
    states = plan.get("mean_model_state_sha256")
    if (
        not isinstance(states, dict)
        or set(states) != set(ASSETS)
        or any(
            not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
            for v in states.values()
        )
    ):
        raise ValueError("Complete inherited mean-state identities required")
    contract = plan.get("legacy_comparator")
    if (contract is None) != (legacy is None):
        raise ValueError("Explicit legacy contract/callback correspondence required")
    if legacy is not None:
        if not isinstance(contract, dict):
            raise ValueError("Separate explicit legacy contract required")
        if set(legacy) != {"callback", "source", "artifacts", "proof"} or not callable(
            legacy["callback"]
        ):
            raise ValueError("Separate authenticated legacy adapter required")
        callback = legacy["callback"]
        source = inspect.getsourcefile(callback)
        if (
            source is None
            or Path(source).resolve() != Path(legacy["source"]).resolve()
            or sha(source) != contract.get("callback_source_sha256")
            or callback.__module__ + "." + callback.__qualname__
            != contract.get("callback_qualified_name")
            or sha(legacy["proof"]) != contract.get("proof_sha256")
        ):
            raise ValueError("Legacy callback/proof identity changed")
        artifact_hashes = {
            name: sha(path) for name, path in legacy["artifacts"].items()
        }
        if not artifact_hashes or artifact_hashes != contract.get("artifact_sha256"):
            raise ValueError("Legacy artifact identity changed")
        lp = read(legacy["proof"])
        if (
            lp.get("schema") != "authenticated_legacy_forecast_v1"
            or lp.get("weights_refitted") is not False
            or lp.get("artifact_sha256") != artifact_hashes
            or lp.get("model_state_sha256") != contract.get("model_state_sha256")
            or lp.get("calibration_identity") != contract.get("calibration_identity")
            or not lp.get("model_state_sha256")
            or not lp.get("calibration_identity")
        ):
            raise ValueError("Independent legacy model/calibration proof required")
    return plan, receipt


def regression(target, prediction):
    if len(target) != len(prediction) or not target:
        raise ValueError("Complete prediction cohort required")
    squared = [(number(y) - number(p)) ** 2 for y, p in zip(target, prediction)]
    value = math.sqrt(math.fsum(squared) / len(squared))
    return {"rows": len(target), "rmse": number(value)}


def intervals(target, lower, upper, alpha=0.1):
    if not len(target) == len(lower) == len(upper) or not target:
        raise ValueError("Complete interval cohort required")
    covered = []
    widths = []
    scores = []
    for y, lo, hi in zip(target, lower, upper):
        y, lo, hi = number(y), number(lo), number(hi)
        if lo > hi:
            raise ValueError("Ordered finite bounds required")
        width = hi - lo
        score = width + 2 / alpha * (lo - y if y < lo else y - hi if y > hi else 0.0)
        widths.append(number(width))
        scores.append(number(score))
        covered.append(lo <= y <= hi)
    return {
        "rows": len(target),
        "nominal_coverage": 1 - alpha,
        "coverage": sum(covered) / len(covered),
        "mean_width": math.fsum(widths) / len(widths),
        "mean_interval_score": math.fsum(scores) / len(scores),
    }


def evaluate_rows(ids, targets, selected, ungated, lower, upper, matched_width=False):
    if len(ids) != len(set(ids)) or not len(ids) == len(targets) == len(
        selected
    ) == len(ungated) == len(lower) == len(upper):
        raise ValueError("Exact unique aligned evaluation rows required")

    def part(indices):
        take = lambda values: [values[i] for i in indices]
        y = take(targets)
        p = take(selected)
        raw = take(ungated)
        lo = take(lower)
        hi = take(upper)
        out = {
            "selected": regression(y, p),
            "ungated_same_mean": regression(y, raw),
            "zero": regression(y, [0.0] * len(y)),
            "selected_intervals": intervals(y, lo, hi),
        }
        if matched_width:
            half = [(b - a) / 2 for a, b in zip(lo, hi)]
            out["matched_width_diagnostics_uncalibrated"] = {
                name: intervals(
                    y,
                    [v - w for v, w in zip(values, half)],
                    [v + w for v, w in zip(values, half)],
                )
                for name, values in [
                    ("ungated_same_mean", raw),
                    ("zero", [0.0] * len(y)),
                ]
            }
        return out

    return {
        "aggregate": part(list(range(len(ids)))),
        "per_asset": {
            a: part([i for i, row in enumerate(ids) if row[0] == a]) for a in ASSETS
        },
    }


def run(
    snapshot,
    provenance,
    calibrated,
    policy,
    release,
    release_sha256,
    output,
    *,
    legacy=None,
):
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh immutable evaluation output required")
    plan, receipt = authorize(
        snapshot, provenance, calibrated, policy, release, release_sha256, legacy
    )
    # All future source parsing and authenticated checkpoint loading starts here.
    content = Path(snapshot).read_bytes()
    observations = json.loads(content)["observations"]
    ids, pending, counts = source_cohort(observations)
    prefix_sha = calibration_prefix_sha256(observations)
    parts = {
        "development": [r for r in observations if r["d"] <= "2025-12-31"],
        "observed2026": [r for r in observations if r["d"] >= "2026-01-01"],
    }
    if (
        {name: len(rows) for name, rows in parts.items()}
        != plan["source_parent_counts"]
        or any(not rows for rows in parts.values())
        or {
            name: {"first": rows[0]["d"], "last": rows[-1]["d"]}
            for name, rows in parts.items()
        }
        != plan["source_parent_ranges"]
    ):
        raise ValueError("Complete two-parent publication coverage changed")
    if (
        canonical(ids) != plan["expected_decision_ids_sha256"]
        or canonical(pending) != plan["expected_unmatured_ids_sha256"]
        or counts != plan["expected_counts"]
        or cohort_ranges(ids) != plan["expected_ranges"]
        or len(observations) != plan["source_rows"]
        or observations[-1]["d"] != plan["maximum_source_date"]
        or prefix_sha != receipt["calibration_source_sha256"]
    ):
        raise ValueError("Exact full source-derived evaluation cohort changed")
    from src.models import gated_forecaster as production
    from src.models.regime_interactions import predict
    from src.models.volatility_gate_inference import predict_records
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import build_panel_features

    production.check_policy(policy)
    bundle, loaded_receipt = production.load_bundle(calibrated)
    if (
        loaded_receipt != receipt
        or bundle.get("quantiles") is None
        or bundle.get("policy_sha256") != plan["mean_policy_sha256"]
        or bundle.get("source_sha256") != plan["production_source_sha256"]
        or production.model_state_fingerprints(bundle["models"])
        != plan["mean_model_state_sha256"]
    ):
        raise ValueError("Saved calibrated model/policy state changed")
    for asset in ASSETS:
        calibration = receipt.get("calibration", {}).get(asset, {})
        if (
            type(calibration.get("rows")) is not int
            or calibration["rows"] <= 0
            or not all(
                isinstance(calibration.get(k), str)
                and calibration[k].startswith("2025-")
                for k in ("first_decision", "last_decision", "last_target_maturity")
            )
            or number(calibration.get("normalized_quantile"))
            != number(bundle["quantiles"][asset])
        ):
            raise ValueError("Saved calibration interval identity changed")
    if bundle["context"].date.max().strftime("%Y-%m-%d") != "2025-12-31":
        raise ValueError("Frozen pre2026 context required")
    panel = parse_panel(content)
    frames = build_panel_features(panel)
    actual = []
    targets = []
    selected = []
    ungated = []
    lower = []
    upper = []
    scales = []
    by_day = {r["d"]: r for r in observations}
    forecasts = production.forecast(bundle, panel)
    forecast_map = {}
    for row in forecasts.to_dict("records"):
        key = (row["asset"], row["date"].strftime("%Y-%m-%d"))
        if key in forecast_map:
            raise ValueError("Duplicate saved forecast identity")
        forecast_map[key] = row
    replay_max = 0.0
    for asset in ASSETS:
        records = production.frame_records(frames[asset])
        wanted = [key for key in ids if key[0] == asset]
        decisions = {key[1] for key in wanted}
        rows = [r for r in records if r["date"] in decisions]
        cohort = [
            (asset, r["date"], r["execution_date"], r["target_date"]) for r in rows
        ]
        if cohort != wanted:
            raise ValueError("Missing/dropped/changed feature cohort")
        raw = predict(bundle["models"][asset], rows, False)
        gated = predict_records(bundle["models"][asset], rows)
        if len(raw) != len(rows) or len(gated) != len(rows):
            raise ValueError("Truncated saved-model predictions")
        for row, before, after, key in zip(rows, raw, gated, cohort):
            target = math.log(number(by_day[key[3]][asset]["v"])) - math.log(
                number(by_day[key[2]][asset]["v"])
            )
            if abs(number(row["target"]) - target) > 1e-14:
                raise ValueError("Source-derived delayed target changed")
            half = max(number(row["ewma_volatility"]), 1e-5) * number(
                bundle["quantiles"][asset]
            )
            saved = forecast_map.get((asset, key[1]))
            if saved is None:
                raise ValueError("Standalone forecast dropped eligible decision")
            differences = [
                abs(number(saved["prediction"]) - number(after)),
                abs(number(saved["half_width"]) - half),
                abs(number(saved["lower"]) - (after - half)),
                abs(number(saved["upper"]) - (after + half)),
            ]
            replay_max = max(replay_max, *differences)
            actual.append(key)
            targets.append(target)
            selected.append(number(after))
            ungated.append(number(before))
            lower.append(number(after - half))
            upper.append(number(after + half))
            scales.append(half)
    if actual != ids or replay_max > 1e-14:
        raise ValueError("Saved standalone forecast replay differs")
    loaded_again, again_receipt = production.load_bundle(calibrated)
    again = production.forecast(loaded_again, panel)
    if (
        again_receipt != receipt
        or list(forecasts.columns) != list(again.columns)
        or len(forecasts) != len(again)
        or not forecasts[["date", "asset"]].equals(again[["date", "asset"]])
    ):
        raise ValueError("Saved checkpoint reload cohort changed")
    for key in ("prediction", "half_width", "lower", "upper"):
        if len(forecasts):
            replay_max = max(
                replay_max,
                max(
                    abs(number(a) - number(b))
                    for a, b in zip(forecasts[key], again[key])
                ),
            )
    if (
        replay_max > 1e-14
        or production.model_state_fingerprints(bundle["models"])
        != plan["mean_model_state_sha256"]
    ):
        raise ValueError("Saved checkpoint forecast replay changed")
    result = {
        "schema": "causal_gate_evaluation_result_v1",
        "release_sha256": release_sha256,
        "source_sha256": plan["source_sha256"],
        "source_provenance_sha256": plan["source_provenance_sha256"],
        "calibrated_bundle_sha256": plan["calibrated_bundle_sha256"],
        "calibrated_receipt_sha256": plan["calibrated_receipt_sha256"],
        "mean_policy_sha256": plan["mean_policy_sha256"],
        "mean_model_state_sha256": plan["mean_model_state_sha256"],
        "decision_ids_sha256": canonical(ids),
        "counts": counts,
        "decision_ranges": cohort_ranges(ids),
        "rows": len(ids),
        "unmatured_excluded": {
            "rows": len(pending),
            "ids_sha256": canonical(pending),
            "per_asset": {a: sum(r[0] == a for r in pending) for a in ASSETS},
        },
        "saved_forecast_replay_max_abs_difference": replay_max,
        "weights_refitted": False,
        "policy_reselected": False,
        "interpretation": "Already observed2026/current-vintage retrospective comparison; serial dependence, no exchangeability or profit guarantee",
        "metrics": evaluate_rows(
            ids,
            targets,
            selected,
            ungated,
            lower,
            upper,
            plan["matched_width_diagnostics"],
        ),
        "true_legacy_comparator": None,
    }
    if legacy is not None:
        rows = legacy["callback"](panel.copy(), list(ids), dict(legacy["artifacts"]))
        if [
            (r["asset"], r["date"], r["execution_date"], r["target_date"]) for r in rows
        ] != ids:
            raise ValueError("Authenticated legacy comparator cohort differs")
        predictions = [number(r["prediction"]) for r in rows]
        lo = [number(r["lower"]) for r in rows]
        hi = [number(r["upper"]) for r in rows]
        result["true_legacy_comparator"] = {
            "contract": plan["legacy_comparator"],
            "regression": regression(targets, predictions),
            "intervals": intervals(targets, lo, hi),
            "per_asset": {
                a: {
                    "regression": regression(
                        [targets[i] for i, k in enumerate(ids) if k[0] == a],
                        [predictions[i] for i, k in enumerate(ids) if k[0] == a],
                    ),
                    "intervals": intervals(
                        [targets[i] for i, k in enumerate(ids) if k[0] == a],
                        [lo[i] for i, k in enumerate(ids) if k[0] == a],
                        [hi[i] for i, k in enumerate(ids) if k[0] == a],
                    ),
                }
                for a in ASSETS
            },
        }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "evaluation.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "snapshot",
        "provenance",
        "calibrated",
        "policy",
        "release",
        "release-sha256",
        "output",
    ):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
