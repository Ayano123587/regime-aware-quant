"""Prospective all-fold mean selection; no 2025 calibration or 2026 input."""

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random
import time

from src.models.regime_interactions import ALPHAS, BASE_FEATURES, SIGNALS, number

ASSETS = ["FXEURCAD", "FXGBPCAD", "FXAUDCAD"]
YEARS = [2021, 2022, 2023, 2024]
FOLD_ROWS = [741, 738, 738, 744]
PARENT_SHA256 = "58c8d02de064898f4387d5727972083d5ea3850aa94009da200f07fbad97df5e"
DERIVED_SHA256 = "21b11ed0fb6a6d8348b39a4553763a68a9a37a4e81aa7474cb6b843c9660bacc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def split_records(records, year):
    if type(year) is not int or year not in YEARS:
        raise ValueError("Declared chronological fold required")
    eligible = []
    for row in records:
        decision = date.fromisoformat(row["date"])
        if decision > date(2024, 12, 31):
            raise ValueError("Post-2024 observation forbidden")
        if row["target_date"] is None:
            continue
        execution = date.fromisoformat(row["execution_date"])
        maturity = date.fromisoformat(row["target_date"])
        if not decision < execution < maturity:
            raise ValueError("Delayed label chronology changed")
        if maturity <= date(2024, 12, 31):
            number(row["target"])
            eligible.append(row)
    validation = [
        r
        for r in eligible
        if date(year, 1, 1) <= date.fromisoformat(r["date"])
        and date.fromisoformat(r["target_date"]) <= date(year, 12, 31)
    ]
    if not validation:
        raise ValueError("Empty validation fold")
    first = min(r["date"] for r in validation)
    training = [r for r in eligible if r["target_date"] < first]
    if not training or max(r["target_date"] for r in training) >= first:
        raise ValueError("Training label purge failed")
    return training, validation


def metrics(rows):
    if not rows:
        raise ValueError("Nonempty scored cohort required")
    return {
        "n": len(rows),
        "rmse": math.sqrt(
            sum((number(r["target"]) - number(r["prediction"])) ** 2 for r in rows)
            / len(rows)
        ),
    }


def choose(runs):
    expected = {"zero"} | {
        f"{kind}_{alpha}" for kind in ("linear", "interaction") for alpha in ALPHAS
    }
    if len(runs) != len(expected) or {r["name"] for r in runs} != expected:
        raise ValueError("Complete fixed control and candidate grid required")
    by_name = {r["name"]: r for r in runs}
    for row in runs:
        if (
            [r["year"] for r in row["folds"]] != YEARS
            or [r["n"] for r in row["folds"]] != FOLD_ROWS
            or row["validation"]["n"] != sum(FOLD_ROWS)
        ):
            raise ValueError("Changed complete chronological validation cohort")
        for value in [row["validation"]["rmse"]] + [r["rmse"] for r in row["folds"]]:
            if number(value) < 0:
                raise ValueError("Negative RMSE")
    retained = by_name["linear_10000"]
    reference = min(
        (by_name[name] for name in ["zero"] + [f"linear_{a}" for a in ALPHAS]),
        key=lambda r: r["validation"]["rmse"],
    )
    diagnostics = []
    for alpha in ALPHAS:
        candidate = by_name[f"interaction_{alpha}"]
        matched = by_name[f"linear_{alpha}"]
        retained_wins = sum(
            c["rmse"] < b["rmse"] for c, b in zip(candidate["folds"], retained["folds"])
        )
        matched_wins = sum(
            c["rmse"] < b["rmse"] for c, b in zip(candidate["folds"], matched["folds"])
        )
        gates = {
            "relative_RMSE_gain": candidate["validation"]["rmse"]
            <= reference["validation"]["rmse"] * 0.999,
            "retained_year_support": retained_wins >= 3,
            "matched_year_support": matched_wins >= 3,
        }
        diagnostics.append(
            {
                "name": candidate["name"],
                "gates": gates,
                "eligible": all(gates.values()),
                "retained_year_wins": retained_wins,
                "matched_year_wins": matched_wins,
            }
        )
    eligible = [by_name[r["name"]] for r in diagnostics if r["eligible"]]
    winner = (
        min(eligible, key=lambda r: (r["validation"]["rmse"], ALPHAS.index(r["alpha"])))
        if eligible
        else retained
    )
    return {
        "selected": winner["name"],
        "promoted": bool(eligible),
        "strongest_control": reference["name"],
        "candidates": diagnostics,
    }


def month_interval(baseline, candidate, draws=5000, seed=20261016):
    if len(baseline) != len(candidate) or type(draws) is not int or draws < 1:
        raise ValueError(
            "Aligned paired predictions and positive bootstrap count required"
        )
    groups = defaultdict(lambda: [0.0, 0.0, 0])
    for left, right in zip(baseline, candidate):
        if (left["date"], left["asset"], left["target"]) != (
            right["date"],
            right["asset"],
            right["target"],
        ):
            raise ValueError("Changed paired date/asset/target")
        group = groups[left["date"][:7]]
        group[0] += (number(left["target"]) - number(left["prediction"])) ** 2
        group[1] += (number(right["target"]) - number(right["prediction"])) ** 2
        group[2] += 1
    if not groups:
        raise ValueError("Empty paired month blocks")
    months = sorted(groups)
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        selected = rng.choices(months, k=len(months))
        count = sum(groups[m][2] for m in selected)
        differences.append(
            math.sqrt(sum(groups[m][0] for m in selected) / count)
            - math.sqrt(sum(groups[m][1] for m in selected) / count)
        )
    differences.sort()

    def quantile(p):
        position = p * (draws - 1)
        lo = int(position)
        hi = min(lo + 1, draws - 1)
        return differences[lo] + (position - lo) * (differences[hi] - differences[lo])

    return {
        "low": quantile(0.025),
        "high": quantile(0.975),
        "months": len(months),
        "draws": draws,
        "seed": seed,
        "measurement": "Retained minus candidate RMSE; descriptive paired month-block interval, not a selection gate",
    }


def check_plan(path):
    plan = read(path)
    expected = {
        "assets": ASSETS,
        "validation_years": YEARS,
        "expected_fold_rows": FOLD_ROWS,
        "alphas": ALPHAS,
        "signals": SIGNALS,
        "base_features": BASE_FEATURES,
        "source_parent_sha256": PARENT_SHA256,
        "source_derived_sha256": DERIVED_SHA256,
        "source_rows": 1995,
        "source_first_date": "2017-01-03",
        "source_last_date": "2024-12-31",
        "maximum_source_date": "2024-12-31",
        "maximum_target_maturity": "2024-12-31",
        "minimum_relative_rmse_gain": 0.001,
        "minimum_year_wins": 3,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 20261016,
        "gate": {
            "volatility_ratio_definition": "volatility_5 / max(volatility_20, 1e-8)",
            "ratio_clip": [0.5, 2.0],
            "transform": "natural_log",
        },
        "feature_order": "base_then_gate_then_fixed_signal_products",
        "scaling": "StandardScaler fitted only on fold training rows",
        "fit_intercept": False,
        "threads": 2,
        "control_reference": "Best pooled RMSE of zero and all three matched linear controls",
        "year_comparators": ["linear_10000", "same_alpha_linear"],
        "tie_break": "declared_alpha_order",
        "bootstrap_unit": "calendar_month_all_assets",
        "bootstrap_quantiles": [0.025, 0.975],
        "bootstrap_selection_gate": False,
    }
    if not isinstance(plan, dict) or json.dumps(
        {k: plan.get(k) for k in expected}, sort_keys=True, allow_nan=False
    ) != json.dumps(expected, sort_keys=True):
        raise ValueError("Changed prespecified interaction study contract")
    return plan


def run(snapshot, provenance, protocol, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh study output required")
    plan = check_plan(protocol)
    receipt = read(provenance)
    if (
        receipt.get("parent_sha256") != PARENT_SHA256
        or receipt.get("source_sha256") != DERIVED_SHA256
        or sha(snapshot) != DERIVED_SHA256
        or receipt.get("rows") != 1995
        or receipt.get("maximum_source_date") != "2024-12-31"
    ):
        raise ValueError("Private development source identity changed")
    content = Path(snapshot).read_bytes()
    observations = json.loads(content)["observations"]
    if not observations or any(
        not "2017-01-01" <= r["d"] <= "2024-12-31" for r in observations
    ):
        raise ValueError("Only 2017–2024 observations may enter mean selection")
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import FEATURES, build_panel_features
    from src.models.regime_interactions import fit, predict
    from threadpoolctl import threadpool_limits
    import joblib
    import pandas as pd

    if FEATURES != BASE_FEATURES:
        raise ValueError("Production feature contract changed")
    started = time.monotonic()
    panel = parse_panel(content)
    frames = build_panel_features(panel)
    records = {}
    for asset, frame in frames.items():
        frame = frame.copy()
        for key in ("date", "execution_date", "target_date"):
            frame[key] = frame[key].map(
                lambda value: None if pd.isna(value) else value.strftime("%Y-%m-%d")
            )
        records[asset] = frame.to_dict("records")
    folds = {
        year: {a: split_records(records[a], year) for a in ASSETS} for year in YEARS
    }
    if [sum(len(folds[y][a][1]) for a in ASSETS) for y in YEARS] != FOLD_ROWS:
        raise ValueError("Full frozen validation cohort count mismatch")
    output.mkdir()
    runs = []
    predictions = {}
    replay_max = 0.0
    smoke = []
    with threadpool_limits(limits=2):
        for asset in ASSETS:
            training, validation = folds[2021][asset]
            for interactions in (False, True):
                model = fit(training[:32], 10000, interactions)
                path = output / f"mechanics-{asset}-{interactions}.joblib"
                joblib.dump(model, path)
                before = predict(model, validation[:8], interactions)
                after = predict(joblib.load(path), validation[:8], interactions)
                delta = max(abs(a - b) for a, b in zip(before, after))
                norm = float(
                    sum(float(v) ** 2 for v in model.named_steps["ridge"].coef_)
                )
                if delta > 1e-14 or not math.isfinite(norm) or norm <= 0:
                    raise ValueError("Tiny learned-fit/save/reload mechanics failed")
                smoke.append(
                    {
                        "asset": asset,
                        "interactions": interactions,
                        "training_rows": 32,
                        "replay_rows": 8,
                        "coefficient_squared_norm": norm,
                        "replay_max_abs_difference": delta,
                    }
                )
        (output / "mechanics.json").write_text(
            json.dumps(smoke, indent=2) + "\n", encoding="utf8"
        )
        for kind, alpha in [("zero", None)] + [
            (kind, a) for kind in ("linear", "interaction") for a in ALPHAS
        ]:
            name = "zero" if kind == "zero" else f"{kind}_{alpha}"
            all_rows = []
            summaries = []
            for year in YEARS:
                scored = []
                for asset in ASSETS:
                    training, validation = folds[year][asset]
                    if kind == "zero":
                        values = [0.0] * len(validation)
                    else:
                        model = fit(training, alpha, kind == "interaction")
                        values = predict(model, validation, kind == "interaction")
                        checkpoint = output / f"{name}-{year}-{asset}.joblib"
                        joblib.dump(model, checkpoint)
                        loaded = predict(
                            joblib.load(checkpoint), validation, kind == "interaction"
                        )
                        error = max(abs(a - b) for a, b in zip(values, loaded))
                        if error > 1e-14:
                            raise ValueError(
                                "Actual saved fold-checkpoint replay changed"
                            )
                        replay_max = max(replay_max, error)
                    scored.extend(
                        {
                            "date": r["date"],
                            "asset": asset,
                            "target": r["target"],
                            "prediction": p,
                        }
                        for r, p in zip(validation, values)
                    )
                summaries.append({"year": year, **metrics(scored)})
                all_rows.extend(scored)
                print(
                    json.dumps(
                        {"policy": name, "year": year, "validation": summaries[-1]}
                    ),
                    flush=True,
                )
            predictions[name] = all_rows
            (output / (name + "-private-predictions.json")).write_text(
                json.dumps(all_rows), encoding="utf8"
            )
            runs.append(
                {
                    "name": name,
                    "kind": kind,
                    "alpha": alpha,
                    "validation": metrics(all_rows),
                    "folds": summaries,
                    "assets": {
                        a: metrics([r for r in all_rows if r["asset"] == a])
                        for a in ASSETS
                    },
                }
            )
    retained = next(r for r in runs if r["name"] == "linear_10000")
    if not math.isclose(
        retained["validation"]["rmse"], 0.0039109948384311215, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            "Retained baseline does not replay the frozen chronological cohort"
        )
    decision = choose(runs)
    import importlib.metadata
    import platform

    report = {
        "source_sha256": sha(snapshot),
        "source_provenance": receipt,
        "protocol_sha256": sha(protocol),
        "source_sha256_files": {
            name: sha(name)
            for name in [
                "src/models/interaction_study.py",
                "src/models/regime_interactions.py",
                "src/features/multi_asset.py",
                "src/data/multi_asset.py",
            ]
        },
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                )
            },
        },
        "checkpoint_sha256": {p.name: sha(p) for p in sorted(output.glob("*.joblib"))},
        "mechanics": smoke,
        "runs": runs,
        "decision": decision,
        "saved_checkpoint_replay_max_abs_difference": replay_max,
        "month_intervals": {
            f"interaction_{a}": month_interval(
                predictions["linear_10000"], predictions[f"interaction_{a}"]
            )
            for a in ALPHAS
        },
        "seconds": time.monotonic() - started,
        "validation_rows": sum(FOLD_ROWS),
        "calibration_2025_accessed": False,
        "test_2026_accessed": False,
        "deployment_changed": False,
        "interpretation": "Previously observed chronological validation; no independent test or trading-profit claim",
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "provenance", "protocol", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        report = run(args.snapshot, args.provenance, args.protocol, args.output)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, "Interaction study refused: " + str(error) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
