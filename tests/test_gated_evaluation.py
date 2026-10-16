"""StageC software fixtures only: no actual source, weights, fits or network."""

import copy, json, math
from pathlib import Path
import pytest
from src.models import gated_evaluation as e


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def test_release_hash_precedes_every_source_or_checkpoint_read(tmp_path, monkeypatch):
    release = put(tmp_path / "release.json", {})
    original = e.sha
    seen = []

    def checked(path):
        seen.append(Path(path))
        assert Path(path) == release
        return original(path)

    monkeypatch.setattr(e, "sha", checked)
    with pytest.raises(ValueError, match="release SHA"):
        e.run(
            "never-read-source",
            "never-read-provenance",
            "never-load-model",
            "never-read-policy",
            release,
            "0" * 64,
            tmp_path / "output",
        )
    assert seen == [release]


def observations(days):
    return [
        {
            "d": d,
            **{
                a: {"v": str(1.0 + i * 0.001 + j * 0.1)} for j, a in enumerate(e.ASSETS)
            },
        }
        for i, d in enumerate(days)
    ]


def test_source_ids_use_publication_successors_not_calendar_days():
    rows = observations(
        ["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    )
    ids, pending, counts = e.source_cohort(rows)
    assert ids[:2] == [
        (e.ASSETS[0], "2026-01-02", "2026-01-05", "2026-01-06"),
        (e.ASSETS[0], "2026-01-05", "2026-01-06", "2026-01-07"),
    ]
    assert counts == {a: 2 for a in e.ASSETS} and len(pending) == 6
    assert pending[-1] == (e.ASSETS[-1], "2026-01-07", None, None)


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "unordered", "gap", "nonfinite", "negative", "missing", "nat"],
)
def test_invalid_source_chronology_is_never_dropped(defect):
    rows = observations(["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06"])
    if defect == "duplicate":
        rows[2]["d"] = rows[1]["d"]
    elif defect == "unordered":
        rows[2]["d"] = "2026-01-01"
    elif defect == "gap":
        rows[2]["d"] = "2026-01-20"
    elif defect == "nonfinite":
        rows[1][e.ASSETS[0]]["v"] = "nan"
    elif defect == "negative":
        rows[1][e.ASSETS[0]]["v"] = "-1"
    elif defect == "missing":
        rows[1].pop(e.ASSETS[0])
    else:
        rows[1]["d"] = "NaT"
    with pytest.raises((ValueError, KeyError)):
        e.source_cohort(rows)


def test_proper_interval_score_and_matched_width_label():
    metric = e.intervals([0.0, 2.0], [-1.0, -1.0], [1.0, 1.0])
    assert metric == {
        "rows": 2,
        "nominal_coverage": 0.9,
        "coverage": 0.5,
        "mean_width": 2.0,
        "mean_interval_score": 12.0,
    }
    ids = [(a, "2026-01-02", "2026-01-05", "2026-01-06") for a in e.ASSETS]
    result = e.evaluate_rows(
        ids,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0] * 3,
        [1.0] * 3,
        True,
    )
    assert result["aggregate"]["selected"]["rmse"] == 0
    assert "matched_width_diagnostics_uncalibrated" in result["aggregate"]
    assert "true_legacy" not in result["aggregate"]
    with pytest.raises(ValueError):
        e.intervals([0.0], [1.0], [-1.0])


def fixture(tmp_path, monkeypatch):
    import pandas as pd
    from src.models import gated_forecaster as production
    from src.models import regime_interactions as mean
    from src.models import volatility_gate_inference as gate
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import build_panel_features

    days = pd.bdate_range("2025-09-01", "2026-01-12").strftime("%Y-%m-%d").tolist()
    rows = [
        {
            "d": d,
            **{
                a: {
                    "v": str(
                        math.exp(0.1 * j + 0.001 * i + 0.002 * math.sin(i / 3 + j))
                    )
                }
                for j, a in enumerate(e.ASSETS)
            },
        }
        for i, d in enumerate(days)
    ]
    source = put(tmp_path / "source.json", {"observations": rows})
    ids, pending, counts = e.source_cohort(rows)
    prefix = [r for r in rows if r["d"] <= "2025-12-31"]
    prefix_sha = (
        __import__("hashlib")
        .sha256(
            (
                json.dumps({"observations": prefix}, separators=(",", ":")) + "\n"
            ).encode()
        )
        .hexdigest()
    )
    root = Path(e.__file__).resolve().parents[2]
    sources = {p: e.sha(root / p) for p in e.PRODUCTION_FILES}
    policy = put(tmp_path / "policy.json", {"production_source_sha256": sources})
    calibrated = tmp_path / "calibrated"
    calibrated.mkdir()
    (calibrated / "bundle.joblib").write_bytes(
        b"NOT A MODEL: explicit software fixture"
    )
    states = {a: "d" * 64 for a in e.ASSETS}
    receipt = {
        "stage": "calibrated_frozen",
        "bundle_sha256": e.sha(calibrated / "bundle.joblib"),
        "policy_sha256": e.sha(policy),
        "mean_model_state_sha256": states,
        "calibration_release_sha256": "a" * 64,
        "calibration_source_sha256": prefix_sha,
        "calibration_decision_ids_sha256": "b" * 64,
        "source_provenance_sha256": "c" * 64,
        "mean_refitted": False,
        "test_accessed": False,
        "calibration": {
            a: {
                "rows": 2,
                "first_decision": "2025-01-02",
                "last_decision": "2025-12-29",
                "last_target_maturity": "2025-12-31",
                "normalized_quantile": 2.0,
            }
            for a in e.ASSETS
        },
    }
    put(calibrated / "receipt.json", receipt)
    provenance = {
        "schema": "causal_gate_evaluation_source_v1",
        "source_sha256": e.sha(source),
        "source_parent_sha256": {"development": "e" * 64, "observed2026": "f" * 64},
        "source_parent_counts": {
            "development": len(prefix),
            "observed2026": len(rows) - len(prefix),
        },
        "source_parent_ranges": {
            "development": {"first": prefix[0]["d"], "last": prefix[-1]["d"]},
            "observed2026": {
                "first": next(r["d"] for r in rows if r["d"] >= "2026-01-01"),
                "last": rows[-1]["d"],
            },
        },
        "calibration_prefix_sha256": prefix_sha,
        "expected_decision_ids_sha256": e.canonical(ids),
        "expected_unmatured_ids_sha256": e.canonical(pending),
        "expected_counts": counts,
        "expected_ranges": e.cohort_ranges(ids),
        "source_rows": len(rows),
        "maximum_source_date": rows[-1]["d"],
    }
    provenance_path = put(tmp_path / "provenance.json", provenance)
    plan = {
        "schema": e.SCHEMA,
        "decision_year": 2026,
        "execution_year": 2026,
        "target_year": 2026,
        "interval_alpha": 0.1,
        "weights_refitted": False,
        "policy_reselected": False,
        "retrospective_observed_2026": True,
        "comparators": ["selected", "ungated_same_mean", "zero"],
        "source_sha256": e.sha(source),
        "source_provenance_sha256": e.sha(provenance_path),
        "calibrated_bundle_sha256": receipt["bundle_sha256"],
        "calibrated_receipt_sha256": e.sha(calibrated / "receipt.json"),
        "mean_policy_sha256": e.sha(policy),
        "evaluator_source_sha256": e.sha(e.__file__),
        "production_source_sha256": sources,
        "mean_model_state_sha256": states,
        "calibration_release_sha256": receipt["calibration_release_sha256"],
        "calibration_source_sha256": prefix_sha,
        "calibration_decision_ids_sha256": receipt["calibration_decision_ids_sha256"],
        "calibration_source_provenance_sha256": receipt["source_provenance_sha256"],
        "source_parent_sha256": {"development": "e" * 64, "observed2026": "f" * 64},
        "source_parent_counts": {
            "development": len(prefix),
            "observed2026": len(rows) - len(prefix),
        },
        "source_parent_ranges": {
            "development": {"first": prefix[0]["d"], "last": prefix[-1]["d"]},
            "observed2026": {
                "first": next(r["d"] for r in rows if r["d"] >= "2026-01-01"),
                "last": rows[-1]["d"],
            },
        },
        "expected_decision_ids_sha256": e.canonical(ids),
        "expected_unmatured_ids_sha256": e.canonical(pending),
        "expected_counts": counts,
        "expected_ranges": e.cohort_ranges(ids),
        "source_rows": len(rows),
        "maximum_source_date": rows[-1]["d"],
        "matched_width_diagnostics": True,
        "legacy_comparator": None,
    }
    release = put(tmp_path / "release.json", plan)
    bundle = {
        "models": {a: 0.0001 * (j + 1) for j, a in enumerate(e.ASSETS)},
        "quantiles": {a: 2.0 for a in e.ASSETS},
        "policy_sha256": e.sha(policy),
        "source_sha256": sources,
        "context": parse_panel(json.dumps({"observations": prefix})),
    }
    monkeypatch.setattr(production, "check_policy", lambda _: plan)
    monkeypatch.setattr(
        production,
        "load_bundle",
        lambda _: (copy.deepcopy(bundle), copy.deepcopy(receipt)),
    )
    monkeypatch.setattr(production, "model_state_fingerprints", lambda _: states)
    monkeypatch.setattr(
        mean, "predict", lambda model, records, interaction: [model] * len(records)
    )
    monkeypatch.setattr(
        gate,
        "predict_records",
        lambda model, records: [
            2 * model if r["volatility_ratio"] > 1.2 else 0.0 for r in records
        ],
    )

    def forecast(saved, panel):
        frames = build_panel_features(panel)
        result = []
        for a in e.ASSETS:
            part = frames[a].loc[frames[a].date > saved["context"].date.max()]
            for r in production.frame_records(part):
                prediction = (
                    2 * saved["models"][a] if r["volatility_ratio"] > 1.2 else 0.0
                )
                half = max(r["ewma_volatility"], 1e-5) * saved["quantiles"][a]
                result.append(
                    {
                        "date": pd.Timestamp(r["date"], tz="UTC"),
                        "asset": a,
                        "prediction": prediction,
                        "half_width": half,
                        "lower": prediction - half,
                        "upper": prediction + half,
                    }
                )
        return pd.DataFrame(result)

    monkeypatch.setattr(production, "forecast", forecast)
    return {
        "snapshot": source,
        "provenance": provenance_path,
        "calibrated": calibrated,
        "policy": policy,
        "release": release,
        "release_sha256": e.sha(release),
        "output": tmp_path / "evaluation",
    }, plan


def test_complete_source_cohort_saved_forecast_replay_and_aggregate_only(
    tmp_path, monkeypatch
):
    args, plan = fixture(tmp_path, monkeypatch)
    result = e.run(**args)
    assert result["counts"] == plan["expected_counts"] and result["rows"] == sum(
        plan["expected_counts"].values()
    )
    assert (
        result["unmatured_excluded"]["rows"] == 6
        and result["saved_forecast_replay_max_abs_difference"] == 0
    )
    assert (
        result["true_legacy_comparator"] is None
        and result["weights_refitted"] is False
        and result["policy_reselected"] is False
    )
    assert set(p.name for p in args["output"].iterdir()) == {"evaluation.json"}
    assert "targets" not in json.dumps(result)
    with pytest.raises(ValueError, match="immutable"):
        e.run(**args)


@pytest.mark.parametrize(
    "defect",
    [
        "source",
        "receipt",
        "policy",
        "provenance",
        "counts",
        "ids",
        "unmatured",
        "legacy_omitted",
        "legacy_callback_missing",
        "reselection",
    ],
)
def test_release_or_complete_cohort_tampering_rejected(tmp_path, monkeypatch, defect):
    args, plan = fixture(tmp_path, monkeypatch)
    if defect == "source":
        args["snapshot"].write_text("{}")
    elif defect == "receipt":
        put(args["calibrated"] / "receipt.json", {})
    elif defect == "policy":
        put(args["policy"], {})
    elif defect == "provenance":
        put(args["provenance"], {})
    else:
        if defect == "counts":
            plan["expected_counts"][e.ASSETS[0]] -= 1
        elif defect == "ids":
            plan["expected_decision_ids_sha256"] = "0" * 64
        elif defect == "unmatured":
            plan["expected_unmatured_ids_sha256"] = "0" * 64
        elif defect == "legacy_omitted":
            plan.pop("legacy_comparator")
        elif defect == "legacy_callback_missing":
            plan["legacy_comparator"] = {"unproven": True}
        else:
            plan["policy_reselected"] = True
        put(args["release"], plan)
        args["release_sha256"] = e.sha(args["release"])
    with pytest.raises(ValueError):
        e.run(**args)


def test_missing_feature_decision_cannot_reduce_evaluation_denominator(
    tmp_path, monkeypatch
):
    args, _ = fixture(tmp_path, monkeypatch)
    from src.features import multi_asset

    actual = multi_asset.build_panel_features

    def dropped(panel):
        frames = actual(panel)
        frame = frames[e.ASSETS[0]]
        first = frame.index[frame.date.dt.year == 2026][0]
        frames[e.ASSETS[0]] = frame.drop(first)
        return frames

    monkeypatch.setattr(multi_asset, "build_panel_features", dropped)
    with pytest.raises(ValueError, match="cohort"):
        e.run(**args)


def test_changed_delayed_target_rejected_against_source(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    from src.features import multi_asset

    actual = multi_asset.build_panel_features

    def changed(panel):
        frames = actual(panel)
        for frame in frames.values():
            frame.loc[frame.date.dt.year == 2026, "target"] += 0.01
        return frames

    monkeypatch.setattr(multi_asset, "build_panel_features", changed)
    with pytest.raises(ValueError, match="delayed target"):
        e.run(**args)


def software_legacy(panel, ids, artifacts):
    # Only a transport fixture; no real historical checkpoint is represented.
    return [
        {
            "asset": a,
            "date": d,
            "execution_date": x,
            "target_date": t,
            "prediction": 0.0,
            "lower": -0.01,
            "upper": 0.01,
        }
        for a, d, x, t in ids
    ]


def authenticated_legacy(tmp_path, plan, args):
    artifact = tmp_path / "legacy.bin"
    artifact.write_bytes(b"NOT A MODEL: software legacy fixture")
    states = {a: "9" * 64 for a in e.ASSETS}
    calibration = {"source_sha256": "8" * 64, "quantiles_sha256": "7" * 64}
    hashes = {"bundle": e.sha(artifact)}
    proof = put(
        tmp_path / "legacy-proof.json",
        {
            "schema": "authenticated_legacy_forecast_v1",
            "weights_refitted": False,
            "artifact_sha256": hashes,
            "model_state_sha256": states,
            "calibration_identity": calibration,
        },
    )
    contract = {
        "callback_source_sha256": e.sha(__file__),
        "callback_qualified_name": software_legacy.__module__
        + "."
        + software_legacy.__qualname__,
        "artifact_sha256": hashes,
        "proof_sha256": e.sha(proof),
        "model_state_sha256": states,
        "calibration_identity": calibration,
    }
    plan["legacy_comparator"] = contract
    put(args["release"], plan)
    args["release_sha256"] = e.sha(args["release"])
    return {
        "callback": software_legacy,
        "source": __file__,
        "artifacts": {"bundle": artifact},
        "proof": proof,
    }


def test_true_legacy_requires_separate_authenticated_artifact_and_calibration(
    tmp_path, monkeypatch
):
    args, plan = fixture(tmp_path, monkeypatch)
    adapter = authenticated_legacy(tmp_path, plan, args)
    result = e.run(**args, legacy=adapter)
    assert result["true_legacy_comparator"]["contract"] == plan["legacy_comparator"]
    assert result["true_legacy_comparator"]["regression"]["rows"] == result["rows"]
    assert result["true_legacy_comparator"]["intervals"]["mean_width"] == pytest.approx(
        0.02
    )
    assert "ungated_same_mean" in result["metrics"]["aggregate"]


@pytest.mark.parametrize("defect", ["artifact", "source", "proof", "callback"])
def test_legacy_identity_tampering_fails_closed(tmp_path, monkeypatch, defect):
    args, plan = fixture(tmp_path, monkeypatch)
    adapter = authenticated_legacy(tmp_path, plan, args)
    if defect == "artifact":
        adapter["artifacts"]["bundle"].write_bytes(b"changed")
    elif defect == "proof":
        put(adapter["proof"], {})
    elif defect == "source":
        plan["legacy_comparator"]["callback_source_sha256"] = "0" * 64
    else:
        plan["legacy_comparator"]["callback_qualified_name"] = "unreviewed.callback"
    put(args["release"], plan)
    args["release_sha256"] = e.sha(args["release"])
    with pytest.raises(ValueError, match="Legacy|legacy"):
        e.run(**args, legacy=adapter)


def test_saved_forecast_reload_difference_is_rejected(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    from src.models import gated_forecaster as production

    original = production.forecast
    calls = []

    def changed(bundle, panel):
        result = original(bundle, panel)
        calls.append(1)
        if len(calls) == 2:
            result.loc[result.index[0], "prediction"] += 0.01
        return result

    monkeypatch.setattr(production, "forecast", changed)
    with pytest.raises(ValueError, match="replay"):
        e.run(**args)


def test_stage_b_and_c_cutoff_serialization_are_identical(tmp_path, monkeypatch):
    from test_gated_calibration_staging import source_fixture
    from src import prepare_gated_calibration as staging

    parent, rows = source_fixture(tmp_path, monkeypatch)
    raw, provenance = staging.cutoff_source(parent)
    assert (
        e.calibration_prefix_sha256(rows)
        == __import__("hashlib").sha256(raw).hexdigest()
        == provenance["derived_source_sha256"]
    )
    assert json.loads(raw)["observations"] == rows[:-1]


@pytest.mark.parametrize("defect", ["ids", "counts", "ranges", "parent_counts"])
def test_self_consistent_forged_provenance_cannot_change_source_cohort(
    tmp_path, monkeypatch, defect
):
    args, plan = fixture(tmp_path, monkeypatch)
    proof = e.read(args["provenance"])
    if defect == "ids":
        plan["expected_decision_ids_sha256"] = proof["expected_decision_ids_sha256"] = (
            "0" * 64
        )
    elif defect == "counts":
        plan["expected_counts"][e.ASSETS[0]] -= 1
        proof["expected_counts"] = plan["expected_counts"]
    elif defect == "ranges":
        plan["expected_ranges"][e.ASSETS[0]]["last_decision"] = "2026-01-02"
        proof["expected_ranges"] = plan["expected_ranges"]
    else:
        plan["source_parent_counts"]["observed2026"] -= 1
        proof["source_parent_counts"] = plan["source_parent_counts"]
    put(args["provenance"], proof)
    plan["source_provenance_sha256"] = e.sha(args["provenance"])
    put(args["release"], plan)
    args["release_sha256"] = e.sha(args["release"])
    with pytest.raises(ValueError, match="cohort|coverage"):
        e.run(**args)


def software_legacy_drop(panel, ids, artifacts):
    return software_legacy(panel, ids, artifacts)[:-1]


def test_authenticated_legacy_callback_still_cannot_drop_query_rows(
    tmp_path, monkeypatch
):
    args, plan = fixture(tmp_path, monkeypatch)
    adapter = authenticated_legacy(tmp_path, plan, args)
    adapter["callback"] = software_legacy_drop
    plan["legacy_comparator"]["callback_qualified_name"] = (
        software_legacy_drop.__module__ + "." + software_legacy_drop.__qualname__
    )
    put(args["release"], plan)
    args["release_sha256"] = e.sha(args["release"])
    with pytest.raises(ValueError, match="comparator cohort"):
        e.run(**args, legacy=adapter)
