"""Legacy authentication software fixtures, never actual estimator loading."""

import copy
import json
from types import SimpleNamespace
import pytest
from src.models import gated_legacy as legacy


@pytest.fixture
def case(tmp_path, monkeypatch):
    info = {
        "rows": 247,
        "last_training_label": "2024-12-31 UTC",
        "first_calibration_decision": "2025-01-02 UTC",
        "last_calibration_label": "2025-12-31 UTC",
        "absolute_quantile": 0.01,
        "normalized_quantile": 1.6,
    }
    selection = {
        "selected": {"kind": "ridge", "alpha": 10000},
        "calibration": {a: info.copy() for a in legacy.ASSETS},
    }
    s = tmp_path / "selection.json"
    s.write_text(json.dumps(selection))
    b = tmp_path / "bundle.joblib"
    b.write_bytes(b"never deserialized fixture")
    monkeypatch.setattr(legacy, "BUNDLE_SHA256", legacy.sha(b))
    monkeypatch.setattr(legacy, "SELECTION_SHA256", legacy.sha(s))
    checkpoint = {
        "version": "uncertainty_v2",
        "assets": legacy.ASSETS.copy(),
        "selection_sha256": legacy.sha(s),
        "bundles": {
            a: {
                "model": object(),
                "absolute_quantile": 0.01,
                "normalized_quantile": 1.6,
            }
            for a in legacy.ASSETS
        },
    }
    calls = []
    monkeypatch.setattr(legacy, "_load", lambda p: calls.append(p) or checkpoint)
    monkeypatch.setattr(
        legacy,
        "_production",
        lambda: SimpleNamespace(
            model_state_fingerprints=lambda m: {a: "canonical-" + a for a in m}
        ),
    )
    return SimpleNamespace(
        artifacts={"bundle": b, "selection": s}, checkpoint=checkpoint, calls=calls
    )


def test_authenticate_preserves_actual_original_intervals(case):
    checkpoint, proof = legacy.authenticate(case.artifacts)
    assert checkpoint is case.checkpoint
    assert proof["weights_refitted"] is False
    assert set(proof["model_state_sha256"]) == set(legacy.ASSETS)
    assert all(
        x["normalized_quantile"] == 1.6 for x in proof["calibration_identity"].values()
    )


@pytest.mark.parametrize("artifact", ["bundle", "selection"])
def test_tampered_artifacts_refused_before_unpickle(case, artifact):
    case.artifacts[artifact].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact bytes"):
        legacy.authenticate(case.artifacts)
    assert case.calls == []


@pytest.mark.parametrize("field", ["version", "assets", "selection_sha256", "bundles"])
def test_wrong_original_checkpoint_identity(case, field):
    case.checkpoint[field] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        legacy.authenticate(case.artifacts)


@pytest.mark.parametrize("value", [float("nan"), -1, 1.61])
def test_original_quantile_revision_rejected(case, value):
    case.checkpoint["bundles"][legacy.ASSETS[0]]["normalized_quantile"] = value
    with pytest.raises(ValueError, match="quantile"):
        legacy.authenticate(case.artifacts)


def test_complete_cohort_order_and_actual_delayed_ids():
    def stamp(day):
        return SimpleNamespace(strftime=lambda _: day)

    ids = [
        (legacy.ASSETS[1], "2030-01-02", "2030-01-03", "2030-01-04"),
        (legacy.ASSETS[0], "2030-01-02", "2030-01-03", "2030-01-04"),
    ]
    predictions = {
        (a, d): {"prediction": 0.2, "lower": 0.1, "upper": 0.3} for a, d, _, _ in ids
    }
    features = {
        (a, d): {"execution_date": stamp(x), "target_date": stamp(t)}
        for a, d, x, t in ids
    }
    output = legacy.align_rows(predictions, features, ids)
    assert [
        (r["asset"], r["date"], r["execution_date"], r["target_date"]) for r in output
    ] == ids
    with pytest.raises(ValueError, match="Unique"):
        legacy.align_rows(predictions, features, ids + ids[:1])
    predictions.pop(ids[0][:2])
    with pytest.raises(ValueError, match="Complete"):
        legacy.align_rows(predictions, features, ids)


@pytest.mark.parametrize("field", ["execution_date", "target_date"])
def test_source_chronology_mismatch_refused(field):
    ids = [(legacy.ASSETS[0], "2030-01-02", "2030-01-03", "2030-01-04")]
    key = ids[0][:2]
    features = {
        key: {
            "execution_date": SimpleNamespace(strftime=lambda _: "2030-01-03"),
            "target_date": SimpleNamespace(strftime=lambda _: "2030-01-04"),
        }
    }
    features[key][field] = SimpleNamespace(strftime=lambda _: "2030-02-01")
    with pytest.raises(ValueError, match="chronology"):
        legacy.align_rows({key: {}}, features, ids)
