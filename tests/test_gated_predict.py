"""Software guard tests; no fitted estimators or private financial data loaded."""

import json
from types import SimpleNamespace

import pytest

from src.models import gated_predict as entry


class Frame:
    def __len__(self):
        return 1

    def to_csv(self, *, index, lineterminator):
        assert index is False and lineterminator == "\n"
        return "date,asset,prediction\n2030-01-01,FIXTURE,0\n"


@pytest.fixture
def case(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    bundle_path = model / "bundle.joblib"
    bundle_path.write_bytes(b"software fixture, never deserialized")
    policy = tmp_path / "policy.json"
    sources = {"fixture.py": "fixture-digest"}
    policy.write_text(json.dumps({"production_source_sha256": sources}))
    receipt = {
        "stage": "mean_frozen",
        "policy_sha256": entry.sha(policy),
        "bundle_sha256": entry.sha(bundle_path),
        "mean_model_state_sha256": {"FIXTURE": "state-digest"},
    }
    receipt_path = model / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    snapshot = tmp_path / "panel.json"
    snapshot.write_text('{"software_fixture": true}')
    bundle = {
        "policy_sha256": entry.sha(policy),
        "source_sha256": sources.copy(),
        "quantiles": None,
    }
    calls = []

    def check_policy(path):
        calls.append("policy")
        return json.loads(path.read_text())

    def load_bundle(path):
        calls.append("load")
        return bundle, receipt.copy()

    backend = SimpleNamespace(
        check_policy=check_policy,
        load_bundle=load_bundle,
        forecast=lambda *_: Frame(),
    )
    monkeypatch.setattr(entry, "_production", lambda: backend)
    monkeypatch.setattr(entry, "_parse_panel", lambda content: json.loads(content))
    args = dict(
        model=model,
        snapshot=snapshot,
        policy=policy,
        policy_sha256=entry.sha(policy),
        receipt_sha256=entry.sha(receipt_path),
        output=tmp_path / "forecast.csv",
    )
    return SimpleNamespace(
        args=args,
        calls=calls,
        backend=backend,
        bundle=bundle,
        receipt=receipt,
        receipt_path=receipt_path,
    )


def test_valid_forecast_and_provenance(case):
    proof = entry.predict(**case.args)
    assert case.calls == ["policy", "load", "policy"]
    assert proof["rows"] == 1 and proof["accuracy_evaluated"] is False
    assert proof["predictions_sha256"] == entry.sha(case.args["output"])
    sidecar = case.args["output"].with_name("forecast.csv.provenance.json")
    assert json.loads(sidecar.read_text()) == proof


@pytest.mark.parametrize("key", ["policy_sha256", "receipt_sha256"])
def test_bad_digest_before_load(case, key):
    case.args[key] = "0" * 64
    with pytest.raises(ValueError, match="digests"):
        entry.predict(**case.args)
    assert "load" not in case.calls


def test_source_tamper_before_load(case):
    def reject(_):
        raise ValueError("Exact frozen production source bytes required")

    case.backend.check_policy = reject
    with pytest.raises(ValueError, match="source bytes"):
        entry.predict(**case.args)
    assert "load" not in case.calls


def test_bundle_tamper_before_load(case):
    (case.args["model"] / "bundle.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        entry.predict(**case.args)
    assert "load" not in case.calls


@pytest.mark.parametrize(
    "target", ["snapshot", "policy", "bundle", "receipt", "inside"]
)
def test_output_alias_or_model_path_refused(case, target):
    paths = {
        "snapshot": case.args["snapshot"],
        "policy": case.args["policy"],
        "bundle": case.args["model"] / "bundle.joblib",
        "receipt": case.receipt_path,
        "inside": case.args["model"] / "new.csv",
    }
    case.args["output"] = paths[target]
    with pytest.raises(ValueError, match="Fresh output"):
        entry.predict(**case.args)
    assert "load" not in case.calls


@pytest.mark.parametrize("field", ["policy_sha256", "source_sha256", "quantiles"])
def test_loaded_bundle_identity_refused(case, field):
    case.bundle[field] = {"changed": 1}
    with pytest.raises(ValueError, match="Loaded checkpoint"):
        entry.predict(**case.args)
    assert not case.args["output"].exists()


def test_loaded_receipt_change_refused(case):
    case.backend.load_bundle = lambda _: (case.bundle, {**case.receipt, "extra": True})
    with pytest.raises(ValueError, match="Loaded checkpoint"):
        entry.predict(**case.args)


def test_input_revision_during_forecast_refused(case):
    def forecast(*_):
        case.args["snapshot"].write_bytes(b"revised")
        return Frame()

    case.backend.forecast = forecast
    with pytest.raises(ValueError, match="inputs changed"):
        entry.predict(**case.args)
    assert not case.args["output"].exists()


def test_existing_sidecar_refused_before_load(case):
    sidecar = case.args["output"].with_name("forecast.csv.provenance.json")
    sidecar.write_text("preserve")
    with pytest.raises(ValueError, match="Fresh output"):
        entry.predict(**case.args)
    assert "load" not in case.calls and sidecar.read_text() == "preserve"


def test_calibrated_stage_valid(case):
    case.receipt["stage"] = "calibrated_frozen"
    case.bundle["quantiles"] = {"FIXTURE": 1.0}
    case.receipt_path.write_text(json.dumps(case.receipt))
    case.args["receipt_sha256"] = entry.sha(case.receipt_path)
    assert entry.predict(**case.args)["stage"] == "calibrated_frozen"
