"""Software fixtures only; actual learned checkpoint replay is a cloud contract."""

from types import SimpleNamespace
import json
import pytest
from src.models import volatility_gate_inference as inference
from src.models.regime_interactions import BASE_FEATURES
from src.models.interaction_study import ASSETS, YEARS, sha


class Estimator:
    def predict(self, values):
        assert all(len(row) == len(BASE_FEATURES) for row in values)
        return [row[0] for row in values]


def test_saved_prediction_transform_strict_edge_and_target_independence():
    rows = [
        {
            **{name: 0.0 for name in BASE_FEATURES},
            "return_1": 0.003,
            "volatility_ratio": ratio,
            "target": object(),
        }
        for ratio in [1.19, 1.2, 1.200001]
    ]
    assert inference.predict_records(Estimator(), rows) == [0.0, 0.0, 0.006]
    rows[-1]["volatility_ratio"] = float("nan")
    with pytest.raises(ValueError):
        inference.predict_records(Estimator(), rows)


@pytest.mark.parametrize("defect", ["steps", "alpha", "intercept", "width"])
def test_retained_model_contract_rejects_changed_estimator(defect):
    model = SimpleNamespace(
        named_steps={
            "standardscaler": object(),
            "ridge": SimpleNamespace(alpha=10000, fit_intercept=False),
        },
        n_features_in_=len(BASE_FEATURES),
    )
    inference.validate_estimator(model)
    if defect == "steps":
        model.named_steps.pop("standardscaler")
    elif defect == "alpha":
        model.named_steps["ridge"].alpha = 1000
    elif defect == "intercept":
        model.named_steps["ridge"].fit_intercept = True
    else:
        model.n_features_in_ += 1
    with pytest.raises(ValueError):
        inference.validate_estimator(model)


def test_all_twelve_checkpoint_bytes_required_before_unpickle(tmp_path):
    hashes = {}
    for year in YEARS:
        for asset in ASSETS:
            name = f"linear_10000-{year}-{asset}.joblib"
            path = tmp_path / name
            path.write_bytes(b"fixture-not-pickle")
            hashes[name] = sha(path)
    assert (
        inference.checkpoint_contract({"checkpoint_sha256": hashes}, tmp_path) == hashes
    )
    name = next(iter(hashes))
    (tmp_path / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes"):
        inference.checkpoint_contract({"checkpoint_sha256": hashes}, tmp_path)
    hashes.pop(name)
    with pytest.raises(ValueError, match="twelve"):
        inference.checkpoint_contract({"checkpoint_sha256": hashes}, tmp_path)


def test_changed_parent_report_refuses_before_heavy_import_or_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(inference, "check_freeze", lambda *args: {})
    monkeypatch.setattr(inference, "load_inputs", lambda *args: ({}, {}, {}))
    (tmp_path / "saved-model-parity-plan.json").write_text(
        json.dumps({"schema": "saved_gate_parity_v1", "parent_report_sha256": "0" * 64})
    )
    report = tmp_path / "report.json"
    report.write_text("{}")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="report identity"):
        inference.run(None, None, None, tmp_path / "freeze.json", report, None, output)
    assert not output.exists()


def test_staging_refuses_dirty_source_before_private_reads(tmp_path, monkeypatch):
    from src import prepare_volatility_parity as staging

    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *args, **kwargs: "?? changed.py"
    )
    output = tmp_path / "stage"
    with pytest.raises(ValueError, match="Commit"):
        staging.prepare(tmp_path / "absent", output, "owner/private-inputs")
    assert not output.exists()


def test_staging_preserves_existing_payload(tmp_path):
    from src import prepare_volatility_parity as staging

    output = tmp_path / "stage"
    output.mkdir()
    sentinel = output / "receipt"
    sentinel.write_text("preserved")
    with pytest.raises(ValueError, match="Fresh"):
        staging.prepare(tmp_path / "absent", output, "owner/private-inputs")
    assert sentinel.read_text() == "preserved"
