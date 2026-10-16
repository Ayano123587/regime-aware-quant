"""Fresh workflow orchestration fixtures; no actual fits, arrays or data."""

import hashlib
import json
from types import SimpleNamespace
import pytest
from src import gated_workflow as workflow


@pytest.fixture
def case(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    rows = [
        {"d": "2017-01-03", "fixture": 1},
        {"d": "2024-12-31", "fixture": 2},
        {"d": "2025-12-31", "fixture": "not used for mean"},
    ]
    parent.write_text(json.dumps({"observations": rows}))
    monkeypatch.setattr(workflow, "PARENT_SHA256", workflow.sha(parent))
    raw = (
        json.dumps({"observations": rows[:2]}, separators=(",", ":")) + "\n"
    ).encode()
    parent.write_bytes(raw)
    backend = SimpleNamespace(
        DERIVED_SHA256=hashlib.sha256(raw).hexdigest(), sha=workflow.sha
    )
    calls = []

    def train(source, policy, out):
        calls.append("train")
        assert source.read_bytes() == raw
        out.mkdir()
        (out / "bundle.joblib").write_bytes(b"own fresh fixture never deserialized")
        receipt = {
            "stage": "mean_frozen",
            "mean_source_sha256": backend.DERIVED_SHA256,
            "training": {
                a: {
                    "rows": 1933,
                    "first_decision": "2017-03-29",
                    "last_decision": "2024-12-27",
                    "last_target_maturity": "2024-12-31",
                }
                for a in workflow.REFERENCE_STATES
            },
            "calibration_accessed": False,
            "test_accessed": False,
            "bundle_sha256": workflow.sha(out / "bundle.joblib"),
            "policy_sha256": workflow.POLICY_SHA256,
            "mean_model_state_sha256": workflow.REFERENCE_STATES.copy(),
        }
        workflow.write(out / "receipt.json", receipt)
        return receipt

    backend.train = train
    monkeypatch.setattr(workflow, "production", lambda _: backend)
    return SimpleNamespace(
        parent=parent,
        policy=tmp_path / "policy",
        out=tmp_path / "train",
        backend=backend,
        calls=calls,
    )


def test_fresh_training_freezes_own_artifact_without_old_bundle(case):
    result = workflow.train(case.parent, case.policy, case.out)
    assert result["mean_bundle_sha256"] == workflow.sha(case.out / "mean/bundle.joblib")
    assert result["mean_receipt_sha256"] == workflow.sha(case.out / "mean/receipt.json")
    assert result["reference_numerical_state_equal"] is True
    assert result["reported_accuracy_reused"] is False
    assert not (case.out / "calibration-release.json").exists()


def test_existing_training_output_rejected_before_private_read(case, monkeypatch):
    case.out.mkdir()
    monkeypatch.setattr(
        workflow, "production", lambda _: pytest.fail("No policy/model/source read")
    )
    with pytest.raises(ValueError, match="Fresh"):
        workflow.train(case.parent, case.policy, case.out)


def test_bad_parent_never_fits(case):
    case.parent.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pre2025 snapshot"):
        workflow.train(case.parent, case.policy, case.out)
    assert case.calls == [] and not case.out.exists()


def test_reference_mismatch_requires_explicit_own_run(case):
    original = case.backend.train

    def different(*args):
        receipt = original(*args)
        receipt["mean_model_state_sha256"] = {"different": "state"}
        path = args[-1] / "receipt.json"
        path.unlink()
        workflow.write(path, receipt)
        return receipt

    case.backend.train = different
    with pytest.raises(ValueError, match="numerical state differs"):
        workflow.train(case.parent, case.policy, case.out)
    assert not (case.out / "mean-freeze.json").exists()
    result = workflow.train(
        case.parent, case.policy, case.out.with_name("own-run"), True
    )
    assert (
        result["reference_numerical_state_equal"] is False
        and result["reported_accuracy_reused"] is False
    )


def test_calibration_requires_frozen_mean_before_full_source(case, monkeypatch):
    workflow.train(case.parent, case.policy, case.out)
    with pytest.raises(ValueError, match="freeze digest"):
        workflow.calibrate(
            case.out,
            "never-read-source",
            case.policy,
            "0" * 64,
            case.out.with_name("calibration"),
        )


def test_release_uses_own_mean_and_preserves_state(case, monkeypatch):
    workflow.train(case.parent, case.policy, case.out)
    from src import prepare_gated_calibration as source

    raw = b'{"software_fixture":true}\n'
    provenance = {
        "expected_decision_ids_sha256": "fixture-ids",
        "expected_counts": {a: 247 for a in workflow.REFERENCE_STATES},
    }

    def cutoff(parent):
        assert (case.out / "mean-freeze.json").exists()
        return raw, provenance

    monkeypatch.setattr(source, "cutoff_source", cutoff)

    def calibrate(model, snapshot, release, out):
        plan = json.loads(release.read_bytes())
        freeze = json.loads((case.out / "mean-freeze.json").read_bytes())
        assert plan["mean_bundle_sha256"] == freeze["mean_bundle_sha256"]
        assert plan["mean_receipt_sha256"] == freeze["mean_receipt_sha256"]
        assert plan["expected_counts"] == provenance["expected_counts"]
        assert snapshot.read_bytes() == raw
        out.mkdir()
        (out / "bundle.joblib").write_bytes(b"own calibrated fixture")
        receipt = {
            "stage": "calibrated_frozen",
            "bundle_sha256": workflow.sha(out / "bundle.joblib"),
            "mean_model_state_sha256": freeze["mean_model_state_sha256"],
        }
        workflow.write(out / "receipt.json", receipt)
        return receipt

    case.backend.calibrate = calibrate
    output = case.out.with_name("calibration")
    result = workflow.calibrate(
        case.out,
        case.parent,
        case.policy,
        workflow.sha(case.out / "mean-freeze.json"),
        output,
    )
    assert result["receipt_sha256"] == workflow.sha(output / "calibrated/receipt.json")
    assert result["reported_accuracy_reused"] is False


def test_parent_with_2025_poison_cannot_enter_training(case, monkeypatch):
    case.parent.write_bytes(
        b'{"observations":[{"d":"2025-01-02","poison": UNPARSEABLE}]}'
    )
    monkeypatch.setattr(
        workflow.json,
        "loads",
        lambda *_: pytest.fail("No 2025 parent parse before mean freeze"),
    )
    with pytest.raises(ValueError, match="pre2025 snapshot"):
        workflow.train(case.parent, case.policy, case.out)
    assert case.calls == [] and not case.out.exists()


@pytest.mark.parametrize(
    "field",
    [
        "parent_snapshot_sha256",
        "mean_source_sha256",
        "production_code_sha256",
        "workflow_sha256",
        "policy_sha256",
        "calibration_started",
        "test_accessed",
        "reported_accuracy_reused",
    ],
)
def test_semantic_freeze_tamper_refused_before_calibration_source(
    case, monkeypatch, field
):
    workflow.train(case.parent, case.policy, case.out)
    path = case.out / "mean-freeze.json"
    freeze = json.loads(path.read_bytes())
    freeze[field] = "tampered"
    path.write_text(json.dumps(freeze))
    from src import prepare_gated_calibration as source

    monkeypatch.setattr(
        source,
        "cutoff_source",
        lambda _: pytest.fail("No full2025 read before authenticatedfreeze"),
    )
    with pytest.raises(ValueError, match="frozen mean identities"):
        workflow.calibrate(
            case.out,
            "never-read-2025-parent",
            case.policy,
            workflow.sha(path),
            case.out.with_name("calibration"),
        )


@pytest.mark.parametrize(
    "field",
    [
        "policy_sha256",
        "mean_source_sha256",
        "training",
        "calibration_accessed",
        "test_accessed",
    ],
)
def test_own_receipt_recipe_tamper_refused_before_full_source(case, monkeypatch, field):
    workflow.train(case.parent, case.policy, case.out)
    receipt_path = case.out / "mean/receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = "tampered"
    receipt_path.write_text(json.dumps(receipt))
    freeze_path = case.out / "mean-freeze.json"
    freeze = json.loads(freeze_path.read_bytes())
    freeze["mean_receipt_sha256"] = workflow.sha(receipt_path)
    freeze_path.write_text(json.dumps(freeze))
    from src import prepare_gated_calibration as source

    monkeypatch.setattr(
        source, "cutoff_source", lambda _: pytest.fail("No calibration source read")
    )
    with pytest.raises(ValueError, match="recipe/cohort"):
        workflow.calibrate(
            case.out,
            "never-read-source",
            case.policy,
            workflow.sha(freeze_path),
            case.out.with_name("calibration"),
        )
