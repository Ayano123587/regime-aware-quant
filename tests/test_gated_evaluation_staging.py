"""Software contract fixtures only; no real source, model loading or benchmark."""

import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from src import prepare_gated_evaluation as staging
from src.cloud import gated_evaluation_runner as cloud


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def arguments(tmp_path):
    return dict(
        development_parent="never-open-development",
        observed_parent="never-open-future",
        calibrated_receipt="never-open-receipt",
        legacy_proof="never-open-proof",
        frozen_inputs="never-open-contract",
        frozen_inputs_sha256="0" * 64,
        output=tmp_path / "stage",
        dataset="owner/dataset",
        kernel="owner/evaluation",
        legacy_bundle="never-unpickle-bundle",
        legacy_selection="never-open-selection",
    )


def test_dirty_git_precedes_every_private_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *a, **k: " M source.py"
    )
    monkeypatch.setattr(
        staging, "check_frozen_inputs", lambda *a: pytest.fail("private read")
    )
    with pytest.raises(ValueError, match="Commit source"):
        staging.prepare(**arguments(tmp_path))
    assert not (tmp_path / "stage").exists()


def test_missing_frozen_proof_blocks_future_source(tmp_path, monkeypatch):
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "")

    def reject(*args):
        raise ValueError("successful legacy proof required")

    monkeypatch.setattr(staging, "check_frozen_inputs", reject)
    monkeypatch.setattr(
        staging, "derive_source", lambda *a: pytest.fail("future source read")
    )
    with pytest.raises(ValueError, match="successful legacy"):
        staging.prepare(**arguments(tmp_path))
    assert not (tmp_path / "stage").exists()


def test_contract_hash_is_checked_before_receipt_or_proof(tmp_path, monkeypatch):
    contract = put(tmp_path / "contract.json", {})
    original = staging.evaluation.sha

    def only_contract(path):
        assert Path(path) == contract
        return original(path)

    monkeypatch.setattr(staging.evaluation, "sha", only_contract)
    with pytest.raises(ValueError, match="frozen-inputs SHA"):
        staging.check_frozen_inputs(
            contract, "0" * 64, "poison-receipt", "poison-proof"
        )


@pytest.mark.parametrize(
    "contract",
    [{}, {"schema": "gated_evaluation_frozen_inputs_v1"}, {"legacy_contract": None}],
)
def test_placeholder_contracts_never_release_source(tmp_path, contract):
    path = put(tmp_path / "inputs.json", contract)
    with pytest.raises(ValueError, match="Complete frozen"):
        staging.check_frozen_inputs(
            path, staging.evaluation.sha(path), "never-read", "never-read"
        )


def fixture_rows(days):
    # Deliberately nonnumeric: the staging cohort must use dates, not values.
    return [
        {
            "d": d,
            **{a: {"v": "software-fixture-only"} for a in staging.evaluation.ASSETS},
        }
        for d in days
    ]


def test_full582_date_only_derivation_and_six_pending(tmp_path, monkeypatch):
    development = [
        (date(2017, 1, 3) + timedelta(days=i)).isoformat() for i in range(2243)
    ] + ["2025-12-31"]
    future = []
    day = date(2026, 1, 2)
    while day <= date(2026, 10, 14):
        if day.weekday() < 5:
            future.append(day.isoformat())
        day += timedelta(days=1)
    while len(future) > 196:
        future.pop(10)
    dp = put(tmp_path / "development.json", {"observations": fixture_rows(development)})
    fp = put(tmp_path / "future.json", {"observations": fixture_rows(future)})
    monkeypatch.setattr(
        staging,
        "PARENTS",
        {
            "development": staging.evaluation.sha(dp),
            "observed2026": staging.evaluation.sha(fp),
        },
    )
    prefix = staging.evaluation.calibration_prefix_sha256(fixture_rows(development))
    monkeypatch.setattr(staging, "PREFIX", prefix)
    raw, proof = staging.derive_source(dp, fp)
    assert len(json.loads(raw)["observations"]) == 2440
    assert proof["expected_counts"] == {a: 194 for a in staging.evaluation.ASSETS}
    assert proof["maximum_source_date"] == "2026-10-14"
    assert proof["expected_unmatured_ids_sha256"]
    assert proof["source_parent_counts"] == {"development": 2244, "observed2026": 196}


def test_both_parent_hashes_before_any_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(staging.evaluation, "sha", lambda _: "0" * 64)
    monkeypatch.setattr(
        staging.evaluation, "read", lambda _: pytest.fail("untrusted source parsed")
    )
    with pytest.raises(ValueError, match="Both original"):
        staging.derive_source("first", "second")


@pytest.mark.parametrize("expected", [None, "", "0" * 63])
def test_unrendered_runner_rejects_before_copy(tmp_path, expected):
    with pytest.raises(ValueError, match="manifest pin"):
        cloud.restore(tmp_path, tmp_path / "workspace", "never-read-program", expected)
    assert not (tmp_path / "workspace").exists()


def test_changed_manifest_rejects_before_json_parse_and_copy(tmp_path):
    (tmp_path / "source-manifest.json").write_text("invalid JSON")
    with pytest.raises(ValueError, match="exact evaluation"):
        cloud.restore(tmp_path, tmp_path / "workspace", "never-read-program", "0" * 64)
    assert not (tmp_path / "workspace").exists()


def test_runtime_mismatch_before_any_model_loading(monkeypatch):
    monkeypatch.setattr(cloud.platform, "python_version", lambda: "3.fixture")
    monkeypatch.setattr(cloud.importlib.metadata, "version", lambda _: "fixture")
    with pytest.raises(ValueError, match="before model loading"):
        cloud.check_runtime({"python": "wrong"})


def test_exact_runtime_contract(monkeypatch):
    monkeypatch.setattr(cloud.platform, "python_version", lambda: "3.fixture")
    monkeypatch.setattr(cloud.importlib.metadata, "version", lambda _: "fixture")
    expected = {
        "python": "3.fixture",
        **{
            n: "fixture"
            for n in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "threadpoolctl",
            )
        },
    }
    assert cloud.check_runtime(expected) == expected


def transport_fixture(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    root = Path(staging.__file__).resolve().parents[1]
    names = sorted(staging.evaluation.PRODUCTION_FILES) + [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/gated_evaluation.py",
        "src/models/gated_legacy.py",
        "src/prepare_gated_evaluation.py",
        "src/cloud/gated_evaluation_runner.py",
        "docs/gated-forecast-policy.json",
    ]
    rows = []
    for name in names + [
        "private/" + n
        for n in (
            "source.json",
            "provenance.json",
            "release.json",
            "frozen-inputs.json",
            "calibrated-receipt.json",
            "legacy-proof.json",
            "legacy-bundle.joblib",
            "legacy-selection.json",
        )
    ]:
        raw = (
            (root / name).read_bytes()
            if not name.startswith("private/")
            else b"software fixture, never deserialized"
        )
        flat = name.replace("/", "__")
        (source / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    path = put(
        source / "source-manifest.json",
        {
            "schema": "gated_evaluation_transport_v1",
            "dispatchable": True,
            "weights_refitted": False,
            "files": rows,
        },
    )
    digest = cloud.sha(path)
    program = tmp_path / "run.py"
    program.write_bytes(
        Path(cloud.__file__)
        .read_bytes()
        .replace(
            b"EXPECTED_MANIFEST_SHA256 = None",
            ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode(),
        )
    )
    return source, program, digest


def test_exact_manifest_template_restores_without_import_or_load(tmp_path):
    source, program, digest = transport_fixture(tmp_path)
    workspace = tmp_path / "workspace"
    cloud.restore(source, workspace, program, digest)
    assert (
        workspace / "private/legacy-bundle.joblib"
    ).read_bytes() == b"software fixture, never deserialized"


@pytest.mark.parametrize("defect", ["program", "payload", "duplicate"])
def test_transport_tamper_fails_before_workspace_copy(tmp_path, defect):
    source, program, digest = transport_fixture(tmp_path)
    if defect == "program":
        program.write_bytes(program.read_bytes() + b"\n# changed")
    elif defect == "payload":
        (source / "private__source.json").write_bytes(b"changed")
    else:
        path = source / "source-manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["files"].append(manifest["files"][0])
        path.write_text(json.dumps(manifest))
        digest = cloud.sha(path)
    with pytest.raises(ValueError):
        cloud.restore(source, tmp_path / "workspace", program, digest)
    assert not (tmp_path / "workspace").exists()


def test_runner_runtime_gate_precedes_model_discovery(tmp_path, monkeypatch):
    workspace = tmp_path / "evaluation-source"
    (workspace / "private").mkdir(parents=True)
    path = put(
        workspace / "private/frozen-inputs.json",
        {"runtime": {"python": "incompatible"}},
    )
    monkeypatch.setattr(
        cloud, "restore", lambda *a: {"frozen_inputs_sha256": cloud.sha(path)}
    )
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        cloud, "locate", lambda *a: pytest.fail("model discovery before runtime gate")
    )
    with pytest.raises(ValueError, match="before model loading"):
        cloud.main(tmp_path, tmp_path, "unused")


def compatibility_fixture():
    receipt = {
        "mean_model_state_sha256": {
            a: str(i) * 64 for i, a in enumerate(staging.evaluation.ASSETS)
        },
        "calibration_decision_ids_sha256": "c" * 64,
        "calibration": {
            a: {"normalized_quantile": float(i + 1)}
            for i, a in enumerate(staging.evaluation.ASSETS)
        },
    }
    proof = {
        "frozen_mean_model_state_sha256": copy.deepcopy(
            receipt["mean_model_state_sha256"]
        ),
        "historical_rows_are_future_forecasts": False,
        "historical_row_mechanics": [
            {
                "asset": a,
                "rows": 5,
                "direct_formula_max_delta": 0.0,
                "reload_max_delta": 0.0,
            }
            for a in staging.evaluation.ASSETS
        ],
        "cross_runtime_current_mean_calibration": {
            "decision_ids_sha256": receipt["calibration_decision_ids_sha256"],
            "accuracy_evaluated": False,
            "per_asset": [
                {
                    "asset": a,
                    "rows": 247,
                    "direct_formula_max_delta": 0.0,
                    "frozen_quantile": receipt["calibration"][a]["normalized_quantile"],
                    "reproduced_quantile": receipt["calibration"][a][
                        "normalized_quantile"
                    ],
                    "quantile_abs_delta": 0.0,
                }
                for a in staging.evaluation.ASSETS
            ],
        },
    }
    return proof, receipt


def test_complete_measured_compatibility_contract():
    proof, receipt = compatibility_fixture()
    staging.check_compatibility(proof, receipt)


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_ids",
        "wrong_states",
        "missing_historical",
        "historical_count",
        "historical_negative",
        "historical_reload_negative",
        "historical_nonfinite",
        "future_claim",
        "accuracy_claim",
        "negative_prediction",
        "negative_quantile_delta",
        "false_quantile_delta",
        "wrong_frozen_quantile",
        "changed_reproduced_quantile",
        "missing_asset",
    ],
)
def test_poisoned_compatibility_cannot_release_future_source(defect):
    proof, receipt = compatibility_fixture()
    compat = proof["cross_runtime_current_mean_calibration"]
    first = compat["per_asset"][0]
    historical = proof["historical_row_mechanics"][0]
    if defect == "wrong_ids":
        compat["decision_ids_sha256"] = "wrong"
    elif defect == "wrong_states":
        proof["frozen_mean_model_state_sha256"][staging.evaluation.ASSETS[0]] = "wrong"
    elif defect == "missing_historical":
        proof.pop("historical_row_mechanics")
    elif defect == "historical_count":
        historical["rows"] = 4
    elif defect == "historical_negative":
        historical["direct_formula_max_delta"] = -1.0
    elif defect == "historical_reload_negative":
        historical["reload_max_delta"] = -1.0
    elif defect == "historical_nonfinite":
        historical["reload_max_delta"] = float("nan")
    elif defect == "future_claim":
        proof["historical_rows_are_future_forecasts"] = True
    elif defect == "accuracy_claim":
        compat["accuracy_evaluated"] = True
    elif defect == "negative_prediction":
        first["direct_formula_max_delta"] = -1.0
    elif defect == "negative_quantile_delta":
        first["quantile_abs_delta"] = -1.0
    elif defect == "false_quantile_delta":
        first["quantile_abs_delta"] = 1e-13
    elif defect == "wrong_frozen_quantile":
        first["frozen_quantile"] += 1.0
        first["reproduced_quantile"] += 1.0
    elif defect == "changed_reproduced_quantile":
        first["reproduced_quantile"] += 1.0
    else:
        compat["per_asset"].pop()
    with pytest.raises(ValueError):
        staging.check_compatibility(proof, receipt)


def test_actual_mounted_legacy_proof_matches_frozen_bytes(tmp_path):
    mounted = put(
        tmp_path / "authenticated-legacy-proof.json", {"software_fixture": True}
    )
    staged = tmp_path / "staged.json"
    staged.write_bytes(mounted.read_bytes())
    assert (
        cloud.check_mounted_legacy_proof(tmp_path, staged, cloud.sha(staged)) == mounted
    )


@pytest.mark.parametrize(
    "defect", ["missing", "ambiguous", "mounted_mismatch", "staged_mismatch"]
)
def test_unusable_mounted_proof_blocks_before_model_discovery(
    tmp_path, defect, monkeypatch
):
    original = put(tmp_path / "staged.json", {"fixture": "expected"})
    digest = cloud.sha(original)
    inputs = tmp_path / "input"
    inputs.mkdir()
    mounted = inputs / "authenticated-legacy-proof.json"
    if defect != "missing":
        mounted.write_bytes(original.read_bytes())
    if defect == "ambiguous":
        (inputs / "duplicate").mkdir()
        (inputs / "duplicate/authenticated-legacy-proof.json").write_bytes(
            original.read_bytes()
        )
    elif defect == "mounted_mismatch":
        mounted.write_bytes(b"changed")
    elif defect == "staged_mismatch":
        original.write_bytes(b"changed")
    with pytest.raises(ValueError):
        cloud.check_mounted_legacy_proof(inputs, original, digest)


def test_model_artifact_lookup_refuses_ambiguity(tmp_path):
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "bundle.joblib").write_bytes(b"software fixture, not a model")
    with pytest.raises(ValueError, match="unique"):
        cloud.locate(
            tmp_path,
            "bundle.joblib",
            hashlib.sha256(b"software fixture, not a model").hexdigest(),
        )


def test_runner_has_one_renderable_manifest_assignment():
    template = Path(cloud.__file__).read_text(encoding="utf-8")
    assert template.count("EXPECTED_MANIFEST_SHA256 = None") == 1
