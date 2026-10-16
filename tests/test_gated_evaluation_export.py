"""Software fixtures for artifact replay barriers; never model training or scoring."""

import copy
import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models import gated_evaluation_export as export


def fixture():
    ids = [
        (
            asset,
            (date(2026, 1, 2) + timedelta(days=i)).isoformat(),
            (date(2026, 1, 3) + timedelta(days=i)).isoformat(),
            (date(2026, 1, 4) + timedelta(days=i)).isoformat(),
        )
        for asset in export.evaluation.ASSETS
        for i in range(194)
    ]
    rows = [
        dict(zip(export.COLUMNS, (*key, 1.0, 0.9, 0.8, 0.0, 0.8, 1.2, 0.9, 0.7, 1.1)))
        for key in ids
    ]

    def metric(n):
        return {
            "selected": {"rows": n, "rmse": 0.1},
            "ungated_same_mean": {"rows": n, "rmse": 0.2},
            "zero": {"rows": n, "rmse": 1.0},
            "selected_intervals": {
                "rows": n,
                "nominal_coverage": 0.9,
                "coverage": 1.0,
                "mean_width": 0.4,
                "mean_interval_score": 0.4,
            },
        }

    def legacy(n):
        return {
            "regression": {"rows": n, "rmse": 0.1},
            "intervals": {
                "rows": n,
                "nominal_coverage": 0.9,
                "coverage": 1.0,
                "mean_width": 0.4,
                "mean_interval_score": 0.4,
            },
        }

    original = {
        "metrics": {
            "aggregate": metric(582),
            "per_asset": {a: metric(194) for a in export.evaluation.ASSETS},
        },
        "true_legacy_comparator": {
            **legacy(582),
            "per_asset": {a: legacy(194) for a in export.evaluation.ASSETS},
        },
    }
    return rows, ids, original


def test_all582_columns_recompute_against_hand_calculated_receipt():
    rows, ids, original = fixture()
    export.verify_rows(rows, ids, original)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "reordered",
        "extra",
        "target",
        "selected",
        "ungated_same_mean",
        "zero",
        "selected_lower",
        "selected_upper",
        "true_legacy",
        "true_legacy_lower",
        "true_legacy_upper",
        "nonfinite",
        "aggregate",
        "per_asset",
    ],
)
def test_changed_predictions_or_aggregates_never_export(defect):
    rows, ids, original = fixture()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[-2]
    elif defect == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "extra":
        rows[0]["unexpected"] = 1
    elif defect == "nonfinite":
        rows[0]["target"] = float("nan")
    elif defect == "aggregate":
        original["metrics"]["aggregate"]["selected"]["rmse"] += 0.1
    elif defect == "per_asset":
        original["true_legacy_comparator"]["per_asset"][export.evaluation.ASSETS[0]][
            "intervals"
        ]["mean_width"] += 0.1
    else:
        rows[0][defect] += 1.0
    with pytest.raises(ValueError):
        export.verify_rows(rows, ids, original)


def test_csv_roundtrip_retains_all_float_columns_and_metrics():
    rows, ids, original = fixture()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=export.COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    restored = list(csv.DictReader(stream))
    for row in restored:
        for column in export.COLUMNS[4:]:
            row[column] = float(row[column])
    export.verify_rows(restored, ids, original)
    assert restored == rows


def test_original_receipt_hash_precedes_any_source_parse(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("{}")
    original_sha = export.evaluation.sha

    def only_result(p):
        assert p == path
        return original_sha(p)

    monkeypatch.setattr(export.evaluation, "sha", only_result)
    monkeypatch.setattr(
        export.evaluation,
        "read",
        lambda *a: pytest.fail("source parsed before receipt identity"),
    )
    with pytest.raises(ValueError, match="Exact completed"):
        export.authenticate_result(path, "never-open-execution", "0" * 64)


def test_runtime_mismatch_is_rejected_before_model_import(monkeypatch):
    monkeypatch.setattr(export.platform, "python_version", lambda: "fixture")
    monkeypatch.setattr(export.importlib.metadata, "version", lambda *a: "fixture")
    with pytest.raises(ValueError, match="before model load"):
        export.check_runtime({"python": "wrong"})


def test_postflight_rechecks_current_production_sources_before_write(tmp_path):
    path = tmp_path / "immutable.json"
    path.write_text("software fixture")

    def changed_policy(_):
        raise ValueError("current production source changed")

    production = SimpleNamespace(
        model_state_fingerprints=lambda _: {"fixture": "state"},
        check_policy=changed_policy,
    )
    with pytest.raises(ValueError, match="production source changed"):
        export.check_postflight(
            {str(path): export.evaluation.sha(path)},
            production,
            "policy",
            {"models": {}},
            {"fixture": "state"},
        )
