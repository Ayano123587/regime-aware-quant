"""Authenticated original uncertainty-v2 comparator; no training or selection."""

import hashlib
import json
import math
from pathlib import Path

BUNDLE_SHA256 = "b5c263242f18a99307436df1545e5a46434deaf12fb8fd877c0e565a8289010c"
SELECTION_SHA256 = "4868528c4e7658211b26490297cbacbecf13a047a71910f82f28d778ca3fe33b"
ASSETS = ["FXEURCAD", "FXGBPCAD", "FXAUDCAD"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    import joblib

    return joblib.load(path)


def _production():
    from src.models import gated_forecaster

    return gated_forecaster


def authenticate(artifacts):
    """Verify fixed own-run identities before deserializing an estimator."""
    if set(artifacts) != {"bundle", "selection"}:
        raise ValueError("Original bundle and selection required")
    expected = {"bundle": BUNDLE_SHA256, "selection": SELECTION_SHA256}
    if {k: sha(p) for k, p in artifacts.items()} != expected:
        raise ValueError("Original legacy artifact bytes changed")
    selection = json.loads(Path(artifacts["selection"]).read_bytes())
    if selection.get("selected") != {"kind": "ridge", "alpha": 10000}:
        raise ValueError("Original selected ridge recipe required")
    checkpoint = _load(artifacts["bundle"])
    if (
        checkpoint.get("version") != "uncertainty_v2"
        or checkpoint.get("assets") != ASSETS
        or checkpoint.get("selection_sha256") != SELECTION_SHA256
        or set(checkpoint.get("bundles", {})) != set(ASSETS)
    ):
        raise ValueError("Original legacy checkpoint schema changed")
    calibration = selection.get("calibration", {})
    if set(calibration) != set(ASSETS):
        raise ValueError("Complete original calibration required")
    for asset in ASSETS:
        info = calibration[asset]
        if (
            info.get("rows") != 247
            or not str(info.get("last_training_label", "")).startswith("2024-12-31")
            or not str(info.get("first_calibration_decision", "")).startswith(
                "2025-01-02"
            )
            or not str(info.get("last_calibration_label", "")).startswith("2025-12-31")
        ):
            raise ValueError("Original chronological calibration identity changed")
        for key in ("absolute_quantile", "normalized_quantile"):
            value = info.get(key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or checkpoint["bundles"][asset].get(key) != value
            ):
                raise ValueError("Original legacy interval quantile changed")
    models = {a: checkpoint["bundles"][a]["model"] for a in ASSETS}
    states = _production().model_state_fingerprints(models)
    return checkpoint, {
        "schema": "authenticated_legacy_forecast_v1",
        "artifact_sha256": expected,
        "model_state_sha256": states,
        "calibration_identity": calibration,
        "weights_refitted": False,
    }


def predict_rows(old, rows):
    """Original ridge and normalized interval formula on already causal rows."""
    import numpy as np
    from src.features.multi_asset import FEATURES
    from src.models.regime_interactions import BASE_FEATURES

    if FEATURES != BASE_FEATURES:
        raise ValueError("Original feature order changed")
    if not np.isfinite(rows[FEATURES].to_numpy(float)).all():
        raise ValueError("Finite original legacy features required")
    values = old["model"].predict(rows[FEATURES])
    widths = (
        np.maximum(rows.ewma_volatility.to_numpy(), 1e-5) * old["normalized_quantile"]
    )
    if not np.isfinite(values).all() or not np.isfinite(widths).all():
        raise ValueError("Finite original legacy predictions required")
    return values, values - widths, values + widths


def forecast(panel, ids, artifacts):
    """Return all requested matured IDs with original means and original intervals."""
    checkpoint, _ = authenticate(artifacts)
    import numpy as np
    import pandas as pd
    from src.data.multi_asset import validate_panel
    from src.features.multi_asset import build_panel_features, FEATURES
    from src.models.regime_interactions import BASE_FEATURES

    if FEATURES != BASE_FEATURES:
        raise ValueError("Original feature order changed")
    panel = validate_panel(panel)
    context = validate_panel(checkpoint["context"])
    if context.date.max().strftime("%Y-%m-%d") != "2025-12-31":
        raise ValueError("Original pre2026 context required")
    overlap = panel.merge(context, on="date", suffixes=("_new", "_old"))
    if any(
        not np.allclose(overlap[a + "_new"], overlap[a + "_old"], atol=1e-12, rtol=0)
        for a in ASSETS
    ):
        raise ValueError("Legacy saved context revised")
    merged = validate_panel(
        pd.concat([context, panel])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    frames = build_panel_features(merged)
    parts = []
    for asset in ASSETS:
        rows = frames[asset].loc[
            frames[asset].date.isin(panel.date)
            & (frames[asset].date > context.date.max())
        ]
        if rows.empty:
            continue
        values, lower, upper = predict_rows(checkpoint["bundles"][asset], rows)
        parts.append(
            pd.DataFrame(
                {
                    "date": rows.date,
                    "asset": asset,
                    "prediction": values,
                    "lower": lower,
                    "upper": upper,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    predictions = {}
    for row in output.to_dict("records"):
        key = (row["asset"], row["date"].strftime("%Y-%m-%d"))
        if key in predictions:
            raise ValueError("Duplicate legacy prediction")
        predictions[key] = row
    features = {}
    for asset, frame in frames.items():
        for row in frame.to_dict("records"):
            key = (asset, row["date"].strftime("%Y-%m-%d"))
            features[key] = row
    return align_rows(predictions, features, ids)


def align_rows(predictions, features, ids):
    """Preserve the released full decision order and delayed source IDs."""
    if len(set(map(tuple, ids))) != len(ids):
        raise ValueError("Unique expected legacy decision IDs required")
    result = []
    for asset, day, execution, maturity in ids:
        key = (asset, day)
        if key not in predictions or key not in features:
            raise ValueError("Complete legacy forecast cohort required")
        row, feature = predictions[key], features[key]
        if (
            feature["execution_date"].strftime("%Y-%m-%d") != execution
            or feature["target_date"].strftime("%Y-%m-%d") != maturity
        ):
            raise ValueError("Legacy delayed source chronology differs")
        result.append(
            {
                "asset": asset,
                "date": day,
                "execution_date": execution,
                "target_date": maturity,
                **{k: float(row[k]) for k in ("prediction", "lower", "upper")},
            }
        )
    return result
