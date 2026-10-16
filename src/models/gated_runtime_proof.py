"""Cross-runtime checks against the frozen complete 2025 calibration artifact."""

from pathlib import Path
import hashlib
import json

A_RECEIPT = "1f92dbb7594f47a289c5a3df09a2b47c49fb41035620686b55825362e938ac86"
A_BUNDLE = "bc918cc4f3940fb56d25b9710c6a7b9c9201160d5b30db9459946927dbfb4be0"
B_RECEIPT = "9b7efb90cfd1a90d087d9e0a810ef98b2daef87ba0f11784565a16c157955462"
B_BUNDLE = "87993d065d29b0e0e10ba71771f8fe3875e92d2a2361f0cb113350cb4818c4a5"
SOURCE = "6f0439d2d3f323ad7aecd21803eb3a27d9a8ac042e277f46db2701cac0058c36"
IDS = "caf7c827abd65d7ffe689bb174c2f37afaa6adc10f5d9f70d5146bee80f71ac3"


def verify(source, inputs):
    """Authenticate A/B/source before imports or model deserialization."""
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source) != SOURCE:
        raise ValueError("Frozen full 2025 source required")
    models = []
    for receipt_hash, bundle_hash in [(A_RECEIPT, A_BUNDLE), (B_RECEIPT, B_BUNDLE)]:
        matches = [
            p for p in Path(inputs).rglob("receipt.json") if sha(p) == receipt_hash
        ]
        if len(matches) != 1 or sha(matches[0].parent / "bundle.joblib") != bundle_hash:
            raise ValueError("Original A/B own-run artifacts required")
        models.append(matches[0].parent)
    from src.models import gated_forecaster as production
    from src.models.regime_interactions import BASE_FEATURES
    from src.data.multi_asset import parse_panel
    from src.features.multi_asset import build_panel_features
    import numpy as np

    a, ar = production.load_bundle(models[0])
    b, br = production.load_bundle(models[1])
    if (
        production.model_state_fingerprints(a["models"])
        != ar["mean_model_state_sha256"]
        or production.model_state_fingerprints(b["models"])
        != ar["mean_model_state_sha256"]
        or br["mean_model_state_sha256"] != ar["mean_model_state_sha256"]
    ):
        raise ValueError("A/B canonical learned state changed across runtime")
    content = Path(source).read_bytes()
    observations = json.loads(content)["observations"]
    if any(r["d"] > "2025-12-31" for r in observations):
        raise ValueError("No evaluation source allowed")
    ids, digest, counts = production.calibration_source_cohort(observations)
    if digest != IDS or counts != {x: 247 for x in production.ASSETS}:
        raise ValueError("Full741 source cohort required")
    panel = parse_panel(content)
    frames = build_panel_features(panel)
    forecasts = production.forecast(a, panel)
    actual = []
    reports = []
    for asset in production.ASSETS:
        frame = frames[asset]
        rows = frame.loc[
            (frame.date.dt.year == 2025)
            & (frame.execution_date.dt.year == 2025)
            & (frame.target_date.dt.year == 2025)
            & frame.target.notna()
        ]
        if len(rows) != 247:
            raise ValueError("Complete calibration feature cohort required")
        records = production.frame_records(rows)
        actual.extend(
            (asset, r["date"], r["execution_date"], r["target_date"]) for r in records
        )
        model = a["models"][asset]
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        x = rows[BASE_FEATURES].to_numpy(float)
        raw = ((x - scaler.mean_) / scaler.scale_) @ ridge.coef_
        direct = raw * 2 * (rows.volatility_ratio.to_numpy() > 1.2)
        joined = rows[["date", "target", "ewma_volatility"]].merge(
            forecasts.loc[forecasts.asset == asset, ["date", "prediction"]],
            on="date",
            validate="one_to_one",
        )
        delta = float(np.max(np.abs(direct - joined.prediction.to_numpy())))
        residual = np.abs(
            joined.target.to_numpy() - joined.prediction.to_numpy()
        ) / np.maximum(joined.ewma_volatility.to_numpy(), 1e-5)
        q = production.quantile90(residual.tolist())
        reference = br["calibration"][asset]["normalized_quantile"]
        qdelta = abs(q - reference)
        if delta > 1e-14 or qdelta > 1e-12 or b["quantiles"][asset] != reference:
            raise ValueError(
                "Cross-runtime mean mathematics or frozen calibration quantile differs"
            )
        reports.append(
            {
                "asset": asset,
                "rows": 247,
                "direct_formula_max_delta": delta,
                "reproduced_quantile": q,
                "frozen_quantile": reference,
                "quantile_abs_delta": qdelta,
            }
        )
    if actual != ids:
        raise ValueError("Full calibration IDs differ")
    return {
        "a_receipt_sha256": A_RECEIPT,
        "a_bundle_sha256": A_BUNDLE,
        "b_receipt_sha256": B_RECEIPT,
        "b_bundle_sha256": B_BUNDLE,
        "source_sha256": SOURCE,
        "decision_ids_sha256": IDS,
        "rows": 741,
        "mean_state_exact": True,
        "per_asset": reports,
        "mean_refitted": False,
        "accuracy_evaluated": False,
        "test_accessed": False,
    }
