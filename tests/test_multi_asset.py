import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.data.multi_asset import ASSETS, parse_panel, validate_panel
from src.features.multi_asset import FEATURES, build_panel_features
from src.models.advanced import feature_split, predict_with_bridge, final_test, forecast_panel
from src.models.multi_asset import fit_asset, predict_asset, REGIME_FEATURES
from src.models.portfolio import RiskPolicy, target_weights, rebalance, simulate
from src.models.drift import DelayedResidualQueue


def panel(n=420):
    rng = np.random.default_rng(42)
    values = np.exp(np.cumsum(rng.normal(0, .003, (n, 3)), axis=0))
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, tz="UTC"), **{asset: values[:, k] for k, asset in enumerate(ASSETS)}})


def test_causal_peer_features_and_exact_delayed_label():
    raw = panel()
    full, prefix = build_panel_features(raw), build_panel_features(raw.iloc[:230])
    for asset in ASSETS:
        np.testing.assert_allclose(full[asset].iloc[:len(prefix[asset])][FEATURES], prefix[asset][FEATURES], atol=1e-12)
        frame = prefix[asset]
        assert frame.target.iloc[-2:].isna().all()
        first = full[asset].iloc[0]
        j = raw.index[raw.date == first.date][0]
        assert first.target == pytest.approx(np.log(raw[asset].iloc[j + 2] / raw[asset].iloc[j + 1]))
        peers = [other for other in ASSETS if other != asset]
        assert first.peer_return_lag1 == pytest.approx(np.log(raw[peers].iloc[j - 1] / raw[peers].iloc[j - 2]).mean())
        assert first.date < first.execution_date < first.target_date


def test_market_alignment_missing_values_and_gaps():
    row = {"d": "2020-01-01", **{asset: {"v": "1"} for asset in ASSETS}}
    del row[ASSETS[1]]
    with pytest.raises(ValueError, match="alignment"):
        parse_panel(json.dumps({"observations": [row]}))
    for value in [np.nan, np.inf, 0, -1]:
        raw = panel(30)
        raw.loc[3, ASSETS[0]] = value
        with pytest.raises(ValueError):
            validate_panel(raw)
    raw = panel(30)
    raw["date"] = pd.date_range("2020-01-01", periods=30, freq="14D", tz="UTC")
    with pytest.raises(ValueError, match="gap"):
        validate_panel(raw)


def test_purge_and_hmm_bridge_continuity_and_label_independence(tmp_path):
    frame = build_panel_features(panel())[ASSETS[0]]
    train, validation, bridge = feature_split(frame, "2020-10-01", "2021-02-01")
    assert train.target_date.max() < validation.date.min()
    assert len(bridge) == 2
    bundle = fit_asset(train, {"max_depth": 2, "reg_lambda": 20}, iterations=3)
    joined = pd.concat([bridge, validation], ignore_index=True)
    direct = predict_asset(bundle, joined)
    bridged = predict_with_bridge(bundle, bridge, validation)
    for a, b in zip(direct, bridged):
        np.testing.assert_allclose(a[len(bridge):], b)
    first = bundle["hmm"].filter(bridge[REGIME_FEATURES].to_numpy(), bundle["last_posterior"])[-1]
    chunked = predict_asset(bundle, validation, previous=first)
    for a, b in zip(chunked, bridged):
        np.testing.assert_allclose(a, b, atol=1e-12)
    changed = validation.copy()
    changed["target"] = np.inf
    for a, b in zip(predict_asset(bundle, changed, previous=first), chunked):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(chunked[2].sum(axis=1), 1)
    checkpoint = tmp_path / "model.joblib"
    joblib.dump(bundle, checkpoint)
    np.testing.assert_allclose(predict_asset(joblib.load(checkpoint), joined)[0], direct[0])


def test_vol_target_caps_and_invalid_prediction_shapes():
    history = np.random.default_rng(9).normal(0, .004, (60, 3))
    policy = RiskPolicy(annual_volatility=.5, gross_cap=.8, per_asset_cap=.3)
    weights = target_weights([.002, -.003, .004], history, policy)
    assert np.abs(weights).sum() <= .8 + 1e-12
    assert np.abs(weights).max() <= .3 + 1e-12
    np.testing.assert_array_equal(target_weights([1, 1, 1], history, RiskPolicy(annual_volatility=0)), np.zeros(3))
    with pytest.raises(ValueError):
        target_weights([[1, 2, 3]], history, policy)
    with pytest.raises(ValueError):
        RiskPolicy(gross_cap=1.5)


def test_post_cost_nav_exact_cash_and_reversal_accounting():
    new, cash, cost, turnover = rebalance(100., np.zeros(3), [.5, -.25, 0], 10)
    nav = 100 / (1 + .001 * .75)
    np.testing.assert_allclose(new, np.array([.5, -.25, 0]) * nav)
    assert cash + new.sum() == pytest.approx(nav)
    assert cost == pytest.approx(turnover * .001)
    assert nav == pytest.approx(100 - cost)
    reversed_values, reversed_cash, reversed_cost, traded = rebalance(nav, new, [-.5, .25, 0], 10)
    assert traded == pytest.approx(np.abs(reversed_values - new).sum())
    assert reversed_cash + reversed_values.sum() == pytest.approx(nav - reversed_cost)
    assert np.abs(reversed_values / (nav - reversed_cost)).sum() == pytest.approx(.75)
    liquidated, final_cash, exit_cost, traded = rebalance(nav - reversed_cost, reversed_values, [0, 0, 0], 10)
    assert final_cash == pytest.approx(nav - reversed_cost - exit_cost)
    assert traded == pytest.approx(np.abs(reversed_values).sum())
    assert not liquidated.any()


def test_one_observation_delay_and_no_same_day_profit():
    raw = panel(30)
    raw.loc[22:, ASSETS[0]] *= 1.1
    signals = pd.DataFrame({"date": [raw.date.iloc[20]], **{asset: [.001] for asset in ASSETS}})
    ledger, report = simulate(raw, signals, RiskPolicy(), cost_bps=2)
    assert ledger.loc[20, "gross_exposure"] == 0
    assert ledger.loc[21, "gross_exposure"] > 0
    assert ledger.loc[21, "decision_date"] == raw.date.iloc[20]
    assert ledger.loc[20, "pnl"] == 0 and ledger.loc[21, "pnl"] == 0
    assert ledger.loc[22, "pnl"] != 0
    assert report["final_nav"] == pytest.approx(report["initial_cash"] + report["total_marked_pnl"] - report["total_cost"])
    assert report["liquidated_cash"] == pytest.approx(report["final_nav"])
    assert abs(report["accounting_residual"]) < 1e-6


def test_cash_strategy_and_cost_sensitivity_and_normalized_duplicate_dates():
    raw = panel(120)
    signals = pd.DataFrame({"date": raw.date.iloc[70:-2], **{asset: np.full(48, .001) for asset in ASSETS}})
    _, cash = simulate(raw, signals, RiskPolicy(annual_volatility=0), start_date=raw.date.iloc[70])
    assert cash["final_nav"] == 100000 and cash["total_turnover"] == 0 and cash["sharpe_zero_cash_rate"] is None
    assert cash["observations"] == 50 and cash["history_observations"] == 120
    values = []
    for cost in [0, 2, 5, 10]:
        ledger, result = simulate(raw, signals, RiskPolicy(), cost_bps=cost, start_date=raw.date.iloc[70])
        assert ledger.cash.min() >= 0
        assert result["max_posttrade_gross"] <= 1 + 1e-10
        assert result["max_posttrade_asset"] <= .5 + 1e-10
        values.append(result["final_nav"])
    assert values == sorted(values, reverse=True)
    duplicate = pd.DataFrame({"date": ["2020-01-01T00:00:00Z", "2019-12-31T19:00:00-05:00"], **{asset: [0, 0] for asset in ASSETS}})
    with pytest.raises(ValueError, match="normalization"):
        simulate(raw, duplicate, RiskPolicy())


def test_monitor_does_not_update_until_label_arrives_or_change_from_future():
    queue = DelayedResidualQueue(ASSETS)
    queue.enqueue(0, ASSETS[0], .01, .02)
    assert queue.arrive(0, {}) == []
    assert queue.arrive(1, {}) == []
    assert queue.monitors[ASSETS[0]].count == 0
    observed = queue.arrive(2, {ASSETS[0]: .03})
    assert observed[0]["standardized_residual"] == pytest.approx(1)
    assert queue.monitors[ASSETS[0]].count == 1 and not queue.pending
    with pytest.raises(ValueError):
        queue.arrive(2, {ASSETS[0]: 100})


def test_missing_delayed_label_and_backdated_prediction_rejected():
    queue = DelayedResidualQueue(ASSETS)
    queue.enqueue(0, ASSETS[0], 0., .01)
    queue.arrive(0, {})
    with pytest.raises(ValueError, match="missing"):
        queue.arrive(2, {})
    with pytest.raises(ValueError, match="Skipped"):
        queue.arrive(3, {})


def test_final_release_guard_precedes_network(tmp_path):
    (tmp_path / "evaluation.json").write_text("{}", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="already"):
        final_test(tmp_path, "absent.json")


def test_recursive_ewm_full_history_matches_saved_context_new_only_predictions():
    raw = panel(340)
    prior = raw.iloc[:280].copy()
    trained = build_panel_features(prior)
    bundles = {asset: fit_asset(trained[asset].dropna(subset=["target", "target_date"]),
                               {"max_depth": 2, "reg_lambda": 20}, iterations=2) for asset in ASSETS}
    checkpoint = {"version": "multi_asset_v1", "assets": ASSETS, "bundles": bundles,
                  "strength": .25, "price_context": prior.copy()}
    new = raw.iloc[280:310].copy()
    full = forecast_panel(checkpoint, raw.iloc[:310].copy())
    new_only = forecast_panel(checkpoint, new)
    expected = full.loc[full.date.isin(new.date)].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, new_only.reset_index(drop=True))
    revised = new.copy()
    overlapping = raw.iloc[279:310].copy()
    overlapping.loc[overlapping.index[0], ASSETS[0]] *= 1.01
    with pytest.raises(ValueError, match="revises"):
        forecast_panel(checkpoint, overlapping)
