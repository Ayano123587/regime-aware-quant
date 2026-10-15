import numpy as np
import pandas as pd
import pytest
from src.data.multi_asset import ASSETS
from src.models.uncertainty import finite_quantile, interval_metrics, forecast


def test_calibration_order_statistic_and_invalid_support():
    assert finite_quantile(np.arange(10.)) == 9
    assert finite_quantile(np.arange(19.), .1) == 17
    for values in [[], [float('nan')], [-1], [1]]:
        with pytest.raises(ValueError):
            finite_quantile(values)


def test_interval_score_penalizes_misses_and_rejects_nonfinite():
    result = interval_metrics(np.array([0., 2.]), np.zeros(2), np.ones(2))
    assert result['coverage'] == .5
    assert result['mean_width'] == 2
    assert result['mean_interval_score'] == 12
    with pytest.raises(ValueError):
        interval_metrics(np.array([np.nan]), np.zeros(1), np.ones(1))


def test_live_intervals_do_not_depend_on_future_rows_or_labels():
    rng = np.random.default_rng(13)
    panel = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=160, freq='B', tz='UTC')})
    for asset in ASSETS:
        panel[asset] = np.exp(np.cumsum(rng.normal(0, .003, len(panel))))
    checkpoint = {'version': 'uncertainty_v2', 'assets': ASSETS, 'context': panel.iloc[:120].copy(),
                  'bundles': {a: {'model': None, 'absolute_quantile': .01, 'normalized_quantile': 2.} for a in ASSETS}}
    first = forecast(checkpoint, panel.iloc[120:140].copy())
    full = forecast(checkpoint, panel.iloc[120:].copy())
    pd.testing.assert_frame_equal(first.reset_index(drop=True), full.loc[full.date <= first.date.max()].reset_index(drop=True))
    assert (first.prediction == 0).all() and (first.upper > first.lower).all()
    changed = panel.iloc[119:140].copy()
    changed.loc[changed.index[0], ASSETS[0]] *= 1.01
    with pytest.raises(ValueError, match='revises'):
        forecast(checkpoint, changed)
    checkpoint['bundles'][ASSETS[0]]['normalized_quantile'] = float('nan')
    with pytest.raises(ValueError, match='calibration'):
        forecast(checkpoint, panel.iloc[120:140].copy())
