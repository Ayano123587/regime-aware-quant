"""Guard tests never read observations or rescore a test cohort."""
import sys
from unittest.mock import Mock
import pytest
from src.models import uncertainty as u


def test_training_rejects_existing_before_fetch(tmp_path, monkeypatch):
    fetch = Mock(side_effect=AssertionError('must not fetch'))
    monkeypatch.setattr(u, 'fetch_snapshot', fetch)
    with pytest.raises(ValueError, match='fresh'):
        u.train_selected(tmp_path)
    fetch.assert_not_called()


def test_evaluation_requires_ack_before_loading(tmp_path, monkeypatch):
    load = Mock(side_effect=AssertionError('must not load'))
    monkeypatch.setattr(u.joblib, 'load', load)
    with pytest.raises(ValueError, match='acknowledgement'):
        u.evaluate_selected(tmp_path/'model', tmp_path/'report', tmp_path/'rows.csv')
    load.assert_not_called()


def test_evaluation_rejects_existing_predictions(tmp_path, monkeypatch):
    path = tmp_path/'rows.csv'
    path.write_text('preserve', encoding='utf-8')
    with pytest.raises(ValueError, match='Fresh'):
        u.evaluate_selected(tmp_path/'model', tmp_path/'report', path, acknowledge_observed=True)
    assert path.read_text() == 'preserve'


def test_forecast_does_not_load_or_overwrite_existing(tmp_path, monkeypatch):
    path = tmp_path/'rows.csv'
    path.write_text('preserve', encoding='utf-8')
    load = Mock(side_effect=AssertionError('must not load'))
    monkeypatch.setattr(u.joblib, 'load', load)
    monkeypatch.setattr(sys, 'argv', ['uncertainty', 'forecast', '--output', str(tmp_path/'model'), '--panel', 'absent.csv', '--predictions', str(path)])
    with pytest.raises(ValueError, match='already exists'):
        u.main()
    load.assert_not_called()
    assert path.read_text() == 'preserve'


def test_combined_run_requires_explicit_paths_before_training(monkeypatch):
    train = Mock(side_effect=AssertionError('must not train'))
    monkeypatch.setattr(u, 'train_selected', train)
    with pytest.raises(ValueError, match='explicit'):
        u.run(acknowledge_observed=True)
    train.assert_not_called()


def test_training_stops_before_holdout_and_saves_context(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from src.features.multi_asset import FEATURES
    dates = pd.date_range('2025-01-01', periods=12, tz='UTC')
    dates = pd.DatetimeIndex([pd.Timestamp('2024-12-20', tz='UTC'), *dates])
    frame = pd.DataFrame({key: np.ones(len(dates)) for key in FEATURES})
    frame['date'] = dates
    frame['target_date'] = dates + pd.Timedelta(days=2)
    frame['target'] = .001
    frame['ewma_volatility'] = .01
    prior = pd.DataFrame({'date': dates})
    fetch = Mock(return_value=(prior, {'fixture': True}))
    monkeypatch.setattr(u, 'fetch_snapshot', fetch)
    monkeypatch.setattr(u, 'build_panel_features', lambda _: {asset: frame for asset in u.ASSETS})
    monkeypatch.setattr(u, 'search_candidates', lambda *_: ([], {'kind': 'zero', 'name': 'zero'}))
    result = u.train_selected(tmp_path/'fresh')
    assert result['test_used_for_selection'] is False
    assert fetch.call_count == 1
    assert fetch.call_args.args[2] == '2025-12-31'
    saved = u.joblib.load(tmp_path/'fresh'/'bundle.joblib')
    assert saved['context'].equals(prior)
    assert not (tmp_path/'fresh'/'results.json').exists()
