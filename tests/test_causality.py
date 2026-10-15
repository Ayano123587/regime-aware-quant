import numpy as np
import pandas as pd
import pytest
from src.config import FEATURES
from src.features.build_features import build_features
from src.models.hmm import GaussianHMM
from src.models.train_model import fit_bundle, forecast, returns_summary, evaluate
from src.data.make_dataset import load_prices


def prices(n=420):
    rng = np.random.default_rng(42)
    returns = rng.normal(.0002, .01, n)
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
                         "close": 100*np.exp(np.cumsum(returns))})


def test_features_prefix_invariant():
    full = build_features(prices())
    prefix = build_features(prices().iloc[:220])
    np.testing.assert_allclose(full.iloc[:len(prefix)][FEATURES], prefix[FEATURES])
    assert pd.isna(prefix.target.iloc[-1])
    assert full.target.iloc[0] == pytest.approx(np.log(prices().close.iloc[60]/prices().close.iloc[59]))


def test_hmm_prefix_and_chunk_invariance():
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(-2, .3, (50,2)), rng.normal(2, .3, (50,2))]
    hmm = GaussianHMM(iterations=15).fit(x)
    full = hmm.filter(x)
    np.testing.assert_allclose(full[:60], hmm.filter(x[:60]), atol=1e-12)
    np.testing.assert_allclose(full[60:], hmm.filter(x[60:], full[59]), atol=1e-12)
    np.testing.assert_allclose(full.sum(axis=1), 1)
    assert np.isfinite(full).all()
    assert np.isfinite(hmm.log_likelihoods).all()


def test_hmm_rejects_invalid_input():
    with pytest.raises(ValueError): GaussianHMM(states=1)
    with pytest.raises(ValueError): GaussianHMM().fit(np.ones((3,2)))
    with pytest.raises(ValueError): GaussianHMM().fit(np.full((20,2), np.nan))


def test_forecast_checkpoint_and_expert_gating(tmp_path):
    import joblib
    frame = build_features(prices()).dropna()
    bundle = fit_bundle(frame.iloc[:200])
    prediction, posterior = forecast(bundle, frame.iloc[200:220])
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(posterior.sum(axis=1), 1)
    path = tmp_path/'bundle.joblib'
    joblib.dump(bundle,path)
    restored = joblib.load(path)
    np.testing.assert_allclose(prediction, forecast(restored,frame.iloc[200:220])[0])


def test_costs_include_initial_entry_and_reversal():
    summary = returns_summary(np.array([1.,-1.,1.]),np.zeros(3),10)
    assert summary['turnover'] == 5
    assert summary['net_log_return'] == pytest.approx(-.005)
    assert summary['max_drawdown'] == pytest.approx(np.exp(-.005)-1)


def test_walkforward_target_purge(tmp_path):
    result = evaluate(prices(), tmp_path, folds=2)
    for fold in result['folds']:
        assert pd.Timestamp(fold['train_last_target']) < pd.Timestamp(fold['test_first'])
    assert (tmp_path/'bundle.joblib').exists()
    assert (tmp_path/'evaluation.json').exists()


def test_duplicate_price_dates_rejected(tmp_path):
    frame = prices(160)
    frame.loc[1,'date']=frame.loc[0,'date']
    path=tmp_path/'bad.csv'; frame.to_csv(path,index=False)
    with pytest.raises(ValueError): load_prices(path)
