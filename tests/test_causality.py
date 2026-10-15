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
    with pytest.raises(ValueError): GaussianHMM(states=2.5)
    with pytest.raises(ValueError): GaussianHMM().fit(np.empty((20,0)))
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


def test_loader_rejects_bad_values_and_unsorted_dates(tmp_path):
    for kind in ['missing','infinite','unsorted']:
        frame=prices(160)
        if kind=='missing':frame.loc[7,'close']=np.nan
        if kind=='infinite':frame.loc[7,'close']=np.inf
        if kind=='unsorted':frame=frame.iloc[::-1]
        path=tmp_path/f'{kind}.csv';frame.to_csv(path,index=False)
        with pytest.raises(ValueError):load_prices(path)


def test_filter_validates_checkpoint_probability_and_dimension():
    x=np.random.default_rng(12).normal(size=(40,2));hmm=GaussianHMM(states=2,iterations=3).fit(x)
    for previous in [np.array([1.,1.]),np.array([-1.,2.]),np.array([np.nan,0.]),np.array([1.,0.,0.])]:
        with pytest.raises(ValueError):hmm.filter(x[:2],previous)
    with pytest.raises(ValueError):hmm.filter(np.ones((4,3)))
    with pytest.raises(ValueError):GaussianHMM(states=2).fit(np.ones((20,2)))


def test_checkpoint_context_reproduces_full_history_and_rejects_gap():
    from src.models.predict_model import predict_prices
    raw=prices(420);frame=build_features(raw).dropna();train=frame.iloc[:200]
    bundle=fit_bundle(train,states=2,hmm_iterations=5,context_prices=raw)
    new=raw.loc[raw.date>train.target_date.max()].iloc[:20]
    full=predict_prices(bundle,raw.loc[raw.date<=new.date.max()]);only_new=predict_prices(bundle,new)
    pd.testing.assert_frame_equal(full.loc[full.date.isin(new.date)].reset_index(drop=True),only_new.reset_index(drop=True))
    with pytest.raises(ValueError,match='bridge'):predict_prices(bundle,raw.iloc[-10:])
    revised=raw.loc[raw.date>=bundle['price_context'].date.min()].copy();revised.loc[revised.index[0],'close']*=1.01
    with pytest.raises(ValueError,match='revises'):predict_prices(bundle,revised)


def test_forecast_does_not_read_labels_and_rejects_bad_time():
    raw=prices();frame=build_features(raw).dropna();bundle=fit_bundle(frame.iloc[:200],states=2,hmm_iterations=5)
    future=frame.iloc[200:210].copy();a=forecast(bundle,future)[0]
    future['target']=np.inf
    np.testing.assert_array_equal(a,forecast(bundle,future)[0])
    with pytest.raises(ValueError):forecast(bundle,future.iloc[::-1])


def test_chronological_validation_purges_label_boundary():
    from src.models.development import frame_for_validation
    frame=build_features(prices(1000)).dropna();train,val,bridge=frame_for_validation(frame,'2021-01-01','2022-01-01')
    assert train.target_date.max()<val.date.min()
    assert val.target_date.max()<=pd.Timestamp('2022-01-01',tz='UTC')
    assert not set(train.date)&set(val.date)


def test_new_year_test_refuses_overwrite_without_network(tmp_path):
    from src.models.development import final_test
    (tmp_path/'evaluation.json').write_text('{}')
    with pytest.raises(ValueError,match='already evaluated'):final_test(tmp_path,'does-not-exist')


def test_zero_forecast_has_no_directional_coverage():
    summary=returns_summary(np.zeros(4),np.array([.01,-.01,.02,-.02]))
    assert summary['direction_accuracy'] is None
    assert summary['prediction_coverage']==0.
    assert summary['turnover']==0.
