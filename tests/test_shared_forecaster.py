import numpy as np
import pandas as pd
import pytest
from src.data.multi_asset import ASSETS
from src.features.multi_asset import build_panel_features
from src.models.shared_forecaster import fit,predict,design,forecast
from src.models.shared_study import choose,run


def panel():
    rng=np.random.default_rng(7);p=pd.DataFrame({'date':pd.date_range('2023-01-02',periods=180,freq='B',tz='UTC')})
    for a in ASSETS:p[a]=np.exp(np.cumsum(rng.normal(0,.003,len(p))))
    return p


def test_pooled_scaler_fits_training_only_and_targets_are_not_inference_features():
    frames=build_panel_features(panel());training={a:f.iloc[:60].copy() for a,f in frames.items()}
    spec={'kind':'pooled_normalized','features':'reduced','alpha':1000}
    model=fit(training,spec)
    expected=pd.concat([design(training[a],spec)[0] for a in ASSETS]).mean().to_numpy()
    np.testing.assert_allclose(model['models']['shared'][0].mean_,expected)
    a=ASSETS[0];future=frames[a].iloc[60:].copy();first=predict(model,a,future);future['target']=1e9
    np.testing.assert_array_equal(first,predict(model,a,future))
    assert set(model['models'])=={'shared'}


def test_zero_and_nonfinite_inputs():
    frames=build_panel_features(panel());training={a:f.dropna(subset=['target']) for a,f in frames.items()}
    model=fit(training,{'kind':'zero'})
    assert (predict(model,ASSETS[0],training[ASSETS[0]])==0).all()
    damaged=training[ASSETS[0]].copy();damaged['return_1']=np.nan
    with pytest.raises(ValueError,match='Finite'):predict(model,ASSETS[0],damaged)


def test_selection_requires_relative_gain_and_three_years_but_zero_stays_eligible():
    def result(name,total,folds):return {'spec':{'name':name},'pooled_validation_rmse':total,'folds':[{'rmse':v} for v in folds]}
    zero=result('zero',1.,[1.]*4);v2=result('v2_full',.999,[.999]*4)
    candidate=result('candidate',.995,[.98,.98,1.01,1.01])
    assert choose([zero,v2,candidate])['name']=='v2_full'
    candidate['folds']=[{'rmse':.995}]*4
    assert choose([zero,v2,candidate])['name']=='candidate'
    zero['pooled_validation_rmse']=.9
    assert choose([zero,v2,candidate])['name']=='zero'


def test_checkpoint_forecast_prefix_invariance_and_revision_rejection(tmp_path):
    import joblib
    raw=panel();frames=build_panel_features(raw.iloc[:120]);training={a:f.dropna(subset=['target']) for a,f in frames.items()}
    model=fit(training,{'kind':'pooled_normalized','features':'reduced','alpha':1000})
    checkpoint={'version':'shared_forecast_v4','point':model,'quantiles':{a:2. for a in ASSETS},'context':raw.iloc[:120].copy()}
    joblib.dump(checkpoint,tmp_path/'bundle.joblib');loaded=joblib.load(tmp_path/'bundle.joblib')
    first=forecast(loaded,raw.iloc[120:140]);whole=forecast(loaded,raw.iloc[120:])
    pd.testing.assert_frame_equal(first.reset_index(drop=True),whole.loc[whole.date<=first.date.max()].reset_index(drop=True))
    changed=raw.iloc[119:140].copy();changed.iloc[0,1]*=1.01
    with pytest.raises(ValueError,match='revises'):forecast(loaded,changed)
    loaded['quantiles'][ASSETS[0]]=float('inf')
    with pytest.raises(ValueError,match='calibration'):forecast(loaded,raw.iloc[120:140])


def test_completed_study_is_preserved(tmp_path):
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError,match='Preserve'):run(tmp_path)


def test_recency_weights_use_training_dates_and_weight_scaler_consistently():
    frames=build_panel_features(panel());training={a:f.iloc[:60].copy() for a,f in frames.items()}
    spec={'kind':'per_asset','features':'reduced','alpha':10000,'half_life_days':365.25}
    model=fit(training,spec)
    a=ASSETS[0];dates=training[a].target_date;age=(dates.max()-dates).dt.total_seconds().to_numpy()/86400
    weights=np.exp2(-age/365.25);weights/=weights.mean()
    expected=np.average(design(training[a],spec)[0],axis=0,weights=weights)
    np.testing.assert_allclose(model['models'][a][0].mean_,expected)
    changed={a:f.copy() for a,f in training.items()}
    for f in changed.values():f['date']=f.date+pd.Timedelta(days=1000)
    second=fit(changed,spec)
    np.testing.assert_array_equal(model['models'][a][-1].coef_,second['models'][a][-1].coef_)


def test_fixed_equal_blend_is_exact_component_average():
    frames=build_panel_features(panel());training={a:f.dropna(subset=['target']) for a,f in frames.items()}
    specs=[{'kind':'per_asset','features':'full','alpha':10000},{'kind':'pooled_normalized','features':'reduced','alpha':10000}]
    model=fit(training,{'kind':'fixed_blend','components':specs});a=ASSETS[0]
    expected=np.mean([predict(c,a,training[a]) for c in model['components']],axis=0)
    np.testing.assert_array_equal(predict(model,a,training[a]),expected)


def test_retained_checkpoint_alias_verifies_hash_before_loading(tmp_path,monkeypatch):
    import json,joblib
    from src.data.multi_asset import sha
    from src.models.shared_study import forecast_directory
    monkeypatch.chdir(tmp_path);path=tmp_path/'models/uncertainty-v2';path.mkdir(parents=True)
    raw=panel();bundle={'version':'uncertainty_v2','assets':ASSETS,'context':raw.iloc[:120],
        'bundles':{a:{'model':None,'absolute_quantile':.01,'normalized_quantile':2.} for a in ASSETS}}
    joblib.dump(bundle,path/'bundle.joblib')
    study=tmp_path/'models/study';study.mkdir()
    (study/'results.json').write_text(json.dumps({'selected':{'name':'v2_full'},'retained_checkpoint_sha256':sha(path/'bundle.joblib')}))
    result=forecast_directory(study,raw.iloc[120:140]);assert len(result)==60
    with (path/'bundle.joblib').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='hash changed'):forecast_directory(study,raw.iloc[120:140])
