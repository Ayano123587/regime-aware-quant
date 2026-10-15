import math
import numpy as np
import pandas as pd
import pytest
from src.data.multi_asset import ASSETS
from src.models.online_intervals import OnlineIntervals, run, replay
from src.features.multi_asset import build_panel_features


def state(policy=60,pending=None):
    return OnlineIntervals.initialize(policy,{a:[.1]*120 for a in ASSETS},
        {a:1. for a in ASSETS},pd.Timestamp('2025-01-01',tz='UTC'),pending or [])


def test_labels_arrive_at_two_observations_with_saved_prediction_and_scale():
    model=state()
    first=model.step(pd.Timestamp('2025-01-02',tz='UTC'),{a:1. for a in ASSETS},{a:.2 for a in ASSETS},{a:2. for a in ASSETS})
    assert model.state['arrived_labels']==0
    model.step(pd.Timestamp('2025-01-03',tz='UTC'),{a:1. for a in ASSETS},{a:0. for a in ASSETS},{a:5. for a in ASSETS})
    assert model.state['arrived_labels']==0
    model.step(pd.Timestamp('2025-01-06',tz='UTC'),{a:2. for a in ASSETS},{a:0. for a in ASSETS},{a:7. for a in ASSETS})
    assert model.state['arrived_labels']==3
    for a in ASSETS:
        assert model.state['history'][a][-1]==pytest.approx(abs(math.log(2)-.2)/2)
        assert len(model.state['history'][a])==60
        assert first[a]['half_width']==pytest.approx(.2)


def test_arrivals_update_before_current_forecast():
    pending=[{'asset':a,'due_index':1,'prediction':100.,'scale':1.,'decision_date':'2024-12-31'} for a in ASSETS]
    model=OnlineIntervals.initialize(60,{a:[.1]*10 for a in ASSETS},{a:1. for a in ASSETS},'2025-01-01',pending)
    bands=model.step(pd.Timestamp('2025-01-02'),{a:1. for a in ASSETS},{a:0. for a in ASSETS},{a:1. for a in ASSETS})
    assert bands[ASSETS[0]]['normalized_quantile']==100.


def test_fixed_policy_does_not_adapt():
    pending=[{'asset':a,'due_index':1,'prediction':100.,'scale':1.,'decision_date':'2024-12-31'} for a in ASSETS]
    model=state('fixed',pending)
    bands=model.step(pd.Timestamp('2025-01-02',tz='UTC'),{a:1. for a in ASSETS},{a:0. for a in ASSETS},{a:1. for a in ASSETS})
    assert bands[ASSETS[0]]['normalized_quantile']==.1
    assert model.state['arrived_labels']==3
    assert len(model.state['history'][ASSETS[0]])==120


def test_saved_state_exact_continuation_and_bad_inputs(tmp_path):
    model=state()
    args=(pd.Timestamp('2025-01-02',tz='UTC'),{a:1. for a in ASSETS},{a:0. for a in ASSETS},{a:1. for a in ASSETS})
    model.step(*args);model.save(tmp_path/'state.json');loaded=OnlineIntervals.load(tmp_path/'state.json')
    later=(pd.Timestamp('2025-01-03',tz='UTC'),*args[1:])
    assert model.step(*later)==loaded.step(*later)
    assert model.state==loaded.state
    with pytest.raises(ValueError,match='increasing'):model.step(*later)
    with pytest.raises(ValueError,match='finite'):model.step(pd.Timestamp('2025-01-06',tz='UTC'),{a:float('nan') for a in ASSETS},*args[2:])


def test_replay_future_values_and_labels_cannot_change_prefix_bands():
    rng=np.random.default_rng(2)
    panel=pd.DataFrame({'date':pd.date_range('2024-10-01',periods=160,freq='B',tz='UTC')})
    for a in ASSETS:panel[a]=np.exp(np.cumsum(rng.normal(0,.01,len(panel))))
    first=panel.iloc[100:120];whole=panel.iloc[100:]
    initial=OnlineIntervals.initialize(60,{a:[.5]*120 for a in ASSETS},panel.iloc[99],panel.date.iloc[99],[])
    before=OnlineIntervals(initial.state);full=OnlineIntervals(initial.state)
    prefix_frames=build_panel_features(panel.iloc[:120]);frames=build_panel_features(panel)
    # Future labels are deliberately corrupted; no label is passed into state.step.
    for frame in frames.values():frame['target']=99999.
    x=replay(before,first,prefix_frames,{a:None for a in ASSETS})
    y=replay(full,whole,frames,{a:None for a in ASSETS})
    columns=['date','asset','prediction','half_width','normalized_quantile']
    pd.testing.assert_frame_equal(x[columns].reset_index(drop=True),y.loc[y.date<=first.date.max(),columns].reset_index(drop=True))


def test_study_preserves_existing_selection(tmp_path):
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError,match='Preserve'):run(tmp_path)

@pytest.mark.parametrize('corruption',[
    lambda s:s.pop('history'),
    lambda s:s['last_prices'].pop(ASSETS[0]),
    lambda s:s['history'].__setitem__('unexpected',[.1]*10),
    lambda s:s['history'][ASSETS[0]].__setitem__(0,-1.),
    lambda s:s['history'][ASSETS[0]].__setitem__(0,float('nan')),
    lambda s:s['history'].__setitem__(ASSETS[0],[.1]),
    lambda s:s['fixed_quantiles'].__setitem__(ASSETS[0],-1.),
    lambda s:s['last_prices'].__setitem__(ASSETS[0],0.),
    lambda s:s.__setitem__('index',True),
    lambda s:s.__setitem__('arrived_labels',-1),
    lambda s:s.__setitem__('last_date','NaT'),
    lambda s:s.__setitem__('last_date',12345),
    lambda s:s['fixed_quantiles'].__setitem__(ASSETS[0],10**1000),
])
def test_malformed_json_state_is_rejected(tmp_path,corruption):
    import json
    damaged=state().state
    corruption(damaged)
    path=tmp_path/'damaged.json';path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError):OnlineIntervals.load(path)


@pytest.mark.parametrize('field,value',[
    ('asset','unexpected'),('due_index',0),('due_index',3),('due_index',True),
    ('scale',0.),('scale',float('inf')),('prediction',float('nan')),
    ('decision_date','2026-01-01'),('decision_date','invalid'),
])
def test_malformed_pending_record_is_rejected(field,value):
    pending=[{'asset':a,'due_index':1,'prediction':0.,'scale':1.,'decision_date':'2024-12-31'} for a in ASSETS]
    damaged=state(pending=pending).state
    damaged['pending'][0][field]=value
    with pytest.raises(ValueError):OnlineIntervals(damaged)


@pytest.mark.parametrize('date',['NaT','invalid','2025-01-09',None,12345])
def test_bad_publication_dates_and_gaps_are_atomic(date):
    import copy
    model=state();before=copy.deepcopy(model.state)
    with pytest.raises(ValueError):model.step(date,{a:1. for a in ASSETS},{a:0. for a in ASSETS},{a:1. for a in ASSETS})
    assert model.state==before


@pytest.mark.parametrize('failure',['width','endpoint','residual','ratio'])
def test_overflow_after_arrival_does_not_partially_mutate_state(failure):
    import copy
    pending=[{'asset':a,'due_index':1,'prediction':0.,'scale':1.,'decision_date':'2024-12-31'} for a in ASSETS]
    model=state('fixed',pending)
    prices={a:1. for a in ASSETS};predictions={a:0. for a in ASSETS};scales={a:1. for a in ASSETS}
    if failure in ['width','endpoint']:
        model.state['fixed_quantiles'][ASSETS[0]]=1e308
        scales[ASSETS[0]]=10. if failure=='width' else 1.
        predictions[ASSETS[0]]=1e308 if failure=='endpoint' else 0.
    elif failure=='residual':
        model.state['pending'][-1]['prediction']=1e308
        model.state['pending'][-1]['scale']=1e-308
    else:
        model.state['last_prices'][ASSETS[-1]]=1e-308
        prices[ASSETS[-1]]=1e308
    before=copy.deepcopy(model.state)
    with pytest.raises(ValueError,match='Nonfinite'):model.step('2025-01-02',prices,predictions,scales)
    assert model.state==before


def test_incomplete_duplicate_pending_and_missing_fields_are_rejected():
    import copy
    pending=[{'asset':a,'due_index':1,'prediction':0.,'scale':1.,'decision_date':'2024-12-31'} for a in ASSETS]
    valid=state(pending=pending).state
    for change in ['incomplete','duplicate','missing']:
        damaged=copy.deepcopy(valid)
        if change=='incomplete':damaged['pending'].pop()
        elif change=='duplicate':damaged['pending'].append(damaged['pending'][0].copy())
        else:damaged['pending'][0].pop('scale')
        with pytest.raises(ValueError):OnlineIntervals(damaged)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path=tmp_path/'duplicate.json'
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError,match='Duplicate'):OnlineIntervals.load(path)
