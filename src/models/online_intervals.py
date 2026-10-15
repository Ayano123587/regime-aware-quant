"""Causal delayed-label rolling residual calibration; no mean-model updates."""
import argparse
import json
import math
import copy
from numbers import Real
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data.multi_asset import ASSETS, fetch_snapshot, sha, validate_panel
from src.features.multi_asset import build_panel_features
from src.models.advanced import write
from src.models.uncertainty import fit, point, finite_quantile, interval_metrics, search_candidates


class OnlineIntervals:
    """JSON-persisted state. A publication matures prior queued targets before forecasting."""
    @staticmethod
    def _date(value):
        try:
            if isinstance(value, Real):
                raise ValueError('Numeric timestamps are unsupported')
            date = pd.Timestamp(value)
            if pd.isna(date):
                raise ValueError('NaT date')
            return date.tz_localize('UTC') if date.tzinfo is None else date.tz_convert('UTC')
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError('A valid finite publication date is required') from error

    @staticmethod
    def _number(value, positive=False, nonnegative=False):
        try:
            valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
        except (OverflowError, TypeError):
            valid = False
        if not valid:
            raise ValueError('State numeric values must be finite')
        if (positive and value <= 0) or (nonnegative and value < 0):
            raise ValueError('State values violate positivity constraints')

    @classmethod
    def _validate(cls, state):
        keys = {'version','policy','alpha','index','last_date','last_prices','history',
                'fixed_quantiles','pending','arrived_labels'}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError('Invalid interval state schema')
        if type(state['version']) is not int or state['version'] != 1 or type(state['policy']) not in [str,int] or state['policy'] not in ['fixed',60,120] or state['alpha'] != .1:
            raise ValueError('Unsupported interval state')
        for key in ['index','arrived_labels']:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('State counters must be nonnegative integers')
        last = cls._date(state['last_date'])
        state['last_date'] = str(last)
        for key in ['last_prices','history','fixed_quantiles']:
            if not isinstance(state[key], dict) or set(state[key]) != set(ASSETS):
                raise ValueError('State asset sets must match exactly')
        for asset in ASSETS:
            cls._number(state['last_prices'][asset], positive=True)
            cls._number(state['fixed_quantiles'][asset], nonnegative=True)
            history = state['history'][asset]
            if not isinstance(history, list) or (state['policy'] != 'fixed' and len(history) > state['policy']):
                raise ValueError('Invalid residual reservoir')
            for value in history:
                cls._number(value, nonnegative=True)
            finite_quantile(history, state['alpha'])
        if not isinstance(state['pending'], list):
            raise ValueError('Pending queue must be a list')
        seen = set()
        for item in state['pending']:
            if not isinstance(item, dict) or set(item) != {'asset','due_index','decision_date','prediction','scale'}:
                raise ValueError('Invalid pending record schema')
            asset, due = item['asset'], item['due_index']
            if not isinstance(asset, str) or asset not in ASSETS or type(due) is not int or due not in [state['index']+1,state['index']+2] or (asset,due) in seen:
                raise ValueError('Invalid or duplicate pending arrival')
            seen.add((asset,due))
            decision = cls._date(item['decision_date'])
            if decision > last or last-decision > pd.Timedelta(days=7) or (due==state['index']+2 and decision!=last) or (due==state['index']+1 and decision>=last):
                raise ValueError('Pending decision date conflicts with delayed arrival')
            item['decision_date'] = str(decision)
            cls._number(item['prediction'])
            cls._number(item['scale'], positive=True)
        for due in {due for _,due in seen}:
            if {asset for asset,index in seen if index==due} != set(ASSETS):
                raise ValueError('Each pending observation must cover every asset')
        return state

    def __init__(self, state):
        try:
            copied = json.loads(json.dumps(state, allow_nan=False))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('Interval state must be finite JSON data') from error
        self.state = self._validate(copied)

    @classmethod
    def initialize(cls, policy, residuals, last_prices, last_date, pending):
        if policy not in ['fixed',60,120]:
            raise ValueError('Unsupported interval policy')
        history={a:list(map(float,residuals[a])) for a in ASSETS}
        quantiles={a:finite_quantile(history[a]) for a in ASSETS}
        if policy!='fixed':
            history={a:v[-policy:] for a,v in history.items()}
        return cls({'version':1,'policy':policy,'alpha':.1,'index':0,'last_date':str(last_date),
                    'last_prices':{a:float(last_prices[a]) for a in ASSETS},'history':history,
                    'fixed_quantiles':quantiles,'pending':pending,'arrived_labels':0})

    def step(self, date, prices, predictions, scales):
        state=self._validate(copy.deepcopy(self.state))
        date=self._date(date)
        if date<=self._date(state['last_date']):
            raise ValueError('Publications must arrive in strictly increasing order')
        if date-self._date(state['last_date'])>pd.Timedelta(days=7):
            raise ValueError('Publication gap exceeds seven calendar days')
        for values in [prices,predictions,scales]:
            if not isinstance(values, dict) or set(values)!=set(ASSETS):
                raise ValueError('Aligned finite asset inputs required')
            for a in ASSETS:
                self._number(values[a])
        if any(prices[a]<=0 or scales[a]<=0 for a in ASSETS):
            raise ValueError('Prices and saved decision scales must be positive')
        current=state['index']+1
        remaining=[]
        for item in state['pending']:
            if item['due_index']<current:
                raise ValueError('Overdue queue indicates skipped state updates')
            if item['due_index']==current:
                a=item['asset']
                ratio=prices[a]/state['last_prices'][a]
                if not math.isfinite(ratio) or ratio<=0:
                    raise ValueError('Nonfinite matured return ratio')
                actual=math.log(ratio)
                residual=abs(actual-item['prediction'])/item['scale']
                if not math.isfinite(residual):
                    raise ValueError('Nonfinite matured residual')
                if state['policy']!='fixed':
                    state['history'][a].append(residual)
                    state['history'][a]=state['history'][a][-state['policy']:]
                state['arrived_labels']+=1
            else:
                remaining.append(item)
        bands={}
        for a in ASSETS:
            q=state['fixed_quantiles'][a] if state['policy']=='fixed' else finite_quantile(state['history'][a])
            width=scales[a]*q
            if not all(math.isfinite(value) for value in [width,predictions[a]-width,predictions[a]+width]):
                raise ValueError('Nonfinite computed interval bands')
            bands[a]={'prediction':predictions[a],'half_width':width,'normalized_quantile':q}
            remaining.append({'asset':a,'due_index':current+2,'decision_date':str(date),
                              'prediction':float(predictions[a]),'scale':float(scales[a])})
        state.update(index=current,last_date=str(date),last_prices={a:float(prices[a]) for a in ASSETS},pending=remaining)
        self.state=self._validate(state)
        return bands

    def save(self,path):
        write(path,self._validate(copy.deepcopy(self.state)))

    @classmethod
    def load(cls,path):
        def unique_keys(pairs):
            result = {}
            for key,value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON state key')
                result[key] = value
            return result
        return cls(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_keys))


def initialize_state(policy, frames, models, panel, calibration_year):
    residuals={};pending=[]
    last_date=panel.date.max()
    tail=list(panel.date.iloc[-2:])
    for a in ASSETS:
        frame=frames[a]
        calibration=frame.loc[(frame.date.dt.year==calibration_year)&frame.target.notna()&(frame.target_date<=last_date)]
        residuals[a]=(np.abs(calibration.target.to_numpy()-point(models[a],calibration))/np.maximum(calibration.ewma_volatility.to_numpy(),1e-5)).tolist()
        for offset,date in zip([-1,0],tail):
            row=frame.loc[frame.date==date]
            if len(row)!=1:
                raise ValueError('Missing pre-period pending feature row')
            pending.append({'asset':a,'due_index':offset+2,'decision_date':str(date),
                            'prediction':float(point(models[a],row)[0]),'scale':float(max(row.ewma_volatility.iloc[0],1e-5))})
    return OnlineIntervals.initialize(policy,residuals,panel.iloc[-1],last_date,pending)


def replay(state, panel, frames, models):
    """Rows can contain future labels, but step receives only current observed prices/features."""
    output=[]
    for observation in panel.itertuples(index=False):
        date=observation.date
        feature_rows={a:frames[a].loc[frames[a].date==date] for a in ASSETS}
        if any(len(row)!=1 for row in feature_rows.values()):
            raise ValueError('Each publication needs one feature row per asset')
        predictions={a:float(point(models[a],feature_rows[a])[0]) for a in ASSETS}
        scales={a:float(max(feature_rows[a].ewma_volatility.iloc[0],1e-5)) for a in ASSETS}
        bands=state.step(date,{a:getattr(observation,a) for a in ASSETS},predictions,scales)
        for a in ASSETS:
            output.append({'date':date,'asset':a,**bands[a],
                           'target':float(feature_rows[a].target.iloc[0])})
    return pd.DataFrame(output)


def summarize(rows):
    rows=rows.loc[rows.target.notna()]
    result={'pooled':interval_metrics(rows.target,rows.prediction,rows.half_width)}
    result['per_asset']={a:interval_metrics(g.target,g.prediction,g.half_width) for a,g in rows.groupby('asset')}
    return result


def run(output='models/online-interval-v3',plan_path='docs/online-interval-plan.json'):
    output=Path(output)
    if (output/'results.json').exists() or (output/'selection.json').exists():
        raise ValueError('Preserve existing online interval study')
    started=time.monotonic()
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['policies']!=['fixed',60,120] or plan['alpha']!=.1:
        raise ValueError('Unsupported frozen policy search')
    prior,provenance=fetch_snapshot('data/external/multi-asset-v1/development','2017-01-01','2025-12-31')
    frames=build_panel_features(prior)
    diagnostic_plan={'candidates':plan['diagnostic_point_candidates'],'validation_years':plan['diagnostic_point_selection_years']}
    with threadpool_limits(limits=2):
        point_candidates,spec=search_candidates(frames,diagnostic_plan)
    cutoff=pd.Timestamp(plan['diagnostic_training_label_end'],tz='UTC')
    with threadpool_limits(limits=2):
        models={a:fit(frames[a].loc[frames[a].target_date<=cutoff],spec) for a in ASSETS}
    seed_panel=prior.loc[prior.date.dt.year<=2024]
    seed_frames=build_panel_features(seed_panel)
    validation_panel=prior.loc[prior.date.dt.year==2025]
    runs=[]
    local=Path('data/interim')/output.name;local.mkdir(parents=True,exist_ok=True)
    for policy in plan['policies']:
        state=initialize_state(policy,seed_frames,models,seed_panel,2024)
        rows=replay(state,validation_panel,frames,models)
        rows.to_csv(local/f'validation-{policy}.csv',index=False,lineterminator='\n')
        runs.append({'policy':policy,'validation':summarize(rows)})
    order={'fixed':0,120:1,60:2}
    selected=min(runs,key=lambda r:(r['validation']['pooled']['mean_interval_score'],order[r['policy']]))['policy']
    selection={'plan_sha256':sha(plan_path),'implementation_sha256':sha(__file__),'point_selection':point_candidates,
               'diagnostic_point_spec':spec,'runs':runs,'selected_policy':selected,'test_used_for_selection':False,
               'development_provenance':provenance,'disclosure':plan['disclosure']}
    write(output/'selection.json',selection)
    checkpoint_path=Path('models/uncertainty-v2/bundle.joblib');bundle=joblib.load(checkpoint_path)
    final_models={a:bundle['bundles'][a]['model'] for a in ASSETS}
    # Freeze interval choice before replay. Preserve exact existing v2 point forecasts.
    test,test_provenance=fetch_snapshot('data/external/multi-asset-v1/holdout','2026-01-01','2026-10-14')
    combined=validate_panel(pd.concat([prior,test],ignore_index=True));final_frames=build_panel_features(combined)
    reports={};verification={}
    for policy in dict.fromkeys(['fixed',selected]):
        state=initialize_state(policy,frames,final_models,prior,2025)
        if policy=='fixed':
            for a in ASSETS:
                if abs(state.state['fixed_quantiles'][a]-bundle['bundles'][a]['normalized_quantile'])>1e-12:
                    raise AssertionError('Fixed baseline differs from v2 calibration')
        state.save(local/f'initial-state-{policy}.json')
        loaded=OnlineIntervals.load(local/f'initial-state-{policy}.json')
        rows=replay(loaded,test,final_frames,final_models)
        rows.to_csv(local/f'final-{policy}.csv',index=False,lineterminator='\n')
        loaded.save(local/f'final-state-{policy}.json')
        # Actual JSON save/load at a midpoint must preserve the entire causal replay.
        resumed=OnlineIntervals.load(local/f'initial-state-{policy}.json')
        midpoint=len(test)//2
        first=replay(resumed,test.iloc[:midpoint],final_frames,final_models)
        resumed.save(local/f'midpoint-state-{policy}.json')
        resumed=OnlineIntervals.load(local/f'midpoint-state-{policy}.json')
        second=replay(resumed,test.iloc[midpoint:],final_frames,final_models)
        pd.testing.assert_frame_equal(rows.reset_index(drop=True),pd.concat([first,second],ignore_index=True))
        if resumed.state!=loaded.state:
            raise AssertionError('Persisted interval queue differs after continuation')
        reports[str(policy)]=summarize(rows)
        verification[str(policy)]={'arrived_labels':loaded.state['arrived_labels'],'pending_labels':len(loaded.state['pending']),
                                    'final_state_sha256':sha(local/f'final-state-{policy}.json'),'midpoint_save_load_exact':True}
    if time.monotonic()-started>plan['maximum_cpu_seconds']:
        raise TimeoutError('Study exceeded declared CPU wall budget')
    report={'selection_sha256':sha(output/'selection.json'),'selected_policy':selected,'observed_2026':reports,
            'state_verification':verification,'point_checkpoint_sha256':sha(checkpoint_path),'test_provenance':test_provenance,
            'runtime_seconds':time.monotonic()-started,'mean_model_changed':False,'risk_policy_changed':False,
            'disclosure':plan['disclosure'],'limitations':['Online order-statistic intervals have no distribution-free guarantee under drift or dependence.','Validation and final point models have different chronological training vintages.','Policies selected by proper interval score, not forced to nominal coverage or width.']}
    write(output/'results.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='models/online-interval-v3')
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2))


if __name__=='__main__':
    main()
