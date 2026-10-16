"""Chronological controlled evaluation of lower-variance shared point procedures."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data.multi_asset import ASSETS,fetch_snapshot,sha,validate_panel
from src.features.multi_asset import build_panel_features
from src.models.advanced import feature_split,write
from src.models.uncertainty import finite_quantile,interval_metrics
from src.models.shared_forecaster import fit,predict,forecast


def choose(runs,relative_gain=.001,year_support=3):
    reference=min((r for r in runs if r['spec']['name'] in ['zero','v2_full']),key=lambda r:r['pooled_validation_rmse'])
    eligible=[]
    for result in runs:
        wins=sum(x['rmse']<y['rmse'] for x,y in zip(result['folds'],reference['folds']))
        if result['spec']['name'] in ['zero','v2_full'] or (result['pooled_validation_rmse']<=reference['pooled_validation_rmse']*(1-relative_gain) and wins>=year_support):eligible.append(result)
    return min(eligible,key=lambda r:r['pooled_validation_rmse'])['spec']


def run(output='models/shared-forecast-v4',plan_path='docs/shared-forecast-plan.json'):
    output=Path(output)
    if (output/'selection.json').exists():raise ValueError('Preserve completed shared study')
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['assets']!=ASSETS or plan['validation_years']!=[2021,2022,2023,2024] or plan['final_training_label_end']!='2024-12-31' or plan['interval_calibration_year']!=2025:raise ValueError('Unsupported chronological study design')
    started=time.monotonic();prior,provenance=fetch_snapshot('data/external/multi-asset-v1/development','2017-01-01','2025-12-31');frames=build_panel_features(prior)
    local=Path('data/interim')/output.name;local.mkdir(parents=True,exist_ok=True)
    runs=[];failures=[]
    with threadpool_limits(limits=2):
        for spec in plan['candidates']:
            if time.monotonic()-started>plan['maximum_cpu_seconds']:
                failures.append({'candidate':spec['name'],'reason':'Declared CPU wall budget exhausted before fit'});break
            begin=time.monotonic()
            try:
                all_rows=[];folds=[]
                for year in plan['validation_years']:
                    parts={a:feature_split(frames[a],f'{year}-01-01',f'{year}-12-31') for a in ASSETS}
                    model=fit({a:parts[a][0] for a in ASSETS},spec)
                    rows=[]
                    for a in ASSETS:
                        val=parts[a][1];prediction=predict(model,a,val)
                        rows.append(pd.DataFrame({'date':val.date,'asset':a,'target':val.target,'prediction':prediction}))
                    fold=pd.concat(rows,ignore_index=True);all_rows.append(fold)
                    folds.append({'year':year,'rows':len(fold),'rmse':float(np.sqrt(np.mean((fold.target-fold.prediction)**2)))})
                combined=pd.concat(all_rows,ignore_index=True);combined.to_csv(local/(spec['name']+'-validation.csv'),index=False,lineterminator='\n')
                runs.append({'spec':spec,'pooled_validation_rmse':float(np.sqrt(np.mean((combined.target-combined.prediction)**2))),'folds':folds,'runtime_seconds':time.monotonic()-begin})
            except Exception as error:failures.append({'candidate':spec['name'],'error':type(error).__name__,'reason':str(error)})
    if {r['spec']['name'] for r in runs}&{'zero','v2_full'}!={'zero','v2_full'}:raise ValueError('Required baselines failed')
    selected=choose(runs)
    selection={'plan_sha256':sha(plan_path),'implementation_sha256':sha(__file__),'procedure_sha256':sha('src/models/shared_forecaster.py'),'runs':runs,'failures':failures,'selected':selected,'test_used_for_selection':False,'development_provenance':provenance,'runtime_seconds':time.monotonic()-started,'disclosure':plan['disclosure']}
    write(output/'selection.json',selection)
    if selected['name']=='v2_full':
        result={**selection,'deployment_changed':False,'test_evaluated':False,'retained_checkpoint_sha256':sha('models/uncertainty-v2/bundle.joblib')}
        write(output/'results.json',result)
    return selection


def release(output='models/shared-forecast-v4',plan_path='docs/shared-forecast-plan.json'):
    output=Path(output)
    if (output/'results.json').exists():raise ValueError('Preserve completed final report')
    selection=json.loads((output/'selection.json').read_text());plan=json.loads(Path(plan_path).read_text())
    if sha(plan_path)!=selection['plan_sha256']:raise ValueError('Frozen selection plan changed')
    spec=selection['selected'];prior,provenance=fetch_snapshot('data/external/multi-asset-v1/development','2017-01-01','2025-12-31');frames=build_panel_features(prior)
    cutoff=pd.Timestamp(plan['final_training_label_end'],tz='UTC')
    with threadpool_limits(limits=2):model=fit({a:frames[a].loc[frames[a].target_date<=cutoff] for a in ASSETS},spec)
    quantiles={};calibration={}
    for a in ASSETS:
        rows=frames[a].loc[(frames[a].date.dt.year==2025)&frames[a].target.notna()]
        scale=np.maximum(rows.ewma_volatility.to_numpy(),1e-5)
        quantiles[a]=finite_quantile(abs(rows.target.to_numpy()-predict(model,a,rows))/scale)
        calibration[a]={'rows':len(rows),'normalized_quantile':quantiles[a],'last_label':str(rows.target_date.max())}
    bundle={'version':'shared_forecast_v4','point':model,'quantiles':quantiles,'context':prior,'selection_sha256':sha(output/'selection.json')}
    joblib.dump(bundle,output/'bundle.joblib')
    # Frozen validation selection and separate2025 calibration precede observed2026 replay.
    test,test_provenance=fetch_snapshot('data/external/multi-asset-v1/holdout','2026-01-01','2026-10-14')
    combined=validate_panel(pd.concat([prior,test],ignore_index=True));final_frames=build_panel_features(combined)
    recorded=[]
    for a in ASSETS:
        rows=final_frames[a].loc[(final_frames[a].date>=test.date.min())&final_frames[a].target.notna()]
        recorded.append(pd.DataFrame({'date':rows.date,'asset':a,'target':rows.target,'prediction':predict(model,a,rows),'half_width':np.maximum(rows.ewma_volatility.to_numpy(),1e-5)*quantiles[a]}))
    recorded=pd.concat(recorded,ignore_index=True);local=Path('data/interim')/output.name;local.mkdir(parents=True,exist_ok=True);recorded.to_csv(local/'observed-predictions.csv',index=False,lineterminator='\n')
    loaded=joblib.load(output/'bundle.joblib');replay=forecast(loaded,test)
    matched=recorded.merge(replay,on=['date','asset'],validate='one_to_one',suffixes=('_recorded','_loaded'))
    delta=float(abs(matched.prediction_recorded-matched.prediction_loaded).max());band_delta=float(abs(matched.half_width_recorded-matched.half_width_loaded).max())
    if len(matched)!=len(recorded) or delta>1e-14 or band_delta>1e-14:raise AssertionError('Actual checkpoint replay differs')
    report={'selection_sha256':sha(output/'selection.json'),'selected':spec,'observed_2026':interval_metrics(recorded.target,recorded.prediction,recorded.half_width),
            'zero_rmse':float(np.sqrt(np.mean(recorded.target**2))),'per_asset':{a:interval_metrics(g.target,g.prediction,g.half_width) for a,g in recorded.groupby('asset')},
            'calibration':calibration,'checkpoint_sha256':sha(output/'bundle.joblib'),'checkpoint_verification':{'matched_rows':len(matched),'point_max_delta':delta,'width_max_delta':band_delta},
            'test_provenance':test_provenance,'disclosure':plan['disclosure'],'risk_policy_changed':False}
    write(output/'results.json',report);return report



def forecast_directory(output,panel):
    """Forecast with either an adopted new procedure or hash-verified retained v2."""
    output=Path(output)
    if (output/'bundle.joblib').exists():
        return forecast(joblib.load(output/'bundle.joblib'),panel)
    metadata=json.loads((output/'results.json').read_text(encoding='utf-8'))
    if metadata['selected']['name']!='v2_full':raise ValueError('Selected checkpoint has not been exported')
    retained=Path('models/uncertainty-v2/bundle.joblib')
    if sha(retained)!=metadata['retained_checkpoint_sha256']:raise ValueError('Retained checkpoint hash changed')
    from src.models.uncertainty import forecast as retained_forecast
    rows=retained_forecast(joblib.load(retained),panel)
    if rows.empty:return rows
    rows['half_width']=(rows.upper-rows.lower)/2
    return rows[['date','asset','prediction','half_width','lower','upper']]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['study','release','forecast']);parser.add_argument('--output',default='models/shared-forecast-v4');parser.add_argument('--panel');parser.add_argument('--plan',default='docs/shared-forecast-plan.json');parser.add_argument('--predictions',default='data/interim/shared-new-forecasts.csv');args=parser.parse_args()
    if args.action=='study':result=run(args.output,args.plan)
    elif args.action=='release':result=release(args.output,args.plan)
    else:
        if not args.panel:parser.error('forecast requires --panel')
        rows=forecast_directory(args.output,pd.read_csv(args.panel));path=Path(args.predictions);path.parent.mkdir(parents=True,exist_ok=True);rows.to_csv(path,index=False,lineterminator='\n');result={'rows':len(rows),'path':str(path)}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
