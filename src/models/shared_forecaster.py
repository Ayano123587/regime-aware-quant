"""Importable shared/per-asset ridge procedures with causal volatility normalization."""
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.multi_asset import ASSETS,validate_panel
from src.features.multi_asset import FEATURES,build_panel_features

REDUCED=['return_1','momentum_3','momentum_20','zscore_20','rsi_14','autocorrelation_20','peer_return_lag1','volatility_ratio']


def design(frame,spec):
    if spec.get('features','full') not in ['full','reduced']:
        raise ValueError('Unsupported feature set')
    columns=FEATURES if spec.get('features','full')=='full' else REDUCED
    values=frame[columns].copy()
    scale=np.maximum(frame.ewma_volatility.to_numpy(float),1e-5)
    if not np.isfinite(values.to_numpy(float)).all() or not np.isfinite(scale).all():
        raise ValueError('Finite causal features and scales required')
    if spec['kind']=='pooled_normalized':
        for column,horizon in [('return_1',1),('peer_return_lag1',1),('momentum_3',3),('momentum_20',20)]:
            if column in values:values[column]=values[column]/(scale*math.sqrt(horizon))
    if not np.isfinite(values.to_numpy(float)).all():raise ValueError('Nonfinite normalized features')
    return values,scale


def fit(frames,spec):
    if spec['kind']=='fixed_blend':
        if len(spec['components'])!=2:raise ValueError('Only two predefined equal components supported')
        return {'spec':spec,'components':[fit(frames,component) for component in spec['components']]}
    if set(frames)!=set(ASSETS) or spec['kind'] not in ['zero','per_asset','pooled','pooled_normalized']:
        raise ValueError('Unsupported procedure or asset set')
    prepared={}
    for asset,frame in frames.items():
        x,scale=design(frame,spec);target=frame.target.to_numpy(float)
        if not len(frame) or not np.isfinite(target).all():raise ValueError('Finite matured training labels required')
        weights=None
        if 'half_life_days' in spec:
            half=spec['half_life_days']
            if not isinstance(half,(int,float)) or isinstance(half,bool) or not math.isfinite(half) or half<=0:raise ValueError('Positive finite recency half-life required')
            dates=pd.to_datetime(frame.target_date,utc=True,errors='raise')
            if dates.isna().any():raise ValueError('Matured training publication dates required')
            age=(dates.max()-dates).dt.total_seconds().to_numpy()/86400
            weights=np.exp2(-age/half)
            if not np.isfinite(weights).all() or weights.mean()<=0:raise ValueError('Invalid recency weights')
            weights=weights/weights.mean()
        prepared[asset]=(x,target/scale if spec['kind']=='pooled_normalized' else target,weights)
    if spec['kind']=='zero':return {'spec':spec,'models':{}}
    if isinstance(spec['alpha'],bool) or not isinstance(spec['alpha'],(float,int)) or not math.isfinite(spec['alpha']) or spec['alpha']<=0:
        raise ValueError('Positive finite ridge alpha required')
    def train(x,y,weights):
        model=make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha'],fit_intercept=False))
        if weights is None:return model.fit(x,y)
        return model.fit(x,y,standardscaler__sample_weight=weights,ridge__sample_weight=weights)
    if spec['kind']=='per_asset':models={a:train(*prepared[a]) for a in ASSETS}
    else:models={'shared':train(pd.concat([prepared[a][0] for a in ASSETS],ignore_index=True),np.concatenate([prepared[a][1] for a in ASSETS]),None if prepared[ASSETS[0]][2] is None else np.concatenate([prepared[a][2] for a in ASSETS]))}
    return {'spec':spec,'models':models}


def predict(bundle,asset,frame):
    if asset not in ASSETS:raise ValueError('Unknown asset')
    spec=bundle['spec']
    if spec['kind']=='fixed_blend':
        values=np.mean([predict(component,asset,frame) for component in bundle['components']],axis=0)
        if not np.isfinite(values).all():raise ValueError('Nonfinite blended forecasts')
        return values
    x,scale=design(frame,spec)
    if spec['kind']=='zero':values=np.zeros(len(frame))
    else:
        values=bundle['models'][asset if spec['kind']=='per_asset' else 'shared'].predict(x)
        if spec['kind']=='pooled_normalized':values=values*scale
    if not np.isfinite(values).all():raise ValueError('Nonfinite point forecasts')
    return values


def forecast(checkpoint,supplied):
    if checkpoint['version']!='shared_forecast_v4':raise ValueError('Unsupported shared checkpoint')
    supplied=validate_panel(supplied);context=validate_panel(checkpoint['context'])
    overlap=supplied.merge(context,on='date',suffixes=('_new','_old'))
    if any(not np.allclose(overlap[a+'_new'],overlap[a+'_old'],atol=1e-12,rtol=0) for a in ASSETS):raise ValueError('Input revises saved context')
    panel=validate_panel(pd.concat([context,supplied]).drop_duplicates('date').sort_values('date').reset_index(drop=True))
    frames=build_panel_features(panel);parts=[]
    for a in ASSETS:
        rows=frames[a].loc[frames[a].date.isin(supplied.date)&(frames[a].date>context.date.max())]
        if rows.empty:continue
        point=predict(checkpoint['point'],a,rows);q=checkpoint['quantiles'][a]
        if not math.isfinite(q) or q<0:raise ValueError('Finite nonnegative interval calibration required')
        width=np.maximum(rows.ewma_volatility.to_numpy(),1e-5)*q
        if not np.isfinite(width).all() or not np.isfinite(point-width).all() or not np.isfinite(point+width).all():raise ValueError('Nonfinite interval bands')
        parts.append(pd.DataFrame({'date':rows.date,'asset':a,'prediction':point,'half_width':width,'lower':point-width,'upper':point+width}))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
