"""Prespecified retrospective ablations, then one new-year test."""
import argparse
import hashlib
import json
import platform
import importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request, urlopen
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from src.config import FEATURES
from src.data.make_dataset import load_prices
from src.features.build_features import build_features
from src.models.hmm import GaussianHMM
from src.models.train_model import fit_bundle, forecast, REGIME_FEATURES, returns_summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_for_validation(frame, start, end):
    start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    validation = frame.loc[(frame.date >= start) & (frame.date <= end) & (frame.target_date <= end)].copy()
    train = frame.loc[(frame.date < validation.date.min()) & (frame.target_date < validation.date.min())].copy()
    bridge = frame.loc[(frame.date > train.date.max()) & (frame.date < validation.date.min())].copy()
    return train, validation, bridge


def development(output='models/development', data='data/raw/prices.csv'):
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    if (output / 'selection.json').exists():
        raise ValueError('Configuration selection already frozen')
    if sha(data) != plan['historical_data_sha256']:
        raise ValueError('Historical development snapshot changed')
    prices = load_prices(data)
    if prices.date.max() > pd.Timestamp('2025-12-31', tz='UTC'):
        raise ValueError('Development cannot access 2026 observations')
    frame = build_features(prices).dropna(subset=['target', 'target_date'])
    scores, specs, folds, local = defaultdict(list), {}, [], []
    zero_errors = []
    for fold, (start, end) in enumerate(plan['validation_windows']):
        train, val, bridge = frame_for_validation(frame, start, end)
        actual = val.target.to_numpy()
        zero_errors.extend(actual**2)
        folds.append({'fold': fold, 'train_rows': len(train), 'validation_rows': len(val),
                      'train_last_target': str(train.target_date.max()),
                      'validation_first': str(val.date.min()), 'validation_last_target': str(val.target_date.max()),
                      'bridge_rows': len(bridge)})
        for states in plan['states']:
            hmm = GaussianHMM(states=states, iterations=plan['hmm_max_iterations']).fit(train[REGIME_FEATURES].to_numpy())
            for tree in plan['tree_settings']:
                bundle = fit_bundle(train, states=states, strength=1., tree_settings=tree,
                                    fitted_hmm=hmm, context_prices=prices)
                previous = bundle['last_posterior']
                if len(bridge):
                    previous = hmm.filter(bridge[REGIME_FEATURES].to_numpy(), previous)[-1]
                gated, _ = forecast(bundle, val, previous)
                global_pred = bundle['baseline'].predict(val[FEATURES].to_numpy())
                for strength in plan['gating_strength']:
                    spec = {'states': states, 'strength': strength, 'tree': tree}
                    key = json.dumps(spec, sort_keys=True)
                    specs[key] = spec
                    pred = strength * gated + (1-strength) * global_pred
                    scores[key].extend((pred-actual)**2)
                    local.append(pd.DataFrame({'date': val.date, 'target': actual, 'prediction': pred,
                                               'fold': fold, 'candidate': key}))
                    print(json.dumps({'fold': fold, 'spec': spec,
                                      'rmse': float(np.sqrt(np.mean((pred-actual)**2))),
                                      'matched_global_rmse': float(np.sqrt(np.mean((global_pred-actual)**2))),
                                      'zero_rmse': float(np.sqrt(np.mean(actual**2)))}), flush=True)
    runs = [{'spec': specs[k], 'pooled_validation_rmse': float(np.sqrt(np.mean(v))),
             'validation_rows': len(v)} for k, v in scores.items()]
    winner = min(runs, key=lambda r: (r['pooled_validation_rmse'], r['spec']['strength'],
                                     r['spec']['states'], r['spec']['tree']['max_depth']))
    report = {'plan_sha256': sha(output/'plan.json'), 'historical_data_sha256': sha(data),
              'winner': winner, 'runs': runs, 'folds': folds, 'candidates': len(runs),
              'new2026values_used': False, 'disclosure': plan['development_disclosure'],
              'zero_pooled_validation_rmse': float(np.sqrt(np.mean(zero_errors))),
              'matched_global_baselines': [run for run in runs if run['spec']['strength']==0 and run['spec']['states']==2]}
    (output/'selection.json').write_text(json.dumps(report, indent=2))
    Path('data/interim').mkdir(parents=True, exist_ok=True)
    pd.concat(local, ignore_index=True).to_csv('data/interim/development-predictions.csv', index=False)
    print(json.dumps({'winner': winner}))


def block_uncertainty(actual, pred, baseline, draws=1000):
    rng = np.random.default_rng(42)
    n, samples = len(actual), []
    for _ in range(draws):
        starts = rng.integers(0, n, size=(n+4)//5)
        indices = np.concatenate([(start+np.arange(5)) % n for start in starts])[:n]
        samples.append(np.sqrt(np.mean((actual[indices]-pred[indices])**2))
                       - np.sqrt(np.mean((actual[indices]-baseline[indices])**2)))
    return {'method': 'paired circular moving-block bootstrap, block5, seed42, 1000draws; descriptive sampling uncertainty only',
            'rmse_difference_95pct': [float(v) for v in np.quantile(samples, [.025, .975])]}


def final_test(output='models/development', data='data/raw/prices.csv'):
    output = Path(output)
    if (output/'evaluation.json').exists():
        raise ValueError('New-year test already evaluated; preserve result')
    plan = json.loads((output/'plan.json').read_text())
    selection = json.loads((output/'selection.json').read_text())
    if sha(output/'plan.json') != selection['plan_sha256'] or sha(data) != selection['historical_data_sha256']:
        raise ValueError('Frozen development state changed')
    # New-year source values are requested only after configuration selection.
    url = ('https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date='
           + plan['holdout_start'] + '&end_date=' + plan['holdout_end'])
    with urlopen(Request(url, headers={'User-Agent': 'regime-aware-quant research'}), timeout=60) as response:
        content = response.read()
    Path('data/external').mkdir(parents=True, exist_ok=True)
    Path('data/external/boc-2026.json').write_bytes(content)
    raw = json.loads(content)['observations']
    new = pd.DataFrame([{'date': r['d'], 'close': r['FXUSDCAD']['v']} for r in raw if 'FXUSDCAD' in r])
    new.to_csv('data/external/prices-2026.csv', index=False)
    new = load_prices('data/external/prices-2026.csv')
    if (new.date.min() < pd.Timestamp(plan['holdout_start'], tz='UTC')
            or new.date.max() > pd.Timestamp(plan['holdout_end'], tz='UTC')):
        raise ValueError('Holdout dates outside published frozen window')
    prior = load_prices(data)
    frame = build_features(pd.concat([prior, new], ignore_index=True))
    test = frame.loc[(frame.date >= pd.Timestamp(plan['holdout_start'], tz='UTC'))
                     & frame.target.notna() & (frame.target_date <= new.date.max())].copy()
    train = build_features(prior).dropna(subset=['target', 'target_date'])
    if train.target_date.max() >= test.date.min():
        raise ValueError('Final label purge failed')
    spec = selection['winner']['spec']
    bundle = fit_bundle(train, states=spec['states'], strength=spec['strength'], tree_settings=spec['tree'],
                        hmm_iterations=plan['hmm_max_iterations'], context_prices=prior)
    bridge = frame.loc[(frame.date > train.date.max()) & (frame.date < test.date.min())]
    previous = bundle['last_posterior']
    if len(bridge):
        previous = bundle['hmm'].filter(bridge[REGIME_FEATURES].to_numpy(), previous)[-1]
    pred, posterior = forecast(bundle, test, previous)
    base = bundle['baseline'].predict(test[FEATURES].to_numpy())
    actual, zero = test.target.to_numpy(), np.zeros(len(test))
    local = Path('data/interim/final-2026-predictions.csv')
    pd.DataFrame({'date': test.date, 'target_date': test.target_date, 'target': actual, 'selected': pred,
                  'global': base, 'zero': zero, 'state': posterior.argmax(axis=1)}).to_csv(local, index=False)
    joblib.dump(bundle, output/'bundle.joblib')
    report = {'selected_spec': spec, 'selection_sha256': sha(output/'selection.json'),
              'plan_sha256': sha(output/'plan.json'), 'new_source_url': url,
              'new_source_sha256': hashlib.sha256(content).hexdigest(),
              'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
              'environment':{'python':platform.python_version(),**{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','xgboost']}},
              'source_observation_first': str(new.date.min()), 'source_observation_last': str(new.date.max()),
              'new_observations': len(new), 'test_rows': len(test), 'test_feature_first': str(test.date.min()),
              'test_feature_last': str(test.date.max()), 'test_target_last': str(test.target_date.max()),
              'train_rows': len(train), 'train_last_feature': str(train.date.max()),
              'train_last_target': str(train.target_date.max()), 'bridge_rows': len(bridge),
              'selected': returns_summary(pred, actual), 'matched_global': returns_summary(base, actual),
              'zero': returns_summary(zero, actual),
              'selected_minus_global_uncertainty': block_uncertainty(actual, pred, base),
              'selected_minus_zero_uncertainty': block_uncertainty(actual, pred, zero),
              'local_predictions_sha256': sha(local), 'checkpoint_context_rows': len(bundle['price_context']),
              'publication_assumption': 'Indicative averages available after official16:30ETpublication; historical vintages unavailable',
              'limitations': ['One currency pair and one new partial year', '2017-2025 used retrospectively for development',
                              'Causal filter only, no future heldout smoothing',
                              'Reference averages not executable quotes; publication delays and revisions not reconstructed',
                              'No test-guided tuning or guaranteed profit']}
    (output/'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['search', 'test'])
    parser.add_argument('--output', default='models/development')
    parser.add_argument('--data', default='data/raw/prices.csv')
    args = parser.parse_args()
    if args.action == 'search':
        development(args.output, args.data)
    else:
        final_test(args.output, args.data)


if __name__ == '__main__':
    main()
