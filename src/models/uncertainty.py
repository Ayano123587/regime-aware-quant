"""Regularized point forecasts and chronological residual interval calibration."""
import argparse
import json
import math
import platform
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.multi_asset import ASSETS, fetch_snapshot, sha, validate_panel
from src.features.multi_asset import FEATURES, build_panel_features
from src.models.advanced import write, feature_split


def finite_quantile(values, alpha=.1):
    """Finite-sample order statistic; no interpolation or heldout residual fitting."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any() or not 0 < alpha < 1:
        raise ValueError("Finite nonnegative calibration residuals and alpha in (0,1) required")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        raise ValueError("Insufficient calibration support for requested coverage")
    return float(np.sort(values)[rank - 1])


def fit(frame, spec):
    if frame.empty or not np.isfinite(frame[FEATURES + ['target']].to_numpy(float)).all():
        raise ValueError("Finite training features and matured targets required")
    if spec['kind'] == 'zero':
        return None
    if spec['kind'] != 'ridge' or not math.isfinite(spec['alpha']) or spec['alpha'] <= 0:
        raise ValueError("Unsupported regularized candidate")
    # No fitted intercept: centered features avoid learning an unstable unconditional drift.
    return make_pipeline(StandardScaler(), Ridge(alpha=spec['alpha'], fit_intercept=False)).fit(frame[FEATURES], frame.target)


def point(model, frame):
    if not np.isfinite(frame[FEATURES].to_numpy(float)).all():
        raise ValueError("Forecast features must be finite")
    return np.zeros(len(frame)) if model is None else model.predict(frame[FEATURES])


def interval_metrics(actual, prediction, width, alpha=.1):
    actual, prediction, width = [np.asarray(x, float) for x in (actual, prediction, width)]
    if actual.ndim != 1 or not len(actual) or actual.shape != prediction.shape or width.shape != actual.shape or not all(np.isfinite(x).all() for x in [actual, prediction, width]) or (width < 0).any() or not 0 < alpha < 1:
        raise ValueError("Aligned finite observations, forecasts and nonnegative half-widths required")
    low, high = prediction - width, prediction + width
    score = 2 * width + (2 / alpha) * (np.maximum(low - actual, 0) + np.maximum(actual - high, 0))
    return {'rows': len(actual), 'rmse': float(np.sqrt(np.mean((actual - prediction) ** 2))),
            'coverage': float(np.mean((actual >= low) & (actual <= high))),
            'mean_width': float(np.mean(2 * width)), 'mean_interval_score': float(score.mean())}


def forecast(checkpoint, supplied):
    if checkpoint.get('version') != 'uncertainty_v2' or checkpoint['assets'] != ASSETS:
        raise ValueError("Unsupported uncertainty checkpoint")
    supplied = validate_panel(supplied)
    context = validate_panel(checkpoint['context'])
    overlap = supplied.merge(context, on='date', suffixes=('_new', '_old'))
    if any(not np.allclose(overlap[a + '_new'], overlap[a + '_old'], atol=1e-12, rtol=0) for a in ASSETS):
        raise ValueError("Input revises saved source context")
    panel = validate_panel(pd.concat([context, supplied]).drop_duplicates('date').sort_values('date').reset_index(drop=True))
    frames, parts = build_panel_features(panel), []
    for asset in ASSETS:
        rows = frames[asset].loc[frames[asset].date.isin(supplied.date) & (frames[asset].date > context.date.max())].copy()
        if rows.empty:
            continue
        bundle = checkpoint['bundles'][asset]
        if not all(math.isfinite(bundle[key]) and bundle[key] >= 0 for key in ['absolute_quantile', 'normalized_quantile']):
            raise ValueError('Checkpoint interval calibration must be finite and nonnegative')
        values = point(bundle['model'], rows)
        dynamic = np.maximum(rows.ewma_volatility.to_numpy(), 1e-5) * bundle['normalized_quantile']
        parts.append(pd.DataFrame({'date': rows.date, 'asset': asset, 'prediction': values,
                                 'lower': values - dynamic, 'upper': values + dynamic,
                                 'constant_lower': values - bundle['absolute_quantile'],
                                 'constant_upper': values + bundle['absolute_quantile']}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def search_candidates(development, plan):
    """Chronological model search without accessing calibration or test rows."""
    candidates = []
    for spec in plan['candidates']:
        squared, count = 0., 0
        folds = []
        for year in plan['validation_years']:
            fold_squared, fold_count = 0., 0
            for asset in ASSETS:
                training, validation, _ = feature_split(development[asset], f'{year}-01-01', f'{year}-12-31')
                residual = validation.target.to_numpy() - point(fit(training, spec), validation)
                fold_squared += float(residual @ residual)
                fold_count += len(residual)
            folds.append({'year': year, 'rows': fold_count, 'rmse': math.sqrt(fold_squared / fold_count)})
            squared += fold_squared
            count += fold_count
        candidates.append({'spec': spec, 'pooled_validation_rmse': math.sqrt(squared / count), 'folds': folds})
    return candidates, min(candidates, key=lambda x: x['pooled_validation_rmse'])['spec']


def train_selected(output, plan_path='docs/uncertainty-plan.json'):
    output = Path(output)
    if output.exists():
        raise ValueError("Retrospective report already exists; use a fresh reproduction directory")
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    # This implementation supports only the frozen chronological design.
    if plan['assets'] != ASSETS or plan['validation_years'] != [2021, 2022, 2023, 2024] or plan['calibration_year'] != 2025 or plan['test_year'] != 2026 or plan['interval_alpha'] != .1 or plan['final_training_label_end'] != '2024-12-31':
        raise ValueError("Unsupported uncertainty study design")
    prior, provenance = fetch_snapshot('data/external/multi-asset-v1/development', '2017-01-01', '2025-12-31')
    development = build_panel_features(prior)
    candidates, selected = search_candidates(development, plan)
    bundles, calibration = {}, {}
    cutoff = pd.Timestamp(plan['final_training_label_end'], tz='UTC')
    for asset in ASSETS:
        frame = development[asset]
        training = frame.loc[frame.target_date <= cutoff]
        rows = frame.loc[(frame.date >= pd.Timestamp('2025-01-01', tz='UTC')) & frame.target.notna()]
        if training.target_date.max() >= rows.date.min():
            raise ValueError("Calibration label purge failed")
        model = fit(training, selected)
        residual = np.abs(rows.target.to_numpy() - point(model, rows))
        absolute = finite_quantile(residual)
        normalized = finite_quantile(residual / np.maximum(rows.ewma_volatility.to_numpy(), 1e-5))
        bundles[asset] = {'model': model, 'absolute_quantile': absolute, 'normalized_quantile': normalized}
        calibration[asset] = {'rows': len(rows), 'last_training_label': str(training.target_date.max()),
                              'first_calibration_decision': str(rows.date.min()), 'last_calibration_label': str(rows.target_date.max()),
                              'absolute_quantile': absolute, 'normalized_quantile': normalized}
    selection = {'plan_sha256': sha(plan_path), 'implementation_sha256': sha(__file__), 'candidates': candidates, 'selected': selected,
                 'calibration': calibration, 'development_provenance': provenance,
                 'test_used_for_selection': False, 'disclosure': plan['disclosure']}
    write(output / 'selection.json', selection)  # Written before observed-test replay.
    checkpoint = {'version': 'uncertainty_v2', 'assets': ASSETS, 'bundles': bundles,
                  'context': prior, 'selection_sha256': sha(output / 'selection.json')}
    joblib.dump(checkpoint, output / 'bundle.joblib')
    return selection


def evaluate_selected(output, report_directory, predictions, *, acknowledge_observed=False):
    if not acknowledge_observed:
        raise ValueError('Explicit acknowledgement of observed 2026 evaluation required')
    output, report_directory, predictions = Path(output), Path(report_directory), Path(predictions)
    if report_directory.exists() or predictions.exists():
        raise ValueError('Fresh report directory and predictions path required')
    if predictions.resolve() == (output / 'bundle.joblib').resolve():
        raise ValueError('Predictions must not replace checkpoint')
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    checkpoint = joblib.load(output / 'bundle.joblib')
    if checkpoint['selection_sha256'] != sha(output / 'selection.json'):
        raise ValueError('Selection fingerprint mismatch')
    prior, bundles, selected = checkpoint['context'], checkpoint['bundles'], selection['selected']
    plan = {'study': 'regularized-uncertainty-v2', 'disclosure': selection['disclosure']}
    test, test_provenance = fetch_snapshot('data/external/multi-asset-v1/holdout', '2026-01-01', '2026-10-14')
    combined = validate_panel(pd.concat([prior, test], ignore_index=True))
    frames = build_panel_features(combined)
    prediction_path = predictions
    predictions, metrics = [], {}
    for asset in ASSETS:
        rows = frames[asset].loc[(frames[asset].date >= test.date.min()) & frames[asset].target.notna()]
        bundle = bundles[asset]
        predicted = point(bundle['model'], rows)
        widths = np.maximum(rows.ewma_volatility.to_numpy(), 1e-5) * bundle['normalized_quantile']
        metrics[asset] = {'volatility_normalized': interval_metrics(rows.target, predicted, widths),
                          'constant': interval_metrics(rows.target, predicted, np.full(len(rows), bundle['absolute_quantile'])),
                          'zero_rmse': float(np.sqrt(np.mean(rows.target.to_numpy() ** 2)))}
        predictions.append(pd.DataFrame({'date': rows.date, 'asset': asset, 'target': rows.target,
                                         'prediction': predicted, 'half_width': widths}))
    recorded = pd.concat(predictions, ignore_index=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open('x', encoding='utf-8', newline='') as stream:
        recorded.to_csv(stream, index=False, lineterminator='\n')
    loaded = joblib.load(output / 'bundle.joblib')
    replay = forecast(loaded, test)
    matched = recorded.merge(replay, on=['date', 'asset'], validate='one_to_one', suffixes=('_recorded', '_loaded'))
    delta = float(np.abs(matched.prediction_recorded - matched.prediction_loaded).max())
    band_delta = float(np.abs((matched.upper - matched.lower) / 2 - matched.half_width).max())
    if len(matched) != len(recorded) or delta > 1e-14 or band_delta > 1e-14:
        raise AssertionError("Actual checkpoint replay differs")
    report = {'study': plan['study'], 'selection_sha256': checkpoint['selection_sha256'],
              'implementation_sha256': sha(__file__),
              'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']}},
              'selected': selected, 'test_provenance': test_provenance, 'per_asset': metrics,
              'pooled_rmse': float(np.sqrt(np.mean((recorded.target - recorded.prediction) ** 2))),
              'pooled_zero_rmse': float(np.sqrt(np.mean(recorded.target ** 2))),
              'checkpoint_verification': {'scored_rows': len(matched), 'inference_rows': len(replay),
                                          'point_max_delta': delta, 'half_width_max_delta': band_delta,
                                          'checkpoint_sha256': sha(output / 'bundle.joblib')},
              'limitations': [plan['disclosure'], 'The point model stays fixed after 2024; 2025 labels calibrate intervals only.',
                              'Nominal 90% intervals have no finite-sample guarantee under temporal dependence or drift.',
                              'Both interval variants are reported, not selected by 2026 score.',
                              'Reference rates are non-executable; no trading policy is changed by this study.']}
    write(report_directory / 'results.json', report)
    print(json.dumps(report, indent=2))
    return report


def run(output='models/uncertainty-v2', plan_path='docs/uncertainty-plan.json', *, acknowledge_observed=False, report_directory=None, predictions=None):
    if not acknowledge_observed or report_directory is None or predictions is None:
        raise ValueError('Observed evaluation requires acknowledgement and explicit fresh report/predictions paths')
    if Path(report_directory).exists() or Path(predictions).exists():
        raise ValueError('Fresh report and predictions paths required')
    train_selected(output, plan_path)
    return evaluate_selected(output, report_directory, predictions, acknowledge_observed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'evaluate', 'study', 'forecast'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--panel')
    parser.add_argument('--predictions')
    parser.add_argument('--report-directory')
    parser.add_argument('--acknowledge-observed', action='store_true')
    args = parser.parse_args()
    if args.action == 'train':
        train_selected(args.output)
    elif args.action in ['evaluate', 'study']:
        if not args.acknowledge_observed or not args.report_directory or not args.predictions:
            parser.error('evaluation requires --acknowledge-observed, --report-directory and --predictions')
        fn = run if args.action == 'study' else evaluate_selected
        if args.action == 'study':
            fn(args.output, acknowledge_observed=True, report_directory=args.report_directory, predictions=args.predictions)
        else:
            fn(args.output, args.report_directory, args.predictions, acknowledge_observed=True)
    else:
        if not args.panel or not args.predictions:
            parser.error('forecast requires --panel and explicit fresh --predictions')
        path = Path(args.predictions)
        if path.exists():
            raise ValueError('Forecast output already exists')
        result = forecast(joblib.load(Path(args.output) / 'bundle.joblib'), pd.read_csv(args.panel))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='') as stream:
            result.to_csv(stream, index=False, lineterminator='\n')
        print(json.dumps({'rows': len(result), 'output': str(path)}))


if __name__ == '__main__':
    main()
