"""Aggregate error and coverage diagnostics; never selects models or policies."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.advanced import write


def block_mean_interval(values, block=10, repeats=1000, seed=42):
    """Resample shared date blocks, keeping correlated asset losses together."""
    values = np.asarray(values, float)
    if not isinstance(block, int) or block < 1 or not isinstance(repeats, int) or repeats < 100 or values.ndim != 1 or len(values) < block or not np.isfinite(values).all():
        raise ValueError('Finite date-level losses, adequate support and >=100 repeats required')
    generator = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        starts = generator.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        indices = np.concatenate([(start + np.arange(block)) % len(values) for start in starts])[:len(values)]
        means.append(values[indices].mean())
    low, high = np.quantile(means, [.025, .975])
    return {'mean_mse_difference': float(values.mean()), 'percentile_95_interval': [float(low), float(high)],
            'block_observations': block, 'replicates': repeats, 'seed': seed,
            'excludes_zero': bool(low > 0 or high < 0),
            'interpretation': 'Negative favors the new predictor; descriptive observed-period uncertainty, not a prospective guarantee.'}


def report(predictions='data/interim/uncertainty-v2/predictions.csv',
           previous='data/interim/multi-asset-v1/final-predictions.csv',
           output='models/uncertainty-v2/error-diagnostics.json'):
    rows = pd.read_csv(predictions, parse_dates=['date'])
    if rows[['date', 'asset']].duplicated().any() or not np.isfinite(rows[['target', 'prediction', 'half_width']].to_numpy()).all():
        raise ValueError('Unique finite forecast observations required')
    rows['quarter'] = rows.date.dt.strftime('%Y') + '-Q' + ((rows.date.dt.month - 1) // 3 + 1).astype(str)
    rows['squared_error'] = (rows.target - rows.prediction) ** 2
    rows['zero_squared_error'] = rows.target ** 2
    rows['covered'] = abs(rows.target - rows.prediction) <= rows.half_width
    summaries = []
    for (asset, quarter), group in rows.groupby(['asset', 'quarter']):
        summaries.append({'asset': asset, 'quarter': quarter, 'rows': len(group),
                          'rmse': float(np.sqrt(group.squared_error.mean())),
                          'zero_rmse': float(np.sqrt(group.zero_squared_error.mean())),
                          'nominal_90_coverage': float(group.covered.mean()),
                          'mean_interval_width': float(2 * group.half_width.mean())})
    differences = (rows.squared_error - rows.zero_squared_error).groupby(rows.date).mean()
    result = {'quarterly': summaries, 'new_minus_zero': block_mean_interval(differences.to_numpy()),
              'selection_or_trading_changes': False,
              'limitations': ['Both periods and earlier test scores have already been observed.',
                              'Assets share CAD exposure; date blocks retain contemporaneous correlation.',
                              'A ten-observation circular bootstrap is a descriptive assumption, not guaranteed inference.',
                              'Small quarterly cohorts have unstable coverage. These diagnostics do not retune intervals.']}
    if Path(previous).exists():
        old = pd.read_csv(previous, parse_dates=['date'])
        joined = rows.merge(old[['date', 'asset', 'target', 'prediction']], on=['date', 'asset'],
                            suffixes=('_new', '_old'), validate='one_to_one')
        if len(joined) != len(rows) or not np.allclose(joined.target_new, joined.target_old, rtol=0, atol=1e-12):
            raise ValueError('Previous forecast cohort/targets do not match')
        delta = ((joined.target_new - joined.prediction_new) ** 2 - (joined.target_old - joined.prediction_old) ** 2).groupby(joined.date).mean()
        result['new_minus_previous_xgboost'] = block_mean_interval(delta.to_numpy())
        result['training_period_confounder'] = 'New point model trained through 2024; previous XGBoost through 2025. Difference is between complete procedures, not a controlled architecture-only effect.'
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/uncertainty-v2/error-diagnostics.json')
    args = parser.parse_args()
    report(output=args.output)
