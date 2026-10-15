# Regularized forecasts and uncertainty intervals

The earlier multiasset XGBoost study did not beat zero return. This round tests a
lower-variance mean model and reports uncertainty rather than adding more unvalidated
trading activity. The original HMM, risk engine, cash selection and scores are preserved.

Four expanding annual validation windows (2021–2024) compare zero return with standardized
ridge regressions at alpha100/1000/10000. Feature scaling is fitted on training rows only;
the regression has no intercept. Labels are still delayed returns from t+1 to t+2, and
training labels mature strictly before each validation decision. The selected alpha10000
achieves pooled validation RMSE0.00391099 versus zero0.00391380.

The selected value was the boundary of the first grid. A subsequent validation-only
range check tested alpha3000/10000/30000/100000/1000000, with zero still eligible.
Alpha10000 remained best; stronger shrinkage converged toward zero performance rather
than providing another gain. [All candidates and folds](../models/uncertainty-v2/validation-followup.json)
retain the weaker settings. No calibration or 2026 rows were scored in this follow-up.

The final point model stays fixed after labels through December2024. Separately, 2025
forecast residuals calibrate per-asset nominal90% intervals using a finite-sample order
statistic: either absolute residuals (constant width) or residuals normalized by causal
EWM volatility. There is no refit on 2025 labels. Both band methods are reported without
choosing between them using test scores. Full feature-only historical prices initialize
recursive EWM state for checkpoint inference.

## Retrospective observed-period results

The 2026 source and earlier scores were already viewed. These results are retrospective
comparisons on the same582 labeled forecasts, not a new untouched holdout. New point
RMSE is **0.00295666**, earlier XGBoost0.00298725, zero0.00295177. Training periods differ:
the new point model trains through2024 and the earlier XGBoost through2025. The difference
is between complete procedures, not an architecture-only effect. Zero remains slightly
better overall, so this study does not establish a tradable edge.

| Asset | New RMSE | Zero RMSE | Volatility-band coverage | Constant-band coverage |
|---|---:|---:|---:|---:|
| EUR/CAD | .00239737 | .00239998 | .88144 | .96392 |
| GBP/CAD | .00269308 | .00268342 | .90722 | .97938 |
| AUD/CAD | .00363668 | .00363017 | .85567 | .85052 |

Volatility-based bands are narrower for EUR/GBP and have lower mean interval scores
than constant bands there. AUD has the opposite interval-score ordering. The nominal90%
coverage target is missed for some assets. Time dependence and nonstationarity invalidate
an unconditional split-conformal guarantee; neither band type is claimed universally better.
The report includes width and proper interval-score penalties as well as coverage.

Date-block diagnostics keep simultaneous asset errors together. With1000 circular
bootstrap replicates and10-observation blocks, the new-minus-zero MSE interval crosses
zero. The difference against the earlier XGBoost is negative over this observed period,
but remains retrospective and confounded by training-period differences. Quarterly
coverage/error tables expose instability and small cohorts; they do not adapt the model
or risk policy. See [error diagnostics](../models/uncertainty-v2/error-diagnostics.json).

## Reproduce and forecast

```sh
python -m src.models.uncertainty study --output models/uncertainty-reproduction
python -m src.models.uncertainty forecast --output models/uncertainty-v2 --panel data/external/multi-asset-v1/holdout/panel.csv
python -m src.visualization.forecast_diagnostics
```

Use a fresh output directory for a repeat; reports refuse overwrites. The actual saved
checkpoint reproduces all582 scored point forecasts (maxdifference2.71e-20) and band
half-widths (4.34e-19). It also emits six later forecasts with no matured target in the
snapshot. The tests verify exact order-statistic behavior, interval penalties, revision
rejection, nonfinite calibration rejection and prefix invariance. Raw snapshots, individual
forecasts and binary checkpoints remain ignored. Only aggregate reports are committed.

Further mean-model search has plateaued on this regularization grid. A credible next
step needs richer, correctly timestamped external predictors or a new future period;
repeatedly tuning this observed2026 snapshot would not provide stronger evidence.
