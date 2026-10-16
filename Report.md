# FX forecasting: method and results

The selected model is per-asset ridge regression (alpha 10,000, no fitted intercept) over 22 causal features. It doubles the predicted return when the decision-time volatility ratio is strictly greater than 1.2. Scaling uses matured training rows only; peer features lag one publication, and targets measure the delayed t+1 to t+2 return.

## Validation and training

Chronological 2021–2024 folds contain 2,961 eligible asset decisions. Validation RMSE is 0.00390641 for the selected gate, 0.00391099 for ungated ridge and 0.00391380 for zero. Repeated validation development makes these differences descriptive; they are not independent evidence of an edge.

Three fresh mean fits use all 5,799 eligible decisions through 2024. Separate 2025 calibration uses all 741 eligible decisions without refitting the mean. Each asset's interval half-width is its fixed 90% normalized residual quantile multiplied by decision-time EWMA volatility, floored at 0.00001.

## Observed cohort

The already observed 2026 snapshot contains 196 publications per asset. Evaluation covers all 582 matured decisions; six pending decisions are excluded from scoring.

| Forecast | RMSE |
| --- | ---: |
| Selected volatility gate | 0.00295114 |
| Ungated ridge | 0.00295666 |
| Zero return | 0.00295177 |

Selected nominal 90% intervals have coverage 0.88316, mean width 0.00949003 and proper interval score 0.01309862. The point improvement over zero is tiny and mixed across assets. Reference rates are not executable quotes; these results do not demonstrate trading profit. Serial dependence and current-vintage data also limit interval guarantees.

## Reproduction

Follow [README.md](README.md) to fetch mean data, train, verify the mean freeze, acquire calibration data, calibrate and forecast. Provider revisions stop pinned reproduction. Fresh training and calibration were verified in a separate cloud run; all 582 matured point forecasts and interval bounds match the audited reference within 6.1e-18, and all 588 saved forecasts reproduce after reload.

[Training and calibration evidence](docs/gated-training-calibration.json) and [observed results](docs/gated-observed-results.json) record the model, source and audit identities. The four notebooks contain executed source-data analysis, visible model code, full-cohort validation, and result plots. Source terms and historical-vintage limitations are in [docs/datasets.md](docs/datasets.md). Data, weights and individual predictions remain ignored.
