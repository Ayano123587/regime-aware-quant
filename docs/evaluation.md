# Executed experiment

The fixed three-fold CPU experiment ran against 2,244 real Bank of Canada USD/CAD
reference-rate observations from 2017-01-03 through 2025-12-31. The 60-observation warmup
and next-observation label leave 2,184 supervised rows. Each holdout fold has 364 rows.

| Model | Overall holdout RMSE | Direction accuracy |
|---|---:|---:|
| Gaussian-HMM gated XGBoost | 0.00364161 | 0.54487 |
| Global XGBoost | 0.00362259 | 0.55311 |

**The regime model did not improve the primary metric.** Keep this negative result rather
than choosing a favorable subperiod. The final fold had a negative cost-sensitive diagnostic
return, and economic profitability is not established. This single currency pair is not
evidence that regime conditioning generally helps or fails.

Full fold dates, strict target purges, training sizes, HMM iterations, turnover and diagnostic
equity metrics appear in `models/run/evaluation.json`. The local bundle can predict new
observations after its training cutoff; joblib weights and row-level CSVs are ignored.

Source bytes SHA256:
`5f1be1ecdb13ea5852819f08734b0bb6198b404c6d678591565b3981d062dd68`.
Cleaned CSV SHA256 appears in the numerical evaluation manifest. API responses may be
revised; retain source bytes locally for an exact rerun.

Commands executed:

```sh
python -m pip install -e ".[dev]"
python -m src.data.make_dataset
python -m pytest -q
python -m src.models.train_model
python -m src.models.predict_model --data data/raw/prices.csv
```

Tests cover causal feature prefix equality, causal filter prefix/chunk equivalence,
finite normalized probabilities, bad input, checkpoint round-trip, expert forecast shape,
strict target-time purging, initial/reversal costs, and duplicate-date rejection.
The fixed run uses seed 42, three states, 100 depth-two trees per expert, and a maximum
60 HMM EM iterations. It ran on CPU; no Kaggle run was needed. No hyperparameter search
was performed on these holdouts.

## Prespecified development and new 2026 holdout

The 2017–2025 data had already been observed, so this is retrospective development. Two chronological validation windows, 2021–2022 and 2023–2025, contain 1,242 labeled observations after strict purges. Twenty-four configurations vary the HMM state count, regime/global blend and tree regularization. Pooled validation RMSE selected two states, a 25% regime / 75% global blend, depth 1 and lambda 30. Validation RMSE was 0.0036214205, versus 0.0036218404 for the matched global model and 0.0036269646 for a zero forecast. This nominal gain is extremely small.

The 2026 observations were requested after selection was frozen. January 2 through October 14 supplies 196 observations and 195 labeled next-observation forecasts. Refit labels end on December 31, 2025; evaluation features begin on January 2, 2026. One purged feature-only bridge advances the posterior. No 2026 observation fits model parameters, and held-out smoothing is never used.

| New 2026 test model | RMSE |
|---|---:|
| Selected regime/global blend | 0.0025302463 |
| Matched global XGBoost | 0.0025309643 |
| Zero-return forecast | 0.0025475597 |

The selected model has a small nominal advantage. A paired circular bootstrap with five-observation blocks, 1,000 draws and seed 42 gives a 95% interval of [-0.0000068207, +0.0000049777] for selected-minus-global RMSE, and [-0.0000662026, +0.0000320518] for selected-minus-zero RMSE. Both include zero: **no statistically clear advantage is established**. This describes sampling variability within one short currency-pair window, not revisions or domain uncertainty. No test-guided switch or further tuning occurred.

The request ends on October 14 because the October 15 publication was not safely available at planning time. Reference averages are not executable closes. Cost and equity fields are hypothetical diagnostics. Point-in-time publication and revision vintages are unavailable.

`models/development` preserves the plan, candidate scores, selection, new-year aggregate results, source hashes, date bounds and runtime versions. Row-level prices, predictions and binary checkpoints remain ignored. The final runner refuses to overwrite results. Inference checkpoints retain 64 raw price observations and reject malformed or unordered input, inconsistent overlapping revisions, long missing bridges and invalid posterior probabilities. Tests compare full-history and saved-context inference.

Checkpoint audit: saved-context inference on the new-year-only CSV reproduces all 195 labeled forecasts with maximum absolute difference 4.1e-20, and emits 196 new supplied observations including the final unlabeled Oct 14 feature. Bridge rows advance state without being emitted. The audit reads the saved checkpoint; it does not refit parameters. Continuous supplied observation history is required; the long-gap heuristic does not certify that every short missing publication has been detected.
