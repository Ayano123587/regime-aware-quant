# FX forecasting: results and reproduction

The retained three-asset model uses train-scaled ridge regression (alpha=10,000), with no fitted intercept. Its previously observed 2026 RMSE is 0.002956659782, compared with 0.002987248013 for the global-tree baseline and 0.002951768213 for zero. The learned forecast improves the tree baseline but does not beat zero. This is not evidence of profit or a deployable trading edge.

Data are daily Bank of Canada EUR/CAD, GBP/CAD and AUD/CAD reference rates. The mean model trains through 2024; 2021-24 chronological folds choose its configuration. Residual calibration is separate in 2025. Features use only published observations, peer covariates lag one observation, labels span delayed t+1 to t+2 returns, and boundary labels are purged. Saved inference retains full causal EWMA context. Current API snapshots lack as-published historical vintages.

## Calibration

Fixed residual calibration won 2025 proper interval score: 0.01601720 versus rolling-120 0.01603791 and rolling-60 0.01638900. No rolling policy was selected by 2026 outcomes. Fixed observed coverage was 0.881443, width 0.00941324 and interval score 0.01314386. Coverage is empirical and not guaranteed under nonstationarity. The online queue records 588 arrivals including six bridge labels, 582 scored test forecasts and six pending decisions; these counts refer to different cohorts.

## Reproduce

Use a fresh repository environment. `python train.py selected --output models/my-forecast` reproduces chronological validation and saves the final mean model and interval calibration without loading 2026 data. Outputs must be fresh. `python evaluation_scripts/evaluate.py --checkpoint-directory models/my-forecast --report-directory models/my-observed-report --predictions data/interim/my-observed-predictions.csv --acknowledge-observed` explicitly evaluates the previously observed cohort. Forecasting uses `python -m src.models.uncertainty forecast --output models/my-forecast --panel YOUR_COMPLETE_PANEL.csv --predictions data/interim/new-forecasts.csv`; existing forecast files are rejected.

Checkpoint SHA256: `b5c263242f18a99307436df1545e5a46434deaf12fb8fd877c0e565a8289010c`. The observed test snapshot hash and exact metrics are recorded in [models/uncertainty-v2/results.json](models/uncertainty-v2/results.json). No weights, source rows, predictions or online states are distributed. Source terms are in [docs/datasets.md](docs/datasets.md).

The three executed analysis notebooks include readable causal feature/fit/calibration code, actual chronological tables, aggregate error/interval plots and saved-checkpoint mechanics. They never rescore an observed test. Standalone figures are in `docs/figures`. Source checks cover causal boundaries, delayed labels and fresh-output guards; notebook execution is recorded separately.
