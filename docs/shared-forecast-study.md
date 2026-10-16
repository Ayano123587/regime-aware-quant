# Shared, reduced and recency-weighted forecasts: v2 retained

Three preregistered hypotheses were evaluated on real cached Bank of Canada FX observations. None justified replacing the existing v2 mean procedure. Zero return remained eligible in every comparison. No new 2026 accuracy or interval scoring was performed, and no risk/trading policy changed.

## Chronological protocol

Plans were committed before each stage: `c990884` (dimension reduction/shared normalization), `a2b9aca` (fixed partial pooling), and `42541d8` (recency weights). Expanding 2021–2024 annual validation folds preserve the delayed t+1 to t+2 target. Training labels mature strictly before the first validation decision; scalers and regressions are fitted only on that fold's training rows. Assets are stacked only within the permitted training date range. Shared CAD exposure means stacked rows are correlated, not independent new market histories.

The reduced design uses eight predefined features: current return, 3/20-observation momentum, z-score, RSI, autocorrelation, lagged peer return and volatility ratio. Normalized models divide returns by current causal EWM volatility and momenta by that volatility times square-root horizon. Training targets are divided by the same decision-time scale, and forecasts are multiplied by the scale known at their decision. No future volatility or target is an inference feature. The floor is 1e-5. Sharing changes the effective ridge penalty/sample count; these are complete-procedure comparisons rather than isolated architecture effects.

Zero and v2 always remain eligible. A new procedure needs at least 0.1% lower pooled validation RMSE than the better baseline and improvements in at least three of four years. Eligible procedures are ranked by pooled RMSE. The gate was never relaxed after observing small gains.

## Stage 1: fewer features and shared normalization

| Procedure | Pooled validation RMSE |
|---|---:|
| Zero return | 0.00391379889 |
| Existing v2 per-asset full ridge | **0.00391099484** |
| Reduced per-asset ridge | 0.00391247743 |
| Reduced pooled raw ridge | 0.00391194704 |
| Pooled normalized alpha 1000 | 0.00391201904 |
| Pooled normalized alpha 10000 | 0.00391120009 |
| Pooled normalized alpha 30000 | 0.00391207096 |

The closest shared procedure improved 2023/24 but worsened 2021/22. All new full-period scores were worse.

## Stage 2: fixed partial pooling

The annual tradeoff motivated two fixed 50/50 blends without weight tuning. V2 plus normalized pooled ridge scored 0.00391060119 and beat v2 in three years, but the relative gain was only about 0.0101%, below the declared 0.1% threshold. V2 plus reduced per-asset ridge scored 0.00391127631. V2 was retained. A small nominal gain was not recast as robust improvement.

## Stage 3: temporal recency

Weights decay from the latest training target publication only, using fixed one-/three-year half-lives. The same mean-normalized weights fit both StandardScaler and ridge. Validation/test dates and labels never determine a training weight. Mean normalization keeps total penalty scale comparable, although effective sample size decreases.

| Procedure | Pooled validation RMSE |
|---|---:|
| V2 reference | **0.00391099484** |
| Full per-asset, one-year half-life | 0.00391180525 |
| Full per-asset, three-year half-life | 0.00391130996 |
| Shared normalized, three-year half-life | 0.00391181075 |

None improved pooled RMSE. The retained v2 and zero baselines replicate across all stages. The stages performed 132 ridge fits across chronological folds (including predefined blend components; zero requires no fitted mean). No fit failed; measured study wall times were 1.47, 3.65, and 1.70 seconds on CPU with two numeric threads. These small linear fits do not need GPU training. Complete candidates, annual metrics, source hashes, plans, failures and source receipts remain in the three `models/shared-*-v4` directories.

## Artifacts, forecast command and checks

The existing v2 checkpoint is retained unchanged. A new forecast command verifies its recorded SHA256 before loading the trusted local artifact, so rejecting a new model still leaves a working inference path:

```sh
python -m src.models.shared_study forecast --output models/shared-recency-v4 --panel data/external/multi-asset-v1/holdout/panel.csv --predictions data/interim/shared-recency-v4/retained-cli.csv
```

For fresh reproductions, use distinct output directories and the corresponding plan:

```sh
python -m src.models.shared_study study --output models/shared-reproduction --plan docs/shared-forecast-plan.json
python -m src.models.shared_study study --output models/blend-reproduction --plan docs/shared-blend-plan.json
python -m src.models.shared_study study --output models/recency-reproduction --plan docs/shared-recency-plan.json
python -m pytest -q
```

Run inside this repository's isolated environment. Completed selections refuse overwrite. Binary artifacts, individual validation/forecast rows, and snapshots remain ignored. Only aggregate reports/configuration/hashes are committed. The actual retained-checkpoint forecast command is an inference-mechanics replay; it does not recompute or select on test accuracy. Tests cover train-only/weighted scaling, exact fixed averaging, zero eligibility and confirmation gates, target exclusion, nonfinite inputs, prefix invariance, checkpoint persistence, revision rejection, and retained-checkpoint hash verification.

Source evolved between stages to add predefined blends and recency weights; historical executed-source hashes are preserved in each selection. The final inference-alias addition is separate from scientific scoring, and final audited-source hashes are recorded in verification metadata. Existing procedures are mathematically unchanged.

All years, earlier scores and 2026 observations have been viewed previously. This is retrospective development, without independent generalization or profit claims. The previously reported v2 observed-period RMSE still trails zero; none of these validation exercises overturns that result. More local parameter searching on these return features has weak support. Richer correctly timestamped external predictors or a genuinely future period would address a different evidence gap; repeated score fishing would not.

The final suite passed 75 tests. Actual retained-model CLI replay emitted 588 finite forecasts/bands and matched 582 earlier scored point predictions within 1.1e-16. This check did not compute accuracy or interval metrics. The six extra forecasts have no matured target in the snapshot. Aggregate verification preserves the checkpoint and ignored forecast-archive hashes.
