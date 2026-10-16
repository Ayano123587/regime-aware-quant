# Regime-aware quantitative forecasting

A reproducible research pipeline that combines a **Gaussian hidden Markov model** with
one **XGBoost expert per latent state** (three by default). Each expert learns with different posterior sample weights;
causal regime probabilities blend their next-observation reference-return forecasts. A global XGBoost model
and zero-return forecast provide fixed comparisons.

## Advanced multi-asset extension

The latest [regularized forecast and uncertainty study](docs/uncertainty-study.md)
reduces retrospective point RMSE to 0.00295666 from 0.00298725, while zero return remains
slightly better at 0.00295177. It adds chronological residual calibration, causal
volatility-scaled intervals, quarterly error/coverage diagnostics and date-block uncertainty
comparisons. A validation-only wider regularization search confirmed the selected setting.
These observed-period results do not establish profitability or guaranteed 90% coverage.

A further [delayed-label calibration comparison](docs/online-interval-study.md) tested
rolling 60/120-observation residual windows against fixed bands. Fixed calibration won
the validation interval score and remains selected. The new JSON state API updates only
matured labels, validates saved state and applies atomic updates; the full suite now
passes 67 tests. This negative experiment leaves the mean model and trading policy intact.

The existing project now also contains a prespecified **EUR/CAD, GBP/CAD and AUD/CAD** study with 22 causal features, asset-specific HMM execution-regime forecasts, validation-selected XGBoost gates, volatility-targeted sizing and a self-financing reference ledger. Decisions are queued for one observation before hypothetical execution; the trained target is the subsequent reference return. Post-cost gross/per-asset caps, cash, entry/reversal/liquidation fees, cost sensitivity and delayed-label drift diagnostics are tested explicitly.

```sh
python -m src.models.advanced search
python -m src.models.advanced test
python -m src.models.advanced report
python -m src.models.advanced forecast --panel data/external/multi-asset-v1/holdout/panel.csv
```

See [the frozen plan](docs/multi-asset-plan.json) and [advanced methodology](docs/multi-asset-study.md). Historical development ends in 2025; new 2026 EUR/GBP/AUD values are requested only after model and risk selection. USD/CAD 2026 was already observed and is excluded from new-test claims. The old checkpoints and commands below remain valid. This is still **non-executable reference-price research**, with no brokerage integration or trading actions.

**Historical validation rejected extra complexity:** global XGBoost won among trained candidates but still trailed the zero forecast; cash won risk selection. The project preserves that result rather than forcing regime gating or trades. Active candidates' validation diagnostics remain available; final cash cost sensitivity is zero by construction, not proof of profitability.

The once-released new three-asset test also favored zero forecasts: pooled RMSE **0.00298725 selected/global versus 0.00295177 zero**, across 582 labeled forecasts. Cash remained unchanged. [Executed results and actual checkpoint verification](docs/multi-asset-results.md) document the negative conclusion, source hashes, exact CLI reproduction and 25 passing tests.

```sh
python -m venv .venv
# Windows PowerShell: activate this repository's environment
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m src.data.make_dataset
python -m src.models.train_model
python -m src.visualization.visualize
python -m pytest -q
```

The data command fetches a fixed historical range of daily USD/CAD reference exchange rates
from the Bank of Canada's Valet API. Original bytes, SHA256, source URL and cleaned CSV stay
under ignored `data/raw/`. The repository distributes code and evaluation summaries.
See [dataset provenance](docs/datasets.md). Other chronological series can be imported as
`date,close` CSVs; `close` denotes the observed daily reference level for this FX experiment.

Training uses expanding chronological folds. Each fold fits its own scaler, HMM and experts
using only training history. Training targets must end strictly before evaluation begins.
Held-out states come from a forward filter; future observations never revise past predictions.
Tests explicitly check prefix invariance and chunked filtering equivalence.

`models/run/evaluation.json` records measured results. `models/run/bundle.joblib` and prediction
CSVs stay local. New checkpoints retain 64 raw price context rows. For inference, supply all
new observations since the cutoff; overlapping prices must agree with the saved context:

```sh
python -m src.models.predict_model --data data/raw/new-prices.csv
```

Load only model files you trained or trust. `make predict` with the original training CSV
can predict its final unlabeled close, if it is after the stored model cutoff.

The latent states have arbitrary identities; they are not certified economic labels such as
"trending" or "mean reverting." Daily reference rates are not executable bid/ask quotes.
Reported equity is a hypothetical reference-price diagnostic with a stated turnover
cost; it omits financing, market impact and publication/execution delay. Daily averages are generally published by 16:30 ET, so same-day
average levels cannot be treated as executable closes. Out-of-sample scores may underperform the
global model or a zero forecast. No profitability or investment recommendation is claimed.

Repository layout follows `src/{data,features,models,visualization}`, `tests`, `docs`,
`notebooks` (EDA only), `.github`, `pyproject.toml`, `requirements.txt`, and `Makefile`.
All data directories and `.env` are ignored. MIT licensed original implementation.

## Validation-selected study and new 2026 evidence

A prespecified search over 24 configurations selected two HMM states, a blend of 25% regime experts and 75% global model, and stronger tree regularization. The new 2026 observations were fetched after selection was frozen. Across 195 labeled forecasts, RMSE was 0.0025302463, versus 0.0025309643 for the matched global model and 0.0025475597 for a zero forecast. The small nominal gain is not statistically clear: paired bootstrap intervals include zero. See [evaluation](docs/evaluation.md).

The frozen [development plan](docs/development-plan.md) and `models/development/selection.json` distinguish retrospective development on 2017–2025 data from the new 2026 evaluation. The test ends on October 14 because of publication availability at planning time. The final runner refuses to overwrite `models/development/evaluation.json`. Binary models and row predictions remain local. After reproducing the study, use its selected checkpoint with:

```sh
python -m src.models.predict_model --data data/external/prices-2026.csv --model models/development/bundle.joblib
```

The input may contain just the new observations since the cutoff because the checkpoint preserves rolling-price context. Supply all intervening observations; revisions to overlapping history or long missing gaps require an explicit refit/history repair. Bridge observations advance the state filter without appearing in output unless supplied by the caller.

Use a separate virtual environment for each repository. These research projects use the requested generic `src` package layout; installing several together in one environment is unsupported. On Linux/macOS, activate with `source .venv/bin/activate` instead of the PowerShell command. All `python -m ...` and installed CLI commands assume the current repository directory and its activated environment. On Windows, direct `.venv\Scripts\python.exe -m ...` commands can also run without activation.

To replicate the prespecified development study without overwriting its recorded evidence, copy `models/development/plan.json` into a fresh output directory and run `python -m src.models.development search --output NEW_DIRECTORY`, then `python -m src.models.development test --output NEW_DIRECTORY`. The historical source checksum must match the plan; API revisions require an explicit new data/version decision. A reproduction of the now-observed 2026 test is a replication, not new independent evidence. Do not use that test to select additional settings.

See [Report.md](Report.md) for the consolidated baseline/candidate table and [the shared/recency study](docs/shared-forecast-study.md) for all negative comparisons. Root `train.py` runs validation-only experiments into fresh directories; `evaluation_scripts/` provides aggregate reporting and explicit observed-cohort evaluation without silently rescoring retained baselines.
