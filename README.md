# Regime-aware quantitative forecasting

A reproducible research pipeline that combines a **Gaussian hidden Markov model** with
three **XGBoost experts**. Each expert learns with a different posterior sample weighting;
causal regime probabilities blend their next-day return forecasts. A global XGBoost model
and zero-return forecast provide fixed comparisons.

```sh
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
CSVs stay local. For inference, pass a price CSV containing sufficient rolling history and
new rows after the model's training cutoff:

```sh
python -m src.models.predict_model --data data/raw/new-prices.csv
```

Load only model files you trained or trust. `make predict` with the original training CSV
can predict its final unlabeled close, if it is after the stored model cutoff.

The latent states have arbitrary identities; they are not certified economic labels such as
"trending" or "mean reverting." Daily reference rates are not executable bid/ask quotes.
Reported equity is a hypothetical reference-price diagnostic with a stated turnover
cost; it omits financing, market impact and publication/execution delay. Out-of-sample scores may underperform the
global model or a zero forecast. No profitability or investment recommendation is claimed.

Repository layout follows `src/{data,features,models,visualization}`, `tests`, `docs`,
`notebooks` (EDA only), `.github`, `pyproject.toml`, `requirements.txt`, and `Makefile`.
All data directories and `.env` are ignored. MIT licensed original implementation.
