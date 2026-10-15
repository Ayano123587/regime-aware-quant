# Dataset and methodology sources

- [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs): daily USD/CAD
  exchange-rate observations (`FXUSDCAD`). These are reference rates, not tradable quotes.
  [Terms and attribution](https://www.bankofcanada.ca/terms/) apply separately from the code license.
- Fixed download URL in `src/data/make_dataset.py` requests 2017-01-01 through 2025-12-31.
  SHA256 and exact first/last dates are recorded per download. Exact repeatability requires
  retaining the downloaded bytes, as historical data can be revised. Data stay local.
- FRED S&P 500 was investigated but its download timed out from this environment. No
  FRED dataset or result is claimed. The implemented default uses the verified Bank of Canada endpoint.
- [XGBoost Python reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html):
  squared-error regression and soft per-observation expert weights.
- [Rabiner's HMM tutorial](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf): forward,
  backward and Baum-Welch recursions. This repository implements diagonal emissions directly
  in NumPy/SciPy, rather than requiring a compiled HMM library.

Data validation rejects duplicate dates, nonfinite or nonpositive reference levels. Missing holidays
are dropped; predictions are for the next observed publication date, not the next calendar day.

## Experiment design

Hypothesis: regime-conditioned experts reduce out-of-sample RMSE versus a global XGBoost
model with the same feature set and hyperparameters. Primary comparison is the concatenated
chronological holdout RMSE. Direction accuracy and cost-sensitive equity are diagnostics.
The hypothesis fails if the regime mixture does not improve RMSE. Do not tune against the
reported holdouts; use a separate earlier development window before a fresh final evaluation.

Five close-available features feed each tree. HMM observations are return and rolling volatility.
HMM scaling and parameters are fit within each training fold. Smoothed responsibilities are
used only for unsupervised parameter estimation on training history; expert weights and
held-out probabilities are forward-filtered. There is no held-out state relabeling.

The execution assumption is intentionally simple. Real strategy research also needs adjusted
tradable prices, next-open or delayed execution, borrow constraints, slippage and fee sensitivity,
walk-forward selection and independent temporal replication.
