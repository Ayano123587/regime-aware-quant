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
