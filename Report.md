# Regime-aware Quant: results overview

The retained v2 point forecast is a per-asset standardized ridge trained through 2024, with separate 2025 residual interval calibration. Mean-model improvements are selected chronologically; zero remains eligible.

| Procedure | Validation RMSE 2021–24 | Outcome |
|---|---:|---|
| Zero return | 0.00391379889 | Eligible baseline |
| V2 per-asset ridge | 0.00391099484 | Retained |
| Best normalized shared ridge | 0.00391120009 | Worse pooled validation |
| Fixed half v2 / half shared | 0.00391060119 | Gain 0.0101%; fails 0.1% gate |
| Recency per-asset three-year | 0.00391130996 | Worse pooled validation |

Observed 2026 v2 RMSE remains 0.00295666 versus zero 0.00295177 and earlier XGBoost 0.00298725. Zero still wins that retrospective period overall. The shared/blend/recency studies did not rescore 2026 or change risk/trading policy. They do not establish profit or independent prospective accuracy.

The separate online interval study also retained fixed calibration: 2025 proper interval score 0.01601720 versus rolling 120: 0.01603791 and rolling 60: 0.01638900. State-API guards were added after the scientific run and are documented separately.

- [Chronological v2 uncertainty study](docs/uncertainty-study.md)
- [Delayed online interval study](docs/online-interval-study.md)
- [Three shared/partial/recency hypotheses](docs/shared-forecast-study.md)
- [Reference-rate publication and execution assumptions](docs/multi-asset-study.md)

## Entry points

```sh
python train.py recency --output models/recency-reproduction
python evaluation_scripts/report.py models/recency-reproduction
python evaluation_scripts/evaluate.py --output models/recency-reproduction --observed-holdout
```

Training copies the declared plan into a fresh output directory and evaluates validation only. Separate evaluation requires explicit acknowledgement of the already-observed cohort, refuses historical report overwrite and does nothing when v2 was retained. The forecast command in the shared study verifies the retained checkpoint hash before loading it. Use one isolated environment per repository.

Implementation remains under `src/`, tests under `tests/`, and notebooks are EDA-only. Only aggregate protocols/results/hashes are committed; market snapshots, predictions, queue state, trained weights and secrets remain ignored. This reporting organization follows the supplied [project-structure reference](https://github.com/yash9439/Finetuning-Opus-MT-en-fr-WikiData), with original forecasting implementation and evidence.
