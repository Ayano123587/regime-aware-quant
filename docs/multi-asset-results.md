# Executed advanced study: no demonstrated forecast or trading advantage

The frozen six-model development search on 2017–2025 selected depth-two global XGBoost, with regime-gating strength zero. Its pooled validation RMSE was 0.00390006 versus zero forecast 0.00386198. All four active risk candidates had negative mean validation utility; cash won with utility zero. The active candidates' full validation risk diagnostics are preserved in [selection.json](../models/multi-asset-v1/selection.json).

Only after selection commit `913da33` were the fixed EUR/CAD, GBP/CAD and AUD/CAD 2026 observations requested. The source snapshot has 196 common dates, January 2 through October 14, and SHA256 `c64e0805e00a265135f122bf1df9c74091918e2f8a3567e821cbf3c96935b7e4`. There are 194 fully matured delayed forecasts per asset: decisions January 2–October 9, with labels through October 14. The shared 2026 macroeconomic period is not wholly unseen because the earlier USD/CAD experiment was already observed; only these fixed asset-value snapshots were newly retrieved after selection.

| Asset | Labeled forecasts | Selected/global RMSE | Zero forecast RMSE |
| --- | --- | --- | --- |
| EUR/CAD | 194 | 0.0024326633 | 0.0023999797 |
| GBP/CAD | 194 | 0.0026967296 | 0.0026834217 |
| AUD/CAD | 194 | 0.0036852070 | 0.0036301725 |
| Pooled | 582 | 0.0029872480 | 0.0029517682 |

Zero forecasts beat the trained selected model on every asset. No test-based retuning, alternative strategy selection or profitable-regime claim follows. The final report is [evaluation.json](../models/multi-asset-v1/evaluation.json), with the frozen plan/selection hashes and full environment/source provenance.

The validation-selected cash policy stays at initial CAD reference NAV 100,000 over all 196 evaluation observations. Cost, turnover, return, exposure and drawdown are zero at every prespecified cost level (0/2/5/10 basis points). This is abstention, **not transaction-cost robustness or profitability**. Active validation policies did exercise the risk engine: fold turnover ranged from about 135 to 250 times NAV, with gross/per-asset caps enforced after costs. Their negative net utility is preserved.

The strictly delayed residual diagnostics processed 194 matured labels per asset, with three/two/three alarms for EUR/GBP/AUD and no pending labels. Alarms did not change forecasts or positions, and are not verified market changes.

## Actual checkpoint and independent checks

The trained advanced checkpoint preserves all 2,244 historical feature-only price rows to replay recursive EWM features exactly. The actual forecast CLI reproduced **all 582 saved labeled predictions exactly**, including global forecasts, regime probabilities and conditional state-volatility output (maximum absolute difference zero). It also returned six additional latest forecasts whose labels are not yet available within the snapshot. Actual saved-checkpoint prefix invariance was checked without refitting. See [checkpoint-verification.json](../models/multi-asset-v1/checkpoint-verification.json).

An independent root audit recalculated pooled RMSE from the ignored row-level CSV: 0.002987248013278064 versus zero 0.00295176821262805, matching the aggregate report within 1e-14. It independently checked the 196-row cash ledger: constant NAV 100,000 and zero costs/traded notional.

The complete existing-plus-extension suite passes **25 tests**. Historical model/risk selection took about 97 seconds on CPU; final fitting, checkpoint inference and exact accounting checks completed locally within the authorized first-pass budget. No GPU or paid compute was needed. Data, model binaries, row predictions and ledgers remain ignored; only aggregate reports/code/docs are committed. Historical single-asset checkpoints and commands remain supported.
