# Advanced multi-asset study

This extension is a research system for reference observations, not a brokerage system. It adds EUR/CAD, GBP/CAD and AUD/CAD to the existing repository without replacing the earlier USD/CAD checkpoint or commands. All three assets, dates, model/risk candidates and selection rules were fixed in multi-asset-plan.json before new 2026 values were requested. The previous USD/CAD 2026 study was already observed and is explicitly excluded from new-test claims. The shared macroeconomic period is not wholly unobserved.

## Reproduce or inspect

```sh
python -m src.models.advanced search
python -m src.models.advanced test
python -m src.models.advanced report
python -m src.models.advanced forecast --panel data/external/multi-asset-v1/holdout/panel.csv
```

After package installation, `regime-advanced` exposes the same actions. Recorded selection and final reports refuse overwrites. For a replication, choose a fresh `--output` directory; this repeats an observed study and does not create new independent test evidence. Downloaded source bytes, CSVs, row forecasts, ledgers and model binaries remain ignored. Only aggregate reports, configuration and source provenance are committed. Trusted locally trained joblib files are required for checkpoint inference.

## Observation and publication contract

The [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) supplies daily [indicative reference exchange rates](https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/). They average aggregated quotes and are generally published by 16:30 Eastern time. They are not executable closes, bids or asks. Attribution is Bank of Canada, without endorsement; [source terms](https://www.bankofcanada.ca/terms/) apply.

All three selected series must have an observation on every returned date. The parser rejects partial rows, nonfinite/nonpositive levels, duplicate/unsorted dates and gaps greater than seven calendar days. It never fills, backfills or fabricates business dates. Holidays are represented by the next published shared observation, not an assumed weekday calendar. A missing common date shorter than seven days cannot be identified from an arbitrary caller CSV; callers must supply complete post-cutoff observations.

Features are considered known only after the date's reference publication. Peer returns, momentum and volatility are additionally lagged one common observation to avoid relying on uncertain within-publication cross-series ordering. This is a synchronous-release assumption, not reconstructed tick-level timing. Source snapshots are SHA256-pinned and immutable once cached; current API snapshots do not reproduce historical as-published vintages or later revisions.

## Causal models and delayed targets

Twenty-two features include multi-horizon momentum, rolling/EWM volatility, volatility ratios, downside risk, drawdown, price-relative z-scores, RSI, autocorrelation, skew, calendar gaps and lagged peer context. Undefined autocorrelation/skew for constant past windows is represented as zero; missing prices are never imputed. Recursive EWM features require their complete initialization history, so advanced checkpoints retain all prior feature-only levels rather than truncating to a misleading finite rolling window.

A decision after observation `t` cannot earn the reference move ending at `t` or be assumed filled at `t`'s earlier average. The forecast target is therefore `log(P[t+2]/P[t+1])`: queue a decision at `t`, hypothetical reference rebalance at the next shared observation `t+1`, and hold until `t+2`. This still does not make a future reference average executable. It is an explicitly delayed reference accounting diagnostic.

Each asset fits its own two-state Gaussian HMM and global/state-weighted XGBoost experts. HMM fitting uses only training data. Forward probabilities are advanced across all purged feature-only bridge rows. The expert gate uses the one-step state forecast `posterior[t] @ transition`, not a smoothed future state. State indices and conditional volatility forecasts are latent model outputs, not validated economic labels.

Validation windows are 2021–2022, 2023 and 2024–2025, with expanding earlier training. A training target must mature strictly before the first validation decision; two intervening feature rows advance the HMM without labels. Six prespecified model candidates compare two tree regularization settings and gate strengths 0/.25/.75. Selection uses pooled multi-asset validation RMSE; zero return is a fixed comparison. Pooled RMSE naturally weights more volatile assets more strongly and does not establish an optimal investment objective.

## Risk and exact reference accounting

Risk candidates compare 4%/8% annual volatility targets, 0/2-basis-point signal deadzones and cash. They use the selected model's validation forecasts only. Utility is mean fold annualized arithmetic reference return minus three times annualized variance, net of two basis points per dollar turnover. Cash remains an eligible policy; an inability to beat cash is preserved rather than forcing activity.

Sizing uses only returns observed through the decision, a 60-observation covariance estimate with 25% diagonal shrinkage, gross exposure at most one and per-asset absolute exposure at most .5. The engine queues target weights for the next observation. Prices then mark existing currency units to CAD; costs are charged on the absolute dollars traded, including entry, reversal and final liquidation. It solves post-cost NAV and desired exposures together:

`NAV_after + cost_rate * sum(abs(target_weight * NAV_after - old_value)) = NAV_before`.

This prevents transaction costs from silently increasing post-trade exposure beyond the stated weight caps. Cash plus marked positions equals NAV, and final NAV equals initial cash plus marked P&L minus all costs within explicit floating tolerance. Sub-nanocent negative cash roundoff is zeroed; material insolvency is rejected. Caps hold after rebalances, not continuously between observations. Short proceeds appear in accounting cash; margin, interest, collateral restrictions and financing are not modeled.

Final reports include zero/2/5/10-basis-point cost sensitivity, turnover, actual evaluation observation count separately from warm-up history, drawdown, volatility, zero-cash-rate Sharpe, tail-loss diagnostics, pre/post-trade exposure and the accounting residual. These are not profitability guarantees or executable return estimates.

## Label-arrival monitoring and evidence limits

Residual forecasts enter a queue with maturity two shared observations later. Only then is the realized reference return supplied to a volatility-standardized, two-sided Page–Hinkley-style diagnostic. Warm-up is 30 matured labels; delta .05 and threshold eight are fixed. Alerts never change models, thresholds or portfolio positions in the heldout study. They are not verified regime changes or proof of concept drift.

Tests cover source alignment, feature prefix invariance, exact delayed labels, bridge filtering/chunk equivalence, post-cost cash/exposure identities, initial entry/reversal/exit costs, delay, gap rejection, strictly delayed monitor updates and complete-context checkpoint inference. The initial fixture test exposed negative cash of roughly 1e-11 from floating arithmetic; the narrowly bounded roundoff correction and full-book identity assertions preserve that finding rather than hiding a substantive loss.

Remaining limitations include one partial year, one seed, correlated CAD exposures, historical revision uncertainty, market timing assumptions and limited candidate coverage. Validation-selected forecast or risk policies may underperform global models, zero returns or cash. No test score is used for further policy selection.

## Historical selection result

The executed six-candidate study took approximately 97 seconds. Depth-two/global XGBoost (gate strength zero) won among trained candidates, with pooled validation RMSE 0.00390006. The zero-return forecast still performed better. Cash won the prespecified validation risk utility comparison. These are negative conclusions for regime gating and active risk taking, not evidence that complexity creates forecasting skill.

Active candidates' real validation ledgers remain in the selection report, demonstrating sizing, costs and exposure accounting. The selected final cash policy must have zero turnover, cost and return at every stated cost level; that is abstention, not evidence of robustness or profitability. No gated or exposed policy is forced into the final study for appearance.
