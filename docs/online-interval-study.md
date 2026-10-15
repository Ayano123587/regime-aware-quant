# Delayed online interval calibration: negative validation result

A fixed residual calibration outperformed the prespecified rolling alternatives on the declared validation objective. No online adaptation is adopted, and the v2 mean model, risk policy, and trading decisions remain unchanged. This is real Bank of Canada reference-rate research on already-observed years, not a fresh independent benchmark or an executable trading system.

## Chronological design

The plan was saved before fitting or comparing interval candidates. A diagnostic point specification was selected using only expanding 2021 and 2022 folds: zero and ridge alpha 100/1000/10000. Ridge alpha 10000 won. Training labels for this diagnostic model end in 2023; training-only standardization is inherited from the existing pipeline. Separate 2024 residuals initialize calibration. All three interval policies are then evaluated on 2025. Neither 2024 calibration labels nor 2025 validation labels select the diagnostic point specification. Training, initial calibration, and interval-policy validation therefore have separate label windows.

Policies are fixed calibration, or the most recent 60/120 matured absolute residuals normalized by the volatility scale saved at decision time. The interval half-width is the finite-sample 90% order statistic multiplied by the current causal EWM scale. Scales are clipped below at 1e-5. The pooled mean proper interval score is the selection criterion; coverage and width are reported alongside it. Ties prefer fixed, then 120, then 60. No candidate is chosen by proximity to 90% coverage alone.

For the final retrospective 2026 replay, the point model is the exact existing v2 checkpoint, trained through 2024. The initial residual calibration uses 2025 only. This is a policy transfer from an earlier diagnostic training vintage; it is not an architecture-controlled estimate. The fixed baseline's initial quantiles are asserted equal to the original v2 checkpoint. Only the validation-selected policy and fixed baseline may be replayed on 2026; because fixed won, those are the same procedure. Rolling 2026 outcomes were not computed.

## Delayed labels and state

A decision at observation t predicts log(P[t+2]/P[t+1]). The state queues the prediction and normalizing scale at t. After publication at t+2, the actual return becomes available and the saved residual enters the rolling reservoir before forming that observation's interval. It never uses the current residual before maturation, recomputes a past scale using later volatility, or replaces the saved point forecast with a newly fitted forecast. Two pending decisions from the end of the initialization period are retained, so the first two later publications do not silently lose their arrivals. The source contract requires complete, ordered shared observations.

State is plain JSON; no executable object is serialized by the online calibrator. The v2 point checkpoint remains a trusted local joblib artifact. Initial state loading and a midpoint save/load reproduce the actual full replay exactly. The final state reports 588 matured asset labels and six pending labels: matured arrivals include two pre-2026 decisions per asset, whereas the 582 scored 2026 decisions exclude the final two observations with unavailable future targets. State, individual residuals, forecasts and source data stay ignored.

## Actual validation results

All policies use the same 741 labeled 2025 asset forecasts and identical point predictions.

| Policy | Coverage | Mean full width | Mean proper interval score |
|---|---:|---:|---:|
| **Fixed (selected)** | 0.929825 | 0.01252474 | **0.01601720** |
| Rolling 60 | 0.904184 | 0.01241924 | 0.01638900 |
| Rolling 120 | 0.904184 | 0.01220322 | 0.01603791 |

Rolling 120 nearly ties fixed (score difference 0.00002070), but this does not reverse the prespecified rule. Its narrower intervals trade off against miss penalties. The result rejects the claim that these rolling windows improve overall interval quality on this validation cohort. Some asset-specific scores improve; choosing a separate post-hoc policy for those assets would be a new search and was not performed.

## Observed 2026 replay

The unchanged fixed procedure has pooled coverage 0.881443, mean width 0.00941324 and mean interval score 0.01314386 across 582 labeled forecasts. Point RMSE remains 0.00295666. Per-asset coverage is 0.881443 EUR/CAD,0.907216 GBP/CAD and 0.855670 AUD/CAD. Thus even the selected procedure does not achieve nominal 90% coverage universally. Matching against the original v2 forecast archive confirms identical targets, means, and interval widths within numerical tolerance.

All these historical observations and earlier scores have already been viewed. Selection is validation-only within this study, but retrospective development and serial dependence prevent an independent evidence or finite-sample coverage claim. Online rolling order statistics would likewise have no distribution-free guarantee under drift. Reference rates are generally known only after publication and are not executable quotes; this extension makes no brokerage or profitability claim.

## Reproduction and checks

From this repository's isolated environment:

```sh
python -m pytest -q
python -m src.models.online_intervals --output models/online-interval-reproduction
```

The cached real development/holdout snapshots and existing `models/uncertainty-v2/bundle.joblib` are required. A complete fresh setup first reproduces the v2 uncertainty study. Use a new output directory; completed selection/results refuse overwrite. The study command is a research replay, not a new live forecasting service. Plans, selection, aggregate metrics, source/checkpoint hashes, and verification are committed under `docs/online-interval-plan.json` and `models/online-interval-v3`. Individual outputs are written only under ignored `data/interim/online-interval-v3`. A subsequent output-path isolation fix makes fresh reproduction directories use matching separate ignored folders, preserving the original forecast/state archive; verification metadata distinguishes the run and current source hashes.

The actual CLI run took 57.96 seconds with native numeric threads limited to two, within the ten-minute CPU allowance. All 34 tests pass, including six new tests for delayed arrival, update-before-forecast ordering, saved decision scales, fixed-policy nonadaptation, prefix/future-label invariance, JSON continuation, invalid inputs, and report preservation. No paid compute or GPU was used.

This bounded hypothesis ends with a negative result. The near-tie of rolling 120 is not a reason to tune windows against the observed 2026 period. A future independent period and broader timestamped predictors would be stronger evidence than continuing local interval-window search.

## State API hardening after the scientific run

Independent review found that malformed caller-supplied state was not fully validated and that a failing update could partially mutate state. These issues did not affect the recorded real-data study. Subsequent guards validate exact schemas and asset sets, nonnegative finite residual histories and quantiles, positive prices and saved scales, integer counters, queue uniqueness/completeness, delayed due indices, and decision dates. Duplicate JSON keys are rejected. Dates normalize to UTC; invalid/NaT dates and publication gaps over seven calendar days are rejected. Shorter missing common dates remain a caller responsibility under the existing source contract.

Each update now validates and deep-copies state, processes matured labels on the copy, verifies finite residuals, widths and interval endpoints, and commits only after every check succeeds. Tests deliberately cause failures after arrivals have been processed and verify that the original state is unchanged. Malformed JSON, invalid gaps, overflow and saved-state continuation are covered.

These guards were added after the scientific run. The study was not replayed or rescored, and no policy or reported metric changed. All three archived initial/midpoint/final states validate successfully without replay. The expanded suite has 67 passing tests (39 online-interval tests plus 28 existing tests). Aggregate verification preserves the executed-source hash and the pre-guard source hash separately from the current audited-source hash. State archives and all row-level data remain ignored.
