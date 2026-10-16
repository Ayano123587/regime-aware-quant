# Experiment ledger

| Study | Selection window/objective | Outcome |
|---|---|---|
| Regularized uncertainty v2 | 2021–24 purged chronological RMSE; zero eligible | Ridge alpha 10000; final mean through 2024, calibration 2025 |
| Shared / blend / recency | Same chronological cohort; >=0.1% gain and >=3/4 year wins | Retain v2; closest blend failed magnitude gate |
| Online intervals v3 | 2025 proper interval score before 2026 replay | Fixed beats rolling 60/120; no adaptive policy forced |

Structured plans are `docs/uncertainty-plan.json`, `docs/shared-forecast-plan.json`, `docs/shared-blend-plan.json`, `docs/shared-recency-plan.json` and `docs/online-interval-plan.json`. Exact candidates, source hashes, runtimes and failures remain in corresponding aggregate model directories. Shared studies used132 ridge fits and approximately 6.8 seconds total scientific CPU wall time; no numerical failures. Previously observed 2026 outcomes were not used for followup selection. State guards were added after the online study; the source/executed-hash distinction remains in its attestation. Raw data, forecasts, ledgers, state histories and weights stay ignored.
