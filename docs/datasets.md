# Source and terms

Rates come from the [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs), with [Bank of Canada terms and attribution](https://www.bankofcanada.ca/terms/). The study uses EUR/CAD (`FXEURCAD`), GBP/CAD (`FXGBPCAD`) and AUD/CAD (`FXAUDCAD`). No Bank of Canada endorsement is implied.

Mean training uses observations from 2017–2024; separate calibration uses 2025. The observed test requests 2026-01-01 through 2026-10-14 and contains 196 published observations per asset. Exact request URLs, retrieved timestamps, raw SHA256, retained dates and counts are saved per snapshot. Mean inputs use sorted-key JSON serialization so object-key ordering cannot change the pinned data identity. Keep downloaded bytes for repeatability: the live API can revise historical data and does not provide the original publication vintage used at each historical decision.

These are reference exchange rates, not executable bid/ask prices. Holidays have no observations; targets use the next observed publication, not calendar-day interpolation. Validation rejects duplicate/invalid timestamps, nonfinite/nonpositive rates and unexpectedly long gaps. Publication timing and missing-observation assumptions are tested in production code.

Training-only scaling, causal filtering and delayed targets are modeling safeguards, not data-license substitutes or guarantees of economic value. Raw source bytes, cleaned rates, predictions, ledgers, drift queues and model binaries remain ignored. Public files contain code, source hashes and aggregate evidence only.
