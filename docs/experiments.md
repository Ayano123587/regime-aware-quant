# Validation selection

Chronological 2021-24 folds compare train-scaled ridge regression with zero return; zero remains eligible. Ridge alpha 10,000 has validation RMSE 0.003910994838, versus zero 0.003913798892. Mean models fit labels through 2024; 2025 labels calibrate intervals separately.

Fixed residual calibration wins 2025 proper interval score: 0.01601720, versus rolling-120 0.01603791 and rolling-60 0.01638900. No interval policy is selected by 2026 outcomes. The observed 2026 cohort is retrospective, with 582 matured targets and six pending decisions. Source snapshots, selected configuration and aggregate metrics are tracked; raw values, predictions, state histories and checkpoints stay ignored.
