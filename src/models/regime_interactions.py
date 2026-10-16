"""Bounded causal volatility interactions with fold-trained ridge scaling."""

import math

BASE_FEATURES = [
    "return_1",
    "momentum_3",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_60",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "ewma_volatility",
    "volatility_ratio",
    "downside_20",
    "drawdown_60",
    "zscore_20",
    "rsi_14",
    "autocorrelation_20",
    "skew_20",
    "gap_days",
    "peer_return_lag1",
    "peer_momentum_lag1",
    "peer_volatility_lag1",
    "relative_momentum",
]
SIGNALS = [
    "return_1",
    "momentum_3",
    "momentum_20",
    "zscore_20",
    "rsi_14",
    "autocorrelation_20",
    "peer_return_lag1",
    "relative_momentum",
]
ALPHAS = [1000, 10000, 30000]


def number(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Finite numeric feature required")
    return float(value)


def columns(interactions):
    if type(interactions) is not bool:
        raise ValueError("Explicit interaction flag required")
    return BASE_FEATURES + (
        ["bounded_volatility_gate"] + [s + "_x_gate" for s in SIGNALS]
        if interactions
        else []
    )


def design(records, interactions):
    """Only decision-time features enter this transform; no target is accessed."""
    columns(interactions)
    values = []
    for record in records:
        row = [number(record[name]) for name in BASE_FEATURES]
        if interactions:
            ratio = number(record["volatility_ratio"])
            if ratio < 0:
                raise ValueError("Negative volatility ratio")
            # Clipping before log includes the limiting case of zero short volatility.
            gate = math.log(min(2.0, max(0.5, ratio)))
            row += [gate] + [number(record[name]) * gate for name in SIGNALS]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("Nonfinite interaction product")
        values.append(row)
    return values


def fit(records, alpha, interactions):
    """Heavy dependencies are imported only in the cloud fitting path."""
    if type(alpha) not in (int, float) or alpha not in ALPHAS or not records:
        raise ValueError("Declared ridge alpha and nonempty matured rows required")
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    targets = [number(record["target"]) for record in records]
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=False)).fit(
        design(records, interactions), targets
    )


def predict(model, records, interactions):
    return [
        number(float(value)) for value in model.predict(design(records, interactions))
    ]
