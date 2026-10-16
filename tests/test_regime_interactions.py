import math

import pytest

from src.models.regime_interactions import BASE_FEATURES, SIGNALS, columns, design


def row(ratio=1.0):
    return {
        **{name: float(i + 1) for i, name in enumerate(BASE_FEATURES)},
        "volatility_ratio": ratio,
    }


def test_exact_31_feature_contract_and_causal_products():
    source = row(2.0)
    values = design([source], True)[0]
    assert len(columns(True)) == len(values) == 31
    assert values[:22] == [source[name] for name in BASE_FEATURES]
    assert values[22] == pytest.approx(math.log(2))
    assert values[23:] == pytest.approx(
        [source[name] * math.log(2) for name in SIGNALS]
    )
    assert design(
        [{**source, "target": "forbidden", "target_date": "2099-01-01"}], True
    ) == [values]


def test_fixed_gate_clipping_and_zero_short_volatility():
    for ratio in (0.0, 0.1, 0.5):
        assert design([row(ratio)], True)[0][22] == -math.log(2)
    for ratio in (2.0, 10.0):
        assert design([row(ratio)], True)[0][22] == math.log(2)
    assert design([row(1.0)], True)[0][22:] == [0.0] * 9
    assert len(design([row()], False)[0]) == 22


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 10**400])
def test_nonfinite_or_boolean_features_are_rejected_without_fitting(bad):
    source = row()
    source["momentum_3"] = bad
    with pytest.raises(ValueError, match="Finite"):
        design([source], True)


def test_negative_gate_or_nonboolean_policy_is_rejected():
    with pytest.raises(ValueError, match="Negative"):
        design([row(-1.0)], True)
    with pytest.raises(ValueError, match="flag"):
        design([row()], 1)
