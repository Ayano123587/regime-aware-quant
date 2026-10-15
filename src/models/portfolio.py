"""Non-executable, self-financing delayed reference-level portfolio accounting."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from src.data.multi_asset import ASSETS, validate_panel


@dataclass(frozen=True)
class RiskPolicy:
    annual_volatility: float = .04
    deadzone_bps: float = 0.
    gross_cap: float = 1.
    per_asset_cap: float = .5

    def __post_init__(self):
        values = [self.annual_volatility, self.deadzone_bps, self.gross_cap, self.per_asset_cap]
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or not 0 < self.gross_cap <= 1 or not 0 < self.per_asset_cap <= 1:
            raise ValueError("Finite nonnegative risk parameters and caps in (0,1] required")


def target_weights(prediction, history, policy):
    prediction, history = np.asarray(prediction, float), np.asarray(history, float)
    if prediction.ndim != 1 or not prediction.size or history.ndim != 2 or history.shape[1] != len(prediction) or len(history) < 20 or not np.isfinite(history).all() or not np.isfinite(prediction).all():
        raise ValueError("Finite aligned predictions and >=20 past return observations required")
    if policy.annual_volatility == 0:
        return np.zeros_like(prediction)
    vol = np.maximum(history[-20:].std(axis=0, ddof=1), 1e-5)
    signal = np.clip(prediction / vol, -1, 1)
    signal[np.abs(prediction) <= policy.deadzone_bps / 10000] = 0
    covariance = np.atleast_2d(np.cov(history[-60:], rowvar=False))
    covariance = .75 * covariance + .25 * np.diag(np.diag(covariance))
    annual = np.sqrt(max(0., float(signal @ covariance @ signal))) * np.sqrt(252)
    weights = signal * (policy.annual_volatility / max(annual, 1e-8))
    weights = np.clip(weights, -policy.per_asset_cap, policy.per_asset_cap)
    weights *= min(1., policy.gross_cap / max(float(np.abs(weights).sum()), 1e-12))
    return weights


def rebalance(nav, old_values, weights, cost_bps):
    old_values, weights = np.asarray(old_values, float), np.asarray(weights, float)
    if nav <= 0 or not math.isfinite(nav) or old_values.shape != weights.shape or not np.isfinite(old_values).all() or not np.isfinite(weights).all() or not math.isfinite(cost_bps) or cost_bps < 0 or cost_bps >= 10000:
        raise ValueError("Invalid solvent reference-book inputs/cost")
    if np.abs(weights).sum() > 1 + 1e-10:
        raise ValueError("No borrowing: gross weight cannot exceed one")
    rate = cost_bps / 10000
    # Solve post-cost NAV = pre-cost NAV - turnover cost, with desired dollar
    # exposures defined against POST-cost NAV so costs cannot violate weight caps.
    lo, hi = 0., nav
    if rate * np.abs(old_values).sum() >= nav:
        raise ValueError("Liquidation cost exhausts reference book")
    for _ in range(70):
        middle = (lo + hi) / 2
        if middle + rate * np.abs(weights * middle - old_values).sum() > nav:
            hi = middle
        else:
            lo = middle
    after = (lo + hi) / 2
    new_values = weights * after
    traded = float(np.abs(new_values - old_values).sum())
    cost = rate * traded
    cash = float(after - new_values.sum())
    if cash < -1e-8 or not np.isfinite([after, traded, cost, cash]).all():
        raise ValueError("Invalid cash/accounting result")
    # Fully invested books can leave a negative sub-nanocent roundoff residual.
    # Zero it; the complete self-financing identity is still asserted below.
    return new_values, max(0., cash), cost, traded


def simulate(panel, signals, policy, cost_bps=2., initial_cash=100000., assets=ASSETS, start_date=None):
    panel = validate_panel(panel, assets)
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Positive finite initial cash required")
    if list(signals.columns) != ["date"] + list(assets) or signals.date.duplicated().any():
        raise ValueError("Signals need unique dates and exact ordered asset list")
    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals.date, utc=True)
    if signals.date.isna().any() or signals.date.duplicated().any():
        raise ValueError("Signal dates must remain unique and finite after UTC normalization")
    if not set(signals.date) <= set(panel.date) or not np.isfinite(signals[list(assets)].to_numpy()).all():
        raise ValueError("Signals must align to observed panel dates with finite values")
    signal_map = signals.set_index("date")
    start_date = panel.date.iloc[0] if start_date is None else pd.Timestamp(start_date)
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")
    if start_date < panel.date.iloc[0] or start_date > panel.date.iloc[-1]:
        raise ValueError("Evaluation start must be inside supplied observation history")
    prices = panel[list(assets)].to_numpy(float)
    returns = np.diff(np.log(prices), axis=0)
    units, cash = np.zeros(len(assets)), float(initial_cash)
    queued, decision_date = np.zeros(len(assets)), None
    rows = []
    for j, date in enumerate(panel.date):
        if date < start_date:
            continue
        old = units * prices[j]
        nav = float(cash + old.sum())
        pnl = float(units @ (prices[j] - prices[j - 1])) if j else 0.
        pre_gross = float(np.abs(old).sum() / nav) if nav > 0 else float("inf")
        weights = np.zeros(len(assets)) if j == len(panel) - 1 else queued
        new, cash, cost, traded = rebalance(nav, old, weights, cost_bps)
        units = new / prices[j]
        after = float(cash + new.sum())
        rows.append({"date": date, "decision_date": decision_date, "nav_before": nav, "nav": after,
                     "cash": cash, "pnl": pnl, "cost": cost, "traded_notional": traded,
                     "turnover": traded / nav, "gross_exposure": float(np.abs(new).sum() / after),
                     "max_asset_exposure": float(np.abs(new).max() / after), "pretrade_gross": pre_gross,
                     **{asset + "_weight": float(new[k] / after) for k, asset in enumerate(assets)}})
        # Queue decisions AFTER the reference observation is known. They execute
        # only at the NEXT shared observation; no current decision earns prior P&L.
        if date in signal_map.index and j >= 20 and j < len(panel) - 2:
            queued = target_weights(signal_map.loc[date, list(assets)].to_numpy(float), returns[:j], policy)
            decision_date = date
        else:
            queued, decision_date = np.zeros(len(assets)), None
    ledger = pd.DataFrame(rows)
    nav = ledger.nav.to_numpy()
    changes = np.diff(np.r_[initial_cash, nav]) / np.r_[initial_cash, nav[:-1]]
    volatility = float(changes.std(ddof=1) * np.sqrt(252)) if len(changes) > 1 else 0.
    tail = changes[changes <= np.quantile(changes, .05)]
    report = {"reference_only": True, "execution_delay_observations": 1, "final_liquidation": True,
              "initial_cash": initial_cash, "final_nav": float(nav[-1]), "total_return": float(nav[-1] / initial_cash - 1),
              "annualized_arithmetic_return": float(changes.mean() * 252), "annualized_volatility": volatility,
              "sharpe_zero_cash_rate": float(changes.mean() / changes.std(ddof=1) * np.sqrt(252)) if len(changes) > 1 and changes.std(ddof=1) > 1e-12 else None,
              "max_drawdown": float((nav / np.maximum.accumulate(np.r_[initial_cash, nav])[1:] - 1).min()),
              "var_95_loss": max(0., -float(np.quantile(changes, .05))),
              "expected_shortfall_95_loss": max(0., -float(tail.mean())),
              "total_cost": float(ledger.cost.sum()), "cost_bps_per_turnover": cost_bps,
              "total_turnover": float(ledger.turnover.sum()), "total_marked_pnl": float(ledger.pnl.sum()),
              "max_posttrade_gross": float(ledger.gross_exposure.max()), "max_posttrade_asset": float(ledger.max_asset_exposure.max()),
              "max_pretrade_gross": float(ledger.pretrade_gross.max()),
              "accounting_residual": float(nav[-1] - initial_cash - ledger.pnl.sum() + ledger.cost.sum()),
              "liquidated_cash": float(ledger.cash.iloc[-1]), "observations": len(ledger), "history_observations": len(panel),
              "limits": "Caps hold after each rebalance, not between observations. Reference levels are non-executable; financing/spread/impact omitted."}
    if abs(report["accounting_residual"]) > max(1e-6, initial_cash * 1e-10):
        raise AssertionError("Self-financing identity failed")
    return ledger, report
