"""A delayed-label diagnostic, never a test-adaptive model or trading rule."""
from dataclasses import dataclass
import math


@dataclass
class ResidualMonitor:
    warmup: int = 30
    delta: float = .05
    threshold: float = 8.
    count: int = 0
    mean: float = 0.
    cumulative: float = 0.
    minimum: float = 0.
    maximum: float = 0.
    alerts: int = 0

    def observe(self, prediction, actual, forecast_volatility):
        if not all(math.isfinite(v) for v in [prediction, actual, forecast_volatility]) or forecast_volatility < 0:
            raise ValueError("Finite matured residual and nonnegative forecast volatility required")
        residual = (actual - prediction) / max(forecast_volatility, 1e-5)
        old_mean = self.mean
        self.count += 1
        self.mean += (residual - self.mean) / self.count
        alert = False
        if self.count > self.warmup:
            self.cumulative += residual - old_mean
            self.minimum = min(self.minimum + self.delta, self.cumulative)
            self.maximum = max(self.maximum - self.delta, self.cumulative)
            alert = max(self.cumulative - self.minimum, self.maximum - self.cumulative) > self.threshold
            if alert:
                self.alerts += 1
                self.cumulative = self.minimum = self.maximum = 0.
        return {"matured_labels": self.count, "standardized_residual": residual, "alert": alert}


class DelayedResidualQueue:
    def __init__(self, assets):
        self.monitors = {asset: ResidualMonitor() for asset in assets}
        self.pending = {}
        self.last_observation = -1

    def enqueue(self, observation_index, asset, prediction, volatility):
        if asset not in self.monitors or observation_index < self.last_observation:
            raise ValueError("Unknown asset or backdated prediction")
        if not all(math.isfinite(v) for v in [prediction, volatility]) or volatility < 0:
            raise ValueError("Invalid queued forecast")
        key = (observation_index + 2, asset)
        if key in self.pending:
            raise ValueError("Duplicate delayed forecast")
        self.pending[key] = (float(prediction), float(volatility))

    def arrive(self, observation_index, realized_return_by_asset):
        if observation_index <= self.last_observation:
            raise ValueError("Observations must arrive strictly once in order")
        due = [(key, values) for key, values in self.pending.items() if key[0] == observation_index]
        if any(key[0] < observation_index for key in self.pending):
            raise ValueError("Skipped label-arrival observation")
        for (_, asset), _ in due:
            if asset not in realized_return_by_asset or not math.isfinite(realized_return_by_asset[asset]):
                raise ValueError("A matured label is missing/nonfinite")
        output = []
        for key, (prediction, volatility) in due:
            asset = key[1]
            output.append({"asset": asset, "arrival_index": observation_index,
                           **self.monitors[asset].observe(prediction, realized_return_by_asset[asset], volatility)})
            del self.pending[key]
        self.last_observation = observation_index
        return output
