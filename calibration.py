"""Serializable PID gain bounds shared by calibration, training, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float

    def as_array(self) -> np.ndarray:
        return np.asarray((self.kp, self.ki, self.kd), dtype=np.float64)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "PIDGains":
        values = tuple(float(value) for value in values)
        if len(values) != 3:
            raise ValueError("PID gains must contain exactly kp, ki, kd")
        if any(value < 0 or not np.isfinite(value) for value in values):
            raise ValueError("PID gains must be finite and non-negative")
        return cls(*values)


@dataclass(frozen=True)
class GainCalibration:
    nominal: PIDGains
    robust: PIDGains
    lower: PIDGains
    upper: PIDGains

    def __post_init__(self) -> None:
        if np.any(self.lower.as_array() >= self.upper.as_array()):
            raise ValueError("each lower gain bound must be below its upper bound")
        for gains in (self.nominal, self.robust):
            values = gains.as_array()
            if np.any(values < self.lower.as_array()) or np.any(values > self.upper.as_array()):
                raise ValueError("nominal and robust gains must lie inside scheduler bounds")

    def map_action(self, action: np.ndarray) -> PIDGains:
        action = np.asarray(action, dtype=np.float64).reshape(3)
        unit = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        mapped = self.lower.as_array() + unit * (self.upper.as_array() - self.lower.as_array())
        return PIDGains.from_iterable(mapped)

    def action_for(self, gains: PIDGains) -> np.ndarray:
        unit = (gains.as_array() - self.lower.as_array()) / (
            self.upper.as_array() - self.lower.as_array()
        )
        return np.clip(2.0 * unit - 1.0, -1.0, 1.0).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GainCalibration":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{key: PIDGains(**data[key]) for key in ("nominal", "robust", "lower", "upper")})

    @classmethod
    def development_default(cls) -> "GainCalibration":
        """Smoke-test values only; final runs must use calibrate_pid.py output."""
        return cls(
            nominal=PIDGains(26.0, 0.5, 4.4),
            robust=PIDGains(24.0, 0.6, 4.8),
            lower=PIDGains(8.0, 0.0, 2.0),
            upper=PIDGains(55.0, 2.0, 8.0),
        )
