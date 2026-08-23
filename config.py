"""Single source of truth for the final experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    physics_hz: int = 500
    policy_hz: int = 50
    history_length: int = 10
    target_speeds: tuple[float, ...] = (0.3, 0.5, 0.7)
    max_time_factor: float = 2.0
    max_time_margin_s: float = 5.0
    corridor_m: float = 1.0
    runaway_m: float = 20.0
    derivative_filter_tau_s: float = 0.01
    actuator_limit: float = 1.0
    delay_severity_s: float = 0.15
    noise_severity_m: float = 0.0003
    transient_start_s: float = 6.0
    final_training_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    development_seed: int = 11

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def frame_skip(self) -> int:
        ratio = self.physics_hz / self.policy_hz
        if not ratio.is_integer():
            raise ValueError("physics_hz must be divisible by policy_hz")
        return int(ratio)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIG = ExperimentConfig()

NOMINAL_MASS_KG = 10.0
NOMINAL_FRICTION = 1.0
NOMINAL_ACTUATOR = 1.0
SPEED_PID_GAINS = (3.0, 1.0, 0.0)
E_CT_SCALE_M = 1.0
E_CT_RATE_SCALE_MPS = 2.0
INTEGRAL_SCALE_MS = 1.0
SPEED_SCALE_MPS = 1.5
YAW_RATE_SCALE_RADPS = 10.0
PROGRESS_WEIGHT = 5.0
TRAIN_DELAY_RANGE_S = (0.0, 0.15)
TRAIN_NOISE_RANGE_M = (0.0, 0.0004)

# Disturbance-training sampler. The training grid mirrors the four evaluation
# conditions so a disturbed policy trains on the operating points it is scored
# on, with randomized severity and event timing. The previous sampler drew
# delay uniformly and then zeroed it whenever it added an event, so the mean
# sampled delay was 0.037 s against a 0.15 s evaluation severity.
DISTURBANCE_CONDITIONS = ("nominal", "delay", "noise", "combined")
DISTURBANCE_CONDITION_WEIGHTS = (0.25, 0.25, 0.15, 0.35)
DISTURBANCE_TRANSIENT_PROB = 0.5
DISTURBANCE_DECLARED_SEVERITY_PROB = 0.5
EVENT_START_MIN_S = 1.0
EVENT_START_IDEAL_FRACTION = 0.8

# A transient episode may change its disturbance more than once, and a change
# may return the channel to zero. A single one-way step taught the policies
# that a disturbance, once seen, is permanent. Evaluation still uses the single
# declared step, so training covers a superset of the scored condition.
DISTURBANCE_CHANGE_COUNT_WEIGHTS = (0.50, 0.30, 0.20)
DISTURBANCE_RECOVERY_PROB = 0.35


PATH_SPLITS: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        {"kind": "arc"},
        {"kind": "scurve"},
        {"kind": "uturn"},
        {"kind": "slalom"},
        {"kind": "boustrophedon"},
        {"kind": "mixed"},
    ),
    "validation": (
        {"kind": "sinusoid"},
        {"kind": "lane_change"},
        {"kind": "random_curvature", "params": {"seed": 5_000, "length": 8.0, "knots": 8, "curvature_max": 1.5, "curvature_rate_max": 2.5}},
        {"kind": "random_curvature", "params": {"seed": 5_001, "length": 8.0, "knots": 8, "curvature_max": 1.5, "curvature_rate_max": 2.5}},
    ),
    "held_out": (
        *(
            {
                "kind": "random_curvature",
                "params": {"seed": seed, "length": 8.0, "knots": 8, "curvature_max": 1.5, "curvature_rate_max": 2.5},
            }
            for seed in (*range(10_000, 10_012), *range(10_013, 10_021))
        ),
    ),
    "structural_ood": (
        {"kind": "figure8"},
        {"kind": "hairpin"},
        {"kind": "square"},
        {"kind": "zigzag60"},
    ),
    "stress": (
        {"kind": "spiral"},
        {"kind": "zigzag"},
        {"kind": "zigzag30"},
        {"kind": "needle"},
    ),
}


CONTROLLERS = (
    "pid_global_nominal",
    "pid_global_robust",
    "ppo_scheduler_nominal",
    "ppo_scheduler_disturbed",
    "ppo_direct_nominal",
    "ppo_direct_disturbed",
)
