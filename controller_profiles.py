"""Proven PPO contracts for the two learned steering controllers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


SCHEDULER_FRAME_FIELDS = (
    "e_ct",
    "e_ct_rate",
    "e_ct_integral",
    "e_theta",
    "speed",
    "yaw_rate",
    "omega_requested",
    "omega_applied",
    "v_applied",
    "previous_action_kp",
    "previous_action_ki",
    "previous_action_kd",
    "v_target",
    "wheel_utilization",
    "pid_saturated",
    "allocation_limited",
    "steer_integrator_hold",
)

DIRECT_FRAME_FIELDS = (
    "e_ct",
    "e_ct_rate",
    "e_ct_integral",
    "e_theta",
    "speed",
    "yaw_rate",
    "previous_steer",
    "omega_applied",
    "v_applied",
    "v_target",
    "wheel_utilization",
    "steer_saturated",
    "allocation_limited",
)

COMMON_PPO_KWARGS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [64, 64]},
}

SCHEDULER_PPO_KWARGS = {
    **COMMON_PPO_KWARGS,
    "gamma": 0.99,
}

DIRECT_PPO_KWARGS = {
    **COMMON_PPO_KWARGS,
    "gamma": 0.999,
    "gae_lambda": 0.995,
    "ent_coef": 0.005,
    "policy_kwargs": {
        "net_arch": [64, 64],
        "log_std_init": math.log(0.1),
    },
}


@dataclass(frozen=True)
class ControllerProfile:
    mode: str
    frame_fields: tuple[str, ...]
    history_length: int
    action_size: int
    default_timesteps: int
    default_eval_freq: int
    tracking_scale_m: float
    tracking_weight: float
    action_smoothness_weight: float
    finish_bonus: float
    failure_penalty: float
    ppo_kwargs: dict[str, Any]
    checkpoint_selection: str

    @property
    def observation_size(self) -> int:
        return len(self.frame_fields) * self.history_length


PROFILES = {
    "scheduled": ControllerProfile(
        mode="scheduled",
        frame_fields=SCHEDULER_FRAME_FIELDS,
        history_length=10,
        action_size=3,
        default_timesteps=300_000,
        default_eval_freq=25_000,
        # The native Blind_PPO 0.005 m scale saturates x^2/(1+x^2) above about
        # 0.02 m. Under the 0.15 s delay the plant tracks at 0.02-0.035 m, so
        # the tracking term was pinned at its maximum and its gradient fell
        # roughly 100-fold, leaving progress and the terminal bonus to drive
        # disturbed training. 0.02 m keeps both the nominal and the disturbed
        # operating points inside the responsive part of the curve.
        tracking_scale_m=0.02,
        tracking_weight=0.035,
        action_smoothness_weight=0.01,
        finish_bonus=20.0,
        failure_penalty=60.0,
        ppo_kwargs=SCHEDULER_PPO_KWARGS,
        checkpoint_selection="manifest_completion_then_failure_adjusted_error",
    ),
    "direct": ControllerProfile(
        mode="direct",
        frame_fields=DIRECT_FRAME_FIELDS,
        history_length=10,
        action_size=1,
        # 1M decisions left validation completion oscillating between 0.46
        # and 1.00 under the dynamic sampler; 2M recovers most of the loss
        # but does not remove the oscillation. See DECISIONS.md 2026-08-23.
        default_timesteps=2_000_000,
        default_eval_freq=100_000,
        tracking_scale_m=0.1,
        tracking_weight=0.035,
        action_smoothness_weight=0.02,
        finish_bonus=20.0,
        failure_penalty=20.0,
        ppo_kwargs=DIRECT_PPO_KWARGS,
        checkpoint_selection="manifest_completion_then_failure_adjusted_error",
    ),
}


def profile_for(mode: str) -> ControllerProfile:
    if mode == "fixed":
        mode = "scheduled"
    try:
        return PROFILES[mode]
    except KeyError as exc:
        raise ValueError(f"no learned-controller profile for mode {mode!r}") from exc
