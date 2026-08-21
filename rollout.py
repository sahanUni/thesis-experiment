"""Shared episode runner used by calibration, validation, evaluation, and UI."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from env import PathFollowingEnv
from scenarios import Scenario


Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RolloutResult:
    metrics: dict
    trace: list[dict]


def fixed_action(env: PathFollowingEnv) -> Predictor:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    return lambda observation: action


def run_episode(
    env: PathFollowingEnv,
    scenario: Scenario,
    predictor: Predictor | None = None,
    *,
    reset_seed: int = 0,
) -> RolloutResult:
    observation, _ = env.reset(seed=reset_seed, options=scenario.to_options())
    predictor = predictor or fixed_action(env)
    inference_ns: list[int] = []
    while True:
        started = time.perf_counter_ns()
        action = np.asarray(predictor(observation), dtype=np.float32)
        inference_ns.append(time.perf_counter_ns() - started)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            metrics = dict(info["episode_metrics"])
            timings = np.asarray(inference_ns, dtype=np.float64) / 1e6
            metrics.update(
                {
                    "mean_inference_ms": float(np.mean(timings)),
                    "p99_inference_ms": float(np.quantile(timings, 0.99)),
                    "policy_decisions": len(inference_ns),
                    "physics_steps": env.physics_steps,
                }
            )
            return RolloutResult(metrics=metrics, trace=env.get_trace())
