"""Controller-neutral MuJoCo environment for all thesis experiment arms."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

import config
from controller_profiles import profile_for
from calibration import GainCalibration, PIDGains
from core import paths
from core.dynamics import (
    DRIVE_SIGNS,
    DT,
    TrackState,
    allocate_steering_priority,
    forward_speed,
    fresh_model,
    track_error,
    yaw_of,
)
from core.pid import ConditionalPIDController
from metrics import failure_adjusted_error


MODES = {"fixed", "scheduled", "direct"}


def tracking_cost_of(distances: Any, scale_m: float) -> float:
    squared = (np.asarray(distances, dtype=np.float64) / scale_m) ** 2
    return float(np.mean(squared / (1.0 + squared)))


class PathFollowingEnv(gym.Env[np.ndarray, np.ndarray]):
    """One plant and observation contract with three steering-controller modes."""

    metadata = {"render_modes": [], "render_fps": config.CONFIG.policy_hz}

    def __init__(
        self,
        *,
        mode: str,
        training: bool = True,
        disturbance_training: bool = False,
        calibration: GainCalibration | None = None,
        fixed_gains: PIDGains | None = None,
        path_specs: tuple[dict[str, Any], ...] | None = None,
        gain_rate_limit: tuple[float, float, float] | None = None,
        derivative_filter_tau_s: float | None = None,
        legacy_derivative_kick: bool | None = None,
        physics_trace: bool = False,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if mode == "fixed" and fixed_gains is None:
            raise ValueError("fixed mode requires fixed_gains")
        self.mode = mode
        self.profile = profile_for(mode)
        self.training = bool(training)
        self.disturbance_training = bool(disturbance_training)
        self.calibration = calibration or GainCalibration.development_default()
        self.fixed_gains = fixed_gains
        self.path_specs = tuple(path_specs or config.PATH_SPLITS["train"])
        if gain_rate_limit is None:
            gain_rate_limit = tuple(
                0.5 * (self.calibration.upper.as_array() - self.calibration.lower.as_array())
            )
        self.gain_rate_limit = np.asarray(gain_rate_limit, dtype=np.float64)
        if derivative_filter_tau_s is None:
            derivative_filter_tau_s = config.CONFIG.derivative_filter_tau_s
        self.derivative_filter_tau_s = float(derivative_filter_tau_s)
        self.legacy_derivative_kick = (
            False if legacy_derivative_kick is None else bool(legacy_derivative_kick)
        )
        self.physics_trace = bool(physics_trace)
        if self.gain_rate_limit.shape != (3,) or np.any(self.gain_rate_limit <= 0):
            raise ValueError("gain_rate_limit must contain three positive values")
        if self.derivative_filter_tau_s < 0:
            raise ValueError("derivative_filter_tau_s must be non-negative")

        action_size = 3 if mode == "fixed" else self.profile.action_size
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.profile.observation_size,),
            dtype=np.float32,
        )
        self.render_mode = None
        self._history: deque[np.ndarray] = deque(maxlen=self.profile.history_length)
        self._needs_reset = True

    @property
    def control_dt(self) -> float:
        return config.CONFIG.frame_skip * DT

    def _sample_episode(self, options: dict[str, Any]) -> dict[str, Any]:
        path_spec = deepcopy(self.path_specs[int(self.np_random.integers(len(self.path_specs)))])
        episode = {
            "scenario_id": "training-sample",
            "path": path_spec,
            "v_target": float(self.np_random.choice(config.CONFIG.target_speeds)),
            "noise_seed": int(self.np_random.integers(0, 2**31 - 1)),
            "initial_delay_s": 0.0,
            "initial_noise_std_m": 0.0,
            "initial_pose": (0.0, 0.0, 0.0),
            "events": [],
        }
        if self.training and self.disturbance_training:
            self._sample_disturbance(episode)
        for key in episode:
            if key in options:
                episode[key] = deepcopy(options[key])
        return episode

    def _sample_severity(self, train_range: tuple[float, float], declared: float) -> float:
        """Draw the declared evaluation severity often enough to train for it."""
        if self.np_random.random() < config.DISTURBANCE_DECLARED_SEVERITY_PROB:
            return float(declared)
        return float(self.np_random.uniform(*train_range))

    def _sample_disturbance(self, episode: dict[str, Any]) -> None:
        """Mirror the evaluation condition grid with randomized severity and timing.

        Event times are stored as a fraction and resolved against the episode's
        own ideal duration in reset(), because training durations span 9 s to
        53 s and a hardcoded window cannot land mid-episode on all of them. In
        the combined transient both channels step together, exactly as the
        evaluation manifest builds them.
        """
        index = int(
            self.np_random.choice(
                len(config.DISTURBANCE_CONDITIONS),
                p=config.DISTURBANCE_CONDITION_WEIGHTS,
            )
        )
        condition = config.DISTURBANCE_CONDITIONS[index]
        if condition == "nominal":
            return
        wants_delay = condition in {"delay", "combined"}
        wants_noise = condition in {"noise", "combined"}
        delay = self._sample_severity(config.TRAIN_DELAY_RANGE_S, config.CONFIG.delay_severity_s)
        noise = self._sample_severity(config.TRAIN_NOISE_RANGE_M, config.CONFIG.noise_severity_m)
        if self.np_random.random() >= config.DISTURBANCE_TRANSIENT_PROB:
            episode["initial_delay_s"] = delay if wants_delay else 0.0
            episode["initial_noise_std_m"] = noise if wants_noise else 0.0
            return
        fractions = self._sample_change_fractions()
        for position, start_fraction in enumerate(fractions):
            recovered = position > 0 and self.np_random.random() < config.DISTURBANCE_RECOVERY_PROB
            if wants_delay:
                episode["events"].append({
                    "kind": "delay_step",
                    "start_fraction": start_fraction,
                    "value": 0.0 if recovered else (
                        delay if position == 0 else
                        self._sample_severity(
                            config.TRAIN_DELAY_RANGE_S, config.CONFIG.delay_severity_s
                        )
                    ),
                })
            if wants_noise:
                episode["events"].append({
                    "kind": "noise_step",
                    "start_fraction": start_fraction,
                    "value": 0.0 if recovered else (
                        noise if position == 0 else
                        self._sample_severity(
                            config.TRAIN_NOISE_RANGE_M, config.CONFIG.noise_severity_m
                        )
                    ),
                })

    def _sample_change_fractions(self) -> list[float]:
        """Sorted change times so later events override earlier ones in _apply_events."""
        count = 1 + int(
            self.np_random.choice(
                len(config.DISTURBANCE_CHANGE_COUNT_WEIGHTS),
                p=config.DISTURBANCE_CHANGE_COUNT_WEIGHTS,
            )
        )
        return sorted(float(self.np_random.random()) for _ in range(count))

    def _resolve_events(
        self,
        events: list[dict[str, Any]],
        ideal_time_s: float,
    ) -> list[dict[str, Any]]:
        """Convert sampled fractional event times to absolute simulation times.

        Serialized evaluation scenarios always carry an absolute start_s and
        pass through untouched.
        """
        latest = max(
            config.EVENT_START_MIN_S,
            config.EVENT_START_IDEAL_FRACTION * ideal_time_s,
        )
        resolved: list[dict[str, Any]] = []
        for event in events:
            event = dict(event)
            if "start_fraction" in event:
                fraction = float(event.pop("start_fraction"))
                event["start_s"] = config.EVENT_START_MIN_S + fraction * (
                    latest - config.EVENT_START_MIN_S
                )
            resolved.append(event)
        return resolved

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode = self._sample_episode(dict(options or {}))
        self.scenario_id = str(episode["scenario_id"])
        self.path_spec = deepcopy(episode["path"])
        self.path = paths.build(self.path_spec)
        self.path_key = str(self.path_spec["kind"])
        self.v_target = float(episode["v_target"])
        self.noise_seed = int(episode["noise_seed"])
        self.ideal_time_s = float(self.path["length"]) / self.v_target
        self.events = self._resolve_events(deepcopy(list(episode["events"])), self.ideal_time_s)
        self._validate_events(self.events)
        self.actuator_delay_s = max(0.0, float(episode["initial_delay_s"]))
        self.sensor_noise_m = max(0.0, float(episode["initial_noise_std_m"]))
        initial_pose = np.asarray(episode["initial_pose"], dtype=np.float64)
        if initial_pose.shape != (3,) or not np.isfinite(initial_pose).all():
            raise ValueError("initial_pose must contain finite x, y, yaw values")
        self._base_actuator_delay_s = self.actuator_delay_s
        self._base_sensor_noise_m = self.sensor_noise_m
        self.delay_steps = int(round(self.actuator_delay_s / DT))
        self.actuator_delay_s = self.delay_steps * DT
        self._command_queue: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * self.delay_steps, maxlen=self.delay_steps + 1
        )
        self._noise_rng = np.random.default_rng(self.noise_seed)

        self.model = fresh_model(
            config.NOMINAL_MASS_KG,
            config.NOMINAL_FRICTION,
            config.NOMINAL_ACTUATOR,
        )
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = (initial_pose[0], initial_pose[1], 0.03)
        self.data.qpos[3:7] = (
            math.cos(initial_pose[2] / 2.0),
            0.0,
            0.0,
            math.sin(initial_pose[2] / 2.0),
        )
        mujoco.mj_forward(self.model, self.data)

        initial_gains = self.fixed_gains or self.calibration.nominal
        self.steer_pid = ConditionalPIDController(
            kp=initial_gains.kp,
            ki=initial_gains.ki,
            kd=initial_gains.kd,
            setpoint=0.0,
            output_limits=(-1.0, 1.0),
            derivative_filter_tau=self.derivative_filter_tau_s,
            legacy_derivative_kick=self.legacy_derivative_kick,
        )
        self.speed_pid = ConditionalPIDController(
            kp=config.SPEED_PID_GAINS[0],
            ki=config.SPEED_PID_GAINS[1],
            kd=config.SPEED_PID_GAINS[2],
            setpoint=self.v_target,
            output_limits=(-1.0, 1.0),
            derivative_filter_tau=self.derivative_filter_tau_s,
        )

        self.max_time_s = (
            config.CONFIG.max_time_factor * self.ideal_time_s + config.CONFIG.max_time_margin_s
        )
        self.max_episode_steps = int(math.ceil(self.max_time_s / self.control_dt))
        if options and "max_episode_steps" in options:
            self.max_episode_steps = int(options["max_episode_steps"])
            self.max_time_s = self.max_episode_steps * self.control_dt

        self.path_index = 0
        self.elapsed_steps = 0
        self.physics_steps = 0
        self.time_s = 0.0
        self.previous_action = (
            self.calibration.action_for(initial_gains).astype(np.float64)
            if self.mode != "direct"
            else np.zeros(3, dtype=np.float64)
        )
        self._previous_direct_action = 0.0
        self._direct_action = 0.0
        self.target_gains = initial_gains.as_array()
        self.applied_gains = initial_gains.as_array()
        self._error_integral = 0.0
        self._last_frame_e_ct = 0.0
        self._last_omega_requested = 0.0
        self._trace: list[dict[str, Any]] = []
        self._physics_trace: list[dict[str, Any]] = []
        self._distance_values: list[float] = []
        self._iae = 0.0
        self._itae = 0.0
        self._steer_sq_sum = 0.0
        self._steer_variation = 0.0
        self._speed_sum = 0.0
        self._gain_variation = 0.0
        self._saturation_count = 0
        self._allocation_count = 0
        self._needs_reset = False

        state = self._state()
        measured = self._measure(state.e_ct)
        self._last_frame_e_ct = measured
        zero = self._zero_control()
        frame = self._frame(state, measured, 0.0, zero)
        self._history.clear()
        for _ in range(self.profile.history_length):
            self._history.append(frame.copy())
        return self._observation(), self._base_info(state)

    @staticmethod
    def _validate_events(events: list[dict[str, Any]]) -> None:
        for event in events:
            if event.get("kind") not in {"delay_step", "noise_step"}:
                raise ValueError(f"unsupported event: {event.get('kind')}")
            if float(event.get("start_s", -1.0)) < 0 or float(event.get("value", -1.0)) < 0:
                raise ValueError("event start_s and value must be non-negative")
            if event.get("end_s") is not None and float(event["end_s"]) <= float(event["start_s"]):
                raise ValueError("event end_s must be after start_s")

    @staticmethod
    def _zero_control() -> dict[str, Any]:
        return {
            "omega_unclamped": 0.0,
            "omega_requested": 0.0,
            "omega_applied": 0.0,
            "v_requested": 0.0,
            "v_applied": 0.0,
            "u_left": 0.0,
            "u_right": 0.0,
            "u_left_applied": 0.0,
            "u_right_applied": 0.0,
            "wheel_utilization": 0.0,
            "steer_saturated": False,
            "allocation_limited": False,
            "steer_integrator_hold": False,
        }

    def _state(self) -> TrackState:
        xy = np.asarray(self.data.qpos[0:2], dtype=np.float64)
        state = track_error(self.path, xy, yaw_of(self.data.qpos[3:7]), self.path_index)
        self.path_index = state.index
        return state

    def _measure(self, true_e_ct: float) -> float:
        if self.sensor_noise_m <= 0:
            return float(true_e_ct)
        return float(true_e_ct + self._noise_rng.normal(0.0, self.sensor_noise_m))

    def _event_active(self, event: dict[str, Any]) -> bool:
        end = event.get("end_s")
        return self.time_s >= float(event["start_s"]) and (end is None or self.time_s < float(end))

    def _set_delay_steps(self, steps: int) -> None:
        steps = max(0, int(steps))
        if steps == self.delay_steps:
            return
        queued = list(self._command_queue)
        if steps > len(queued):
            queued = [queued[0] if queued else (0.0, 0.0)] * (steps - len(queued)) + queued
        else:
            queued = queued[-steps:] if steps else []
        self.delay_steps = steps
        self.actuator_delay_s = steps * DT
        self._command_queue = deque(queued, maxlen=steps + 1)

    def _apply_events(self) -> None:
        delay = self._base_actuator_delay_s
        noise = self._base_sensor_noise_m
        for event in self.events:
            if self._event_active(event):
                if event["kind"] == "delay_step":
                    delay = float(event["value"])
                elif event["kind"] == "noise_step":
                    noise = float(event["value"])
        self._set_delay_steps(int(round(delay / DT)))
        self.sensor_noise_m = noise

    def _delayed_command(self, left: float, right: float) -> tuple[float, float]:
        self._command_queue.append((float(left), float(right)))
        return self._command_queue.popleft()

    def _set_controller_action(self, action: np.ndarray) -> None:
        if self.mode == "direct":
            self._previous_direct_action = self._direct_action
            self._direct_action = float(action[0])
            return
        if self.mode == "fixed":
            gains = self.fixed_gains
            assert gains is not None
            self.previous_action[:] = self.calibration.action_for(gains)
        else:
            self.previous_action[:] = action
            gains = self.calibration.map_action(action)
        self.target_gains = gains.as_array()
        previous_gains = self.applied_gains.copy()
        limit = self.gain_rate_limit * self.control_dt
        self.applied_gains += np.clip(self.target_gains - self.applied_gains, -limit, limit)
        self._gain_variation += float(np.sum(np.abs(self.applied_gains - previous_gains)))
        self.steer_pid.kp, self.steer_pid.ki, self.steer_pid.kd = self.applied_gains

    def _steer(self, measured_e_ct: float) -> float:
        self._error_integral = float(np.clip(
            self._error_integral + measured_e_ct * DT,
            -config.INTEGRAL_SCALE_MS,
            config.INTEGRAL_SCALE_MS,
        ))
        if self.mode == "direct":
            return self._direct_action
        return float(self.steer_pid.update(measured_e_ct, DT))

    def _frame(self, state: TrackState, measured: float, rate: float, control: dict[str, Any]) -> np.ndarray:
        yaw = yaw_of(self.data.qpos[3:7])
        common = (
            rate / config.E_CT_RATE_SCALE_MPS,
            state.e_theta / np.pi,
            forward_speed(self.data, yaw) / config.SPEED_SCALE_MPS,
            float(self.data.qvel[5]) / config.YAW_RATE_SCALE_RADPS,
        )
        if self.mode == "direct":
            values = np.asarray(
                (
                    measured / 0.2,
                    common[0],
                    self._error_integral / config.INTEGRAL_SCALE_MS,
                    *common[1:],
                    self._previous_direct_action,
                    control["omega_applied"],
                    control["v_applied"],
                    self.v_target / max(config.CONFIG.target_speeds),
                    control["wheel_utilization"],
                    float(control["steer_saturated"]),
                    float(control["allocation_limited"]),
                ),
                dtype=np.float32,
            )
        else:
            values = np.asarray(
                (
                    measured / config.E_CT_SCALE_M,
                    common[0],
                    self.steer_pid._integral / config.INTEGRAL_SCALE_MS,
                    *common[1:],
                    control["omega_requested"],
                    control["omega_applied"],
                    control["v_applied"],
                    *self.previous_action,
                    self.v_target / max(config.CONFIG.target_speeds),
                    control["wheel_utilization"],
                    float(control["steer_saturated"]),
                    float(control["allocation_limited"]),
                    float(control["steer_integrator_hold"]),
                ),
                dtype=np.float32,
            )
        return np.clip(values, -1.0, 1.0)

    def _observation(self) -> np.ndarray:
        return np.concatenate(tuple(self._history)).astype(np.float32, copy=False)

    def _base_info(self, state: TrackState) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "path_key": self.path_key,
            "v_target": self.v_target,
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_m": self.sensor_noise_m,
            "progress_m": state.progress_m,
            "distance_m": state.distance_m,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("reset() must be called before step() or after an episode ends")
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = 1 if self.mode == "direct" else 3
        if action.shape != (expected,) or not np.isfinite(action).all():
            raise ValueError(f"action must contain {expected} finite value(s)")
        action = np.clip(action, -1.0, 1.0)
        previous_policy_action = self.previous_action.copy()
        self._set_controller_action(action)
        start = self._state().progress_m
        distances: list[float] = []
        finished = False
        failure_reason = ""
        terminated = False
        control = self._zero_control()
        previous_omega = self._last_omega_requested

        for _ in range(config.CONFIG.frame_skip):
            self._apply_events()
            state_for_control = self._state()
            yaw = yaw_of(self.data.qpos[3:7])
            measured = self._measure(state_for_control.e_ct)
            omega_requested = self._steer(measured)
            v_requested = float(self.speed_pid.update(forward_speed(self.data, yaw), DT))
            _, _, left, right, v_applied, omega_applied = allocate_steering_priority(
                v_requested, omega_requested
            )
            if self.mode != "direct":
                self.steer_pid.apply_output_feedback(omega_applied)
            self.speed_pid.apply_output_feedback(v_applied)
            left_applied, right_applied = self._delayed_command(left, right)
            self.data.ctrl[0] = DRIVE_SIGNS[0] * left_applied
            self.data.ctrl[1] = DRIVE_SIGNS[1] * right_applied
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            self.time_s = self.physics_steps * DT

            state = self._state()
            distance = state.distance_m
            distances.append(distance)
            self._distance_values.append(distance)
            self._iae += distance * DT
            self._itae += self.time_s * distance * DT
            steer_saturated = (
                abs(omega_requested) >= 1.0 - 1e-12
                if self.mode == "direct"
                else abs(self.steer_pid.last_unclamped_output) > 1.0 + 1e-12
            )
            allocation_limited = not (
                np.isclose(v_requested, v_applied) and np.isclose(omega_requested, omega_applied)
            )
            self._saturation_count += int(max(abs(left), abs(right)) > 0.99)
            self._allocation_count += int(allocation_limited)
            self._steer_sq_sum += omega_requested**2
            self._speed_sum += forward_speed(self.data, yaw_of(self.data.qpos[3:7]))
            self._steer_variation += abs(omega_requested - self._last_omega_requested)
            self._last_omega_requested = omega_requested
            control = {
                "omega_unclamped": omega_requested if self.mode == "direct" else self.steer_pid.last_unclamped_output,
                "omega_requested": omega_requested,
                "omega_applied": omega_applied,
                "v_requested": v_requested,
                "v_applied": v_applied,
                "u_left": left,
                "u_right": right,
                "u_left_applied": left_applied,
                "u_right_applied": right_applied,
                "wheel_utilization": max(abs(left), abs(right)),
                "steer_saturated": steer_saturated,
                "allocation_limited": allocation_limited,
                "steer_integrator_hold": self.mode != "direct" and (
                    self.steer_pid.last_internal_hold or self.steer_pid.last_downstream_hold
                ),
            }
            if self.physics_trace:
                self._append_physics_trace(state, measured, control)

            if not np.isfinite(self.data.qpos).all() or np.abs(self.data.qpos[0:2]).max() > config.CONFIG.runaway_m:
                terminated, failure_reason = True, "invalid_state"
            elif abs(state.e_ct) > config.CONFIG.corridor_m or distance > config.CONFIG.corridor_m:
                terminated, failure_reason = True, "corridor_exit"
            elif state.index >= len(self.path["pts"]) - 1:
                terminated, finished = True, True
            if terminated:
                break

        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        if truncated:
            failure_reason = "time_limit"
        measured = self._measure(state.e_ct)
        elapsed = max(len(distances), 1) * DT
        rate = (measured - self._last_frame_e_ct) / elapsed
        self._last_frame_e_ct = measured
        self._history.append(self._frame(state, measured, rate, control))

        progress = max(0.0, state.progress_m - start)
        steer_delta = self._last_omega_requested - previous_omega
        reward_progress = config.PROGRESS_WEIGHT * progress
        reward_tracking = -self.profile.tracking_weight * tracking_cost_of(
            distances, self.profile.tracking_scale_m
        )
        if self.mode == "direct":
            reward_steer = -self.profile.action_smoothness_weight * (
                self._direct_action - self._previous_direct_action
            ) ** 2
        else:
            reward_steer = -self.profile.action_smoothness_weight * float(
                np.sum((self.previous_action - previous_policy_action) ** 2)
            )
        reward_terminal = (
            self.profile.finish_bonus
            if finished
            else -self.profile.failure_penalty
            if (terminated or truncated)
            else 0.0
        )
        reward = reward_progress + reward_tracking + reward_steer + reward_terminal

        info = self._base_info(state)
        info.update({
            "finished": finished,
            "failure_reason": failure_reason,
            "e_ct_measured": measured,
            "action": action.copy(),
            "gains": self.applied_gains.copy() if self.mode != "direct" else None,
            "reward_progress": reward_progress,
            "reward_tracking": reward_tracking,
            "reward_steer": reward_steer,
            "reward_terminal": reward_terminal,
            **control,
        })
        self._append_trace(state, reward, info)
        if terminated or truncated:
            self._needs_reset = True
            info["episode_metrics"] = self.episode_metrics(finished, failure_reason)
        return self._observation(), float(reward), terminated, truncated, info

    def _append_trace(self, state: TrackState, reward: float, info: dict[str, Any]) -> None:
        row = {
            "t": self.time_s,
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "e_ct": state.e_ct,
            "e_ct_measured": info["e_ct_measured"],
            "distance_m": state.distance_m,
            "progress_m": state.progress_m,
            "speed_mps": forward_speed(self.data, yaw_of(self.data.qpos[3:7])),
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_m": self.sensor_noise_m,
            "reward": reward,
        }
        for key in ("omega_requested", "omega_applied", "v_requested", "v_applied", "u_left", "u_right", "u_left_applied", "u_right_applied"):
            row[key] = float(info[key])
        if self.mode != "direct":
            row.update({"kp": float(self.applied_gains[0]), "ki": float(self.applied_gains[1]), "kd": float(self.applied_gains[2])})
        self._trace.append(row)

    def _append_physics_trace(
        self,
        state: TrackState,
        measured: float,
        control: dict[str, Any],
    ) -> None:
        row = {
            "t": self.time_s,
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "e_ct": state.e_ct,
            "e_ct_measured": measured,
            "distance_m": state.distance_m,
            "progress_m": state.progress_m,
            "speed_mps": forward_speed(self.data, yaw_of(self.data.qpos[3:7])),
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_m": self.sensor_noise_m,
        }
        for key in ("omega_requested", "omega_applied", "v_requested", "v_applied", "u_left", "u_right", "u_left_applied", "u_right_applied"):
            row[key] = float(control[key])
        if self.mode != "direct":
            row.update({"kp": float(self.applied_gains[0]), "ki": float(self.applied_gains[1]), "kd": float(self.applied_gains[2])})
        self._physics_trace.append(row)

    def episode_metrics(self, finished: bool, failure_reason: str) -> dict[str, Any]:
        values = np.asarray(self._distance_values or [0.0], dtype=np.float64)
        count = max(len(self._distance_values), 1)
        result = {
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "finished": bool(finished),
            "failure_reason": failure_reason,
            "duration_s": self.time_s,
            "max_time_s": self.max_time_s,
            "progress_pct": 100.0 * float(self.path["s"][self.path_index]) / float(self.path["length"]),
            "failure_adjusted_error_m": failure_adjusted_error(
                self._iae,
                finished=finished,
                duration_s=self.time_s,
                max_time_s=max(self.max_time_s, DT),
                corridor_m=config.CONFIG.corridor_m,
            ),
            "mean_distance_m": float(np.mean(values)),
            "rms_distance_m": float(np.sqrt(np.mean(values**2))),
            "p95_distance_m": float(np.quantile(values, 0.95)),
            "max_distance_m": float(np.max(values)),
            "iae_m_s": self._iae,
            "itae_m_s2": self._itae,
            "mean_speed_mps": self._speed_sum / count,
            "steer_rms": math.sqrt(self._steer_sq_sum / count),
            "steer_total_variation_per_s": self._steer_variation / max(self.time_s, DT),
            "gain_total_variation_per_s": self._gain_variation / max(self.time_s, DT) if self.mode != "direct" else 0.0,
            "wheel_saturation_fraction": self._saturation_count / count,
            "allocation_limited_fraction": self._allocation_count / count,
        }
        result.update(self._recovery_metrics(values))
        return result

    def _recovery_metrics(self, distances: np.ndarray) -> dict[str, float | None]:
        starts = [float(event["start_s"]) for event in self.events]
        if not starts:
            return {
                "pre_event_mean_distance_m": None,
                "post_event_peak_distance_m": None,
                "post_event_excess_iae_m_s": None,
                "post_event_excess_itae_m_s2": None,
                "recovery_time_s": None,
            }
        start = min(starts)
        times = np.arange(1, len(distances) + 1, dtype=np.float64) * DT
        pre = distances[(times >= max(0.0, start - 1.0)) & (times < start)]
        post = (times >= start) & (times <= start + 2.0)
        baseline = float(np.mean(pre)) if len(pre) else 0.0
        post_values = distances[post]
        relative_t = times[post] - start
        excess = np.maximum(post_values - baseline, 0.0)
        threshold = max(1.25 * baseline, 0.02)
        sustained = max(1, int(round(0.5 / DT)))
        recovery: float | None = None
        after = np.flatnonzero(times >= start)
        for index in after:
            if index + sustained <= len(distances) and np.all(distances[index : index + sustained] <= threshold):
                recovery = float(times[index] - start)
                break
        return {
            "pre_event_mean_distance_m": baseline,
            "post_event_peak_distance_m": float(np.max(post_values)) if len(post_values) else None,
            "post_event_excess_iae_m_s": float(np.sum(excess) * DT),
            "post_event_excess_itae_m_s2": float(np.sum(relative_t * excess) * DT),
            "recovery_time_s": recovery,
        }

    def get_trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._physics_trace if self.physics_trace else self._trace)

    def close(self) -> None:
        self._needs_reset = True
