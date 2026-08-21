import numpy as np
import pytest

from calibration import GainCalibration
from core.dynamics import DT
from env import PathFollowingEnv
from scenarios import DisturbanceEvent, Scenario


CALIBRATION = GainCalibration.development_default()


def scenario(**overrides):
    values = {
        "scenario_id": "test-arc",
        "path": {"kind": "arc"},
        "target_speed": 0.3,
        "evaluation_mode": "stationary",
        "condition": "nominal",
        "noise_seed": 9,
    }
    values.update(overrides)
    return Scenario(**values)


@pytest.mark.parametrize("mode,shape", [("fixed", (3,)), ("scheduled", (3,)), ("direct", (1,))])
def test_modes_share_observation_contract(mode, shape):
    kwargs = {"fixed_gains": CALIBRATION.nominal} if mode == "fixed" else {}
    env = PathFollowingEnv(mode=mode, training=False, calibration=CALIBRATION, **kwargs)
    try:
        observation, _ = env.reset(seed=1, options={**scenario().to_options(), "max_episode_steps": 1})
        assert observation.shape == (170,)
        assert env.action_space.shape == shape
        _, _, _, truncated, info = env.step(np.zeros(shape, dtype=np.float32))
        assert truncated
        assert "failure_adjusted_error_m" in info["episode_metrics"]
    finally:
        env.close()


def test_scheduler_uses_absolute_per_gain_rate_limits():
    rates = np.asarray((20.0, 0.5, 1.5))
    env = PathFollowingEnv(
        mode="scheduled",
        training=False,
        calibration=CALIBRATION,
        gain_rate_limit=tuple(rates),
    )
    try:
        env.reset(seed=1, options={**scenario().to_options(), "max_episode_steps": 1})
        initial = env.applied_gains.copy()
        env.step(np.ones(3, dtype=np.float32))
        expected = initial + rates * env.control_dt
        np.testing.assert_allclose(env.applied_gains, expected)
    finally:
        env.close()


def test_fixed_and_equivalent_scheduler_replay_exactly():
    fixed = PathFollowingEnv(mode="fixed", training=False, calibration=CALIBRATION, fixed_gains=CALIBRATION.nominal)
    scheduled = PathFollowingEnv(mode="scheduled", training=False, calibration=CALIBRATION)
    options = {**scenario().to_options(), "max_episode_steps": 8}
    try:
        fixed_obs, _ = fixed.reset(seed=2, options=options)
        scheduled_obs, _ = scheduled.reset(seed=2, options=options)
        np.testing.assert_array_equal(fixed_obs, scheduled_obs)
        action = CALIBRATION.action_for(CALIBRATION.nominal)
        while True:
            fixed_step = fixed.step(np.zeros(3, dtype=np.float32))
            scheduled_step = scheduled.step(action)
            np.testing.assert_allclose(fixed_step[0], scheduled_step[0], atol=1e-12, rtol=1e-7)
            assert fixed_step[1] == pytest.approx(scheduled_step[1], abs=1e-12)
            assert fixed_step[2:4] == scheduled_step[2:4]
            if fixed_step[2] or fixed_step[3]:
                break
        fixed_trace = fixed.get_trace()
        scheduled_trace = scheduled.get_trace()
        for field in ("x", "y", "e_ct", "distance_m", "omega_requested", "omega_applied"):
            np.testing.assert_allclose(
                [row[field] for row in fixed_trace],
                [row[field] for row in scheduled_trace],
                atol=1e-12,
                rtol=1e-7,
            )
    finally:
        fixed.close()
        scheduled.close()


def test_delay_and_noise_events_apply_at_simulation_time():
    event_scenario = scenario(
        evaluation_mode="transient",
        condition="combined",
        events=(
            DisturbanceEvent("delay_step", 0.01, 0.006),
            DisturbanceEvent("noise_step", 0.01, 0.001),
        ),
    )
    env = PathFollowingEnv(mode="fixed", training=False, calibration=CALIBRATION, fixed_gains=CALIBRATION.nominal)
    try:
        env.reset(seed=3, options={**event_scenario.to_options(), "max_episode_steps": 1})
        env.step(np.zeros(3, dtype=np.float32))
        assert env.time_s == pytest.approx(10 * DT)
        assert env.actuator_delay_s == pytest.approx(0.006)
        assert env.sensor_noise_m == pytest.approx(0.001)
    finally:
        env.close()


def test_noise_and_delay_replay_match_between_pid_adapters():
    paired = scenario(
        condition="combined",
        initial_delay_s=0.006,
        initial_noise_std_m=0.001,
    )
    fixed = PathFollowingEnv(mode="fixed", training=False, calibration=CALIBRATION, fixed_gains=CALIBRATION.nominal)
    scheduled = PathFollowingEnv(mode="scheduled", training=False, calibration=CALIBRATION)
    options = {**paired.to_options(), "max_episode_steps": 5}
    try:
        fixed.reset(seed=8, options=options)
        scheduled.reset(seed=8, options=options)
        action = CALIBRATION.action_for(CALIBRATION.nominal)
        for _ in range(5):
            fixed_result = fixed.step(np.zeros(3, dtype=np.float32))
            scheduled_result = scheduled.step(action)
            assert fixed_result[2:4] == scheduled_result[2:4]
        np.testing.assert_allclose(
            [row["e_ct_measured"] for row in fixed.get_trace()],
            [row["e_ct_measured"] for row in scheduled.get_trace()],
            atol=1e-12,
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            [row["u_left_applied"] for row in fixed.get_trace()],
            [row["u_left_applied"] for row in scheduled.get_trace()],
            atol=1e-12,
            rtol=1e-7,
        )
    finally:
        fixed.close()
        scheduled.close()


def test_failure_adjusted_error_pads_unused_horizon():
    env = PathFollowingEnv(mode="direct", training=False, calibration=CALIBRATION)
    try:
        env.reset(seed=4, options={**scenario().to_options(), "max_episode_steps": 1})
        _, _, _, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        assert truncated
        metrics = info["episode_metrics"]
        expected = (metrics["iae_m_s"] + env.max_time_s - env.time_s) / env.max_time_s
        assert metrics["failure_adjusted_error_m"] == pytest.approx(expected)
    finally:
        env.close()


def test_optional_trace_records_every_physics_step():
    env = PathFollowingEnv(mode="direct", training=False, calibration=CALIBRATION, physics_trace=True)
    try:
        env.reset(seed=5, options={**scenario().to_options(), "max_episode_steps": 1})
        env.step(np.zeros(1, dtype=np.float32))
        assert len(env.get_trace()) == env.physics_steps == 10
    finally:
        env.close()
