import numpy as np
import pytest

import config

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


@pytest.mark.parametrize(
    "mode,action_shape,observation_shape",
    [
        ("fixed", (3,), (170,)),
        ("scheduled", (3,), (170,)),
        ("direct", (1,), (130,)),
    ],
)
def test_modes_use_native_controller_contracts(mode, action_shape, observation_shape):
    kwargs = {"fixed_gains": CALIBRATION.nominal} if mode == "fixed" else {}
    env = PathFollowingEnv(mode=mode, training=False, calibration=CALIBRATION, **kwargs)
    try:
        observation, _ = env.reset(seed=1, options={**scenario().to_options(), "max_episode_steps": 1})
        assert observation.shape == observation_shape
        assert env.action_space.shape == action_shape
        _, _, _, truncated, info = env.step(np.zeros(action_shape, dtype=np.float32))
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


def test_scheduler_default_rate_limit_matches_blind_ppo_box_rate():
    env = PathFollowingEnv(mode="scheduled", training=False, calibration=CALIBRATION)
    try:
        env.reset(seed=1, options={**scenario().to_options(), "max_episode_steps": 1})
        initial = env.applied_gains.copy()
        env.step(np.ones(3, dtype=np.float32))
        box_width = CALIBRATION.upper.as_array() - CALIBRATION.lower.as_array()
        expected = initial + 0.5 * box_width * env.control_dt
        np.testing.assert_allclose(env.applied_gains, expected)
    finally:
        env.close()


def test_scheduler_uses_shared_filtered_pid_contract_by_default():
    fixed = PathFollowingEnv(
        mode="fixed",
        training=False,
        calibration=CALIBRATION,
        fixed_gains=CALIBRATION.nominal,
    )
    scheduled = PathFollowingEnv(mode="scheduled", training=False, calibration=CALIBRATION)
    try:
        fixed.reset(seed=1, options={**scenario().to_options(), "max_episode_steps": 1})
        scheduled.reset(seed=1, options={**scenario().to_options(), "max_episode_steps": 1})
        assert fixed.steer_pid.derivative_filter_tau == pytest.approx(0.01)
        assert scheduled.steer_pid.derivative_filter_tau == pytest.approx(0.01)
        assert not fixed.steer_pid.legacy_derivative_kick
        assert not scheduled.steer_pid.legacy_derivative_kick
    finally:
        fixed.close()
        scheduled.close()


def test_fixed_and_equivalent_scheduler_replay_exactly():
    fixed = PathFollowingEnv(
        mode="fixed",
        training=False,
        calibration=CALIBRATION,
        fixed_gains=CALIBRATION.nominal,
    )
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
    fixed = PathFollowingEnv(
        mode="fixed",
        training=False,
        calibration=CALIBRATION,
        fixed_gains=CALIBRATION.nominal,
    )
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


def disturbed_env():
    return PathFollowingEnv(
        mode="scheduled",
        training=True,
        disturbance_training=True,
        calibration=CALIBRATION,
    )


def test_disturbed_sampler_reaches_the_declared_evaluation_severity():
    env = disturbed_env()
    try:
        env.reset(seed=7)
        delays = []
        for _ in range(2000):
            episode = env._sample_episode({})
            steps = [float(e["value"]) for e in episode["events"] if e["kind"] == "delay_step"]
            delays.append(max([float(episode["initial_delay_s"]), *steps]))
    finally:
        env.close()
    delays = np.asarray(delays)
    assert np.isclose(delays, config.CONFIG.delay_severity_s).mean() > 0.25
    assert delays.mean() > 0.06


def test_disturbed_sampler_covers_every_evaluation_condition():
    env = disturbed_env()
    try:
        env.reset(seed=13)
        seen = set()
        for _ in range(2000):
            episode = env._sample_episode({})
            kinds = {e["kind"] for e in episode["events"]}
            delay = bool(episode["initial_delay_s"]) or "delay_step" in kinds
            noise = bool(episode["initial_noise_std_m"]) or "noise_step" in kinds
            seen.add({(False, False): "nominal", (True, False): "delay",
                      (False, True): "noise", (True, True): "combined"}[(delay, noise)])
            if kinds:
                seen.add("transient")
            elif delay or noise:
                seen.add("stationary")
    finally:
        env.close()
    assert seen == {*config.DISTURBANCE_CONDITIONS, "stationary", "transient"}


def test_combined_transient_steps_delay_and_noise_together():
    env = disturbed_env()
    try:
        env.reset(seed=3)
        paired = 0
        for _ in range(2000):
            events = env._sample_episode({})["events"]
            kinds = {e["kind"] for e in events}
            if kinds != {"delay_step", "noise_step"}:
                continue
            paired += 1
            by_kind = {
                kind: sorted(e["start_fraction"] for e in events if e["kind"] == kind)
                for kind in kinds
            }
            assert by_kind["delay_step"] == by_kind["noise_step"]
    finally:
        env.close()
    assert paired > 0


def test_sampled_event_times_resolve_inside_the_driving_window():
    env = disturbed_env()
    try:
        for ideal_time_s in (9.0, 52.6):
            for fraction in (0.0, 0.5, 1.0):
                resolved = env._resolve_events(
                    [{"kind": "delay_step", "start_fraction": fraction, "value": 0.15}],
                    ideal_time_s,
                )
                assert "start_fraction" not in resolved[0]
                assert (
                    config.EVENT_START_MIN_S
                    <= resolved[0]["start_s"]
                    <= config.EVENT_START_IDEAL_FRACTION * ideal_time_s
                )
    finally:
        env.close()


def test_serialized_scenario_events_keep_their_absolute_times():
    env = PathFollowingEnv(mode="scheduled", training=False, calibration=CALIBRATION)
    try:
        env.reset(seed=1, options=scenario(
            evaluation_mode="transient",
            condition="combined",
            events=(
                DisturbanceEvent("delay_step", 6.0, 0.15),
                DisturbanceEvent("noise_step", 6.0, 0.0003),
            ),
        ).to_options())
        assert [event["start_s"] for event in env.events] == [6.0, 6.0]
    finally:
        env.close()


def test_nominal_training_regime_stays_undisturbed():
    env = PathFollowingEnv(
        mode="scheduled", training=True, disturbance_training=False, calibration=CALIBRATION
    )
    try:
        env.reset(seed=5)
        for _ in range(200):
            episode = env._sample_episode({})
            assert episode["initial_delay_s"] == 0.0
            assert episode["initial_noise_std_m"] == 0.0
            assert episode["events"] == []
    finally:
        env.close()


def test_transient_disturbances_can_change_more_than_once_and_recover():
    env = disturbed_env()
    try:
        env.reset(seed=17)
        multi = recovered = 0
        for _ in range(3000):
            events = [e for e in env._sample_episode({})["events"] if e["kind"] == "delay_step"]
            if len(events) > 1:
                multi += 1
                fractions = [e["start_fraction"] for e in events]
                assert fractions == sorted(fractions)
                if any(e["value"] == 0.0 for e in events):
                    recovered += 1
            if events:
                assert events[0]["value"] > 0.0
    finally:
        env.close()
    assert multi > 0
    assert recovered > 0


def test_delay_recovery_restores_undelayed_actuation():
    env = PathFollowingEnv(mode="scheduled", training=False, calibration=CALIBRATION)
    try:
        env.reset(seed=1, options=scenario(
            evaluation_mode="transient",
            condition="delay",
            events=(
                DisturbanceEvent("delay_step", 0.5, 0.15),
                DisturbanceEvent("delay_step", 1.0, 0.0),
            ),
        ).to_options())
        seen = []
        for _ in range(80):
            env.step(np.zeros(3, dtype=np.float32))
            seen.append(env.actuator_delay_s)
        assert max(seen) == pytest.approx(0.15)
        assert seen[-1] == 0.0
    finally:
        env.close()
