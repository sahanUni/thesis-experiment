import math

import controller_profiles


def test_scheduler_profile_matches_blind_ppo_checkpoint_contract():
    profile = controller_profiles.profile_for("scheduled")
    assert profile.observation_size == 170
    assert profile.action_size == 3
    assert profile.default_timesteps == 300_000
    assert profile.ppo_kwargs["gamma"] == 0.99
    assert profile.ppo_kwargs["gae_lambda"] == 0.95
    assert profile.ppo_kwargs["n_steps"] == 2048
    assert profile.ppo_kwargs["batch_size"] == 64
    assert profile.ppo_kwargs["ent_coef"] == 0.0
    assert profile.failure_penalty == 60.0
    assert profile.ppo_kwargs["policy_kwargs"] == {"net_arch": [64, 64]}
    assert profile.checkpoint_selection == "manifest_completion_then_failure_adjusted_error"


def test_direct_profile_matches_50hz_end_to_end_checkpoint_contract():
    profile = controller_profiles.profile_for("direct")
    assert profile.observation_size == 130
    assert profile.action_size == 1
    assert profile.default_timesteps == 2_000_000
    assert profile.default_eval_freq == 100_000
    assert profile.ppo_kwargs["gamma"] == 0.999
    assert profile.ppo_kwargs["gae_lambda"] == 0.995
    assert profile.ppo_kwargs["n_steps"] == 2048
    assert profile.ppo_kwargs["batch_size"] == 64
    assert profile.ppo_kwargs["ent_coef"] == 0.005
    assert profile.failure_penalty == 20.0
    assert profile.ppo_kwargs["policy_kwargs"]["net_arch"] == [64, 64]
    assert profile.ppo_kwargs["policy_kwargs"]["log_std_init"] == math.log(0.1)
    assert profile.checkpoint_selection == "manifest_completion_then_failure_adjusted_error"


def test_disturbance_protocol_uses_150ms_delay():
    import config

    assert config.CONFIG.delay_severity_s == 0.15
    assert config.TRAIN_DELAY_RANGE_S == (0.0, 0.15)


def test_scheduler_tracking_scale_still_responds_at_the_delayed_operating_point():
    profile = controller_profiles.profile_for("scheduled")
    # Under the 0.15 s delay the plant tracks at roughly 0.02-0.035 m. The
    # saturating tracking term must still have gradient there instead of
    # sitting pinned at its maximum, which is what stalled disturbed training.
    x = 0.030 / profile.tracking_scale_m
    assert x**2 / (1.0 + x**2) < 0.75
