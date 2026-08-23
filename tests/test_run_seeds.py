import argparse

import config
import run_seeds


def namespace(**overrides):
    values = {
        "output": str(config.ROOT / "artifacts" / "models" / "final"),
        "scheduler_timesteps": 0,
        "direct_timesteps": 0,
        "scheduler_eval_freq": 25_000,
        "direct_eval_freq": 100_000,
        "sampler_profile": "dynamic",
        "scheduler_calibration": str(
            config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"
        ),
        "validation_manifest": str(config.ROOT / "manifests" / "validation.json"),
        "threads_per_job": 1,
        "only_mode": None,
        "only_regime": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_final_matrix_is_four_arms_by_five_seeds():
    jobs = run_seeds.build_jobs(namespace(), config.CONFIG.final_training_seeds)
    assert len(jobs) == 20
    assert {job.arm for job in jobs} == {
        name for name in config.CONTROLLERS if name.startswith("ppo_")
    }
    assert {job.seed for job in jobs} == set(config.CONFIG.final_training_seeds)


def test_each_run_gets_its_own_log_path():
    jobs = run_seeds.build_jobs(namespace(), config.CONFIG.final_training_seeds)
    assert len({job.log_path for job in jobs}) == len(jobs)


def test_profile_budgets_are_used_when_not_overridden():
    jobs = run_seeds.build_jobs(namespace(), (11,))
    budgets = {
        job.mode: int(job.command[job.command.index("--timesteps") + 1]) for job in jobs
    }
    assert budgets["scheduled"] == 300_000
    assert budgets["direct"] == 2_000_000


def test_only_scheduler_runs_carry_the_scheduler_calibration():
    for job in run_seeds.build_jobs(namespace(), (11,)):
        assert ("--calibration" in job.command) == (job.mode == "scheduled")


def test_every_run_carries_the_declared_sampler_profile():
    for job in run_seeds.build_jobs(namespace(sampler_profile="single_step"), (11,)):
        index = job.command.index("--sampler-profile")
        assert job.command[index + 1] == "single_step"


def test_arm_filters_narrow_the_matrix():
    jobs = run_seeds.build_jobs(
        namespace(only_mode="direct", only_regime="disturbed"),
        config.CONFIG.final_training_seeds,
    )
    assert {job.arm for job in jobs} == {"ppo_direct_disturbed"}
    assert len(jobs) == 5


def test_usable_cores_prefers_the_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "20")
    assert run_seeds.usable_cores() == 20
