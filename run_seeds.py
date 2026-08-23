"""Launch the full PPO training matrix as one concurrent process per run.

One process per (arm, seed), single-threaded each. Env stepping and the serial
PPO gradient update split training wall time roughly evenly, so parallel
environments inside one run saturate quickly while independent runs scale close
to linearly. Twenty single-threaded runs on twenty cores finish in about the
wall time of the slowest single run.

`run_training_matrix.py` remains the sequential, one-command-at-a-time form for
development. This launcher is for the frozen matrix on the compute server.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import config
from controller_profiles import profile_for


ARMS = (
    ("scheduled", "nominal"),
    ("scheduled", "disturbed"),
    ("direct", "nominal"),
    ("direct", "disturbed"),
)


def usable_cores() -> int:
    """Cores granted to this job, not cores present on the node.

    os.cpu_count() reports the whole compute node even inside a small SLURM
    allocation, which would silently oversubscribe the granted cores.
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdigit():
        return int(allocated)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def package_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")
    }


@dataclass
class Job:
    arm: str
    mode: str
    regime: str
    seed: int
    command: list[str]
    log_path: Path


def build_jobs(args: argparse.Namespace, seeds: tuple[int, ...]) -> list[Job]:
    output = Path(args.output).resolve()
    jobs: list[Job] = []
    for mode, regime in ARMS:
        if args.only_mode and mode != args.only_mode:
            continue
        if args.only_regime and regime != args.only_regime:
            continue
        profile = profile_for(mode)
        timesteps = args.scheduler_timesteps if mode == "scheduled" else args.direct_timesteps
        arm = f"ppo_{'scheduler' if mode == 'scheduled' else 'direct'}_{regime}"
        for seed in seeds:
            run_dir = output / arm / f"seed_{seed}"
            command = [
                sys.executable,
                str(config.ROOT / "train.py"),
                "--mode", mode,
                "--regime", regime,
                "--seed", str(seed),
                "--timesteps", str(timesteps or profile.default_timesteps),
                "--eval-freq", str(
                    args.scheduler_eval_freq if mode == "scheduled" else args.direct_eval_freq
                ),
                "--sampler-profile", args.sampler_profile,
                "--validation-manifest", str(Path(args.validation_manifest).resolve()),
                "--output", str(output),
                "--torch-threads", str(args.threads_per_job),
            ]
            if mode == "scheduled":
                command += ["--calibration", str(Path(args.scheduler_calibration).resolve())]
            jobs.append(Job(arm, mode, regime, seed, command, run_dir / "train.log"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "final"), default="final")
    parser.add_argument("--seeds", help="Comma-separated override, e.g. 11,23,37,53,71")
    parser.add_argument("--scheduler-timesteps", type=int, default=0, help="0 uses the profile default")
    parser.add_argument("--direct-timesteps", type=int, default=0, help="0 uses the profile default")
    parser.add_argument("--scheduler-eval-freq", type=int, default=25_000)
    parser.add_argument("--direct-eval-freq", type=int, default=100_000)
    parser.add_argument("--sampler-profile", choices=tuple(config.SAMPLER_PROFILES), default="dynamic")
    parser.add_argument(
        "--scheduler-calibration",
        default=str(config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"),
    )
    parser.add_argument(
        "--validation-manifest",
        default=str(config.ROOT / "manifests" / "validation.json"),
    )
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "models" / "final"))
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="0 uses the granted core count divided by --threads-per-job",
    )
    parser.add_argument("--only-mode", choices=("scheduled", "direct"))
    parser.add_argument("--only-regime", choices=("nominal", "disturbed"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads_per_job < 1:
        parser.error("--threads-per-job must be at least 1")

    if args.seeds:
        seeds = tuple(int(value) for value in args.seeds.split(","))
    elif args.phase == "final":
        seeds = config.CONFIG.final_training_seeds
    else:
        seeds = (config.CONFIG.development_seed,)

    for path in (args.scheduler_calibration, args.validation_manifest):
        if not Path(path).is_file():
            parser.error(f"missing required input: {path}")

    jobs = build_jobs(args, seeds)
    if not jobs:
        parser.error("arm selection is empty")

    cores = usable_cores()
    max_parallel = args.max_parallel or max(1, cores // args.threads_per_job)
    wanted = len(jobs) * args.threads_per_job

    print(f"host          : {platform.node()}")
    print(f"slurm job     : {os.environ.get('SLURM_JOB_ID', 'none')}")
    print(f"granted cores : {cores}")
    print(f"runs          : {len(jobs)} x {args.threads_per_job} thread(s) = {wanted} threads")
    print(f"concurrency   : {max_parallel} at a time")
    print(f"output        : {Path(args.output).resolve()}")
    print(f"sampler       : {args.sampler_profile}")
    print(f"seeds         : {','.join(str(s) for s in seeds)}")
    if wanted > cores:
        print(f"  NOTE: {wanted} threads exceeds {cores} cores; runs are queued, not oversubscribed")
    print()

    if args.dry_run:
        for job in jobs:
            print(subprocess.list2cmdline(job.command))
        print(f"\ndry run: {len(jobs)} commands")
        return

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    (output / "batch_manifest.json").write_text(
        json.dumps(
            {
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
                "phase": args.phase,
                "seeds": list(seeds),
                "arms": [f"{mode}/{regime}" for mode, regime in ARMS],
                "sampler_profile": args.sampler_profile,
                "threads_per_job": args.threads_per_job,
                "max_parallel": max_parallel,
                "granted_cores": cores,
                "host": platform.node(),
                "python": platform.python_version(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "packages": package_versions(),
                "config": config.CONFIG.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # torch honours --torch-threads, but numpy's BLAS reads these and would
    # otherwise spawn a full thread pool inside every one of the child runs.
    child_env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        child_env[name] = str(args.threads_per_job)

    pending = list(jobs)
    running: list[tuple[Job, subprocess.Popen, float]] = []
    failures: list[tuple[Job, int]] = []

    def launch(job: Job) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = job.log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            job.command,
            cwd=config.ROOT,
            env=child_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        process._log_handle = handle  # type: ignore[attr-defined]
        running.append((job, process, time.time()))
        print(f"  start {job.arm} seed {job.seed} (pid {process.pid})", flush=True)

    while pending or running:
        while pending and len(running) < max_parallel:
            launch(pending.pop(0))
        time.sleep(2.0)
        for entry in list(running):
            job, process, began = entry
            code = process.poll()
            if code is None:
                continue
            running.remove(entry)
            process._log_handle.close()  # type: ignore[attr-defined]
            minutes = (time.time() - began) / 60.0
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"  done  {job.arm} seed {job.seed} in {minutes:.1f} min: {status}", flush=True)
            if code != 0:
                failures.append((job, code))

    total = (time.time() - started_at) / 60.0
    print(f"\n{len(jobs) - len(failures)}/{len(jobs)} runs completed in {total:.1f} min")
    if failures:
        for job, code in failures:
            print(f"  {job.arm} seed {job.seed}: exit {code}, see {job.log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
