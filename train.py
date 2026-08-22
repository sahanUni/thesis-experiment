"""Train one PPO arm with deterministic manifest-based checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
import platform
from pathlib import Path
import time

import numpy as np
import torch

import config
from calibration import GainCalibration
from controller_profiles import profile_for
from env import PathFollowingEnv
from rollout import run_episode
from scenarios import ScenarioManifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("scheduled", "direct"), required=True)
    parser.add_argument("--regime", choices=("nominal", "disturbed"), required=True)
    parser.add_argument("--seed", type=int, default=config.CONFIG.development_seed)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--calibration")
    parser.add_argument("--validation-manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--validation-limit", type=int, default=0, help="Development-only cap; 0 uses the full manifest")
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "models" / "protocol_v2"))
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    profile = profile_for(args.mode)
    args.timesteps = args.timesteps or profile.default_timesteps
    args.eval_freq = profile.default_eval_freq if args.eval_freq is None else args.eval_freq
    if args.timesteps <= 0 or args.eval_freq <= 0:
        parser.error("timesteps and eval-freq must be positive")
    torch.set_num_threads(max(1, args.torch_threads))

    default_calibration = (
        config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"
        if args.mode == "scheduled"
        else config.ROOT / "artifacts" / "calibration" / "calibration.json"
    )
    calibration_path = Path(args.calibration or default_calibration).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    calibration = GainCalibration.load(calibration_path)
    manifest = ScenarioManifest.load(manifest_path)
    validation = tuple(
        scenario
        for scenario in manifest.scenarios
        if (
            scenario.evaluation_mode == "stationary"
            and scenario.condition == "nominal"
        )
        or (
            scenario.evaluation_mode == "transient"
            and scenario.condition == "combined"
        )
    )
    validation = validation[: args.validation_limit or None]
    if not validation:
        parser.error("checkpoint scenario selection is empty")
    arm = f"ppo_{'scheduler' if args.mode == 'scheduled' else 'direct'}_{args.regime}"
    run_dir = Path(args.output).resolve() / arm / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def make_env(training: bool) -> PathFollowingEnv:
        return PathFollowingEnv(
            mode=args.mode,
            training=training,
            disturbance_training=args.regime == "disturbed",
            calibration=calibration,
        )

    class ManifestEvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=1)
            self.best_key = (-np.inf, -np.inf)
            self.history_path = run_dir / "validation_history.jsonl"
            self.last_eval_t = -1

        def _on_step(self) -> bool:
            if self.num_timesteps % args.eval_freq != 0:
                return True
            self._evaluate()
            return True

        def _on_training_end(self) -> None:
            if self.last_eval_t != self.num_timesteps:
                self._evaluate()

        def _evaluate(self) -> None:
            evaluation_env = make_env(training=False)
            try:
                rows = [
                    run_episode(
                        evaluation_env,
                        scenario,
                        lambda obs: self.model.predict(obs, deterministic=True)[0],
                    ).metrics
                    for scenario in validation
                ]
            finally:
                evaluation_env.close()
            completion = float(np.mean([row["finished"] for row in rows]))
            adjusted_error = float(np.mean([row["failure_adjusted_error_m"] for row in rows]))
            key = (completion, -adjusted_error)
            record = {
                "timesteps": self.num_timesteps,
                "completion_rate": completion,
                "failure_adjusted_error_m": adjusted_error,
                "episodes": len(rows),
            }
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self.last_eval_t = self.num_timesteps
            if key > self.best_key:
                self.best_key = key
                self.model.save(run_dir / "best_model")

    env = Monitor(make_env(training=True), filename=str(run_dir / "monitor.csv"))
    started = time.time()
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tensorboard"),
        device="cpu",
        verbose=1,
        **profile.ppo_kwargs,
    )
    callback = ManifestEvalCallback()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
        model.save(run_dir / "final_model")
        if not (run_dir / "best_model.zip").exists():
            model.save(run_dir / "best_model")
    finally:
        env.close()
    metadata = {
        "arm": arm,
        "mode": args.mode,
        "regime": args.regime,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "wall_time_s": time.time() - started,
        "validation_manifest": str(manifest_path),
        "validation_manifest_sha256": manifest.digest(),
        "checkpoint_scenarios": [scenario.scenario_id for scenario in validation],
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "python": platform.python_version(),
        "config": config.CONFIG.to_dict(),
        "gain_rate_limit_per_s": (
            (0.5 * (calibration.upper.as_array() - calibration.lower.as_array())).tolist()
            if args.mode == "scheduled"
            else None
        ),
        "controller_profile": {
            "observation_size": profile.observation_size,
            "action_size": profile.action_size,
            "tracking_scale_m": profile.tracking_scale_m,
            "tracking_weight": profile.tracking_weight,
            "action_smoothness_weight": profile.action_smoothness_weight,
            "finish_bonus": profile.finish_bonus,
            "failure_penalty": profile.failure_penalty,
            "checkpoint_selection": profile.checkpoint_selection,
        },
        "ppo": profile.ppo_kwargs,
        "torch_threads": max(1, args.torch_threads),
        "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "packages": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")},
        "source_sha256": {
            str(path.relative_to(config.ROOT)): sha256(path)
            for path in (
                config.ROOT / "train.py",
                config.ROOT / "env.py",
                config.ROOT / "config.py",
                config.ROOT / "controller_profiles.py",
                config.ROOT / "core" / "dynamics.py",
                config.ROOT / "core" / "paths.py",
                config.ROOT / "core" / "pid.py",
                config.ROOT / "core" / "car_model.xml",
            )
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved {arm}, seed {args.seed}, to {run_dir}")


if __name__ == "__main__":
    main()
