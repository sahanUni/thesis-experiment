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

import config
from calibration import GainCalibration
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
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--calibration", default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"))
    parser.add_argument("--validation-manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=0, help="Development-only cap; 0 uses the full manifest")
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "models"))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_freq <= 0:
        parser.error("timesteps and eval-freq must be positive")

    calibration_path = Path(args.calibration).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    calibration = GainCalibration.load(calibration_path)
    manifest = ScenarioManifest.load(manifest_path)
    validation = manifest.scenarios[: args.validation_limit or None]
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
        verbose=1,
        **config.PPO_KWARGS,
    )
    callback = ManifestEvalCallback()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
        model.save(run_dir / "final_model")
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
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "python": platform.python_version(),
        "config": config.CONFIG.to_dict(),
        "gain_rate_limit_per_s": config.GAIN_RATE_LIMIT_PER_S,
        "ppo": config.PPO_KWARGS,
        "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "packages": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")},
        "source_sha256": {
            str(path.relative_to(config.ROOT)): sha256(path)
            for path in (
                config.ROOT / "train.py",
                config.ROOT / "env.py",
                config.ROOT / "config.py",
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
