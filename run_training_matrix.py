"""Print or execute the one-seed development or frozen five-seed PPO matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--calibration", default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"))
    parser.add_argument("--validation-manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "models"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("timesteps must be positive")
    calibration = Path(args.calibration).resolve()
    validation = Path(args.validation_manifest).resolve()
    if not calibration.is_file() or not validation.is_file():
        parser.error("calibration and validation manifest must already exist")

    seeds = (
        (config.CONFIG.development_seed,)
        if args.phase == "development"
        else config.CONFIG.final_training_seeds
    )
    commands: list[list[str]] = []
    for mode in ("scheduled", "direct"):
        for regime in ("nominal", "disturbed"):
            for seed in seeds:
                commands.append(
                    [
                        sys.executable,
                        str(config.ROOT / "train.py"),
                        "--mode",
                        mode,
                        "--regime",
                        regime,
                        "--seed",
                        str(seed),
                        "--timesteps",
                        str(args.timesteps),
                        "--eval-freq",
                        str(args.eval_freq),
                        "--calibration",
                        str(calibration),
                        "--validation-manifest",
                        str(validation),
                        "--output",
                        str(Path(args.output).resolve()),
                    ]
                )
    for command in commands:
        print(subprocess.list2cmdline(command))
        if args.execute:
            subprocess.run(command, check=True, cwd=config.ROOT)
    if not args.execute:
        print(f"dry run: {len(commands)} commands; add --execute to run sequentially")


if __name__ == "__main__":
    main()
