"""Deterministic paired evaluation over a frozen scenario manifest."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import time
from typing import Any

from calibration import GainCalibration, PIDGains
import config
from env import PathFollowingEnv
from metrics import summarize, write_csv
from rollout import run_episode
from scenarios import ScenarioManifest


SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def machine_description() -> dict[str, Any]:
    """Identify the host, because on a cluster the node varies per allocation."""
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "processor": model or platform.processor(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_models(entries: list[str]) -> list[tuple[str, int, Path]]:
    models = []
    for entry in entries:
        try:
            identity, raw_path = entry.split("=", 1)
            arm, raw_seed = identity.rsplit(":", 1)
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError("model format is ARM:SEED=PATH") from exc
        if not SAFE_NAME.fullmatch(arm) or not arm.startswith(("ppo_scheduler_", "ppo_direct_")):
            raise ValueError(f"invalid learned arm name: {arm}")
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        models.append((arm, seed, path))
    return models


def _trace_path(root: Path, controller: str, scenario_id: str) -> Path:
    if not SAFE_NAME.fullmatch(controller) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", scenario_id):
        raise ValueError("unsafe controller or scenario identifier")
    return root / controller / f"{scenario_id}.csv"


def main() -> None:
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--calibration", default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"))
    parser.add_argument(
        "--scheduler-calibration",
        default=str(config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"),
    )
    parser.add_argument("--model", action="append", default=[], metavar="ARM:SEED=PATH")
    parser.add_argument("--models-root", help="Require and load the complete four-arm, five-seed matrix")
    parser.add_argument("--per-path-gains")
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "results"))
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--save-traces", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    calibration_path = Path(args.calibration).resolve()
    scheduler_calibration_path = Path(args.scheduler_calibration).resolve()
    manifest = ScenarioManifest.load(manifest_path)
    calibration = GainCalibration.load(calibration_path)
    scheduler_calibration = GainCalibration.load(scheduler_calibration_path)
    model_entries = list(args.model)
    if args.models_root:
        root = Path(args.models_root).resolve()
        for arm in (
            "ppo_scheduler_nominal",
            "ppo_scheduler_disturbed",
            "ppo_direct_nominal",
            "ppo_direct_disturbed",
        ):
            for seed in config.CONFIG.final_training_seeds:
                path = root / arm / f"seed_{seed}" / "best_model.zip"
                model_entries.append(f"{arm}:{seed}={path}")
    models = parse_models(model_entries)
    output = Path(args.output).resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, output / "scenario_manifest.json")
    shutil.copy2(calibration_path, output / "calibration.json")
    shutil.copy2(scheduler_calibration_path, output / "scheduler_calibration.json")

    controllers: list[tuple[str, int | None, str, GainCalibration, PIDGains | None, Any]] = [
        ("pid_global_nominal", None, "fixed", calibration, calibration.nominal, None),
        ("pid_global_robust", None, "fixed", calibration, calibration.robust, None),
    ]
    for arm, seed, path in models:
        mode = "scheduled" if arm.startswith("ppo_scheduler_") else "direct"
        model = PPO.load(path)
        controller_calibration = scheduler_calibration if mode == "scheduled" else calibration
        controllers.append((arm, seed, mode, controller_calibration, None, model))

    per_path: dict[str, Any] | None = None
    if args.per_path_gains:
        per_path = json.loads(Path(args.per_path_gains).read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    expected_primary: set[tuple[str, str, str]] = set()
    traces_dir = output / "traces"
    for controller, training_seed, mode, controller_calibration, gains, model in controllers:
        env = PathFollowingEnv(
            mode=mode,
            training=False,
            calibration=controller_calibration,
            fixed_gains=gains,
            physics_trace=args.save_traces,
        )
        try:
            predictor = None if model is None else lambda obs, loaded=model: loaded.predict(obs, deterministic=True)[0]
            for scenario in manifest.scenarios:
                result = run_episode(env, scenario, predictor)
                row = {
                    "controller": controller,
                    "training_seed": "" if training_seed is None else training_seed,
                    "evaluation_mode": scenario.evaluation_mode,
                    "condition": scenario.condition,
                    "path_kind": scenario.path["kind"],
                    "target_speed": scenario.target_speed,
                    **result.metrics,
                }
                rows.append(row)
                expected_primary.add((controller, str(training_seed or ""), scenario.scenario_id))
                if args.save_traces:
                    write_csv(_trace_path(traces_dir, f"{controller}_seed{training_seed or 0}", scenario.scenario_id), result.trace)
        finally:
            env.close()

    if per_path is not None:
        for regime in ("nominal", "robust"):
            controller = f"pid_per_path_{regime}"
            for scenario in manifest.scenarios:
                gain_data = per_path.get(scenario.path["kind"], {}).get(regime)
                if gain_data is None:
                    continue
                env = PathFollowingEnv(
                    mode="fixed",
                    training=False,
                    calibration=calibration,
                    fixed_gains=PIDGains(**gain_data),
                    physics_trace=args.save_traces,
                )
                try:
                    result = run_episode(env, scenario)
                finally:
                    env.close()
                rows.append({
                    "controller": controller,
                    "training_seed": "",
                    "evaluation_mode": scenario.evaluation_mode,
                    "condition": scenario.condition,
                    "path_kind": scenario.path["kind"],
                    "target_speed": scenario.target_speed,
                    **result.metrics,
                })
                if args.save_traces:
                    write_csv(
                        _trace_path(traces_dir, f"{controller}_seed0", scenario.scenario_id),
                        result.trace,
                    )

    primary_rows = [
        row for row in rows if not str(row["controller"]).startswith("pid_per_path_")
    ]
    actual_primary = {
        (str(row["controller"]), str(row["training_seed"]), str(row["scenario_id"]))
        for row in primary_rows
    }
    if actual_primary != expected_primary or len(primary_rows) != len(expected_primary):
        raise RuntimeError("primary controller/scenario result matrix is incomplete or duplicated")
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.digest(),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "scheduler_calibration": str(scheduler_calibration_path),
        "scheduler_calibration_sha256": sha256(scheduler_calibration_path),
        "controllers": [
            {
                "arm": arm,
                "training_seed": seed,
                "model": str(path),
                "model_sha256": sha256(path),
                "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
            }
            for (arm, seed, path), model in zip(models, [item[5] for item in controllers[2:]])
        ],
        "config": config.CONFIG.to_dict(),
        "scheduler_gain_rate_limit_per_s": (
            0.5
            * (
                scheduler_calibration.upper.as_array()
                - scheduler_calibration.lower.as_array()
            )
        ).tolist(),
        "save_traces": args.save_traces,
        "packages": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")},
        # PLAN.md requires final results from one declared machine, and on a
        # cluster the node varies per allocation. Record which one produced
        # this run so a mixed-node result set is detectable afterwards.
        "machine": machine_description(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"evaluated {len(controllers)} controllers on {len(manifest.scenarios)} paired scenarios")
    print(f"results saved to {output}")


if __name__ == "__main__":
    main()
