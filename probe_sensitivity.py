"""Training-only sweeps for delay, noise, and derivative-filter decisions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import config
from calibration import GainCalibration, PIDGains
from env import PathFollowingEnv
from metrics import write_csv
from rollout import run_episode
from scenarios import Scenario, ScenarioManifest


DELAY_LEVELS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
NOISE_LEVELS = (0.0, 0.0001, 0.0003, 0.0005, 0.0010)
FILTER_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10)


def training_bases(manifest: ScenarioManifest) -> list[Scenario]:
    return [
        scenario
        for scenario in manifest.scenarios
        if scenario.evaluation_mode == "stationary"
        and scenario.condition == "nominal"
        and scenario.target_speed == 0.5
    ]


def gain_probes(calibration: GainCalibration) -> dict[str, PIDGains]:
    low = calibration.lower.as_array()
    high = calibration.upper.as_array()
    return {
        "low": PIDGains.from_iterable(low + 0.25 * (high - low)),
        "nominal": calibration.nominal,
        "high": PIDGains.from_iterable(low + 0.75 * (high - low)),
    }


def modified(base: Scenario, probe: str, value: float) -> Scenario:
    return replace(
        base,
        scenario_id=f"probe-{probe}-{value:g}-{base.path['kind']}",
        condition="delay" if probe == "delay" else "noise",
        initial_delay_s=value if probe == "delay" else 0.0,
        initial_noise_std_m=value if probe in {"noise", "filter"} else 0.0,
    )


def run_probe(
    probe: str,
    levels: tuple[float, ...],
    bases: list[Scenario],
    gains_by_name: dict[str, PIDGains],
    calibration: GainCalibration,
) -> list[dict[str, Any]]:
    rows = []
    for level in levels:
        tau = level if probe == "filter" else config.CONFIG.derivative_filter_tau_s
        scenario_value = config.CONFIG.noise_severity_m if probe == "filter" else level
        for gain_name, gains in gains_by_name.items():
            env = PathFollowingEnv(
                mode="fixed",
                training=False,
                calibration=calibration,
                fixed_gains=gains,
                derivative_filter_tau_s=tau,
            )
            try:
                metrics = [
                    run_episode(env, modified(base, probe, scenario_value)).metrics
                    for base in bases
                ]
            finally:
                env.close()
            rows.append(
                {
                    "probe": probe,
                    "level": level,
                    "gain_probe": gain_name,
                    "episodes": len(metrics),
                    "completion_rate": float(np.mean([row["finished"] for row in metrics])),
                    "failure_adjusted_error_m": float(np.mean([row["failure_adjusted_error_m"] for row in metrics])),
                    "mean_distance_m": float(np.mean([row["mean_distance_m"] for row in metrics])),
                    "steer_total_variation_per_s": float(np.mean([row["steer_total_variation_per_s"] for row in metrics])),
                    "wheel_saturation_fraction": float(np.mean([row["wheel_saturation_fraction"] for row in metrics])),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=("delay", "noise", "filter", "all"), default="all")
    parser.add_argument("--manifest", default=str(config.ROOT / "manifests" / "train.json"))
    parser.add_argument("--calibration", default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"))
    parser.add_argument("--development-default", action="store_true")
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "probes"))
    args = parser.parse_args()

    manifest = ScenarioManifest.load(Path(args.manifest))
    calibration = (
        GainCalibration.development_default()
        if args.development_default
        else GainCalibration.load(Path(args.calibration))
    )
    bases = training_bases(manifest)
    if not bases:
        parser.error("training manifest contains no nominal 0.5 m/s probe scenarios")
    selected = ("delay", "noise", "filter") if args.probe == "all" else (args.probe,)
    levels = {"delay": DELAY_LEVELS, "noise": NOISE_LEVELS, "filter": FILTER_LEVELS}
    output = Path(args.output).resolve()
    for probe in selected:
        destination = output / f"{probe}_sensitivity.csv"
        write_csv(destination, run_probe(probe, levels[probe], bases, gain_probes(calibration), calibration))
        print(f"{probe} sensitivity saved to {destination}")


if __name__ == "__main__":
    main()
