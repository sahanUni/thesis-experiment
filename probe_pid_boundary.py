"""Small validation-only Kp sweep above the calibration search boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from calibration import GainCalibration, PIDGains
from calibrate_pid import _subset, score_gains
import config
from metrics import write_csv
from scenarios import ScenarioManifest


def unique_levels(selected_kp: float, requested: list[float]) -> list[float]:
    return list(dict.fromkeys([selected_kp, *requested]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--regime", choices=("nominal", "robust", "both"), default="both")
    parser.add_argument(
        "--calibration",
        default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"),
    )
    parser.add_argument("--nominal-kp", nargs="+", type=float, default=[50.0, 60.0, 70.0, 80.0])
    parser.add_argument("--robust-kp", nargs="+", type=float, default=[40.0, 50.0, 60.0, 70.0])
    parser.add_argument(
        "--output",
        default=str(config.ROOT / "artifacts" / "probes" / "kp_boundary_sensitivity.csv"),
    )
    args = parser.parse_args()

    manifest = ScenarioManifest.load(Path(args.manifest))
    calibration = GainCalibration.load(Path(args.calibration))
    rows = []
    requested_regimes = (
        ("nominal", calibration.nominal, args.nominal_kp),
        ("robust", calibration.robust, args.robust_kp),
    )
    for regime, selected, requested in requested_regimes:
        if args.regime not in {"both", regime}:
            continue
        scenarios = _subset(manifest, robust=regime == "robust")
        for kp in unique_levels(selected.kp, requested):
            gains = PIDGains(kp, selected.ki, selected.kd)
            rows.append(
                {
                    "regime": regime,
                    "scenario_count": len(scenarios),
                    "is_selected_kp": kp == selected.kp,
                    "kp": kp,
                    "ki_fixed": selected.ki,
                    "kd_fixed": selected.kd,
                    **score_gains(gains, scenarios),
                }
            )

    destination = Path(args.output).resolve()
    write_csv(destination, rows)
    print(f"Kp boundary sensitivity saved to {destination}")


if __name__ == "__main__":
    main()
