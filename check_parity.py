"""Confirm a new machine reproduces the laptop's plant before spending compute.

The two fixed-PID arms contain no neural network, so their results are a pure
function of the physics, the path geometry, and the calibration artifact. Their
completion counts must reproduce on any machine that is going to generate
official results, and their errors must agree to within a small relative band.
A difference beyond that means MuJoCo, numpy, or the calibration differs, and
nothing else is worth running until it is fixed.

    python check_parity.py --write-reference     # once, on the declared machine
    python check_parity.py                       # everywhere else

PPO arms are deliberately excluded: torch on a different CPU can reorder
floating-point reductions and shift an action in the last bits, which is
expected and is not evidence of a broken plant.

This is a sanity check, not a reproducibility guarantee. MuJoCo is not
bit-identical across CPU architectures, so a small drift between machines is
expected; PLAN.md handles that by requiring every final result to come from one
declared machine. What must not drift is completion, which is discrete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from calibration import GainCalibration
import config
from env import PathFollowingEnv
from rollout import run_episode
from scenarios import ScenarioManifest


DEFAULT_REFERENCE = config.ROOT / "artifacts" / "parity" / "reference.json"
FIXED_ARMS = ("pid_global_nominal", "pid_global_robust")

# Completion is discrete and is the primary outcome, so it must match exactly.
# The continuous metrics may not: MuJoCo is not bit-identical across CPUs, and
# a last-bit difference in one contact force amplifies over a 20-50 s
# closed-loop episode. A 1% relative band separates that drift from a genuine
# plant difference, which shows up as changed completion or as whole-percent
# error changes. Version pinning is enforced separately by tools/show_versions.
DEFAULT_TOLERANCE = 0.01

# For an arm that fails, J_FA is dominated by corridor * (T_max - t_end), a
# discontinuous term: a millisecond of difference in the corridor exit moves it
# by percent. Those arms are checked on completion and failure mode only.
FAILING_ARM_NOTE = "completion-only (failed arms: J_FA is dominated by exit-time padding)"


def measure(manifest_path: Path, calibration_path: Path) -> dict:
    manifest = ScenarioManifest.load(manifest_path)
    calibration = GainCalibration.load(calibration_path)
    scenarios = tuple(
        s for s in manifest.scenarios
        if (s.evaluation_mode == "stationary" and s.condition == "nominal")
        or (s.evaluation_mode == "transient" and s.condition == "combined")
    )
    if not scenarios:
        raise SystemExit("parity scenario selection is empty")

    results: dict[str, dict] = {}
    for arm in FIXED_ARMS:
        gains = calibration.nominal if arm.endswith("nominal") else calibration.robust
        env = PathFollowingEnv(mode="fixed", training=False, calibration=calibration,
                               fixed_gains=gains)
        try:
            rows = [run_episode(env, scenario).metrics for scenario in scenarios]
        finally:
            env.close()
        per_condition: dict[str, dict] = {}
        for condition in ("nominal", "combined"):
            subset = [
                row for row, scenario in zip(rows, scenarios)
                if scenario.condition == condition
            ]
            per_condition[condition] = {
                "episodes": len(subset),
                "completion_rate": float(np.mean([r["finished"] for r in subset])),
                "failure_adjusted_error_m": float(
                    np.mean([r["failure_adjusted_error_m"] for r in subset])
                ),
                "mean_distance_m": float(np.mean([r["mean_distance_m"] for r in subset])),
            }
        results[arm] = {"gains": [gains.kp, gains.ki, gains.kd], "conditions": per_condition}
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": manifest.digest(),
        "scenario_count": len(scenarios),
        "arms": results,
    }


def compare(reference: dict, measured: dict, tolerance: float) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    if reference.get("manifest_sha256") != measured.get("manifest_sha256"):
        problems.append(
            "scenario manifest differs from the reference; the comparison is meaningless"
        )
    for arm in FIXED_ARMS:
        want, got = reference["arms"].get(arm), measured["arms"].get(arm)
        if want is None or got is None:
            problems.append(f"{arm}: missing from reference or measurement")
            continue
        if want["gains"] != got["gains"]:
            problems.append(f"{arm}: gains {got['gains']} != reference {want['gains']}")
        for condition, expected in want["conditions"].items():
            actual = got["conditions"][condition]
            # Discrete outcomes first. These are the primary result and must be
            # identical; a difference here is a real plant difference.
            for key in ("episodes", "completion_rate"):
                if expected[key] != actual[key]:
                    problems.append(
                        f"{arm}/{condition}/{key}: {actual[key]!r} != reference {expected[key]!r}"
                    )
            if expected["completion_rate"] < 1.0:
                notes.append(f"{arm}/{condition}: {FAILING_ARM_NOTE}")
                continue
            for key in ("failure_adjusted_error_m", "mean_distance_m"):
                want_value, got_value = expected[key], actual[key]
                scale = max(abs(want_value), 1e-12)
                relative = abs(got_value - want_value) / scale
                if relative > tolerance:
                    problems.append(
                        f"{arm}/{condition}/{key}: {got_value:.9g} vs reference "
                        f"{want_value:.9g} ({relative:.2%} > {tolerance:.2%})"
                    )
                elif relative > 0:
                    notes.append(
                        f"{arm}/{condition}/{key}: {relative:.3%} drift, within tolerance"
                    )
    return problems, notes


def render(measured: dict) -> None:
    print(f"{'arm':<22}{'gains':<22}{'condition':<12}{'compl':>7}{'J_FA mm':>10}")
    for arm, payload in measured["arms"].items():
        gains = "(" + ", ".join(f"{g:g}" for g in payload["gains"]) + ")"
        for condition, values in payload["conditions"].items():
            print(f"{arm:<22}{gains:<22}{condition:<12}"
                  f"{values['completion_rate']:7.2f}"
                  f"{values['failure_adjusted_error_m'] * 1000:10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument(
        "--calibration",
        default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"),
    )
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Relative band on continuous metrics for fully-completing arms",
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Record this machine as the declared reference instead of comparing",
    )
    args = parser.parse_args()

    measured = measure(Path(args.manifest).resolve(), Path(args.calibration).resolve())
    measured["machine"] = {
        "node": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    render(measured)

    reference_path = Path(args.reference).resolve()
    if args.write_reference:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
        print(f"\nreference written to {reference_path}")
        print(f"declared machine: {measured['machine']['node']}")
        return

    if not reference_path.is_file():
        raise SystemExit(f"no reference at {reference_path}; run --write-reference first")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    problems, notes = compare(reference, measured, args.tolerance)
    print(f"\nreference machine: {reference.get('machine', {}).get('node', 'unknown')}")
    print(f"this machine     : {measured['machine']['node']}")
    print(f"tolerance        : {args.tolerance:.2%} relative on completed arms")
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print(f"\nPARITY FAILED ({len(problems)} difference(s)):")
        for problem in problems:
            print(f"  {problem}")
        print("\nThe fixed-PID arms use no neural network. Changed completion, or")
        print("whole-percent error changes, mean the plant itself differs: check")
        print("mujoco and numpy with tools/show_versions.py before training.")
        sys.exit(1)
    print("\nPARITY OK: completion is identical and the continuous metrics agree")
    print("within tolerance. Remember that PLAN.md requires every final result,")
    print("fixed PID included, to come from one declared machine.")


if __name__ == "__main__":
    main()
