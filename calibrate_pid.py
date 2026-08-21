"""Tune strong global and per-path fixed PID baselines by differential evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

import config
from calibration import GainCalibration, PIDGains
from env import PathFollowingEnv
from metrics import write_csv
from rollout import run_episode
from scenarios import Scenario, ScenarioManifest


SEARCH_BOUNDS = ((0.25, 50.0), (0.0, 5.0), (0.0, 10.0))
MIN_SPAN = np.asarray((2.0, 0.5, 0.5), dtype=np.float64)
PRACTICAL_ERROR_REL_TOL = 0.005
VALIDATION_POOL_REL_TOL = 0.02
SCHEDULER_MARGIN_FRACTION = 0.15


def _subset(manifest: ScenarioManifest, robust: bool, path_kind: str | None = None) -> list[Scenario]:
    selected = []
    for scenario in manifest.scenarios:
        if path_kind is not None and scenario.path["kind"] != path_kind:
            continue
        nominal = scenario.evaluation_mode == "stationary" and scenario.condition == "nominal"
        if not robust and not nominal:
            continue
        if robust:
            medium_stationary = scenario.evaluation_mode == "stationary" and scenario.target_speed == 0.5
            medium_transient_combined = (
                scenario.evaluation_mode == "transient"
                and scenario.condition == "combined"
                and scenario.target_speed == 0.5
            )
            if not (nominal or medium_stationary or medium_transient_combined):
                continue
        selected.append(scenario)
    if not selected:
        raise ValueError("calibration scenario selection is empty")
    return selected


def score_gains(gains: PIDGains, scenarios: list[Scenario]) -> dict[str, float]:
    env = PathFollowingEnv(mode="fixed", training=False, fixed_gains=gains)
    try:
        rows = [run_episode(env, scenario).metrics for scenario in scenarios]
    finally:
        env.close()
    completion = float(np.mean([row["finished"] for row in rows]))
    return {
        "completion_rate": completion,
        "failure_adjusted_error_m": float(np.mean([row["failure_adjusted_error_m"] for row in rows])),
        "progress_pct": float(np.mean([row["progress_pct"] for row in rows])),
        "mean_distance_m": float(np.mean([row["mean_distance_m"] for row in rows])),
        "steer_total_variation_per_s": float(np.mean([row["steer_total_variation_per_s"] for row in rows])),
        "wheel_saturation_fraction": float(np.mean([row["wheel_saturation_fraction"] for row in rows])),
        "gain_l1": gains.kp + gains.ki + gains.kd,
    }


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"candidate cache does not exist: {path}")
    unique: dict[tuple[float, float, float], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = tuple(np.round((row["kp"], row["ki"], row["kd"]), 8))
        unique[key] = row
    if not unique:
        raise ValueError(f"candidate cache is empty: {path}")
    return list(unique.values())


def completion_band(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot select from an empty candidate set")
    best_completion = max(row["completion_rate"] for row in rows)
    return [row for row in rows if row["completion_rate"] >= best_completion - 0.01]


def practical_error_band(
    rows: list[dict[str, Any]],
    relative_tolerance: float = PRACTICAL_ERROR_REL_TOL,
) -> list[dict[str, Any]]:
    eligible = completion_band(rows)
    best_error = min(row["failure_adjusted_error_m"] for row in eligible)
    return [
        row
        for row in eligible
        if row["failure_adjusted_error_m"] <= best_error * (1.0 + relative_tolerance)
    ]


def select_practical_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        practical_error_band(rows),
        key=lambda row: (
            row["steer_total_variation_per_s"],
            row.get("gain_l1", row["kp"] + row["ki"] + row["kd"]),
            row["failure_adjusted_error_m"],
        ),
    )


def validation_shortlist(
    rows: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[tuple[str, dict[str, Any]]]:
    eligible = completion_band(rows)
    practical = practical_error_band(eligible)
    expanded = practical_error_band(eligible, VALIDATION_POOL_REL_TOL)
    proposals = [
        ("best_error", min(eligible, key=lambda row: row["failure_adjusted_error_m"])),
        ("smoothest_practical", min(practical, key=lambda row: row["steer_total_variation_per_s"])),
        (
            "lowest_gain_practical",
            min(practical, key=lambda row: row.get("gain_l1", row["kp"] + row["ki"] + row["kd"])),
        ),
        ("smoothest_nearby", min(expanded, key=lambda row: row["steer_total_variation_per_s"])),
        (
            "lowest_gain_nearby",
            min(expanded, key=lambda row: row.get("gain_l1", row["kp"] + row["ki"] + row["kd"])),
        ),
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[tuple[float, float, float]] = set()

    def add(role: str, row: dict[str, Any]) -> None:
        key = tuple(np.round((row["kp"], row["ki"], row["kd"]), 6))
        if key not in used and len(selected) < top_k:
            used.add(key)
            selected.append((role, row))

    for role, row in proposals:
        add(role, row)

    search_span = np.asarray([high - low for low, high in SEARCH_BOUNDS], dtype=np.float64)
    while len(selected) < min(top_k, len(expanded)):
        remaining = [
            row
            for row in expanded
            if tuple(np.round((row["kp"], row["ki"], row["kd"]), 6)) not in used
        ]
        if not remaining:
            break

        def separation(row: dict[str, Any]) -> float:
            point = np.asarray((row["kp"], row["ki"], row["kd"]), dtype=np.float64)
            return min(
                float(np.linalg.norm((point - np.asarray((other["kp"], other["ki"], other["kd"]))) / search_span))
                for _, other in selected
            )

        add(
            "diverse_nearby",
            max(remaining, key=lambda row: (separation(row), -row["failure_adjusted_error_m"])),
        )
    return selected


def optimize(
    scenarios: list[Scenario],
    *,
    seed: int,
    maxiter: int,
    popsize: int,
    cache_path: Path | None = None,
) -> tuple[PIDGains, list[dict[str, Any]]]:
    cache: dict[tuple[float, float, float], dict[str, Any]] = {}
    if cache_path is not None and cache_path.exists():
        for row in load_candidate_rows(cache_path):
            key = tuple(np.round((row["kp"], row["ki"], row["kd"]), 8))
            cache[key] = row

    def objective(values: np.ndarray) -> float:
        key = tuple(np.round(values, 8))
        if key not in cache:
            gains = PIDGains.from_iterable(values)
            metrics = score_gains(gains, scenarios)
            cache[key] = {"evaluation_index": len(cache), "kp": gains.kp, "ki": gains.ki, "kd": gains.kd, **metrics}
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(cache[key]) + "\n")
        row = cache[key]
        return 100.0 * (1.0 - row["completion_rate"]) + row["failure_adjusted_error_m"]

    convergence_path = (
        cache_path.with_name(cache_path.stem + "_convergence.jsonl")
        if cache_path is not None
        else None
    )
    generation = 0
    if convergence_path is not None and convergence_path.exists():
        existing = [json.loads(line) for line in convergence_path.read_text(encoding="utf-8").splitlines()]
        generation = max((int(row["generation"]) for row in existing), default=-1) + 1

    def log_convergence(best: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        if convergence_path is not None:
            with convergence_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "generation": generation,
                            "convergence": float(convergence),
                            "best_kp": float(best[0]),
                            "best_ki": float(best[1]),
                            "best_kd": float(best[2]),
                            "objective": objective(best),
                        }
                    )
                    + "\n"
                )
        generation += 1
        return False

    differential_evolution(
        objective,
        SEARCH_BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        updating="immediate",
        workers=1,
        callback=log_convergence,
    )
    rows = list(cache.values())
    chosen = select_practical_candidate(rows)
    return PIDGains(chosen["kp"], chosen["ki"], chosen["kd"]), rows


def confirm_on_validation(
    rows: list[dict[str, Any]],
    scenarios: list[Scenario],
    *,
    top_k: int = 5,
) -> tuple[PIDGains, list[dict[str, Any]]]:
    checked = []
    for role, candidate in validation_shortlist(rows, top_k=top_k):
        gains = PIDGains(candidate["kp"], candidate["ki"], candidate["kd"])
        checked.append(
            {
                "selection_role": role,
                "kp": gains.kp,
                "ki": gains.ki,
                "kd": gains.kd,
                **score_gains(gains, scenarios),
            }
        )
    chosen = select_practical_candidate(checked)
    return PIDGains(chosen["kp"], chosen["ki"], chosen["kd"]), checked


def scheduler_bounds(
    nominal: PIDGains,
    robust: PIDGains,
) -> tuple[PIDGains, PIDGains]:
    samples = np.asarray([nominal.as_array(), robust.as_array()], dtype=np.float64)
    low = np.min(samples, axis=0)
    high = np.max(samples, axis=0)
    span = np.maximum(high - low, MIN_SPAN)
    margin = span * SCHEDULER_MARGIN_FRACTION
    search_low = np.asarray([bound[0] for bound in SEARCH_BOUNDS])
    search_high = np.asarray([bound[1] for bound in SEARCH_BOUNDS])
    low = np.maximum(low - margin, search_low)
    high = np.minimum(high + margin, search_high)
    return PIDGains.from_iterable(low), PIDGains.from_iterable(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(config.ROOT / "manifests" / "train.json"))
    parser.add_argument("--validation-manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--output", default=str(config.ROOT / "artifacts" / "calibration"))
    parser.add_argument("--seed", type=int, default=config.CONFIG.development_seed)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--per-path", action="store_true")
    parser.add_argument(
        "--reselect-only",
        action="store_true",
        help="reuse cached candidate evaluations without running differential evolution",
    )
    args = parser.parse_args()

    manifest = ScenarioManifest.load(Path(args.manifest))
    validation_manifest = ScenarioManifest.load(Path(args.validation_manifest))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    nominal_cache = output / "nominal_candidates.jsonl"
    robust_cache = output / "robust_candidates.jsonl"
    if args.reselect_only:
        nominal_rows = load_candidate_rows(nominal_cache)
        robust_rows = load_candidate_rows(robust_cache)
    else:
        _, nominal_rows = optimize(
            _subset(manifest, robust=False),
            seed=args.seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
            cache_path=nominal_cache,
        )
        _, robust_rows = optimize(
            _subset(manifest, robust=True),
            seed=args.seed + 1,
            maxiter=args.maxiter,
            popsize=args.popsize,
            cache_path=robust_cache,
        )
    nominal, nominal_validation = confirm_on_validation(
        nominal_rows, _subset(validation_manifest, robust=False)
    )
    robust, robust_validation = confirm_on_validation(
        robust_rows, _subset(validation_manifest, robust=True)
    )
    lower, upper = scheduler_bounds(nominal, robust)
    calibration = GainCalibration(nominal=nominal, robust=robust, lower=lower, upper=upper)
    calibration.save(output / "calibration.json")
    write_csv(output / "nominal_candidates.csv", nominal_rows)
    write_csv(output / "robust_candidates.csv", robust_rows)
    write_csv(output / "nominal_validation.csv", nominal_validation)
    write_csv(output / "robust_validation.csv", robust_validation)

    if args.per_path:
        per_path: dict[str, Any] = {}
        kinds = sorted({scenario.path["kind"] for scenario in manifest.scenarios if scenario.path["kind"] != "random_curvature"})
        for index, kind in enumerate(kinds):
            nominal_path_cache = output / f"per_path_{kind}_nominal.jsonl"
            robust_path_cache = output / f"per_path_{kind}_robust.jsonl"
            if args.reselect_only:
                n_row = select_practical_candidate(load_candidate_rows(nominal_path_cache))
                r_row = select_practical_candidate(load_candidate_rows(robust_path_cache))
                n = PIDGains(n_row["kp"], n_row["ki"], n_row["kd"])
                r = PIDGains(r_row["kp"], r_row["ki"], r_row["kd"])
            else:
                n, _ = optimize(
                    _subset(manifest, robust=False, path_kind=kind),
                    seed=args.seed + 10 + index * 2,
                    maxiter=args.maxiter,
                    popsize=args.popsize,
                    cache_path=nominal_path_cache,
                )
                r, _ = optimize(
                    _subset(manifest, robust=True, path_kind=kind),
                    seed=args.seed + 11 + index * 2,
                    maxiter=args.maxiter,
                    popsize=args.popsize,
                    cache_path=robust_path_cache,
                )
            per_path[kind] = {"nominal": n.__dict__, "robust": r.__dict__}
        (output / "per_path_gains.json").write_text(json.dumps(per_path, indent=2) + "\n", encoding="utf-8")

    print(f"nominal={nominal} robust={robust}")
    print(f"calibration saved to {output / 'calibration.json'}")


if __name__ == "__main__":
    main()
