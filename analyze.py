"""Paired effect estimates and hierarchical bootstrap intervals for thesis tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from metrics import read_csv, write_csv


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _bootstrap(values_by_seed: dict[str, np.ndarray], rng: np.random.Generator, draws: int) -> tuple[float, float]:
    seeds = tuple(values_by_seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            values = values_by_seed[str(seed)]
            seed_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        estimates[index] = np.mean(seed_means)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def compare(
    rows: list[dict[str, str]],
    reference: str,
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    reference_rows = {
        (row["scenario_id"], row["evaluation_mode"], row["condition"]): row
        for row in rows
        if row["controller"] == reference
    }
    learned = sorted({row["controller"] for row in rows if row["controller"] != reference})
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for controller in learned:
        for mode in ("stationary", "transient"):
            for condition in ("nominal", "delay", "noise", "combined"):
                groups: dict[str, dict[str, list[float]]] = defaultdict(
                    lambda: {"error": [], "completion": [], "co_completed": []}
                )
                for row in rows:
                    if row["controller"] != controller or row["evaluation_mode"] != mode or row["condition"] != condition:
                        continue
                    key = (row["scenario_id"], mode, condition)
                    baseline = reference_rows.get(key)
                    if baseline is None:
                        continue
                    training_seed = row["training_seed"] or "fixed"
                    groups[training_seed]["error"].append(
                        float(row["failure_adjusted_error_m"]) - float(baseline["failure_adjusted_error_m"])
                    )
                    groups[training_seed]["completion"].append(
                        float(_as_bool(row["finished"])) - float(_as_bool(baseline["finished"]))
                    )
                    if _as_bool(row["finished"]) and _as_bool(baseline["finished"]):
                        groups[training_seed]["co_completed"].append(
                            float(row["mean_distance_m"]) - float(baseline["mean_distance_m"])
                        )
                if not groups:
                    continue
                result: dict[str, Any] = {
                    "controller": controller,
                    "reference": reference,
                    "evaluation_mode": mode,
                    "condition": condition,
                    "training_seeds": len(groups),
                }
                for label in ("error", "completion", "co_completed"):
                    values = {key: np.asarray(value[label]) for key, value in groups.items() if value[label]}
                    if not values:
                        result[f"{label}_difference"] = ""
                        result[f"{label}_ci_low"] = ""
                        result[f"{label}_ci_high"] = ""
                        continue
                    estimate = float(np.mean([np.mean(item) for item in values.values()]))
                    low, high = _bootstrap(values, rng, draws)
                    result[f"{label}_difference"] = estimate
                    result[f"{label}_ci_low"] = low
                    result[f"{label}_ci_high"] = high
                output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--reference", default="pid_global_robust")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("draws must be at least 100")
    source = Path(args.episodes).resolve()
    destination = Path(args.output).resolve() if args.output else source.with_name("paired_effects.csv")
    write_csv(destination, compare(read_csv(source), args.reference, draws=args.draws, seed=args.seed))
    print(f"paired effects saved to {destination}")


if __name__ == "__main__":
    main()
