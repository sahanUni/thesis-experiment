"""Result-table aggregation without hiding failed episodes."""

from __future__ import annotations

from collections.abc import Iterable
import csv
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_METRICS = (
    "completion_rate",
    "failure_adjusted_error_m",
    "mean_distance_m",
    "rms_distance_m",
    "p95_distance_m",
    "max_distance_m",
    "progress_pct",
    "steer_total_variation_per_s",
    "wheel_saturation_fraction",
)


def failure_adjusted_error(
    iae_m_s: float,
    *,
    finished: bool,
    duration_s: float,
    max_time_s: float,
    corridor_m: float,
) -> float:
    if max_time_s <= 0 or duration_s < 0 or corridor_m <= 0:
        raise ValueError("metric horizon, duration, and corridor must be valid")
    padding = 0.0 if finished else corridor_m * max(max_time_s - duration_s, 0.0)
    return (float(iae_m_s) + padding) / max_time_s


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["controller"]), str(row["evaluation_mode"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (controller, mode, condition), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "controller": controller,
            "evaluation_mode": mode,
            "condition": condition,
            "episodes": len(group),
            "completion_rate": float(np.mean([bool(item["finished"]) for item in group])),
        }
        for metric in PRIMARY_METRICS[1:]:
            result[metric] = float(np.mean([float(item[metric]) for item in group]))
        output.append(result)
    return output


def paired_differences(
    rows: Iterable[dict[str, Any]],
    controller_a: str,
    controller_b: str,
    metric: str = "failure_adjusted_error_m",
) -> np.ndarray:
    indexed: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["controller"] in {controller_a, controller_b}:
            indexed.setdefault(str(row["scenario_id"]), {})[str(row["controller"])] = float(row[metric])
    pairs = [values[controller_a] - values[controller_b] for values in indexed.values() if controller_a in values and controller_b in values]
    return np.asarray(pairs, dtype=np.float64)
