"""Frozen, replayable scenarios for paired controller evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import CONFIG, PATH_SPLITS
from core import paths


EVENT_KINDS = {"delay_step", "noise_step"}
CONDITIONS = ("nominal", "delay", "noise", "combined")


@dataclass(frozen=True)
class DisturbanceEvent:
    kind: str
    start_s: float
    value: float
    end_s: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unsupported disturbance event: {self.kind}")
        if self.start_s < 0 or self.value < 0:
            raise ValueError("event time and value must be non-negative")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("event end_s must be after start_s")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    path: dict[str, Any]
    target_speed: float
    evaluation_mode: str
    condition: str
    noise_seed: int
    initial_delay_s: float = 0.0
    initial_noise_std_m: float = 0.0
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    events: tuple[DisturbanceEvent, ...] = field(default_factory=tuple)
    geometry: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evaluation_mode not in {"stationary", "transient"}:
            raise ValueError("evaluation_mode must be stationary or transient")
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {self.condition}")
        if self.target_speed <= 0:
            raise ValueError("target_speed must be positive")

    def to_options(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "path": self.path,
            "v_target": self.target_speed,
            "noise_seed": self.noise_seed,
            "initial_delay_s": self.initial_delay_s,
            "initial_noise_std_m": self.initial_noise_std_m,
            "initial_pose": self.initial_pose,
            "events": [asdict(event) for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        data = dict(data)
        data["events"] = tuple(DisturbanceEvent(**event) for event in data.get("events", ()))
        return cls(**data)


@dataclass(frozen=True)
class ScenarioManifest:
    name: str
    split: str
    scenarios: tuple[Scenario, ...]
    schema_version: int = 1

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "split": self.split,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScenarioManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            split=payload["split"],
            schema_version=int(payload["schema_version"]),
            scenarios=tuple(Scenario.from_dict(item) for item in payload["scenarios"]),
        )

    def digest(self) -> str:
        canonical = json.dumps([asdict(s) for s in self.scenarios], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _slug(path: dict[str, Any]) -> str:
    if path["kind"] != "random_curvature":
        return path["kind"]
    return f"random_curvature_{path.get('params', {}).get('seed', 'unknown')}"


def build_manifest(
    split: str,
    *,
    name: str | None = None,
    speeds: Iterable[float] = CONFIG.target_speeds,
    include_transients: bool = True,
    seed: int = 20260819,
) -> ScenarioManifest:
    if split not in PATH_SPLITS:
        raise ValueError(f"unknown split: {split}")
    modes = ("stationary", "transient") if include_transients else ("stationary",)
    scenarios: list[Scenario] = []
    index = 0
    for path_spec in PATH_SPLITS[split]:
        built_path = paths.build(path_spec)
        path_geometry = (
            paths.validate_geometry(built_path)
            if path_spec["kind"] == "random_curvature"
            else paths.geometry(built_path)
        )
        for speed in speeds:
            for mode in modes:
                for condition in CONDITIONS:
                    delay = CONFIG.delay_severity_s if mode == "stationary" and condition in {"delay", "combined"} else 0.0
                    noise = CONFIG.noise_severity_m if mode == "stationary" and condition in {"noise", "combined"} else 0.0
                    events: list[DisturbanceEvent] = []
                    if mode == "transient" and condition in {"delay", "combined"}:
                        events.append(DisturbanceEvent("delay_step", CONFIG.transient_start_s, CONFIG.delay_severity_s))
                    if mode == "transient" and condition in {"noise", "combined"}:
                        events.append(DisturbanceEvent("noise_step", CONFIG.transient_start_s, CONFIG.noise_severity_m))
                    scenario_id = f"{split}-{_slug(path_spec)}-v{speed:.1f}-{mode}-{condition}"
                    scenarios.append(
                        Scenario(
                            scenario_id=scenario_id,
                            path=dict(path_spec),
                            target_speed=float(speed),
                            evaluation_mode=mode,
                            condition=condition,
                            noise_seed=seed + index,
                            initial_delay_s=delay,
                            initial_noise_std_m=noise,
                            events=tuple(events),
                            geometry=path_geometry,
                        )
                    )
                    index += 1
    return ScenarioManifest(name=name or f"{split}_evaluation", split=split, scenarios=tuple(scenarios))
