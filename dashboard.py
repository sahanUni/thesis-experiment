"""Interactive runner for the three-way controller comparison.

Fixed PID, scheduled PPO and direct RL on one plant, one path and one
disturbance, driven from the drawer and overlaid on shared axes.

Nothing this page produces is an official result and it never writes into an
artifact directory (DECISIONS.md); every number in the thesis comes from
evaluate.py (PLAN.md). It once carried a second tab that read those frozen
artifacts back, which is a job `analyze.py` does better and without a browser.

Every episode is built as a real `scenarios.Scenario` and run through
`rollout.run_episode` -- the same two objects the batch evaluator uses. That is
deliberate: a dashboard with its own private episode loop would drift from the
evaluator, and the replay-equivalence gate in TODO.md section 9 could never be
met. `tests/test_dashboard.py` asserts the two agree sample for sample.

The visual system is ported from Path_Following_PPO so the two projects read
the same way.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

import config
from calibration import GainCalibration, PIDGains
from config import CONFIG, PATH_SPLITS
from core import paths
from env import PathFollowingEnv
from rollout import run_episode
from scenarios import CONDITIONS, DisturbanceEvent, Scenario


# --- palette ---------------------------------------------------------------

MUTED, ACCENT = "#93a1b8", "#4fc3f7"
PANEL, LINE = "#1a2233", "#2c3a57"
GOOD, BAD, WARN = "#66d9a3", "#ff7a7a", "#ffce6b"
PATH_COLOR = "#8b97ad"
# One per controller panel, in load order. Distinct enough to tell four
# trajectories apart on one corridor plot.
CONTROLLER_COLORS = ("#4fc3f7", "#c792ea", "#ffb26b", "#6bd6c4", "#ff8fa3", "#a3e635")

# The comparison this experiment is about is between three methods, not between
# whichever controllers happen to be selected. Colour is therefore a property of
# the method, so fixed PID is the same amber in every chart of every session and
# the eye can carry a reading from one figure to the next. Seeds and regimes
# within a method get successive shades of that method's hue.
METHOD_COLORS = {
    "pid": ("#ffb26b", "#e08a3c", "#ffd7a8", "#b56c22"),
    "scheduled": ("#4fc3f7", "#2d8fbf", "#a5e4ff", "#1d6a90"),
    "direct": ("#c792ea", "#9a63c0", "#e3c6f7", "#6f3f93"),
}
METHOD_LABELS = {
    "pid": "Fixed PID",
    "scheduled": "Scheduled PPO",
    "direct": "Direct RL",
}
METHOD_ORDER = ("pid", "scheduled", "direct")

# Kept clear of every METHOD_COLORS hue: the disturbance bands are drawn on top
# of the method traces, and a band the colour of one of the methods reads as
# that method's own shading.
DELAY_TINT, NOISE_TINT = "#ff8fa3", "#6bd6c4"
EVENT_TINTS = {"delay_step": DELAY_TINT, "noise_step": NOISE_TINT}
EVENT_LABELS = {"delay_step": "dead time step", "noise_step": "sensor noise step"}

CORRIDOR_M = float(CONFIG.corridor_m)
CORRIDOR_GRID = 240
CORRIDOR_STRIDE = 5

SAFE_ARM = re.compile(r"^[a-z0-9_]+$")

# The sealed five-seed matrix the thesis reports. `protocol_v2` was the default
# while that run was the newest one; pointing at it now would open the dashboard
# on superseded policies trained against the pre-fix reward and gain box.
DEFAULT_MODELS_ROOT = config.ROOT / "artifacts" / "models" / "final"
DEFAULT_CALIBRATION = config.ROOT / "artifacts" / "calibration" / "calibration.json"

# The external conditions, and only those.
#
# Named because the slider and the validator behind it must agree. They did not:
# the slider offered 300 ms of dead time and the validator rejected anything over
# 200, so the top third of the track produced an error instead of an episode.
#
# Neither ceiling is a plant limit. The dead-time queue in env.py resizes to
# whatever it is given, and sensor noise perturbs only the e_ct the controller
# reads. They are the range worth exploring: 400 ms is over two and a half times
# the declared severity, and 200 mm of noise is a fifth of the corridor, which is
# far past the point where the measurement stops being informative.
DELAY_MAX_MS = 400.0
NOISE_MAX_MM = 200.0
EVENT_START_MAX_S = 60.0

# (id, label, min, max, step, default)
SCENARIO_CONTROLS = (
    ("delay", "Dead time (ms)", 0.0, DELAY_MAX_MS, 1.0, 1000.0 * CONFIG.delay_severity_s),
    ("noise", "Sensor noise (mm)", 0.0, NOISE_MAX_MM, 0.01, 1000.0 * CONFIG.noise_severity_m),
    ("event-time", "Disturbance starts at (s)", 0.0, EVENT_START_MAX_S, 0.5, CONFIG.transient_start_s),
    ("seed", "Noise seed", 0.0, 100000.0, 1.0, 20260819.0 % 100000),
)
GAIN_LABELS = ("Kp", "Ki", "Kd")
GAIN_STEPS = (0.1, 0.01, 0.05)

# Metrics shown in the compact header row, in this order. Everything else in
# `metrics` still appears in the full table underneath.
HEADLINE_METRICS = (
    ("finished", "Completed"),
    ("failure_adjusted_error_m", "J_FA"),
    ("mean_distance_m", "Mean distance"),
    ("max_distance_m", "Max distance"),
    ("progress_pct", "Progress"),
    ("duration_s", "Duration"),
)


# --- small helpers ---------------------------------------------------------


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return parsed


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _allocation_spans(rows: list[dict[str, Any]]) -> list[list[float]]:
    """Contiguous stretches where the allocator clipped a wheel command.

    Derived rather than read from the trace: `env.py` computes the same test
    (`u_applied != u_requested`) but only records the fraction in the episode
    metrics, and the dashboard must not change the trace schema the evaluator
    writes.

    Merged into runs because an episode is ~1000 control steps; one plotly
    shape per sample is a chart that takes seconds to draw and says nothing
    more than a handful of spans do.
    """
    times = _values(rows, "t")
    if not times:
        return []
    step = (times[-1] - times[0]) / max(len(times) - 1, 1)
    spans: list[list[float]] = []
    for row in rows:
        try:
            moment = float(row["t"])
            limited = not (
                math.isclose(float(row["u_left"]), float(row["u_left_applied"]), abs_tol=1e-9)
                and math.isclose(float(row["u_right"]), float(row["u_right_applied"]), abs_tol=1e-9)
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not limited:
            continue
        if spans and moment - spans[-1][1] <= step * 1.5:
            spans[-1][1] = moment + step
        else:
            spans.append([moment, moment + step])
    return spans


def path_label(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    params = spec.get("params", {})
    if kind == "random_curvature":
        return f"random curvature (seed {params.get('seed', '?')})"
    return kind


def path_value(spec: dict[str, Any]) -> str:
    """A JSON string, so a whole path spec survives a dropdown round trip."""
    return json.dumps(spec, sort_keys=True)


def resolve_calibration(explicit: str | Path | None = None) -> tuple[GainCalibration, str]:
    """The calibration artifact, or the development default with a warning.

    A missing calibration is fatal in the batch evaluator and must stay that
    way. Here it is only degraded: the interactive tab is useful for hand
    tuning against the live plant before `calibrate_pid.py` has ever run, and
    refusing to start would remove the tool exactly when it is most wanted. The
    banner says which one is loaded so no one mistakes a smoke-test gain box
    for the calibrated one.
    """
    candidate = Path(explicit) if explicit is not None else DEFAULT_CALIBRATION
    if candidate.exists():
        if candidate.suffix.lower() != ".json":
            raise ValueError(f"calibration is not a JSON artifact: {candidate}")
        return GainCalibration.load(candidate.resolve(strict=True)), candidate.name
    if explicit is not None:
        raise FileNotFoundError(f"no calibration at {candidate}")
    return GainCalibration.development_default(), "development default (NOT calibrated)"


def discover_models(root: Path) -> list[dict[str, Any]]:
    """Every `<arm>/seed_<n>/best_model.zip` under the models root.

    The arm name carries the mode, and `metadata.json` states it outright; both
    are checked because loading a scheduler artifact into a direct-mode
    environment silently reinterprets the action vector.
    """
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        if not SAFE_ARM.fullmatch(arm) or not arm.startswith(("ppo_scheduler_", "ppo_direct_")):
            continue
        for seed_dir in sorted(arm_dir.glob("seed_*")):
            model_path = seed_dir / "best_model.zip"
            if not model_path.exists():
                continue
            declared = arm.split("_")[1]
            mode = "scheduled" if declared == "scheduler" else "direct"
            metadata_path = seed_dir / "metadata.json"
            if metadata_path.exists():
                recorded = json.loads(metadata_path.read_text(encoding="utf-8")).get("mode")
                if recorded and recorded != mode:
                    raise ValueError(
                        f"{model_path} sits under arm '{arm}' but its metadata records "
                        f"mode '{recorded}'. The artifact and its directory disagree."
                    )
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            regime = arm.rsplit("_", 1)[1]
            found.append(
                {
                    "key": f"{arm}:{seed}",
                    # Short inside its method group, where the method is the
                    # heading; the panels and legends prepend the method name.
                    "label": f"{regime} · seed {seed}",
                    "arm": arm,
                    "seed": seed,
                    "mode": mode,
                    "path": model_path.resolve(),
                }
            )
    return found


# --- the runner ------------------------------------------------------------


class InteractiveRunner:
    """Build scenarios, run controllers on them, cache the results.

    One instance per app. PPO artifacts are loaded once and kept; episodes are
    cached on (scenario, controller, gains) so that dragging a gain slider
    re-runs only the PID panel and leaves every policy panel untouched --
    otherwise the comparison would silently be against a different episode.
    """

    def __init__(
        self,
        models_root: Path | None = None,
        calibration_path: str | Path | None = None,
        scheduler_calibration_path: str | Path | None = None,
    ) -> None:
        self.calibration, self.calibration_label = resolve_calibration(calibration_path)
        scheduler_path = scheduler_calibration_path or (
            config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"
        )
        self.scheduler_calibration = GainCalibration.load(Path(scheduler_path))
        self.models_root = Path(models_root or DEFAULT_MODELS_ROOT)
        self.models = discover_models(self.models_root)
        self.model_error = (
            ""
            if self.models
            else (
                "No PPO artifact found under "
                f"{self.models_root}. Train one first:  python train.py --mode scheduled "
                "--regime nominal --seed 11"
            )
        )
        self._loaded: dict[str, Any] = {}
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._lock = Lock()

    # -- controllers --

    def controller_options(self) -> dict[str, list[dict[str, str]]]:
        """Selectable controllers, grouped by the three compared methods.

        Grouped rather than one flat list because the flat list made a
        twenty-three-entry checklist in which the three-way comparison the
        experiment is about was something you had to reconstruct by reading
        arm-name prefixes.
        """
        grouped: dict[str, list[dict[str, str]]] = {
            "pid": [
                {"label": "calibrated nominal (Kp 250)", "value": "pid_global_nominal"},
                {"label": "calibrated robust (Kp 8)", "value": "pid_global_robust"},
                {"label": "hand-tuned (advanced sliders)", "value": "pid_custom"},
            ],
            "scheduled": [],
            "direct": [],
        }
        for entry in self.models:
            grouped[method_of(entry["key"])].append(
                {"label": entry["label"], "value": entry["key"]}
            )
        return grouped

    def default_selection(self) -> dict[str, list[str]]:
        """One controller per method, so the first run is already a comparison.

        The robust PID is the chosen default because it is the baseline the
        learned arms have to beat; opening on the nominal arm would show a
        controller that simply leaves the corridor whenever dead time is on.
        """
        grouped = self.controller_options()
        selection = {"pid": ["pid_global_robust"], "scheduled": [], "direct": []}
        for method in ("scheduled", "direct"):
            options = grouped[method]
            if options:
                selection[method] = [options[0]["value"]]
        return selection

    def _model_entry(self, key: str) -> dict[str, Any]:
        for entry in self.models:
            if entry["key"] == key:
                return entry
        raise ValueError(f"controller '{key}' is no longer loaded")

    def _policy(self, entry: dict[str, Any]):
        model = self._loaded.get(entry["key"])
        if model is None:
            from stable_baselines3 import PPO  # imported late: torch is slow

            model = PPO.load(entry["path"], device="cpu")
            self._loaded[entry["key"]] = model
        return lambda observation: np.asarray(
            model.predict(observation, deterministic=True)[0], dtype=np.float32
        )

    # -- scenarios --

    def scenario_from_controls(
        self,
        path_spec_json: str,
        target_speed: Any,
        condition: str,
        evaluation_mode: str,
        delay_ms: Any,
        noise_mm: Any,
        event_time: Any,
        noise_seed: Any,
    ) -> Scenario:
        """A Scenario assembled exactly the way `build_manifest` assembles one.

        Same field-for-field construction, so a hand-built scenario that
        happens to match a manifest entry runs the identical episode. The only
        difference is the id, which is prefixed `interactive-` so nothing
        downstream can mistake it for a manifest scenario.
        """
        try:
            spec = json.loads(path_spec_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("select a valid path") from error
        if not isinstance(spec, dict) or not isinstance(spec.get("kind"), str):
            raise ValueError("select a valid path")
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        if evaluation_mode not in {"stationary", "transient"}:
            raise ValueError("evaluation mode must be stationary or transient")
        speed = _number(target_speed, "target speed", 0.05, 5.0)

        delay_s = 0.001 * _number(delay_ms, "dead time", 0.0, DELAY_MAX_MS)
        noise_m = 0.001 * _number(noise_mm, "sensor noise", 0.0, NOISE_MAX_MM)
        start_s = _number(event_time, "transient start", 0.0, EVENT_START_MAX_S)
        # Bounded by the field, not by the slider: manifest seeds are
        # 20260819 + index, far outside the slider's convenience range, and
        # a scenario copied out of a manifest has to be reproducible here.
        seed = int(_number(noise_seed, "noise seed", 0.0, 2**31 - 1))

        wants_delay = condition in {"delay", "combined"}
        wants_noise = condition in {"noise", "combined"}
        stationary = evaluation_mode == "stationary"
        events: list[DisturbanceEvent] = []
        if not stationary and wants_delay:
            events.append(DisturbanceEvent("delay_step", start_s, delay_s))
        if not stationary and wants_noise:
            events.append(DisturbanceEvent("noise_step", start_s, noise_m))

        built = paths.build(spec)
        geometry = (
            paths.validate_geometry(built)
            if spec["kind"] == "random_curvature"
            else paths.geometry(built)
        )
        return Scenario(
            scenario_id=f"interactive-{_slugify(path_label(spec))}-v{speed:.1f}-{evaluation_mode}-{condition}",
            path=dict(spec),
            target_speed=float(speed),
            evaluation_mode=evaluation_mode,
            condition=condition,
            noise_seed=seed,
            initial_delay_s=delay_s if (stationary and wants_delay) else 0.0,
            initial_noise_std_m=noise_m if (stationary and wants_noise) else 0.0,
            events=tuple(events),
            geometry=geometry,
        )

    # -- episodes --

    def run(
        self,
        scenario: Scenario,
        controller_keys: Iterable[str],
        gains: tuple[float, float, float],
    ) -> list[dict[str, Any]]:
        keys = [key for key in controller_keys if key]
        if not keys:
            raise ValueError("select at least one controller")
        custom = PIDGains.from_iterable(gains)
        lower, upper = self.calibration.lower.as_array(), self.calibration.upper.as_array()
        if np.any(custom.as_array() < lower) or np.any(custom.as_array() > upper):
            raise ValueError("drawer gains are outside the calibrated scheduler bounds")

        # Method order, not selection order: the legend then reads PID, then
        # scheduled, then direct in every chart regardless of the order the
        # checkboxes were ticked in.
        keys.sort(key=lambda key: METHOD_ORDER.index(method_of(key)))
        scenario_key = json.dumps(asdict(scenario), sort_keys=True)
        used: dict[str, int] = {}
        results: list[dict[str, Any]] = []
        with self._lock:
            for key in keys:
                # Only `pid_custom` depends on the gain sliders; keying the
                # others without the gains is what keeps their episodes stable
                # while the sliders move.
                cache_key = (scenario_key, key) + (
                    tuple(custom.as_array().tolist()) if key == "pid_custom" else ()
                )
                result = self._cache.get(cache_key)
                if result is None:
                    result = self._run_one(scenario, key, custom)
                    self._cache[cache_key] = result
                method = method_of(key)
                shades = METHOD_COLORS[method]
                shade = used.get(method, 0)
                used[method] = shade + 1
                results.append(
                    {
                        **result,
                        "method": method,
                        # Legends and panel headings name the method first: a
                        # bare "disturbed · seed 11" does not say whether it
                        # steered the wheels or scheduled a gain.
                        "label": f"{METHOD_LABELS[method]} — {result['label']}",
                        "color": shades[shade % len(shades)],
                    }
                )
        return results

    def _run_one(
        self, scenario: Scenario, key: str, custom: PIDGains
    ) -> dict[str, Any]:
        if key.startswith("pid_"):
            gains = {
                "pid_global_nominal": self.calibration.nominal,
                "pid_global_robust": self.calibration.robust,
                "pid_custom": custom,
            }[key]
            env = PathFollowingEnv(
                mode="fixed",
                training=False,
                calibration=self.calibration,
                fixed_gains=gains,
            )
            predictor = None
            label = {
                "pid_global_nominal": "calibrated nominal",
                "pid_global_robust": "calibrated robust",
                "pid_custom": "hand-tuned",
            }[key]
            note = f"Kp {gains.kp:.3f}, Ki {gains.ki:.3f}, Kd {gains.kd:.3f}"
        else:
            entry = self._model_entry(key)
            env = PathFollowingEnv(
                mode=entry["mode"],
                training=False,
                calibration=(
                    self.scheduler_calibration
                    if entry["mode"] == "scheduled"
                    else self.calibration
                ),
            )
            predictor = self._policy(entry)
            label = entry["label"]
            note = f"{entry['mode']} mode · {entry['path'].parent.name} · {entry['path'].name}"
        try:
            result = run_episode(env, scenario, predictor)
        finally:
            env.close()
        return {
            "key": key,
            "label": label,
            "note": note,
            "metrics": result.metrics,
            "trace": result.trace,
        }


def method_of(key: str) -> str:
    """Which of the three compared methods a controller key belongs to.

    Everything user-visible is grouped and coloured by this, so it is derived
    from the key in exactly one place rather than re-guessed per chart.
    """
    if key.startswith("pid_"):
        return "pid"
    if key.startswith("ppo_scheduler_"):
        return "scheduled"
    if key.startswith("ppo_direct_"):
        return "direct"
    raise ValueError(f"controller '{key}' belongs to no known method")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "path"


# --- figures ---------------------------------------------------------------


# The chart header is two stacked bands inside the top margin: the title on
# the first line, the horizontal legend on the second. Plotly positions each
# independently, so the two used to be given overlapping coordinates and the
# title was drawn straight through the legend entries. These constants keep the
# two apart in one place instead of at each of the nine call sites.
TITLE_BAND_PX = 26
LEGEND_BAND_PX = 30
CHART_TOP_PX = TITLE_BAND_PX + LEGEND_BAND_PX


def _base_figure(
    title: str, y_label: str, *, height: int, x_label: str, legend_rows: int = 1
) -> go.Figure:
    top = TITLE_BAND_PX + LEGEND_BAND_PX * legend_rows
    figure = go.Figure()
    figure.update_layout(
        template="plotly_dark",
        height=height + top - CHART_TOP_PX,
        margin=dict(l=55, r=15, t=top, b=40),
        paper_bgcolor=PANEL,
        plot_bgcolor="#0a0f1a",
        # Pinned to the top of the figure container rather than floated above
        # the plotting area, so it cannot drift down into the legend band when
        # the margin changes.
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=1.0,
            yanchor="top",
            yref="container",
            pad=dict(t=6),
            font=dict(size=13, color=MUTED),
        ),
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(title=x_label, gridcolor="#1e2a42", zeroline=False)
    figure.update_yaxes(title=y_label, gridcolor="#1e2a42", zeroline=False)
    return figure


def _show_legend(figure: go.Figure, *, x: float = 0.0) -> go.Figure:
    """Turn the legend on in the band the top margin already reserved for it."""
    figure.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=x,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            font=dict(size=10),
        ),
    )
    return figure


def _corridor_field(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks = points[::CORRIDOR_STRIDE]
    padding = CORRIDOR_M * 1.35
    low, high = points.min(axis=0) - padding, points.max(axis=0) + padding
    grid_x = np.linspace(low[0], high[0], CORRIDOR_GRID)
    grid_y = np.linspace(low[1], high[1], CORRIDOR_GRID)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    distance_squared = np.full(mesh_x.shape, np.inf)
    for point_x, point_y in marks:
        np.minimum(
            distance_squared,
            (mesh_x - point_x) ** 2 + (mesh_y - point_y) ** 2,
            out=distance_squared,
        )
    return grid_x, grid_y, np.sqrt(distance_squared)


@lru_cache(maxsize=64)
def _corridor(spec_json: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _corridor_field(paths.build(json.loads(spec_json))["pts"])


def _event_bands(figure: go.Figure, scenario: Scenario, trace_end: float, label: bool) -> None:
    for event in scenario.events:
        if float(event.start_s) > trace_end:
            continue
        end = event.end_s
        tint = EVENT_TINTS.get(event.kind, WARN)
        figure.add_vrect(
            x0=float(event.start_s),
            x1=min(float(end) if end is not None else trace_end, trace_end),
            fillcolor=tint,
            opacity=0.10,
            line_width=0,
            annotation_text=EVENT_LABELS.get(event.kind, event.kind) if label else None,
            annotation_position="top left",
            annotation_font=dict(size=10, color=tint),
        )


def trajectory_figure(
    results: list[dict[str, Any]], scenario: Scenario, *, height: int = 430
) -> go.Figure:
    spec_json = path_value(scenario.path)
    reference = paths.build(scenario.path)
    points = reference["pts"]
    figure = _base_figure(
        f"trajectory — {reference['name']}", "y (m)", height=height, x_label="x (m)"
    )
    grid_x, grid_y, distance = _corridor(spec_json)
    figure.add_contour(
        x=grid_x,
        y=grid_y,
        z=distance,
        showscale=False,
        hoverinfo="skip",
        name="corridor",
        line=dict(width=0),
        contours=dict(type="constraint", operation="<", value=CORRIDOR_M),
        fillcolor="rgba(139,151,173,0.10)",
    )
    figure.add_scatter(
        x=points[::10, 0],
        y=points[::10, 1],
        mode="lines",
        line=dict(color=PATH_COLOR, width=1.6, dash="dash"),
        hoverinfo="skip",
        name="reference path",
    )
    for result in results:
        figure.add_scatter(
            x=_values(result["trace"], "x"),
            y=_values(result["trace"], "y"),
            mode="lines",
            line=dict(color=result["color"], width=2.2),
            name=result["label"],
        )
    figure.add_scatter(
        x=[points[0, 0]], y=[points[0, 1]], mode="markers",
        marker=dict(color=GOOD, size=10), name="start",
    )
    figure.add_scatter(
        x=[points[-1, 0]], y=[points[-1, 1]], mode="markers",
        marker=dict(color=WARN, size=10, symbol="square"), name="finish",
    )
    figure.update_layout(hovermode="closest")
    _show_legend(figure)
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def deviation_figure(rows: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    figure = _base_figure(
        "tracking error — e_ct (measured) vs true distance off path",
        "e (m)", height=285, x_label="time (s)",
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    _event_bands(figure, scenario, trace_end, label=True)
    for limit in (CORRIDOR_M, -CORRIDOR_M):
        figure.add_hline(y=limit, line=dict(color=BAD, width=1, dash="dot"))
    figure.add_hline(y=0.0, line=dict(color=LINE, width=1))
    figure.add_scatter(
        x=times, y=_values(rows, "distance_m"), mode="lines",
        line=dict(color=PATH_COLOR, width=1.2, dash="dot"), name="distance off path",
    )
    figure.add_scatter(
        x=times, y=_values(rows, "e_ct"), mode="lines",
        line=dict(color=ACCENT, width=1.8), name="e_ct (true)",
    )
    measured = _values(rows, "e_ct_measured")
    if measured:
        figure.add_scatter(
            x=times, y=measured, mode="lines",
            line=dict(color=WARN, width=1.0), opacity=0.65,
            name="e_ct (what the PID reads)",
        )
    _show_legend(figure)
    return figure


def speed_figure(rows: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    """Achieved speed against target, with the authority conflict shaded.

    Steering has first claim on the wheels, so the speed channel is what gives
    way in a corner. Shading the allocation-limited samples makes that visible
    instead of leaving it as an unexplained dip.
    """
    figure = _base_figure(
        "speed — achieved vs target", "v (m/s)", height=250, x_label="time (s)"
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    _event_bands(figure, scenario, trace_end, label=False)
    figure.add_hline(
        y=scenario.target_speed, line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text="target", annotation_position="top left",
        annotation_font=dict(size=10, color=MUTED),
    )
    spans = _allocation_spans(rows)
    for start, end in spans[:200]:
        figure.add_vrect(
            x0=start, x1=end, fillcolor=WARN, opacity=0.16, line_width=0, layer="below"
        )
    if spans:
        figure.add_annotation(
            text=f"shaded: steering has taken wheel authority ({len(spans)} spans)",
            showarrow=False, font=dict(size=10, color=WARN),
            xref="paper", yref="paper", x=0.99, y=1.13, xanchor="right",
        )
    figure.add_scatter(
        x=times, y=_values(rows, "speed_mps"), mode="lines",
        line=dict(color=GOOD, width=1.8), name="speed",
    )
    _show_legend(figure, x=0.18)
    return figure


def command_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """What was asked of the wheels and what they were actually given.

    Requested and applied are drawn together because the gap between them IS
    the saturation: a policy whose steering demand is clipped for half the
    corner is not the same as one that was never limited, and the two look
    identical on a trajectory plot.
    """
    figure = _base_figure(
        "wheel commands — requested vs applied", "u (normalised)",
        height=250, x_label="time (s)",
    )
    times = _values(rows, "t")
    limit = float(CONFIG.actuator_limit)
    for bound in (limit, -limit):
        figure.add_hline(y=bound, line=dict(color=BAD, width=1, dash="dot"))
    figure.add_hline(y=0.0, line=dict(color=LINE, width=1))
    for key, name, color, dash in (
        ("u_left", "left requested", ACCENT, "dot"),
        ("u_left_applied", "left applied", ACCENT, "solid"),
        ("u_right", "right requested", DELAY_TINT, "dot"),
        ("u_right_applied", "right applied", DELAY_TINT, "solid"),
    ):
        values = _values(rows, key)
        if not values:
            continue
        figure.add_scatter(
            x=times, y=values, mode="lines",
            line=dict(color=color, width=1.4, dash=dash), name=name,
        )
    _show_legend(figure)
    return figure


def gain_figure(rows: list[dict[str, Any]], calibration: GainCalibration) -> go.Figure:
    """What the controller did with its gains, against the box it may use.

    Flat lines here are a real result, not a missing plot: a scheduler that
    settles on one constant has learned nothing a fixed gain could not do, and
    a mean alone hides that perfectly.
    """
    figure = _base_figure(
        "steering gains — applied over the episode", "gain", height=250, x_label="time (s)"
    )
    times = _values(rows, "t")
    if not _values(rows, "kp"):
        figure.add_annotation(
            text="direct-mode controller: no PID gains to show",
            showarrow=False, font=dict(size=12, color=MUTED),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return figure
    for bound in (calibration.lower.kp, calibration.upper.kp):
        figure.add_hline(y=float(bound), line=dict(color=LINE, width=1, dash="dot"))
    for key, name, color in (("kp", "Kp", ACCENT), ("ki", "Ki", GOOD), ("kd", "Kd", WARN)):
        figure.add_scatter(
            x=times, y=_values(rows, key), mode="lines",
            line=dict(color=color, width=1.6), name=name,
        )
    _show_legend(figure, x=0.18)
    return figure


def disturbance_figure(rows: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    """When the plant changed under the controller.

    The two disturbances in this experiment are dead time and sensor noise, and
    both are step changes recorded per sample, so the trace itself is the
    timeline -- no need to trust the scenario declaration.
    """
    figure = _base_figure(
        "plant condition — dead time and sensor noise", "value",
        height=230, x_label="time (s)",
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    _event_bands(figure, scenario, trace_end, label=True)
    figure.add_scatter(
        x=times, y=[1000.0 * value for value in _values(rows, "actuator_delay_s")],
        mode="lines", line=dict(color=DELAY_TINT, width=1.8, shape="hv"),
        name="dead time (ms)",
    )
    figure.add_scatter(
        x=times, y=[1000.0 * value for value in _values(rows, "sensor_noise_m")],
        mode="lines", line=dict(color=NOISE_TINT, width=1.8, shape="hv"),
        name="sensor noise sigma (mm)",
    )
    _show_legend(figure)
    return figure


# --- method comparison -----------------------------------------------------
#
# The per-controller panels below answer "what did this controller do". These
# answer "how did the three methods differ", which is the question the thesis
# actually asks, and it cannot be read off three panels stacked vertically --
# a 4 mm difference in tracking error is invisible unless the traces share an
# axis. Every chart here is one signal, all selected controllers, one time base.


def _derive(rows: list[dict[str, Any]], key: str) -> list[float]:
    """One trace column, including the two that are differences of columns."""
    if key == "steer_differential":
        left = _values(rows, "u_left_applied")
        right = _values(rows, "u_right_applied")
        return [a - b for a, b in zip(left, right)]
    if key == "delay_ms":
        return [1000.0 * value for value in _values(rows, "actuator_delay_s")]
    return _values(rows, key)


def comparison_figure(
    results: list[dict[str, Any]],
    scenario: Scenario,
    key: str,
    title: str,
    y_label: str,
    *,
    height: int = 270,
    scale: float = 1.0,
    label_events: bool = False,
    reference_lines: tuple[float, ...] = (),
    empty_note: str = "",
) -> go.Figure:
    figure = _base_figure(title, y_label, height=height, x_label="time (s)")
    trace_end = max(
        (max(_values(result["trace"], "t"), default=0.0) for result in results),
        default=0.0,
    )
    _event_bands(figure, scenario, trace_end, label=label_events)
    for line in reference_lines:
        figure.add_hline(
            y=line,
            line=dict(color=BAD if line else LINE, width=1, dash="dot" if line else "solid"),
        )
    drawn = 0
    for result in results:
        values = _derive(result["trace"], key)
        if not values:
            continue
        drawn += 1
        figure.add_scatter(
            x=_values(result["trace"], "t"),
            y=[value * scale for value in values],
            mode="lines",
            line=dict(color=result["color"], width=1.8),
            name=result["label"],
        )
    if not drawn and empty_note:
        figure.add_annotation(
            text=empty_note,
            showarrow=False,
            font=dict(size=12, color=MUTED),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    _show_legend(figure)
    return figure


def comparison_view(results: list[dict[str, Any]], scenario: Scenario) -> list[Any]:
    """The five overlaid charts, in the order the argument is made."""
    graph = {"displaylogo": False, "responsive": True}
    return [
        dcc.Graph(
            figure=comparison_figure(
                results,
                scenario,
                "distance_m",
                "tracking error — distance off the path, all methods",
                "distance (mm)",
                height=300,
                scale=1000.0,
                label_events=True,
                reference_lines=(1000.0 * CORRIDOR_M,),
            ),
            config=graph,
        ),
        dcc.Graph(
            figure=comparison_figure(
                results,
                scenario,
                "delay_ms",
                "the disturbance — dead time actually in force",
                "dead time (ms)",
                height=200,
            ),
            config=graph,
        ),
        dcc.Graph(
            figure=comparison_figure(
                results,
                scenario,
                "omega_applied",
                "steering actuator — yaw rate delivered to the plant",
                "omega (rad/s)",
                reference_lines=(0.0,),
            ),
            config=graph,
        ),
        dcc.Graph(
            figure=comparison_figure(
                results,
                scenario,
                "steer_differential",
                "wheel differential — applied left minus right (saturates at the dotted line)",
                "u_left - u_right",
                reference_lines=(
                    2.0 * float(CONFIG.actuator_limit),
                    -2.0 * float(CONFIG.actuator_limit),
                    0.0,
                ),
            ),
            config=graph,
        ),
        dcc.Graph(
            figure=comparison_figure(
                results,
                scenario,
                "kp",
                "proportional gain — the scheduler's decision variable",
                "Kp",
                empty_note=(
                    "No selected controller exposes a PID gain. Direct RL has none: "
                    "it commands the wheels itself."
                ),
            ),
            config=graph,
        ),
    ]


# --- metric rendering ------------------------------------------------------


def _format_metric(key: str, value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool) or key == "finished":
        truthy = value if isinstance(value, bool) else str(value).lower() == "true"
        return "Yes" if truthy else "No"
    if isinstance(value, str) and not _is_number(value):
        return value
    number = float(value)
    if key.endswith("_pct"):
        return f"{number:.1f}%"
    if key.endswith("_m"):
        return f"{number * 100:.2f} cm"
    if key.endswith("_s"):
        return f"{number:.2f} s"
    if key.endswith("_fraction"):
        return f"{100.0 * number:.1f}%"
    if key.endswith("_mps"):
        return f"{number:.3f} m/s"
    if abs(number) >= 1000 or (number and abs(number) < 0.001):
        return f"{number:.3e}"
    return f"{number:.4f}"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _metric(label: str, value: str) -> html.Div:
    return html.Div([html.B(label), html.Span(value)], className="metric")


def headline_metrics(metrics: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            _metric(label, _format_metric(key, metrics.get(key)))
            for key, label in HEADLINE_METRICS
        ],
        className="metrics",
    )


def full_metrics(metrics: dict[str, Any]) -> html.Details:
    shown = {key for key, _ in HEADLINE_METRICS}
    rest = [(key, value) for key, value in sorted(metrics.items()) if key not in shown]
    return html.Details(
        [
            html.Summary(f"All {len(metrics)} recorded metrics"),
            html.Div(
                [_metric(key.replace("_", " "), _format_metric(key, value)) for key, value in rest],
                className="metrics wide",
            ),
        ]
    )


SCOREBOARD_COLUMNS = (
    ("finished", "Completed", "{}"),
    ("failure_adjusted_error_m", "J_FA (mm)", "{:.2f}"),
    ("mean_distance_m", "Mean err (mm)", "{:.2f}"),
    ("max_distance_m", "Max err (mm)", "{:.2f}"),
    ("steer_total_variation_per_s", "Steer TV /s", "{:.2f}"),
    ("gain_total_variation_per_s", "Gain TV /s", "{:.2f}"),
)
# Columns whose stored unit is metres and whose readable unit is millimetres.
SCOREBOARD_MM = {"failure_adjusted_error_m", "mean_distance_m", "max_distance_m"}


def scoreboard(results: list[dict[str, Any]]) -> html.Table:
    """One row per controller, the numbers the comparison turns on.

    J_FA is the primary outcome and is reported in millimetres here because
    metres puts four leading zeros in front of every difference that matters.
    """
    header = html.Tr(
        [html.Th("Controller")] + [html.Th(label) for _key, label, _fmt in SCOREBOARD_COLUMNS]
    )
    body = []
    for result in results:
        cells = [
            html.Td(
                [
                    html.Span(
                        className="swatch",
                        style={"backgroundColor": result["color"]},
                    ),
                    result["label"],
                ],
                className="scoreboard-name",
            )
        ]
        for key, _label, fmt in SCOREBOARD_COLUMNS:
            value = result["metrics"].get(key)
            if key == "finished":
                text = "Yes" if value else "No"
            elif key.startswith("gain_") and result["method"] == "direct":
                # env.py records 0.0 here for direct mode. Printed as a number it
                # reads as "this policy held its gains perfectly steady", which is
                # the opposite of the truth: it has no gains at all.
                text = "n/a"
            elif value is None or value == "":
                text = "—"
            else:
                number = float(value)
                text = fmt.format(number * 1000.0 if key in SCOREBOARD_MM else number)
            cells.append(html.Td(text))
        body.append(html.Tr(cells))
    return html.Table([html.Thead(header), html.Tbody(body)], className="scoreboard")


def _gain_activity(rows: list[dict[str, Any]]) -> str:
    parts = []
    for key, name in (("kp", "Kp"), ("ki", "Ki"), ("kd", "Kd")):
        values = _values(rows, key)
        if not values:
            continue
        low, high = min(values), max(values)
        parts.append(
            f"{name} {np.mean(values):.2f}"
            + (f" ({low:.2f}–{high:.2f})" if high - low > 1e-6 else " (constant)")
        )
    return "  ·  ".join(parts)


def controller_panel(
    result: dict[str, Any], scenario: Scenario, calibration: GainCalibration
) -> html.Section:
    rows = result["trace"]
    graph = {"displaylogo": False, "responsive": True}
    note = result["note"]
    activity = _gain_activity(rows)
    if activity:
        note += "  ·  " + activity
    return html.Section(
        [
            html.Div(
                [
                    html.H2(result["label"]),
                    # Per panel rather than in the drawer: the button is next to
                    # the charts for the controller it replays, so there is no
                    # step where you pick a controller twice and have to trust
                    # that the two picks agreed.
                    html.Button(
                        "▶  Watch in MuJoCo",
                        id={"role": "replay", "key": result["key"]},
                        className="drawer-close replay-button",
                        n_clicks=0,
                    ),
                ],
                className="panel-head",
            ),
            html.P(note, className="controller-note"),
            headline_metrics(result["metrics"]),
            full_metrics(result["metrics"]),
            dcc.Graph(figure=deviation_figure(rows, scenario), config=graph),
            dcc.Graph(figure=speed_figure(rows, scenario), config=graph),
            dcc.Graph(figure=command_figure(rows), config=graph),
            dcc.Graph(figure=gain_figure(rows, calibration), config=graph),
            dcc.Graph(figure=disturbance_figure(rows, scenario), config=graph),
        ],
        className="comparison-panel",
    )


def interactive_view(
    results: list[dict[str, Any]], scenario: Scenario, runner: InteractiveRunner
) -> html.Div:
    graph = {"displaylogo": False, "responsive": True}
    events = (
        ", ".join(
            f"{EVENT_LABELS.get(event.kind, event.kind)} at {event.start_s:g} s"
            for event in scenario.events
        )
        or "none"
    )
    header = (
        f"{scenario.scenario_id}  ·  {scenario.evaluation_mode}/{scenario.condition}  ·  "
        f"{scenario.target_speed:.1f} m/s  ·  noise seed {scenario.noise_seed}  ·  "
        f"initial dead time {1000 * scenario.initial_delay_s:g} ms  ·  "
        f"initial noise {1000 * scenario.initial_noise_std_m:g} mm  ·  events: {events}"
    )
    return html.Div(
        [
            html.Section(
                [
                    html.H2(
                        [
                            "Selected scenario",
                            html.Span("interactive run", className="badge interactive"),
                        ]
                    ),
                    html.P(header, className="controller-note"),
                    scoreboard(results),
                    dcc.Graph(
                        figure=trajectory_figure(results, scenario, height=520), config=graph
                    ),
                ],
                className="comparison-panel overlay-panel",
            ),
            html.Section(
                [
                    html.H2("Method comparison"),
                    html.P(
                        "Every selected controller on one time base. The shaded band "
                        "marks the disturbance, so what happens after its left edge is "
                        "the whole result.",
                        className="controller-note",
                    ),
                    *comparison_view(results, scenario),
                ],
                className="comparison-panel overlay-panel",
            ),
            html.Div(
                [
                    controller_panel(
                        result,
                        scenario,
                        # The scheduler acts inside its own box (Kp 5-300), not
                        # the PID search box. Drawing its gain trace against the
                        # wrong bounds is how a policy pinned at its ceiling
                        # gets mistaken for one sitting comfortably mid-range.
                        runner.scheduler_calibration
                        if result["method"] == "scheduled"
                        else runner.calibration,
                    )
                    for result in results
                ],
                className="comparison-grid",
            ),
        ]
    )


# --- controls --------------------------------------------------------------


def _declared_mark(minimum: float, maximum: float, default: float) -> dict[float, Any]:
    """A single labelled tick at the control's declared value.

    Returned empty when the default sits on an end of the track, where the tick
    would only restate the bound already printed there.
    """
    if not minimum < default < maximum:
        return {}
    style = {"color": MUTED, "fontSize": "10px", "whiteSpace": "nowrap"}
    return {default: {"label": f"{default:g}", "style": style}}


def slider_row(
    control_id: str, label: str, minimum: float, maximum: float, step: float, default: float
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label),
                    dcc.Input(
                        id=f"n-{control_id}",
                        type="number",
                        value=default,
                        min=minimum,
                        max=maximum,
                        step=step,
                        debounce=True,
                    ),
                ],
                className="slider-label",
            ),
            dcc.Slider(
                minimum, maximum, step,
                value=default,
                id=f"s-{control_id}",
                # A tick at the value the protocol declares. The disturbance
                # tracks are long enough that the declared severity is a
                # fraction of a percent along -- 0.3 mm of noise on a 200 mm
                # track is not findable by dragging -- and without the tick
                # there is nothing to say which part of the range is the
                # experiment and which part is exploration past it.
                marks=_declared_mark(minimum, maximum, default),
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="knob",
    )


def gain_controls(calibration: GainCalibration) -> tuple[tuple[Any, ...], ...]:
    lower, upper = calibration.lower.as_array(), calibration.upper.as_array()
    base = calibration.nominal.as_array()
    return tuple(
        (f"gain-{name}", GAIN_LABELS[index], float(lower[index]), float(upper[index]),
         GAIN_STEPS[index], float(base[index]))
        for index, name in enumerate(("kp", "ki", "kd"))
    )


def path_options() -> list[dict[str, str]]:
    """Every split's paths, labelled with the split they belong to.

    The split is in the label rather than a separate control because it is the
    single most important thing to know before running: a held-out path shown
    without that label invites exactly the leak the manifests exist to prevent.
    """
    options: list[dict[str, str]] = []
    for split, specs in PATH_SPLITS.items():
        for spec in specs:
            options.append(
                {"label": f"{split} · {path_label(spec)}", "value": path_value(dict(spec))}
            )
    return options


# --- app -------------------------------------------------------------------


def create_app(
    models_root: str | Path | None = None,
    calibration_path: str | Path | None = None,
) -> Dash:
    runner = InteractiveRunner(
        Path(models_root) if models_root else None, calibration_path
    )
    gains = gain_controls(runner.calibration)
    all_controls = SCENARIO_CONTROLS + gains

    options = path_options()
    controller_options = runner.controller_options()
    default_selection = runner.default_selection()
    app = Dash(__name__, title="Thesis Experiment")
    app.layout = html.Div(
        [
            # Hidden to start with, because the drawer starts open. Its
            # visibility is the inverse of the drawer's, and `toggle_drawer`
            # keeps the two in step from there.
            html.Button(
                "☰  Controls",
                id="panel-toggle",
                className="panel-toggle",
                n_clicks=0,
                style={"display": "none"},
            ),
            html.Header(
                [
                    html.H1("Thesis Experiment"),
                    html.P(
                        "Fixed PID, scheduled PPO and direct RL on one plant, one path "
                        "and one disturbance. Set the dead time and the sensor noise in "
                        "the drawer, then run. Nothing this page produces is an "
                        "official result: those come from evaluate.py alone. "
                        f"Gain box: {runner.calibration_label}.",
                        className="kicker",
                    ),
                ],
                className="masthead",
            ),
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("Controls", className="drawer-title"),
                            html.Button("✕", id="panel-close", className="drawer-close", n_clicks=0),
                        ],
                        className="drawer-head",
                    ),
                    html.Div("Scenario", className="section-title"),
                    html.Div(
                        [
                            html.Label("Path", htmlFor="path-select"),
                            dcc.Dropdown(
                                id="path-select",
                                value=options[0]["value"],
                                options=options,
                                clearable=False,
                            ),
                        ],
                        className="selector",
                    ),
                    html.Div(
                        [
                            html.Label("Target speed", htmlFor="speed-select"),
                            dcc.Dropdown(
                                id="speed-select",
                                value=CONFIG.target_speeds[1],
                                options=[
                                    {"label": f"{speed:.1f} m/s", "value": speed}
                                    for speed in CONFIG.target_speeds
                                ],
                                clearable=False,
                            ),
                        ],
                        className="selector",
                    ),
                    html.Div(
                        [
                            html.Label("Evaluation mode", htmlFor="mode-select"),
                            dcc.Dropdown(
                                id="mode-select",
                                value="stationary",
                                options=[
                                    {"label": "stationary — condition holds all episode", "value": "stationary"},
                                    {"label": "transient — condition steps mid-episode", "value": "transient"},
                                ],
                                clearable=False,
                            ),
                        ],
                        className="selector",
                    ),
                    html.Div(
                        [
                            html.Label("Condition", htmlFor="condition-select"),
                            dcc.Dropdown(
                                id="condition-select",
                                value="nominal",
                                options=[{"label": name, "value": name} for name in CONDITIONS],
                                clearable=False,
                            ),
                        ],
                        className="selector",
                    ),
                    html.Div(
                        "Dead time queues the wheel command; sensor noise is the "
                        "standard deviation in millimetres of the corruption applied "
                        "to the e_ct the controller reads, never to the score. Both "
                        "are ignored by a nominal condition. The tick on each track "
                        f"marks the declared severity "
                        f"({1000.0 * CONFIG.delay_severity_s:g} ms and "
                        f"{1000.0 * CONFIG.noise_severity_m:g} mm); past it you are "
                        "exploring, not reproducing. Type in the box for a value the "
                        "track is too long to drag to.",
                        className="group-hint",
                    ),
                    *[slider_row(*control) for control in SCENARIO_CONTROLS],
                    html.Div("Methods to compare", className="section-title"),
                    html.Div(
                        "One row per method. Open a row to pick which trained "
                        "artifact runs; picking two seeds of the same method is how "
                        "seed spread shows up. An empty row sits the method out.",
                        className="group-hint",
                    ),
                    *[
                        html.Div(
                            [
                                html.Label(METHOD_LABELS[method]),
                                # A multi-select dropdown rather than a checklist:
                                # ten seeds per learned method is ten permanent
                                # rows of drawer, and the drawer is where the
                                # disturbance sliders have to stay reachable. The
                                # dropdown collapses to its chips and stays one
                                # row tall however many artifacts exist.
                                #
                                # Always in the tree, even with no options: Dash
                                # raises on a callback whose State id is absent,
                                # so an untrained method would take the page down
                                # rather than simply offer nothing to pick.
                                dcc.Dropdown(
                                    id=f"select-{method}",
                                    options=controller_options[method],
                                    value=default_selection[method],
                                    multi=True,
                                    placeholder=(
                                        "not compared"
                                        if controller_options[method]
                                        else "no trained artifact found"
                                    ),
                                    disabled=not controller_options[method],
                                ),
                            ],
                            className="selector method-group",
                        )
                        for method in METHOD_ORDER
                    ],
                    html.Div(runner.model_error, className="group-hint") if runner.model_error else html.Div(),
                    html.Details(
                        [
                            html.Summary("Advanced — hand-tuned PID gains"),
                            html.Div(
                                "These drive the 'hand-tuned' PID arm only. The calibrated "
                                "nominal and robust arms are unaffected, so moving a slider "
                                "never changes what you are comparing against.",
                                className="group-hint",
                            ),
                            *[slider_row(*control) for control in gains],
                            html.Button(
                                "Reset to calibrated nominal", id="gain-reset",
                                className="drawer-close", n_clicks=0,
                            ),
                        ],
                        className="advanced",
                    ),
                    html.Div("Run", className="section-title"),
                    html.Div(
                        "Episodes run only when you press this. Nothing runs while you "
                        "drag a slider.",
                        className="group-hint",
                    ),
                    html.Button("▶  Run scenario", id="run-button", className="run-button", n_clicks=0),
                ],
                id="controls-drawer",
                className="controls panel drawer open",
            ),
            html.Main(
                [
                    html.Div(id="replay-status", className="replay-status"),
                    html.Div(id="tab-content"),
                ],
                className="results",
            ),
        ],
        className="app",
    )

    @app.callback(
        Output("controls-drawer", "className"),
        Output("panel-toggle", "style"),
        Input("panel-toggle", "n_clicks"),
        Input("panel-close", "n_clicks"),
        State("controls-drawer", "className"),
        prevent_initial_call=True,
    )
    def toggle_drawer(_open_clicks: int, _close_clicks: int, current: str):
        """Open and close the drawer, and hide the opener while it is open.

        The button sits over the masthead, so leaving it visible with the
        drawer already open offers a control that does the same thing as the
        ✕ two centimetres to its right. Driven from the callback rather than
        from CSS because the open state lives in the className this same
        callback owns.
        """
        closed = "controls panel drawer"
        opening = ctx.triggered_id != "panel-close" and "open" not in (current or "")
        if opening:
            return f"{closed} open", {"display": "none"}
        return closed, {}

    for control_id, _label, _minimum, _maximum, _step, _default in SCENARIO_CONTROLS:
        app.callback(
            Output(f"s-{control_id}", "value"),
            Output(f"n-{control_id}", "value"),
            Input(f"s-{control_id}", "value"),
            Input(f"n-{control_id}", "value"),
            prevent_initial_call=True,
        )(
            lambda slider_value, number_value, cid=control_id: (
                (number_value, no_update)
                if ctx.triggered_id == f"n-{cid}"
                else (no_update, slider_value)
            )
        )

    # The gain rows carry a third input, the reset button. It has to live in the
    # SAME callback as the slider/number sync: two callbacks writing one Output
    # is a duplicate-output error in Dash.
    for control_id, _label, _minimum, _maximum, _step, default in gains:
        app.callback(
            Output(f"s-{control_id}", "value"),
            Output(f"n-{control_id}", "value"),
            Input(f"s-{control_id}", "value"),
            Input(f"n-{control_id}", "value"),
            Input("gain-reset", "n_clicks"),
            prevent_initial_call=True,
        )(
            lambda slider_value, number_value, _clicks, cid=control_id, base=default: (
                (base, base)
                if ctx.triggered_id == "gain-reset"
                else (number_value, no_update)
                if ctx.triggered_id == f"n-{cid}"
                else (no_update, slider_value)
            )
        )

    @app.callback(
        Output("tab-content", "children"),
        Input("run-button", "n_clicks"),
        State("path-select", "value"),
        State("speed-select", "value"),
        State("mode-select", "value"),
        State("condition-select", "value"),
        State("s-delay", "value"),
        State("s-noise", "value"),
        State("s-event-time", "value"),
        State("s-seed", "value"),
        State("select-pid", "value"),
        State("select-scheduled", "value"),
        State("select-direct", "value"),
        State("s-gain-kp", "value"),
        State("s-gain-ki", "value"),
        State("s-gain-kd", "value"),
    )
    def render(
        run_clicks,
        path_spec, speed, evaluation_mode, condition,
        delay_ms, noise_mm, event_time, seed,
        pid_keys, scheduled_keys, direct_keys, kp, ki, kd,
    ):
        controller_keys = (pid_keys or []) + (scheduled_keys or []) + (direct_keys or [])
        if not run_clicks:
            return html.Div(
                "Choose a scenario and controllers in the drawer, then press "
                "“Run scenario”. Interactive episodes are exploratory only — every "
                "number in the thesis comes from evaluate.py.",
                className="message",
            )
        try:
            scenario = runner.scenario_from_controls(
                path_spec, speed, condition, evaluation_mode,
                delay_ms, noise_mm, event_time, seed,
            )
            results_list = runner.run(
                scenario, controller_keys or [], (float(kp), float(ki), float(kd))
            )
            return interactive_view(results_list, scenario, runner)
        except (ValueError, KeyError, FileNotFoundError) as error:
            return html.Div(str(error), className="message error")

    @app.callback(
        Output("replay-status", "children"),
        Input({"role": "replay", "key": ALL}, "n_clicks"),
        State("path-select", "value"),
        State("speed-select", "value"),
        State("mode-select", "value"),
        State("condition-select", "value"),
        State("s-delay", "value"),
        State("s-noise", "value"),
        State("s-event-time", "value"),
        State("s-seed", "value"),
        State("s-gain-kp", "value"),
        State("s-gain-ki", "value"),
        State("s-gain-kd", "value"),
        prevent_initial_call=True,
    )
    def launch_replay(
        clicks, path_spec, speed, evaluation_mode, condition,
        delay_ms, noise_mm, event_time, seed, kp, ki, kd,
    ):
        """Open the MuJoCo viewer on the episode this panel is showing.

        A separate process because MuJoCo's viewer wants the main thread and
        would fight the Dash server for it. The scenario is passed as the same
        drawer values the charts were built from, and `replay.py` rebuilds it
        through `scenario_from_controls`, so the window cannot end up showing a
        different episode from the one on screen.
        """
        triggered = ctx.triggered_id
        # The ALL pattern fires once on render with every count at zero. Without
        # this a viewer window would open by itself every time a run finished.
        # `clicks` is normally the list of counts, one per matched button, but
        # is a bare int when a single component matches the pattern.
        counts = clicks if isinstance(clicks, (list, tuple)) else [clicks]
        if not isinstance(triggered, dict) or not any(counts or []):
            return no_update
        command = [
            sys.executable,
            str(config.ROOT / "replay.py"),
            "--controller", str(triggered["key"]),
            "--path", str(path_spec),
            "--speed", str(float(speed)),
            "--mode", str(evaluation_mode),
            "--condition", str(condition),
            "--delay-ms", str(float(delay_ms)),
            "--noise-mm", str(float(noise_mm)),
            "--event-time", str(float(event_time)),
            "--noise-seed", str(float(seed)),
            "--gains", f"{float(kp)},{float(ki)},{float(kd)}",
            "--models-root", str(runner.models_root),
        ]
        try:
            subprocess.Popen(command, cwd=str(config.ROOT))
        except OSError as error:
            return html.Div(f"Could not start the viewer: {error}", className="message error")
        return html.Div(
            f"Opening {triggered['key']} in a MuJoCo window. It re-simulates the "
            "episode, so give it a few seconds. The pink ghost is where the "
            "corrupted measurement puts the car; the metrics are printed in its "
            "console to check against the panel.",
            className="message",
        )

    @app.server.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        default=str(DEFAULT_MODELS_ROOT),
        help="Directory scanned for <arm>/seed_<n>/best_model.zip artifacts.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Gain box and baseline gains. Defaults to "
            f"{DEFAULT_CALIBRATION}, and falls back to the development default "
            "with a visible warning when that file does not exist yet."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.models_root, args.calibration)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
