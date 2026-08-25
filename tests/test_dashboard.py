"""Dashboard tests.

The important one is `test_interactive_run_matches_batch_evaluator`: the gate in
TODO.md section 9 is that the dashboard reproduces batch traces for the same
scenario. It holds only because the dashboard runs the SAME `Scenario` through
the SAME `rollout.run_episode` the evaluator uses; the test is what stops a
future shortcut (a private episode loop, a quietly different reset seed) from
breaking that silently.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard
from calibration import GainCalibration
from env import PathFollowingEnv
from rollout import run_episode
from scenarios import Scenario, ScenarioManifest, build_manifest


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> dashboard.InteractiveRunner:
    calibration_path = tmp_path_factory.mktemp("dashboard") / "calibration.json"
    GainCalibration.development_default().save(calibration_path)
    return dashboard.InteractiveRunner(calibration_path=calibration_path)


@pytest.fixture(scope="module")
def scenario(runner: dashboard.InteractiveRunner) -> Scenario:
    return runner.scenario_from_controls(
        dashboard.path_value({"kind": "arc"}),
        target_speed=0.5,
        condition="combined",
        evaluation_mode="transient",
        delay_ms=60.0,
        noise_mm=0.3,
        event_time=3.0,
        noise_seed=12345,
    )


def test_scenario_matches_manifest_construction() -> None:
    """A hand-built scenario is field-for-field a manifest scenario.

    Only the id may differ: interactive ones are prefixed so nothing downstream
    can mistake one for a frozen manifest entry.
    """
    manifest = build_manifest("train", speeds=(0.5,))
    reference = next(
        s
        for s in manifest.scenarios
        if s.path["kind"] == "arc"
        and s.evaluation_mode == "transient"
        and s.condition == "combined"
    )
    built = dashboard.InteractiveRunner().scenario_from_controls(
        dashboard.path_value({"kind": "arc"}),
        target_speed=0.5,
        condition="combined",
        evaluation_mode="transient",
        delay_ms=1000.0 * reference.events[0].value,
        noise_mm=1000.0 * reference.events[1].value,
        event_time=reference.events[0].start_s,
        noise_seed=reference.noise_seed,
    )
    assert built.scenario_id.startswith("interactive-")
    assert built.path == reference.path
    assert built.target_speed == reference.target_speed
    assert built.evaluation_mode == reference.evaluation_mode
    assert built.condition == reference.condition
    assert built.noise_seed == reference.noise_seed
    assert built.initial_delay_s == reference.initial_delay_s
    assert built.initial_noise_std_m == reference.initial_noise_std_m
    assert built.events == reference.events
    assert built.geometry == reference.geometry


def test_interactive_run_matches_batch_evaluator(
    runner: dashboard.InteractiveRunner, scenario: Scenario
) -> None:
    """The replay-equivalence gate: same scenario, same trace, sample for sample."""
    calibration = GainCalibration.development_default()
    env = PathFollowingEnv(
        mode="fixed", training=False, calibration=calibration, fixed_gains=calibration.nominal
    )
    try:
        expected = run_episode(env, scenario)
    finally:
        env.close()

    result = runner.run(scenario, ["pid_global_nominal"], (26.0, 0.5, 4.4))[0]

    # Wall-clock inference timings are measurements of this machine, not of the
    # episode, and differ run to run by design. Everything else must match.
    timing = {"mean_inference_ms", "p99_inference_ms"}
    produced_metrics = {k: v for k, v in result["metrics"].items() if k not in timing}
    expected_metrics = {k: v for k, v in expected.metrics.items() if k not in timing}
    assert produced_metrics == expected_metrics
    assert len(result["trace"]) == len(expected.trace)
    for produced, reference in zip(result["trace"], expected.trace):
        assert produced == reference


def test_gain_sliders_do_not_disturb_the_other_arms(
    runner: dashboard.InteractiveRunner, scenario: Scenario
) -> None:
    """Moving a gain slider re-runs only the drawer arm.

    If the cache key ever picks up the gains for every controller, the panels
    would silently be comparing against different episodes.
    """
    first = runner.run(scenario, ["pid_global_nominal", "pid_custom"], (26.0, 0.5, 4.4))
    second = runner.run(scenario, ["pid_global_nominal", "pid_custom"], (40.0, 0.5, 4.4))

    assert first[0]["metrics"] == second[0]["metrics"]
    assert first[1]["metrics"] != second[1]["metrics"]


def test_gains_outside_the_calibrated_box_are_rejected(
    runner: dashboard.InteractiveRunner, scenario: Scenario
) -> None:
    with pytest.raises(ValueError, match="outside the calibrated"):
        runner.run(scenario, ["pid_custom"], (500.0, 0.5, 4.4))


def test_no_controller_selected_is_an_error(
    runner: dashboard.InteractiveRunner, scenario: Scenario
) -> None:
    with pytest.raises(ValueError, match="at least one controller"):
        runner.run(scenario, [], (26.0, 0.5, 4.4))


@pytest.mark.parametrize("bad", ["", "not json", json.dumps({"nokind": 1}), json.dumps(["arc"])])
def test_bad_path_specification_is_rejected(
    runner: dashboard.InteractiveRunner, bad: str
) -> None:
    with pytest.raises(ValueError, match="valid path"):
        runner.scenario_from_controls(bad, 0.5, "nominal", "stationary", 0.0, 0.0, 3.0, 1)


def test_allocation_spans_are_merged() -> None:
    rows = [
        {"t": 0.00, "u_left": 0.5, "u_left_applied": 0.5, "u_right": 0.5, "u_right_applied": 0.5},
        {"t": 0.02, "u_left": 1.5, "u_left_applied": 1.0, "u_right": 0.5, "u_right_applied": 0.5},
        {"t": 0.04, "u_left": 1.5, "u_left_applied": 1.0, "u_right": 0.5, "u_right_applied": 0.5},
        {"t": 0.06, "u_left": 0.5, "u_left_applied": 0.5, "u_right": 0.5, "u_right_applied": 0.5},
        {"t": 0.50, "u_left": 0.5, "u_left_applied": 0.5, "u_right": 1.5, "u_right_applied": 1.0},
    ]
    spans = dashboard._allocation_spans(rows)
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(0.02)


# --- batch tab -------------------------------------------------------------


def _write_results(directory: Path) -> Path:
    """A minimal but realistic results directory, including a ragged arm.

    `pid_per_path_nominal` deliberately has no row for the second scenario, the
    way evaluate.py skips a scenario whose path kind has no calibrated gains.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest = ScenarioManifest(
        name="test",
        split="train",
        scenarios=(
            Scenario(
                scenario_id="train-arc-v0.5-stationary-nominal",
                path={"kind": "arc"},
                target_speed=0.5,
                evaluation_mode="stationary",
                condition="nominal",
                noise_seed=1,
            ),
            Scenario(
                scenario_id="train-slalom-v0.5-stationary-nominal",
                path={"kind": "slalom"},
                target_speed=0.5,
                evaluation_mode="stationary",
                condition="nominal",
                noise_seed=2,
            ),
        ),
    )
    manifest.save(directory / "scenario_manifest.json")
    rows = [
        {
            "controller": controller,
            "training_seed": "",
            "evaluation_mode": "stationary",
            "condition": "nominal",
            "path_kind": kind,
            "target_speed": "0.5",
            "scenario_id": f"train-{kind}-v0.5-stationary-nominal",
            "finished": "True",
            "failure_adjusted_error_m": "0.0123",
            "mean_distance_m": "0.0100",
            "max_distance_m": "0.0300",
            "progress_pct": "100.0",
            "duration_s": "12.5",
        }
        for kind in ("arc", "slalom")
        for controller in ("pid_global_nominal", "pid_per_path_nominal")
        if not (controller == "pid_per_path_nominal" and kind == "slalom")
    ]
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return directory


def test_load_batch_reads_controllers_and_scenarios(tmp_path: Path) -> None:
    batch = dashboard.load_batch(_write_results(tmp_path / "run"))
    assert batch["controllers"] == ["pid_global_nominal", "pid_per_path_nominal"]
    assert len(batch["scenarios"]) == 2
    assert batch["specs"]["train-arc-v0.5-stationary-nominal"] == {"kind": "arc"}


def test_batch_view_reports_a_missing_combination_instead_of_raising(tmp_path: Path) -> None:
    """The regression this replaces used to 500 the whole page."""
    batch = dashboard.load_batch(_write_results(tmp_path / "run"))
    view = dashboard.batch_view(
        batch["root"], batch["episodes"], batch["specs"],
        "pid_per_path_nominal", "train-slalom-v0.5-stationary-nominal",
    )
    assert "has no result for" in str(view)


def test_batch_view_renders_without_traces(tmp_path: Path) -> None:
    batch = dashboard.load_batch(_write_results(tmp_path / "run"))
    view = dashboard.batch_view(
        batch["root"], batch["episodes"], batch["specs"],
        "pid_global_nominal", "train-arc-v0.5-stationary-nominal",
    )
    assert "No trace saved" in str(view)


def test_empty_results_directory_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no episodes.csv"):
        dashboard.load_batch(empty)


def test_unsafe_identifiers_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsafe path characters"):
        dashboard.validate_batch_rows(
            [{"controller": "../escape", "scenario_id": "ok", "training_seed": ""}]
        )


def test_app_builds_without_models_or_calibration() -> None:
    app = dashboard.create_app()
    assert app.layout is not None


def test_discover_models_rejects_a_mismatched_metadata_mode(tmp_path: Path) -> None:
    seed_dir = tmp_path / "ppo_scheduler_nominal" / "seed_11"
    seed_dir.mkdir(parents=True)
    (seed_dir / "best_model.zip").write_bytes(b"")
    (seed_dir / "metadata.json").write_text(json.dumps({"mode": "direct"}), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        dashboard.discover_models(tmp_path)


# --- three-method comparison ------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        ("pid_global_robust", "pid"),
        ("pid_custom", "pid"),
        ("ppo_scheduler_disturbed:11", "scheduled"),
        ("ppo_direct_nominal:71", "direct"),
    ),
)
def test_method_is_derived_from_the_controller_key(key: str, expected: str) -> None:
    assert dashboard.method_of(key) == expected


def test_an_unknown_controller_key_has_no_method() -> None:
    with pytest.raises(ValueError, match="no known method"):
        dashboard.method_of("something_else")


def test_results_come_back_in_method_order_whatever_order_was_ticked(
    runner: dashboard.InteractiveRunner, scenario: Scenario
) -> None:
    """Selection order must not reorder the legend.

    The charts overlay every controller on one axis, so a legend that reshuffles
    when a checkbox is re-ticked makes two screenshots of the same comparison
    disagree about which line is which.
    """
    results = runner.run(scenario, ["pid_custom", "pid_global_robust"], (26.0, 0.5, 4.4))
    assert [result["method"] for result in results] == ["pid", "pid"]
    assert [result["color"] for result in results] == list(
        dashboard.METHOD_COLORS["pid"][:2]
    )


def test_every_method_keeps_its_own_hue(
    runner: dashboard.InteractiveRunner, scenario: Scenario
) -> None:
    results = runner.run(scenario, ["pid_global_nominal", "pid_global_robust"], (26.0, 0.5, 4.4))
    for result in results:
        assert result["color"] in dashboard.METHOD_COLORS[result["method"]]
        assert result["label"].startswith(dashboard.METHOD_LABELS["pid"])


def test_the_default_selection_offers_the_plain_pid_baseline() -> None:
    selection = dashboard.InteractiveRunner().default_selection()
    assert selection["pid"] == ["pid_global_robust"]
    assert set(selection) == set(dashboard.METHOD_ORDER)


def test_direct_rl_is_not_credited_with_a_steady_gain() -> None:
    """env.py records gain_total_variation_per_s as 0.0 for direct mode.

    Printed as a number that reads as perfect gain discipline, which inverts the
    truth: the arm schedules no gain at all.
    """
    table = dashboard.scoreboard(
        [
            {
                "method": "direct",
                "label": "Direct RL — disturbed · seed 11",
                "color": "#c792ea",
                "metrics": {
                    "finished": True,
                    "failure_adjusted_error_m": 0.0066,
                    "mean_distance_m": 0.012,
                    "max_distance_m": 0.03,
                    "steer_total_variation_per_s": 1.04,
                    "gain_total_variation_per_s": 0.0,
                },
            }
        ]
    )
    cells = [cell.children for cell in table.children[1].children[0].children]
    assert "n/a" in cells
    assert "6.60" in cells  # J_FA reported in millimetres, not metres


def test_the_steering_differential_is_derived_from_the_applied_commands() -> None:
    rows = [
        {"t": 0.0, "u_left_applied": 0.4, "u_right_applied": 0.1},
        {"t": 0.02, "u_left_applied": -0.2, "u_right_applied": 0.3},
    ]
    assert dashboard._derive(rows, "steer_differential") == pytest.approx([0.3, -0.5])
    assert dashboard._derive([{"actuator_delay_s": 0.15}], "delay_ms") == pytest.approx([150.0])


def test_the_gain_chart_skips_arms_that_have_no_gains() -> None:
    """A direct-mode trace carries no kp column and must not fake one."""
    scenario_stub = Scenario(
        scenario_id="stub", path={"kind": "arc"}, target_speed=0.5,
        evaluation_mode="stationary", condition="nominal", noise_seed=0,
    )
    results = [
        {
            "method": "scheduled",
            "label": "Scheduled PPO — disturbed · seed 11",
            "color": "#4fc3f7",
            "trace": [{"t": 0.0, "kp": 26.0}, {"t": 0.02, "kp": 30.0}],
        },
        {
            "method": "direct",
            "label": "Direct RL — disturbed · seed 11",
            "color": "#c792ea",
            "trace": [{"t": 0.0}, {"t": 0.02}],
        },
    ]
    figure = dashboard.comparison_figure(results, scenario_stub, "kp", "gains", "Kp")
    assert [trace.name for trace in figure.data] == ["Scheduled PPO — disturbed · seed 11"]


def test_a_batch_run_is_plotted_against_the_gain_box_that_produced_it(tmp_path: Path) -> None:
    """Old runs carry their own scheduler box; the box moved mid-project."""
    results = tmp_path / "run"
    results.mkdir()
    (results / "scheduler_calibration.json").write_text(
        json.dumps(
            {
                "nominal": {"kp": 26.0, "ki": 0.5, "kd": 4.4},
                "robust": {"kp": 26.0, "ki": 0.5, "kd": 4.4},
                "lower": {"kp": 5.0, "ki": 0.0, "kd": 2.5},
                "upper": {"kp": 300.0, "ki": 1.5, "kd": 6.0},
            }
        ),
        encoding="utf-8",
    )
    assert dashboard._batch_gain_box(results).upper.kp == 300.0
    # A results directory without one still renders, on the documented default.
    assert dashboard._batch_gain_box(tmp_path / "absent").upper.kp == (
        GainCalibration.development_default().upper.kp
    )
