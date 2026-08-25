"""Dashboard tests.

The important one is `test_interactive_run_matches_batch_evaluator`: the gate in
TODO.md section 9 is that the dashboard reproduces batch traces for the same
scenario. It holds only because the dashboard runs the SAME `Scenario` through
the SAME `rollout.run_episode` the evaluator uses; the test is what stops a
future shortcut (a private episode loop, a quietly different reset seed) from
breaking that silently.
"""

from __future__ import annotations

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


def test_interactive_run_matches_the_batch_evaluator(
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
