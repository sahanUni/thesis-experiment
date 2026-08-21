import pytest

from metrics import failure_adjusted_error, paired_differences, summarize


def row(controller, scenario_id, finished, adjusted):
    return {
        "controller": controller,
        "scenario_id": scenario_id,
        "evaluation_mode": "stationary",
        "condition": "nominal",
        "finished": finished,
        "failure_adjusted_error_m": adjusted,
        "mean_distance_m": 0.1,
        "rms_distance_m": 0.1,
        "p95_distance_m": 0.1,
        "max_distance_m": 0.2,
        "progress_pct": 50.0,
        "steer_total_variation_per_s": 1.0,
        "wheel_saturation_fraction": 0.0,
    }


def test_summary_keeps_failed_episode_in_primary_error():
    rows = [row("a", "one", True, 0.1), row("a", "two", False, 0.9)]
    result = summarize(rows)[0]
    assert result["completion_rate"] == 0.5
    assert result["failure_adjusted_error_m"] == 0.5


def test_paired_difference_matches_scenario_ids():
    rows = [row("a", "one", True, 0.3), row("b", "one", True, 0.1)]
    assert paired_differences(rows, "a", "b").tolist() == pytest.approx([0.2])


@pytest.mark.parametrize(
    "finished,duration,expected",
    [
        (False, 2.0, 0.85),
        (False, 8.0, 0.25),
        (False, 10.0, 0.05),
        (True, 2.0, 0.05),
    ],
)
def test_failure_adjustment_handles_early_late_timeout_and_completion(finished, duration, expected):
    assert failure_adjusted_error(
        0.5,
        finished=finished,
        duration_s=duration,
        max_time_s=10.0,
        corridor_m=1.0,
    ) == pytest.approx(expected)
