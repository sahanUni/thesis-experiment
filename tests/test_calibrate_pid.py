from calibration import PIDGains
from calibrate_pid import (
    scheduler_bounds,
    select_practical_candidate,
    validation_shortlist,
)


def candidate(kp: float, ki: float, kd: float, error: float, variation: float) -> dict[str, float]:
    return {
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "completion_rate": 1.0,
        "failure_adjusted_error_m": error,
        "steer_total_variation_per_s": variation,
        "gain_l1": kp + ki + kd,
    }


def test_practical_selection_prefers_smoother_controller_within_half_percent() -> None:
    absolute_best = candidate(50.0, 4.5, 9.0, 0.001000, 0.50)
    smoother = candidate(48.0, 4.0, 5.0, 0.001004, 0.25)
    outside_tolerance = candidate(40.0, 3.0, 3.0, 0.001006, 0.10)

    assert select_practical_candidate([absolute_best, smoother, outside_tolerance]) is smoother


def test_validation_shortlist_contains_distinct_tradeoffs() -> None:
    rows = [
        candidate(50.0, 4.5, 9.0, 0.001000, 0.50),
        candidate(50.0 + 1e-9, 4.5, 9.0, 0.001000001, 0.50),
        candidate(48.0, 4.0, 5.0, 0.001004, 0.25),
        candidate(47.0, 3.5, 6.0, 0.0010045, 0.30),
        candidate(44.0, 3.0, 4.0, 0.001010, 0.20),
        candidate(42.0, 2.0, 3.0, 0.001018, 0.15),
        candidate(30.0, 1.0, 1.0, 0.001100, 0.05),
    ]

    shortlist = validation_shortlist(rows)
    gain_keys = {(round(row["kp"], 6), round(row["ki"], 6), round(row["kd"], 6)) for _, row in shortlist}

    assert len(shortlist) == 5
    assert len(gain_keys) == 5
    assert shortlist[0][0] == "best_error"
    assert any(role == "smoothest_practical" for role, _ in shortlist)


def test_scheduler_bounds_enclose_selected_gains_with_a_small_margin() -> None:
    lower, upper = scheduler_bounds(
        PIDGains(48.0, 4.2, 5.0),
        PIDGains(29.0, 4.5, 2.2),
    )

    assert lower.kp > 20.0
    assert lower.ki > 3.0
    assert upper.kd < 6.0
    assert lower.kp <= 29.0 <= upper.kp
    assert lower.kp <= 48.0 <= upper.kp
