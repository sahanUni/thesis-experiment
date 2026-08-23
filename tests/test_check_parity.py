import copy

import pytest

import check_parity


REFERENCE = {
    "manifest_sha256": "abc",
    "arms": {
        "pid_global_nominal": {
            "gains": [250.0, 4.5454, 0.3967],
            "conditions": {
                "nominal": {
                    "episodes": 12,
                    "completion_rate": 1.0,
                    "failure_adjusted_error_m": 0.0011585981831828752,
                    "mean_distance_m": 0.0025111056851964105,
                },
                "combined": {
                    "episodes": 12,
                    "completion_rate": 0.0,
                    "failure_adjusted_error_m": 0.15196456316227175,
                    "mean_distance_m": 0.15192431384417304,
                },
            },
        },
        "pid_global_robust": {
            "gains": [8.0, 0.5, 4.4],
            "conditions": {
                "nominal": {
                    "episodes": 12,
                    "completion_rate": 1.0,
                    "failure_adjusted_error_m": 0.0023542035866620137,
                    "mean_distance_m": 0.005110214891253787,
                },
                "combined": {
                    "episodes": 12,
                    "completion_rate": 1.0,
                    "failure_adjusted_error_m": 0.009377856833512492,
                    "mean_distance_m": 0.018224531158566204,
                },
            },
        },
    },
}


def measured(**mutations):
    payload = copy.deepcopy(REFERENCE)
    for path, value in mutations.items():
        arm, condition, key = path.split("__")
        payload["arms"][arm]["conditions"][condition][key] = value
    return payload


def test_identical_measurement_passes():
    problems, _ = check_parity.compare(REFERENCE, measured(), check_parity.DEFAULT_TOLERANCE)
    assert problems == []


def test_the_observed_cross_machine_drift_passes():
    """The real cpunode13 numbers against the laptop reference."""
    problems, notes = check_parity.compare(
        REFERENCE,
        measured(
            pid_global_nominal__nominal__failure_adjusted_error_m=0.0011600911373899159,
            pid_global_nominal__nominal__mean_distance_m=0.002514042985849821,
            pid_global_nominal__combined__failure_adjusted_error_m=0.15752354872029165,
            pid_global_nominal__combined__mean_distance_m=0.15747981383352175,
            pid_global_robust__nominal__failure_adjusted_error_m=0.0023623016132293907,
            pid_global_robust__nominal__mean_distance_m=0.005128505008361512,
            pid_global_robust__combined__failure_adjusted_error_m=0.009379022663809005,
            pid_global_robust__combined__mean_distance_m=0.01823950907184743,
        ),
        check_parity.DEFAULT_TOLERANCE,
    )
    assert problems == []
    assert any("completion-only" in note for note in notes)


def test_changed_completion_always_fails():
    problems, _ = check_parity.compare(
        REFERENCE,
        measured(pid_global_robust__combined__completion_rate=0.9166666666666666),
        check_parity.DEFAULT_TOLERANCE,
    )
    assert any("completion_rate" in problem for problem in problems)


def test_a_whole_percent_error_change_fails_on_a_completing_arm():
    problems, _ = check_parity.compare(
        REFERENCE,
        measured(pid_global_robust__combined__failure_adjusted_error_m=0.0096),
        check_parity.DEFAULT_TOLERANCE,
    )
    assert any("failure_adjusted_error_m" in problem for problem in problems)


def test_failed_arms_skip_the_continuous_check_entirely():
    problems, _ = check_parity.compare(
        REFERENCE,
        measured(pid_global_nominal__combined__failure_adjusted_error_m=0.9),
        check_parity.DEFAULT_TOLERANCE,
    )
    assert problems == []


def test_a_different_manifest_invalidates_the_comparison():
    payload = measured()
    payload["manifest_sha256"] = "different"
    problems, _ = check_parity.compare(REFERENCE, payload, check_parity.DEFAULT_TOLERANCE)
    assert any("manifest" in problem for problem in problems)


def test_changed_gains_fail():
    payload = measured()
    payload["arms"]["pid_global_robust"]["gains"] = [18.0, 0.5, 4.4]
    problems, _ = check_parity.compare(REFERENCE, payload, check_parity.DEFAULT_TOLERANCE)
    assert any("gains" in problem for problem in problems)


@pytest.mark.parametrize("tolerance", (0.0, 1e-9))
def test_a_strict_tolerance_still_catches_the_drift(tolerance):
    problems, _ = check_parity.compare(
        REFERENCE,
        measured(pid_global_robust__nominal__failure_adjusted_error_m=0.0023623016132293907),
        tolerance,
    )
    assert problems
