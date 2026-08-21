import pytest

from core.pid import ConditionalPIDController, PIDController


def test_derivative_filter_has_no_reset_kick_and_smooths_step():
    pid = PIDController(kp=0.0, kd=1.0, derivative_filter_tau=0.02)
    assert pid.update(1.0, 0.002) == pytest.approx(0.0)
    response = pid.update(0.0, 0.002)
    assert 0.0 < response < 500.0


def test_conditional_integration_rolls_back_rejected_increment():
    pid = ConditionalPIDController(kp=0.0, ki=1.0, output_limits=(-1.0, 1.0))
    requested = pid.update(-0.5, 1.0)
    assert requested == pytest.approx(0.5)
    assert pid.apply_output_feedback(0.0)
    assert pid._integral == pytest.approx(0.0)


def test_parallel_pid_terms_and_output_limit():
    proportional = PIDController(kp=2.0, ki=0.0, kd=0.0)
    assert proportional.update(-0.25, 0.1) == pytest.approx(0.5)

    integral = PIDController(kp=0.0, ki=2.0, kd=0.0)
    assert integral.update(-0.5, 0.2) == pytest.approx(0.2)
    assert integral.update(-0.5, 0.2) == pytest.approx(0.4)

    saturated = ConditionalPIDController(kp=10.0, ki=1.0, kd=0.0, output_limits=(-1.0, 1.0))
    assert saturated.update(-1.0, 0.1) == pytest.approx(1.0)
    assert saturated.last_internal_hold
    assert saturated._integral == pytest.approx(0.0)
