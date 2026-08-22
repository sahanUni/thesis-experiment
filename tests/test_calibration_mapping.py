import numpy as np

from calibration import GainCalibration, PIDGains


CALIBRATION = GainCalibration(
    nominal=PIDGains(26.0, 0.5, 4.4),
    robust=PIDGains(26.0, 0.5, 4.4),
    lower=PIDGains(10.0, 0.0, 2.5),
    upper=PIDGains(50.0, 1.5, 6.0),
)


def test_piecewise_mapping_keeps_zero_at_calibrated_baseline():
    np.testing.assert_allclose(
        CALIBRATION.map_action(np.zeros(3)).as_array(),
        CALIBRATION.nominal.as_array(),
    )
    np.testing.assert_allclose(
        CALIBRATION.map_action(-np.ones(3)).as_array(),
        CALIBRATION.lower.as_array(),
    )
    np.testing.assert_allclose(
        CALIBRATION.map_action(np.ones(3)).as_array(),
        CALIBRATION.upper.as_array(),
    )


def test_piecewise_action_inverse_round_trips_gains():
    gains = PIDGains(18.0, 1.0, 5.2)
    action = CALIBRATION.action_for(gains)
    np.testing.assert_allclose(CALIBRATION.map_action(action).as_array(), gains.as_array())
