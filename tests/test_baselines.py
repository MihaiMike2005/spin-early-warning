"""
Tests for spinwarn.baselines: B1, B2 and B3 scores from the dataset's raw signals.

A few hand-made rows with the columns the baselines read.
"""

import numpy as np
import pandas as pd
import pytest

from spinwarn import baselines


def rows():
    return pd.DataFrame({
        "abs_slip":      [10.0, 45.0, 50.0, 60.0, 85.0],
        "abs_slip_rate": [90.0, 80.0, 70.0, -5.0, 40.0],
        "time_to_90":    [5.0, 0.6, 0.57, 5.0, 0.125],
    })


def test_b1_is_abs_slip():
    assert baselines.score("b1", rows()).tolist() == [10.0, 45.0, 50.0, 60.0, 85.0]


def test_b3_is_minus_time_to_90_so_it_rises_toward_a_spin():
    assert baselines.score("b3", rows()).tolist() == [-5.0, -0.6, -0.57, -5.0, -0.125]


def test_b2_is_the_rate_over_theta1_and_minus_infinity_at_or_under_it():
    score = baselines.score("b2", rows(), {"theta1": 50})
    assert score.tolist() == [-np.inf, -np.inf, -np.inf, -5.0, 40.0]


@pytest.mark.parametrize("theta2", [-np.inf, -10.0, 0.0, 50.0])
def test_b2_alarms_exactly_where_slip_is_over_theta1_and_its_rate_over_theta2(theta2):
    df = rows()
    alarm = baselines.score("b2", df, {"theta1": 40}) > theta2
    expected = (df["abs_slip"] > 40) & (df["abs_slip_rate"] > theta2)
    assert alarm.tolist() == expected.tolist()


@pytest.mark.parametrize("method, params, message", [
    ("b1", {"theta1": 50}, r"b1 takes params \[\], not \['theta1'\]"),
    ("b2", {}, r"b2 takes params \['theta1'\], not \[\]"),
    ("b2", {"theta1": 50, "theta2": 1}, r"not \['theta1', 'theta2'\]"),
    ("b2", {"theta1": "50"}, "b2's theta1 is a number, not '50'"),
    ("b2", {"theta1": True}, "b2's theta1 is a number, not True"),
])
def test_params_must_be_exactly_the_ones_a_baseline_takes(method, params, message):
    with pytest.raises(ValueError, match=message):
        baselines.score(method, rows(), params)


def test_every_baseline_lists_the_signals_it_reads():
    assert set(baselines.SIGNALS) == set(baselines.BASELINES)
    for method, signals in baselines.SIGNALS.items():
        params = {name: 0 for name in baselines.params_of(method)}
        baselines.score(method, rows()[signals], params)
