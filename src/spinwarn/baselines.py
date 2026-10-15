"""
Physics baselines: B1 slip threshold, B2 slip plus rate, B3 time to 90°.

Each is a score per frame from the raw signals the dataset keeps, higher
nearer a spin, with nothing to fit. A frame alarms when its score is over
the threshold, so the threshold is each one's tuned parameter, set on CV
scores like any model's: θ for B1, θ2 for B2, and −τ for B3. B2's other
one, θ1, gates its score; run.py cv picks it from B2_THETA1_DEG by CV
headline recall (cv.choose_theta1).
"""

import inspect

import numpy as np


def b1(rows) -> np.ndarray:
    """|slip angle|: alarm when it's over θ."""
    return rows["abs_slip"].to_numpy(dtype=float)


def b2(rows, theta1) -> np.ndarray:
    """d|slip|/dt where |slip| is over theta1 (°), −∞ elsewhere: alarm when both are over their θ.

    −∞ never alarms, whatever the threshold: evaluate alarms on score > threshold, strictly.
    """
    over = rows["abs_slip"].to_numpy(dtype=float) > theta1
    return np.where(over, rows["abs_slip_rate"].to_numpy(dtype=float), -np.inf)


def b3(rows) -> np.ndarray:
    """−time to 90°: alarm when time to 90° drops under τ, so when this is over −τ."""
    return -rows["time_to_90"].to_numpy(dtype=float)


BASELINES = {"b1": b1, "b2": b2, "b3": b3}
# The dataset's raw signals each one reads.
SIGNALS = {"b1": ["abs_slip"], "b2": ["abs_slip", "abs_slip_rate"], "b3": ["time_to_90"]}


def params_of(method) -> list:
    """The params a baseline takes, by name: theta1 for b2, none for b1 and b3."""
    return list(inspect.signature(BASELINES[method]).parameters)[1:]


def check_params(method, params):
    """ValueError unless params are exactly the ones method takes, each a number."""
    expected = params_of(method)
    if sorted(params) != sorted(expected):
        raise ValueError(f"{method} takes params {expected}, not {sorted(params)}")
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{method}'s {name} is a number, not {value!r}")


def score(method, rows, params=None) -> np.ndarray:
    """A baseline's score on every row of rows, a slice of the dataset."""
    params = params or {}
    check_params(method, params)
    return BASELINES[method](rows, **params)
