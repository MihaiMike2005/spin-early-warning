"""
Causal window features over the last WINDOW_S, per segment.

make_features turns a session from data.load_session into one row of
features per grid frame, each from t and the WINDOW_S before it alone
(windows over (t − WINDOW_S, t], slopes from t − WINDOW_S to t): nothing
from the future, and nothing from before a gap. FEATURES names them, in
order, for everything downstream.
"""

import numpy as np
import pandas as pd

from spinwarn.config import GRID_HZ, WINDOW_S


# Signals with a left and a right, taken relative to the slide frame by
# frame, as <signal>_rel: × sign(slip), the way data.add_signals builds
# countersteer from steer. Signed, a left and a right drift read opposite,
# which a linear model can't use, and the only extra they carry is which way
# the track's corners go. In held drifts, yaw_rate_rel and lat_g_rel read negative.
RELATIVE = ("yaw_rate", "yaw_acc", "lat_g")

# The Project Plan's feature table. Each feature is <signal>_<stat>: last is
# the value at t; mean and max are over the window; slope is the change over
# the last WINDOW_S, per second. Steering is countersteer, steer relative to
# the slide, and lock, how close to full lock.
SIGNAL_STATS = {
    "abs_slip": ("last", "mean", "max", "slope"),
    "abs_slip_rate": ("last", "max"),
    "time_to_90": ("last",),
    "yaw_rate_rel": ("last", "mean", "slope"),
    "yaw_acc_rel": ("last",),
    "lat_g_rel": ("last", "mean"),
    "lon_g": ("last", "mean"),
    "countersteer": ("last", "mean"),
    "lock": ("last",),
    "gas": ("last", "mean"),
    "brake": ("last", "mean"),
    "speed_kmh": ("last", "slope"),
    "rear_slip": ("mean", "slope"),
}
FEATURES = [f"{signal}_{stat}" for signal, stats in SIGNAL_STATS.items() for stat in stats]
# The data.add_signals columns make_features reads.
INPUTS = [signal.removesuffix("_rel") for signal in SIGNAL_STATS] + ["slip_deg"]


def make_features(df) -> pd.DataFrame:
    """The FEATURES of a grid session, one row per frame, on df's index.

    Computed segment by segment, so no window or slope reaches back across
    a gap. A segment's first WINDOW_S has partial windows and no slope yet
    (NaN); label() excludes those frames, and every feature is known after.
    Not quite with YAW_AXIS None: yaw rate from heading is itself unknown
    for a segment's first RATE_WINDOW_S, so yaw_rate_rel_slope is too, that
    much longer.
    """
    df = df.assign(**{f"{name}_rel": df[name] * np.sign(df["slip_deg"]) for name in RELATIVE})
    pieces = [_segment_features(segment) for _, segment in df.groupby("segment")]
    return pd.concat(pieces).loc[df.index]


def _segment_features(segment) -> pd.DataFrame:
    """make_features for the frames of one segment."""
    signals = segment[list(SIGNAL_STATS)]
    # A time index for rolling(): to the nanosecond, so WINDOW_S back lands
    # exactly on a grid frame, and the window holds WINDOW_S × GRID_HZ frames.
    timed = signals.set_axis(pd.to_timedelta(segment["t"].to_numpy(), unit="s"))
    window = timed.rolling(pd.Timedelta(seconds=WINDOW_S))   # (t − WINDOW_S, t]: closed right, never centred
    n = round(WINDOW_S * GRID_HZ)
    stats = {
        "last": signals,
        "mean": window.mean().set_axis(segment.index),
        "max": window.max().set_axis(segment.index),
        # Frames inside a segment are exactly 1/GRID_HZ apart, so n back is t − WINDOW_S.
        "slope": (signals - signals.shift(n)) / WINDOW_S,
    }
    return pd.DataFrame({
        f"{signal}_{stat}": stats[stat][signal]
        for signal, names in SIGNAL_STATS.items() for stat in names
    })
