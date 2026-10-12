"""
Tests for spinwarn.features: causal windows, slopes and segment isolation.

Synthetic grid sessions with every signal make_features reads, so they run
anywhere without data/raw/.
"""

import numpy as np
import pandas as pd
import pytest

from driftscore.logger import COLUMNS

from spinwarn import data, features
from spinwarn.config import GRID_HZ, WINDOW_S, YAW_AXIS


SIGNALS = features.INPUTS
RAW_SIGNALS = [name for name in COLUMNS if name != "t"]
WINDOW_FRAMES = round(WINDOW_S * GRID_HZ)


def session(seconds, segment=0, start=0.0, seed=0):
    """A grid session of `seconds` with every signal a random walk."""
    t = (round(start * GRID_HZ) + np.arange(round(seconds * GRID_HZ))) / GRID_HZ
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.normal(size=len(t)).cumsum() for name in SIGNALS})
    df.insert(0, "t", t)
    df.insert(1, "segment", segment)
    return df


def two_segments():
    """5 s in segment 0, a 0.5 s gap (under WINDOW_S), then 5 s in segment 1."""
    return pd.concat([session(5.0, segment=0), session(5.0, segment=1, start=5.5, seed=1)],
                     ignore_index=True)


def packets(seed=0):
    """Raw logger rows at ~333 Hz, every column a random walk at 60 km/h: 3 s, a 0.5 s pause, 2.5 s."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.arange(0.0, 3.0, 0.003), np.arange(3.5, 6.0, 0.003)])
    raw = pd.DataFrame({name: rng.normal(size=len(t)).cumsum() for name in COLUMNS})
    raw["t"], raw["speed_kmh"] = t, 60.0
    return raw


def noise(df, rows, columns=SIGNALS, seed=2):
    """A copy of df with `columns` in `rows` replaced by large noise."""
    changed = df.copy()
    changed.loc[rows, columns] = np.random.default_rng(seed).normal(size=(rows.sum(), len(columns))) * 100
    return changed


# --- causality and segments ---------------------------------------------------

@pytest.mark.parametrize("t_star", [2.5, 5.2, 6.0, 8.25])   # segment 0, the gap, segment 1's first window, later
def test_changing_everything_after_t_star_leaves_every_feature_up_to_t_star_alone(t_star):
    df = two_segments()
    later = df["t"] > t_star
    before, after = features.make_features(df), features.make_features(noise(df, later))

    past = df["t"] <= t_star
    pd.testing.assert_frame_equal(before[past], after[past])
    # The noise did reach the features after t*. Not !=: NaN != NaN, so the
    # NaN slopes of segment 1's first window would pass that on their own.
    assert not before[later].equals(after[later])


@pytest.mark.parametrize("yaw_axis", [YAW_AXIS, None])   # the configured axis, and yaw rate from heading
@pytest.mark.parametrize("t_star", [1.7, 3.2, 4.0])      # segment 0, the pause, segment 1's first window
def test_from_raw_packets_nothing_after_t_star_reaches_the_features_up_to_it(monkeypatch, yaw_axis, t_star):
    # The same check through to_grid and add_signals: a forward join or a
    # centred rate there would leak the future just as surely.
    monkeypatch.setattr(data, "YAW_AXIS", yaw_axis)
    raw = packets()
    later = raw["t"] > t_star
    df = data.add_signals(data.to_grid(raw))
    before = features.make_features(df)
    after = features.make_features(data.add_signals(data.to_grid(noise(raw, later, RAW_SIGNALS))))

    past = df["t"] <= t_star
    pd.testing.assert_frame_equal(before[past], after[past])
    assert not before[~past].equals(after[~past])


def test_segment_1_is_blind_to_segment_0():
    # The gap is under WINDOW_S, so a window over the whole session would reach back.
    df = two_segments()
    before = features.make_features(df)
    after = features.make_features(noise(df, df["segment"] == 0))

    second = df["segment"] == 1
    pd.testing.assert_frame_equal(before[second], after[second])


@pytest.mark.parametrize("yaw_axis", [YAW_AXIS, None])
def test_a_drift_and_its_mirror_image_give_the_same_features(monkeypatch, yaw_axis):
    # Left for right: world x, heading, yaw and roll, lateral G and steering
    # change sign, and the wheels swap sides.
    monkeypatch.setattr(data, "YAW_AXIS", yaw_axis)
    raw = packets()
    mirrored = raw.copy()
    for name in ("vel_x", "heading", "ang_vel_y", "ang_vel_z", "acc_g_x", "steer"):
        mirrored[name] = -raw[name]
    for left, right in (("wheel_slip_fl", "wheel_slip_fr"), ("wheel_slip_rl", "wheel_slip_rr")):
        mirrored[[left, right]] = raw[[right, left]].to_numpy()

    def build(rows):
        return features.make_features(data.add_signals(data.to_grid(rows)))

    pd.testing.assert_frame_equal(build(mirrored), build(raw))


# --- what the windows hold ----------------------------------------------------

def test_windows_hold_exactly_the_last_window_s_and_slopes_are_per_second():
    # abs_slip counts frames, so each window's mean and max name its frames.
    # Late in a session, as around 194646's pause at 552 s, where t carries float rounding.
    df = session(60.0, start=551.9)
    k = np.arange(len(df))
    df["abs_slip"] = k.astype(float)
    f = features.make_features(df)

    assert list(f.columns) == features.FEATURES
    assert f.index.equals(df.index)
    assert (f["abs_slip_last"] == k).all() and (f["abs_slip_max"] == k).all()
    # From the first full window on: the WINDOW_FRAMES frames ending at t, no more.
    full = k >= WINDOW_FRAMES - 1
    assert f.loc[full, "abs_slip_mean"].to_numpy() == pytest.approx(k[full] - (WINDOW_FRAMES - 1) / 2)
    # One frame per 1/GRID_HZ s: GRID_HZ per second, known once WINDOW_S has passed.
    assert f.loc[k >= WINDOW_FRAMES, "abs_slip_slope"].to_numpy() == pytest.approx(GRID_HZ)
    assert f.loc[k < WINDOW_FRAMES, "abs_slip_slope"].isna().all()


def test_every_feature_is_known_once_a_segment_has_run_window_s():
    # From raw packets, so every NaN data.add_signals leaves (abs_slip_rate,
    # yaw_acc) is in play. Two segments, 0.5 s apart. With the configured
    # YAW_AXIS: see make_features for the heading fallback's longer start.
    df = data.add_signals(data.to_grid(packets()))
    f = features.make_features(df)

    assert df["segment"].nunique() == 2
    scored = df.groupby("segment").cumcount() >= WINDOW_FRAMES   # label()'s first-WINDOW_S rule
    assert not f[scored].isna().any(axis=None)
