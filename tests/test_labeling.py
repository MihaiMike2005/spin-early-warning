"""
Tests for spinwarn.labeling: onsets, re-arm, targets and exclusions.

Synthetic grid sessions, built straight on the GRID_HZ grid with the few
columns the labeler reads, so they run anywhere without data/raw/.
"""

import numpy as np
import pandas as pd
import pytest

from spinwarn import labeling
from spinwarn.config import GRID_HZ, HORIZON_S, REVERSE_GEAR, WINDOW_S


STEP = 1 / GRID_HZ


def session(seconds, slip, speed_kmh=50.0, gear=3, segment=0, start=0.0):
    """A grid session of `seconds`: slip and speed are numbers or functions of t."""
    t = (round(start * GRID_HZ) + np.arange(round(seconds * GRID_HZ))) / GRID_HZ
    value = lambda x: x(t) if callable(x) else np.full(len(t), float(x))
    return pd.DataFrame({
        "t": t, "segment": segment, "speed_kmh": value(speed_kmh),
        "abs_slip": value(slip), "gear": gear,
    })


def between(t, start, end):
    """Grid times in [start, end), with a margin well under a grid step."""
    return (t > start - STEP / 2) & (t < end - STEP / 2)


def times(df, mask):
    return df.loc[mask, "t"].to_numpy()


def grid(start, end):
    """The grid times in [start, end)."""
    return np.arange(round(start * GRID_HZ), round(end * GRID_HZ)) / GRID_HZ


# --- find_onsets ------------------------------------------------------------

@pytest.mark.parametrize("horizon", [0.5, 1.0])
def test_a_crossing_at_5_s_gives_one_onset_and_positives_exactly_in_the_horizon_before_it(horizon):
    # |slip| climbs 20°/s and first passes 90° on the frame at 5.0 s.
    df = session(10.0, lambda t: 20 * t - 9.9)
    onsets, _ = labeling.find_onsets(df)
    assert onsets == pytest.approx([5.0])

    labelled = labeling.label(df, horizon=horizon)
    assert times(labelled, labelled["target"]) == pytest.approx(grid(5.0 - horizon, 5.0))
    assert not labelled.loc[labelled["target"], "excluded"].any()


def test_a_second_crossing_before_re_arm_is_not_a_new_spin():
    # Over 90° at 5 s, back to 30° for 1.5 s (under REARM_S), over again at 7.5 s.
    slip = lambda t: np.where(between(t, 5.0, 6.0) | (t >= 7.5 - STEP / 2), 120.0, 30.0)
    df = session(12.0, slip)
    onsets, spinning = labeling.find_onsets(df)

    assert onsets == pytest.approx([5.0])
    assert times(df, spinning) == pytest.approx(grid(5.0, 12.0))
    assert not labeling.label(df)["target"].to_numpy()[df["t"] >= 6.0].any()


def test_a_crossing_after_re_arm_is_a_new_spin():
    # The same, but 2.5 s under 45° in between: re-armed at 8.0 s.
    slip = lambda t: np.where(between(t, 5.0, 6.0) | (t >= 8.5 - STEP / 2), 120.0, 30.0)
    df = session(12.0, slip)
    onsets, spinning = labeling.find_onsets(df)

    assert onsets == pytest.approx([5.0, 8.5])
    assert times(df, spinning) == pytest.approx(np.concatenate([grid(5.0, 8.0), grid(8.5, 12.0)]))


def test_a_spin_a_stop_and_a_drive_off_re_arm_only_after_2_s_of_moving_under_45():
    # Spin at 5 s and slow to a stop by 6 s; stand still to 9 s, with slip
    # still over 90° at first (noise near standstill); drive off at 9 s.
    speed = lambda t: np.select([t < 5.0, t < 6.0, t < 9.0], [50.0, 50.0 * (6.0 - t), 0.0], 40.0)
    slip = lambda t: np.select([t < 5.0 - STEP / 2, t < 7.0, t < 9.0 - STEP / 2], [20.0, 130.0, 0.0], 10.0)
    df = session(15.0, slip, speed_kmh=speed)
    onsets, spinning = labeling.find_onsets(df)

    assert onsets == pytest.approx([5.0])
    # Standing still for 3 s re-arms nothing; driving off does, 2 s later.
    assert times(df, spinning) == pytest.approx(grid(5.0, 11.0))


def test_any_frame_at_45_or_more_restarts_the_re_arm():
    # 1.5 s under 45°, one frame at 50°, 1.5 s under again: still disarmed at 9.5 s.
    blip = lambda t: between(t, 7.5, 7.5 + STEP)
    slip = lambda t: np.select([between(t, 5.0, 6.0), blip(t), t >= 9.5 - STEP / 2], [120.0, 50.0, 120.0], 30.0)
    df = session(12.0, slip)
    onsets, spinning = labeling.find_onsets(df)

    assert onsets == pytest.approx([5.0])
    assert spinning[df["t"] >= 5.0 - STEP / 2].all()


def test_over_90_below_min_speed_is_not_a_spin():
    df = session(10.0, lambda t: np.where(t >= 5.0 - STEP / 2, 120.0, 20.0), speed_kmh=15.0)
    onsets, spinning = labeling.find_onsets(df)
    assert len(onsets) == 0 and not spinning.any()


def test_reversing_at_speed_is_not_a_spin():
    # Backing up at 30 km/h reads |slip| near 180°.
    df = session(10.0, lambda t: np.where(between(t, 4.0, 7.0), 178.0, 0.0), speed_kmh=30.0)
    df.loc[between(df["t"], 4.0, 7.0), "gear"] = REVERSE_GEAR
    onsets, spinning = labeling.find_onsets(df)
    assert len(onsets) == 0 and not spinning.any()


# --- SPIN_HOLD_S --------------------------------------------------------------

def test_with_a_hold_a_short_spike_over_90_is_not_a_spin(monkeypatch):
    monkeypatch.setattr(labeling, "SPIN_HOLD_S", 0.25)
    df = session(10.0, lambda t: np.where(between(t, 5.0, 5.1), 120.0, 20.0))
    onsets, spinning = labeling.find_onsets(df)
    assert len(onsets) == 0 and not spinning.any()


def test_with_a_hold_a_held_crossing_is_a_spin_from_its_first_frame(monkeypatch):
    monkeypatch.setattr(labeling, "SPIN_HOLD_S", 0.25)
    # Over 90° for exactly 0.25 s, first frame to last.
    df = session(10.0, lambda t: np.where(between(t, 5.0, 5.25 + STEP), 120.0, 20.0))
    onsets, _ = labeling.find_onsets(df)
    assert onsets == pytest.approx([5.0])

    # One frame short of that is not.
    df = session(10.0, lambda t: np.where(between(t, 5.0, 5.25), 120.0, 20.0))
    assert len(labeling.find_onsets(df)[0]) == 0


# --- segments ---------------------------------------------------------------

def two_segments(slip_1=20.0, slip_2=20.0):
    """6 s in segment 0, a 0.5 s gap, then 6 s in segment 1."""
    return pd.concat([
        session(6.0, slip_1, segment=0),
        session(6.0, slip_2, segment=1, start=6.5),
    ], ignore_index=True)


@pytest.mark.parametrize("horizon", [HORIZON_S, 1.0])
def test_the_first_window_and_last_horizon_of_each_segment_are_excluded(horizon):
    df = labeling.label(two_segments(), horizon=horizon)
    expected = (
        between(df["t"], 0.0, WINDOW_S) | between(df["t"], 6.0 - horizon, 6.0)
        | between(df["t"], 6.5, 6.5 + WINDOW_S) | between(df["t"], 12.5 - horizon, 12.5)
    )
    assert (df["excluded"] == expected).all()
    for s in (0, 1):
        frames = df[df["segment"] == s]
        assert frames["excluded"].iloc[0] and frames["excluded"].iloc[-1]


def test_a_spin_running_through_a_gap_is_one_spin():
    # Over 90° from 5 s to the end of segment 0 and on into segment 1:
    # still spinning after the gap, and re-armed 2 s after it ends.
    df = two_segments(slip_1=lambda t: np.where(t >= 5.0 - STEP / 2, 120.0, 20.0),
                      slip_2=lambda t: np.where(t < 7.0, 120.0, 20.0))
    onsets, spinning = labeling.find_onsets(df)

    assert onsets == pytest.approx([5.0])
    assert times(df, spinning) == pytest.approx(np.concatenate([grid(5.0, 6.0), grid(6.5, 9.0)]))


def test_the_re_arm_starts_over_after_a_gap():
    # 1.5 s under 45° either side of the gap: 3 s in all, but never 2 s in a row.
    df = two_segments(slip_1=lambda t: np.where(between(t, 3.0, 4.5), 120.0, 20.0),
                      slip_2=lambda t: np.where(t >= 8.0 - STEP / 2, 120.0, 20.0))
    onsets, _ = labeling.find_onsets(df)
    assert onsets == pytest.approx([3.0])


def test_targets_never_reach_back_across_a_gap():
    # An onset 0.1 s into segment 1 is under HORIZON_S after segment 0 ends.
    df = labeling.label(two_segments(slip_2=lambda t: np.where(t >= 6.6 - STEP / 2, 120.0, 20.0)))
    assert not df.loc[df["segment"] == 0, "target"].any()
    assert times(df, df["target"]) == pytest.approx(grid(6.5, 6.6))


# --- label's other exclusions -------------------------------------------------

def test_slow_reverse_and_spinning_frames_are_excluded():
    slow = lambda t: np.where(between(t, 2.0, 3.0), 10.0, 50.0)
    df = session(12.0, lambda t: np.where(between(t, 6.0, 7.0), 120.0, 20.0), speed_kmh=slow)
    df.loc[between(df["t"], 4.0, 5.0), "gear"] = REVERSE_GEAR
    df = labeling.label(df)

    middle = between(df["t"], WINDOW_S, 12.0 - HORIZON_S)
    expected = between(df["t"], 2.0, 3.0) | between(df["t"], 4.0, 5.0) | between(df["t"], 6.0, 9.0)
    assert (df.loc[middle, "excluded"] == expected[middle]).all()


def test_label_returns_a_copy():
    df = session(10.0, 20.0)
    labelled = labeling.label(df)
    assert {"target", "excluded"} <= set(labelled.columns)
    assert "target" not in df.columns
