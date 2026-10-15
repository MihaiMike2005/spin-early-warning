"""
Spin onsets, re-arm, targets and exclusions.

    python -m spinwarn.labeling

prints how many spins the rule finds in each dev session.

Everything here works on a session from data.load_session: the GRID_HZ grid,
cut into segments at gaps. Inside a segment the frames are exactly 1/GRID_HZ
apart, so the seconds in config.py become whole frame counts, and nothing
hangs on a float landing an ulp either side of an edge.
"""

import argparse

import numpy as np

from spinwarn.config import (
    GRID_HZ, HORIZON_S, MIN_SPEED_KMH, REARM_DEG, REARM_S, REVERSE_GEAR, SPIN_DEG, SPIN_HOLD_S,
    WINDOW_S,
)
from spinwarn.data import dev_sessions, load_session


def find_onsets(df) -> tuple:
    """(onset_times, spinning) for a grid session: when each spin starts, and which frames are in one.

    A state machine run segment by segment. Armed, a spin starts at the
    first frame at MIN_SPEED_KMH or more with |slip| over SPIN_DEG, if both
    hold for SPIN_HOLD_S; the onset is that first frame, and the car is
    disarmed. Disarmed, it re-arms once it has driven REARM_S without a
    break at MIN_SPEED_KMH or more with |slip| under REARM_DEG, out of
    reverse. spinning covers each onset up to its re-arm. Slip angle is
    noise near standstill, so a stopped car stays spinning: it never re-arms
    by itself.

    Reversing reads |slip| near 180°, so reverse gear disarms the car too:
    backing up at speed would otherwise be a spin, with every frame before
    it excluded as reverse, so no model could ever catch it. Disarming, not
    just skipping reverse frames, also covers a shift out of reverse while
    the car still rolls backward at speed (2026-10-14_225818 at 604 s).
    A reverse inside a spin is part of the spin.

    The session starts armed. Armed or not carries across a gap, which only
    hides time (a pause freezes the car), so a spin running through one is
    still one spin. The hold and the re-arm start over after a gap, since
    nothing is known inside it.

    onset_times is an array of grid times; spinning is a bool per row of df.
    """
    onsets, spinning, _reversing = _onset_frames(df)
    return df["t"].to_numpy()[onsets], spinning


def _onset_frames(df) -> tuple:
    """find_onsets, with each onset as a row position in df instead of a time.

    Also returns reversing: each reverse that disarmed the car, up to its
    re-arm. label() excludes it like spinning.
    """
    segment = df["segment"].tolist()
    at_speed = df["speed_kmh"].to_numpy() >= MIN_SPEED_KMH
    in_reverse = df["gear"].to_numpy() == REVERSE_GEAR
    over = (at_speed & (df["abs_slip"].to_numpy() > SPIN_DEG)).tolist()
    under = (at_speed & ~in_reverse & (df["abs_slip"].to_numpy() < REARM_DEG)).tolist()
    in_reverse = in_reverse.tolist()
    hold_frames = round(SPIN_HOLD_S * GRID_HZ)
    rearm_frames = round(REARM_S * GRID_HZ)

    onsets = []
    spinning = np.zeros(len(df), dtype=bool)
    reversing = np.zeros(len(df), dtype=bool)
    armed = True
    # Whichever disarmed the car, a spin or a reverse, takes its frames up to the re-arm.
    disarmed = spinning
    # Frames in a row toward a spin (armed) or a re-arm (disarmed). A run of
    # n frames spans n - 1 grid steps, so run > hold_frames is SPIN_HOLD_S
    # from its first frame to this one, and likewise for REARM_S.
    run = 0
    for i in range(len(df)):
        if i and segment[i] != segment[i - 1]:
            run = 0
        if armed and in_reverse[i]:
            armed, run, disarmed = False, 0, reversing
        if armed:
            run = run + 1 if over[i] else 0
            if run > hold_frames:
                start = i - hold_frames
                onsets.append(start)
                spinning[start:i + 1] = True
                armed, run, disarmed = False, 0, spinning
        else:
            run = run + 1 if under[i] else 0
            if run > rearm_frames:
                armed, run = True, 0
            else:
                disarmed[i] = True
    return np.array(onsets, dtype=int), spinning, reversing


def label(df, horizon=HORIZON_S):
    """A copy of a grid session with a bool `target` and `excluded` per frame.

    target: a spin onset falls in (t, t + horizon], in the same segment.
    excluded, neither trained on nor scored: under MIN_SPEED_KMH, in reverse
    or not yet re-armed after it, spinning, the first WINDOW_S of a segment
    (the feature window isn't full yet), and its last `horizon` (what comes
    next is unknown).
    """
    df = df.copy()
    onsets, spinning, reversing = _onset_frames(df)
    segment = df["segment"].to_numpy()
    ahead_frames = round(horizon * GRID_HZ)

    # On the grid, the ahead_frames frames before an onset are the ones with
    # it in (t, t + horizon]. Those across a gap don't count: their future
    # is unknown, and they're excluded anyway.
    target = np.zeros(len(df), dtype=bool)
    for j in onsets:
        before = slice(max(j - ahead_frames, 0), j)
        target[before] |= segment[before] == segment[j]

    by_segment = df.groupby("segment")
    since_start = by_segment.cumcount().to_numpy()
    to_end = by_segment.cumcount(ascending=False).to_numpy()
    df["target"] = target
    df["excluded"] = (
        (df["speed_kmh"].to_numpy() < MIN_SPEED_KMH)
        | (df["gear"].to_numpy() == REVERSE_GEAR)
        | reversing
        | spinning
        | (since_start < round(WINDOW_S * GRID_HZ))
        | (to_end < ahead_frames)
    )
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.labeling",
        description="Count the spin onsets the label rule finds in every dev session.",
    )
    parser.parse_args(argv)
    total, minutes = 0, 0.0
    for session_id in dev_sessions():
        df = load_session(session_id)
        onsets, _spinning = find_onsets(df)
        session_min = len(df) / GRID_HZ / 60
        print(f"{session_id}: {len(onsets)} onsets in {session_min:.1f} min, {len(onsets) / session_min:.1f}/min")
        total, minutes = total + len(onsets), minutes + session_min
    print(f"total: {total} onsets in {minutes:.1f} min, {total / minutes:.1f}/min")


if __name__ == "__main__":
    main()
