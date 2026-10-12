"""
Seeded-random spin onsets as video timestamps, to check the label rule on the recordings.

    python -m spinwarn.spotcheck <session> --n 20 --video-offset 3.2

prints n of a dev session's spin onsets, picked at random with SEED and
listed in the order they happen, as a table to paste into notes.md: each
onset's time in the screen recording (mm:ss.s), its peak |slip| and its
speed, and an empty Hit column to mark while watching.

The video offset is the video time at which the logger's status line shows
t = 0. The logger's t keeps counting through a pause, AC's own at the start
included, and so does the recording, so one offset holds for the whole
session.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from spinwarn.config import MIN_SPEED_KMH, RAW_DIR, SEED
from spinwarn.data import dev_sessions, list_sessions, load_session
from spinwarn.labeling import find_onsets


def spins(df) -> pd.DataFrame:
    """One row per spin in a grid session: its number, onset t, peak |slip| and speed at onset.

    The peak is over the spin's frames, onset to re-arm, at MIN_SPEED_KMH or
    more: slip angle is noise near standstill, and the label rule ignores it
    there too. The onset frame itself is always at speed, so there's a peak.
    """
    onsets, spinning = find_onsets(df)
    t = df["t"].to_numpy()
    abs_slip = df["abs_slip"].to_numpy()
    speed = df["speed_kmh"].to_numpy()
    # A spin runs from its onset up to the first frame that isn't spinning: its re-arm.
    ends = np.append(np.flatnonzero(~spinning), len(df))
    rows = []
    for number, start in enumerate(np.searchsorted(t, onsets), 1):
        frames = slice(start, ends[np.searchsorted(ends, start)])
        fast = speed[frames] >= MIN_SPEED_KMH
        rows.append({"spin": number, "t": t[start], "peak_slip": abs_slip[frames][fast].max(),
                     "speed_kmh": speed[start]})
    return pd.DataFrame(rows, columns=["spin", "t", "peak_slip", "speed_kmh"])


def pick(spins, n, seed=SEED) -> pd.DataFrame:
    """n of the spins, picked at random with seed, in time order; all of them if there are n or fewer."""
    if n >= len(spins):
        return spins
    chosen = np.random.default_rng(seed).choice(len(spins), size=n, replace=False)
    return spins.iloc[np.sort(chosen)]


def video_time(t, offset) -> str:
    """Logger time t as mm:ss.s in a recording where the logger's t = 0 falls at `offset` s."""
    # Rounded to tenths first, so 59.96 s reads 01:00.0, not 00:60.0.
    minutes, tenths = divmod(round((t + offset) * 10), 600)
    return f"{minutes:02d}:{tenths // 10:02d}.{tenths % 10}"


def table(spins, offset) -> list:
    """The spins as the lines of a markdown table, with an empty Hit column to fill in on video."""
    lines = [
        "| Spin | Video | t (s) | Peak slip (°) | Speed (km/h) | Hit? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in spins.itertuples():
        lines.append(f"| {s.spin} | {video_time(s.t, offset)} | {s.t:.1f} | {s.peak_slip:.0f} "
                     f"| {s.speed_kmh:.0f} |  |")
    return lines


def parse_n(text) -> int:
    """argparse type for --n: a whole number, 1 or more."""
    try:
        value = int(text)
    except ValueError:
        # Otherwise argparse says "invalid parse_n value".
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {text}")
    return value


def parse_offset(text) -> float:
    """argparse type for --video-offset: seconds, 0 or more.

    The recording starts before the logger, so t = 0 is never before the
    video's start. nan and inf are refused too.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0 <= value < math.inf:   # chained comparisons are False for nan, so it's refused too
        raise argparse.ArgumentTypeError(f"must be 0 or more seconds, got {text}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.spotcheck",
        description="List seeded-random spin onsets of a session as video timestamps, to check on the recording.",
    )
    parser.add_argument("session", help="a dev session id, e.g. 2026-10-11_194646; its path in data/raw/ works too")
    parser.add_argument("--n", type=parse_n, default=20,
                        help="how many onsets (default: 20, or all of them if the session has fewer)")
    parser.add_argument("--video-offset", type=parse_offset, required=True,
                        help="the video time, in seconds, at which the logger's status line shows t = 0")
    args = parser.parse_args(argv)

    # Tab completion offers data/raw/<id>.csv, so take the id from a path.
    session = Path(args.session).stem
    if session not in dev_sessions():
        if session in list_sessions():
            parser.error(f"{session} is a locked test session: only the final run loads it")
        parser.error(f"no session {session} in {RAW_DIR}")

    every = spins(load_session(session))
    picked = pick(every, args.n)
    print(f"{session}: {len(picked)} of {len(every)} spin onsets (seed {SEED}), "
          f"t = 0 at {args.video_offset:g} s in the video\n")
    print("\n".join(table(picked, args.video_offset)))


if __name__ == "__main__":
    main()
