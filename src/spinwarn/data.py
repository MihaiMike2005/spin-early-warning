"""
Load sessions, resample them onto the GRID_HZ grid, add derived signals, list the splits.

    python -m spinwarn.data [--rebuild]

loads every dev session into the cache, data/processed/<id>.parquet, and
prints what it got. A cached session remembers the constants that built it
and is rebuilt when any of them changes. --rebuild forces it, for what the
cache can't see: a change to the code, or a raw CSV copied over again.

Test sessions are left out of everything here except load_test_sessions,
which only the final run calls.
"""

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from driftscore.logcheck import read_meta
from driftscore.signals import SLIP_SPEED_FLOOR_KMH, slip_angle_deg

from spinwarn.config import (
    COUNTERSTEER_SIGN, GAP_S, GRID_HZ, LAT_G_AXIS, LAT_G_SIGN, LON_G_AXIS, LON_G_SIGN,
    PROCESSED_DIR, RATE_WINDOW_S, RAW_DIR, SPIN_DEG, SPLITS_PATH, TTS_CAP_S, YAW_AXIS, YAW_SIGN,
)


AXES = "xyz"   # the logger's suffixes for axis 0, 1, 2


def load_raw(csv_path) -> tuple:
    """(rows, meta) for a logger CSV: a DataFrame, and its JSON as a dict.

    meta is None without a JSON of the same name, or with a damaged one.
    A last line without a newline was cut off mid-row (the logger was
    stopped without Ctrl+C), so it's left out, as logcheck does.
    """
    csv_path = Path(csv_path)
    text = csv_path.read_bytes()
    if not text.endswith(b"\n"):
        text = text[: text.rfind(b"\n") + 1]
    df = pd.read_csv(io.BytesIO(text))
    if df.empty:
        # The logger writes nothing while AC is OFF, as in 191522 and 191538.
        raise ValueError(f"{csv_path.name} has no rows, so AC was never live: leave it out of data/raw/")
    json_path = csv_path.with_suffix(".json")
    meta = read_meta(json_path.read_bytes()) if json_path.exists() else None
    return df, meta


def segment_ids(t) -> np.ndarray:
    """0, 1, 2, ... per row, one higher after each gap over GAP_S."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > GAP_S)])


def to_grid(df, hz=GRID_HZ) -> pd.DataFrame:
    """The packets resampled onto a uniform grid at hz, plus a `segment` column.

    Grid times are k / hz from the logger's t = 0. Each grid point takes the
    last packet at or before it: a backward as-of join, so nothing comes from
    the future, and it's what a live loop running at hz would see. A gap over
    GAP_S (a pause, a stall) ends a segment, and no grid point falls inside
    it, so nothing later windows or differences across one.
    """
    t = df["t"].to_numpy()
    seg = segment_ids(t)
    pieces = []
    for s in np.unique(seg):
        rows = np.flatnonzero(seg == s)
        k = np.arange(math.ceil(t[rows[0]] * hz), math.floor(t[rows[-1]] * hz) + 1)
        # max(): k / hz can round to an ulp before the segment's first packet.
        last = np.maximum(np.searchsorted(t[rows], k / hz, side="right") - 1, 0)
        pieces.append(df.iloc[rows[last]].assign(t=k / hz, segment=s))
    out = pd.concat(pieces, ignore_index=True)
    out.insert(1, "segment", out.pop("segment"))
    return out


def _rate(df, column) -> pd.Series:
    """d(column)/dt over the last RATE_WINDOW_S, inside each segment.

    Backward, so causal. Over RATE_WINDOW_S, not one grid step: each grid
    point's packet is up to one AC packet old, which makes a 1/60 s
    difference jittery. NaN until a segment has run RATE_WINDOW_S.
    """
    n = round(RATE_WINDOW_S * GRID_HZ)
    by_segment = df.groupby("segment")
    return by_segment[column].diff(n) / by_segment["t"].diff(n)


def time_to_90(abs_slip, abs_slip_rate) -> np.ndarray:
    """Seconds for |slip| to reach SPIN_DEG at its current rate, in [0, TTS_CAP_S].

    The B3 baseline, also a feature so the models start from it. TTS_CAP_S
    while |slip| isn't growing, or its rate isn't known yet; 0 past SPIN_DEG.
    """
    abs_slip = np.asarray(abs_slip, dtype=float)
    rate = np.asarray(abs_slip_rate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):   # np.where evaluates both branches
        tts = np.where(rate > 0, (SPIN_DEG - abs_slip) / rate, TTS_CAP_S)
    return np.clip(tts, 0.0, TTS_CAP_S)


def add_signals(df) -> pd.DataFrame:
    """A copy of a grid session with the derived signals labels, features and baselines share.

    slip_deg and abs_slip; abs_slip_rate and time_to_90; yaw_rate (deg/s)
    and yaw_acc (deg/s²); lat_g and lon_g; rear_slip; countersteer and lock.
    """
    df = df.copy()

    # Row by row through driftscore, so labels and features use its tested math.
    # Below its speed floor the slip angle is noise; zeroed, as app.py does.
    slip = np.array([
        slip_angle_deg((vx, 0.0, vz), h)
        for vx, vz, h in zip(df["vel_x"].tolist(), df["vel_z"].tolist(), df["heading"].tolist())
    ])
    slip[df["speed_kmh"].to_numpy() < SLIP_SPEED_FLOOR_KMH] = 0.0
    df["slip_deg"] = slip
    df["abs_slip"] = np.abs(slip)
    df["abs_slip_rate"] = _rate(df, "abs_slip")
    df["time_to_90"] = time_to_90(df["abs_slip"], df["abs_slip_rate"])

    if YAW_AXIS is None:
        # ang_vel_y = -d(heading)/dt (logcheck's "sign -"), so this matches YAW_SIGN = +1.
        heading = df.groupby("segment")["heading"].transform(np.unwrap)
        df["yaw_rate"] = -np.degrees(_rate(df.assign(heading=heading), "heading"))
    else:
        df["yaw_rate"] = np.degrees(df[f"ang_vel_{AXES[YAW_AXIS]}"]) * YAW_SIGN
    df["yaw_acc"] = _rate(df, "yaw_rate")

    df["lat_g"] = df[f"acc_g_{AXES[LAT_G_AXIS]}"] * LAT_G_SIGN
    df["lon_g"] = df[f"acc_g_{AXES[LON_G_AXIS]}"] * LON_G_SIGN
    df["rear_slip"] = (df["wheel_slip_rl"] + df["wheel_slip_rr"]) / 2
    df["countersteer"] = df["steer"] * np.sign(df["slip_deg"]) * COUNTERSTEER_SIGN
    df["lock"] = df["steer"].abs()
    return df


def built_with() -> dict:
    """The constants a cached session depends on: a change to any of them rebuilds it."""
    return {
        "GRID_HZ": GRID_HZ, "GAP_S": GAP_S, "RATE_WINDOW_S": RATE_WINDOW_S,
        "SPIN_DEG": SPIN_DEG, "TTS_CAP_S": TTS_CAP_S, "SLIP_SPEED_FLOOR_KMH": SLIP_SPEED_FLOOR_KMH,
        "YAW_AXIS": YAW_AXIS, "YAW_SIGN": YAW_SIGN, "LAT_G_AXIS": LAT_G_AXIS, "LAT_G_SIGN": LAT_G_SIGN,
        "LON_G_AXIS": LON_G_AXIS, "LON_G_SIGN": LON_G_SIGN, "COUNTERSTEER_SIGN": COUNTERSTEER_SIGN,
    }


def load_session(session_id, rebuild=False) -> pd.DataFrame:
    """A session on the grid with its signals, from the cache if it's current.

    The cache stores built_with() in the Parquet file's metadata (df.attrs),
    so a change to config.py rebuilds it, and so does a cache from before
    it stored them. rebuild forces it, for what the cache can't see: a change
    to the code here or in driftscore, or a raw CSV copied over again.
    """
    cache = PROCESSED_DIR / f"{session_id}.parquet"
    if cache.exists() and not rebuild:
        cached = pd.read_parquet(cache)
        if cached.attrs.get("built_with") == built_with():
            return cached
    df, _meta = load_raw(RAW_DIR / f"{session_id}.csv")
    df = add_signals(to_grid(df))
    df.attrs["built_with"] = built_with()
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Written aside, then renamed over the cache: a write cut short (Ctrl+C)
    # can't leave a damaged cache for every later load to trip on.
    partial = cache.with_name(f"{cache.name}.partial")
    df.to_parquet(partial, index=False)
    partial.replace(cache)
    return df


def list_sessions() -> list:
    """Every recorded session id, the CSV names in data/raw/, oldest first."""
    return sorted(p.stem for p in RAW_DIR.glob("*.csv"))


def _test_ids() -> list:
    """The locked test session ids: splits.json's "test" list, empty until Step 10 locks it."""
    if not SPLITS_PATH.exists():
        return []
    return json.loads(SPLITS_PATH.read_text())["test"]


def dev_sessions() -> list:
    """Every session but the locked test ones: all of them until splits.json exists."""
    test = set(_test_ids())
    return [s for s in list_sessions() if s not in test]


def load_test_sessions(confirm=False) -> dict:
    """The locked test sessions, {id: DataFrame}. Only the final run calls this.

    Raises unless confirm=True: the test sessions are scored once, at the
    very end, and a stray call shouldn't peek at them. Always rebuilt from
    the raw CSVs, never from the cache: --rebuild only reaches dev sessions,
    so a test session's cache can predate a code change. It's one call, and
    about a second.
    """
    if not confirm:
        raise RuntimeError("test sessions are scored once, by the final run: pass confirm=True only there")
    ids = _test_ids()
    if not ids:
        raise RuntimeError(f"no test sessions locked in {SPLITS_PATH}: run `python -m spinwarn.run lock-test` first")
    return {s: load_session(s, rebuild=True) for s in ids}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.data",
        description="Load every dev session onto the grid, cache it, and print what it holds.",
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="build every dev session again, cached or not; run after changing the code "
                             "or copying a raw CSV over again (a change to config.py rebuilds by itself)")
    args = parser.parse_args(argv)
    for session_id in dev_sessions():
        df = load_session(session_id, args.rebuild)
        print(f"{session_id}: {len(df)} frames, {len(df) / GRID_HZ / 60:.1f} min, "
              f"{df['segment'].nunique()} segment(s)")


if __name__ == "__main__":
    main()
