"""
Tests for spinwarn.data: the grid, the derived signals and the session lists.

Synthetic traces only, so they run anywhere, CI included, without the
recordings in data/raw/.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from driftscore.logger import COLUMNS

from spinwarn import data
from spinwarn.config import GRID_HZ, RATE_WINDOW_S, TTS_CAP_S


def packets(t, **columns):
    """A logger-shaped DataFrame with packets at times t: every column 0 but those given."""
    df = pd.DataFrame({name: np.zeros(len(t)) for name in COLUMNS})
    df["t"] = t
    df["packet_id"] = np.arange(len(t))
    for name, values in columns.items():
        df[name] = values
    return df


def jittered(start, end, rate=333.3, seed=0):
    """Packet times from start to end at about `rate`, unevenly spaced like the poll."""
    rng = np.random.default_rng(seed)
    steps = rng.uniform(0.5, 1.5, int((end - start) * rate)) / rate
    return start + np.concatenate([[0.0], np.cumsum(steps)])


def moving(t, travel_deg, pointing_deg, speed_kmh=50.0, **columns):
    """Packets of a car moving toward travel_deg while pointing toward pointing_deg."""
    v = speed_kmh / 3.6
    r = math.radians(travel_deg)
    return packets(
        t, speed_kmh=speed_kmh, vel_x=v * math.sin(r), vel_z=v * math.cos(r),
        heading=-math.radians(pointing_deg),   # AC's heading is mirrored yaw
        **columns,
    )


def settled(df):
    """Rows after each segment's first RATE_WINDOW_S, where every rate is known."""
    start = df.groupby("segment")["t"].transform("min")
    return df[df["t"] - start >= RATE_WINDOW_S]


# --- to_grid ----------------------------------------------------------------

def test_uneven_packets_give_an_even_grid_of_the_last_packet_before_each_point():
    t = jittered(0.0, 3.0)
    grid = data.to_grid(packets(t, vel_x=t))   # vel_x carries each packet's own time

    assert np.allclose(np.diff(grid["t"]), 1 / GRID_HZ)
    assert (grid["segment"] == 0).all()
    # The as-of join: the packet at or before each grid point, and no later one.
    assert (grid["vel_x"] <= grid["t"]).all()
    following = t[np.searchsorted(t, grid["vel_x"], side="right")[:-1]]
    assert (following > grid["t"].iloc[:-1]).all()


def test_a_gap_over_gap_s_starts_a_new_segment_with_no_grid_points_inside():
    before, after = jittered(0.0, 1.0), jittered(1.3, 2.3, seed=1)
    grid = data.to_grid(packets(np.concatenate([before, after])))

    assert sorted(grid["segment"].unique()) == [0, 1]
    assert not ((grid["t"] > before[-1]) & (grid["t"] < after[0])).any()
    assert grid.loc[grid["segment"] == 0, "t"].max() <= before[-1]
    assert grid.loc[grid["segment"] == 1, "t"].min() >= after[0]


def test_a_gap_under_gap_s_is_bridged_by_the_last_packet():
    before = jittered(0.0, 1.0)
    t = np.concatenate([before, jittered(before[-1] + 0.08, 2.0, seed=1)])
    grid = data.to_grid(packets(t, vel_x=t))

    assert (grid["segment"] == 0).all()
    assert np.allclose(np.diff(grid["t"]), 1 / GRID_HZ)
    inside = grid[(grid["t"] > before[-1]) & (grid["t"] < before[-1] + 0.08)]
    assert len(inside) and (inside["vel_x"] == before[-1]).all()


# --- time_to_90 -------------------------------------------------------------

def test_time_to_90_is_capped_unless_slip_is_growing():
    assert data.time_to_90([60.0, 60.0, 60.0], [0.0, -50.0, np.nan]).tolist() == [TTS_CAP_S] * 3


def test_time_to_90_at_the_current_rate():
    tts = data.time_to_90([60.0, 89.0, 95.0], [120.0, 0.1, 30.0])
    # 30° to go at 120°/s; 1° at 0.1°/s is past the cap; already past 90°.
    assert tts == pytest.approx([0.25, TTS_CAP_S, 0.0])


# --- add_signals ------------------------------------------------------------

@pytest.mark.parametrize("direction_deg", [0.0, 30.0, 135.0, -100.0])
def test_a_straight_line_reads_zero_slip(direction_deg):
    t = jittered(0.0, 1.0)
    df = data.add_signals(data.to_grid(moving(t, direction_deg, direction_deg)))
    assert df["slip_deg"].to_numpy() == pytest.approx(0.0, abs=1e-6)


def test_a_slide_reads_driftscores_slip_angle():
    t = jittered(0.0, 1.0)
    df = data.add_signals(data.to_grid(moving(t, travel_deg=30.0, pointing_deg=0.0)))
    assert df["slip_deg"].to_numpy() == pytest.approx(30.0)
    assert df["abs_slip"].to_numpy() == pytest.approx(30.0)


def test_slip_is_zeroed_below_driftscores_speed_floor():
    t = jittered(0.0, 1.0)
    df = data.add_signals(data.to_grid(moving(t, 120.0, 0.0, speed_kmh=2.0)))
    assert (df["slip_deg"] == 0).all()


def test_abs_slip_rate_is_the_change_over_the_last_rate_window():
    # |slip| grows at 100°/s; the car points along +z and its travel turns.
    t = jittered(0.0, 1.0)
    travel = 10.0 + 100.0 * t
    v = 50.0 / 3.6
    df = packets(t, speed_kmh=50.0, vel_x=v * np.sin(np.radians(travel)), vel_z=v * np.cos(np.radians(travel)))
    df = data.add_signals(data.to_grid(df))

    young = df["t"] - df["t"].iloc[0] < RATE_WINDOW_S - 1e-9
    assert df.loc[young, "abs_slip_rate"].isna().all()
    # Each grid point holds a packet up to ~4.5 ms old: a few °/s of jitter.
    assert settled(df)["abs_slip_rate"].to_numpy() == pytest.approx(100.0, abs=10.0)


def test_yaw_rate_from_the_axis_and_from_heading_agree(monkeypatch):
    # Turning at 2 rad/s for 3 s wraps heading across ±π more than once.
    omega = 2.0
    t = jittered(0.0, 3.0)
    heading = (-omega * t + math.pi) % (2 * math.pi) - math.pi   # ang_vel_y = -d(heading)/dt
    df = data.to_grid(packets(t, speed_kmh=50.0, vel_z=13.9, heading=heading, ang_vel_y=omega))

    from_axis = settled(data.add_signals(df))
    monkeypatch.setattr(data, "YAW_AXIS", None)
    from_heading = settled(data.add_signals(df))

    assert from_axis["yaw_rate"].to_numpy() == pytest.approx(math.degrees(omega))
    # Packets up to 4.5 ms old, differenced over RATE_WINDOW_S: under 5% off.
    assert from_heading["yaw_rate"].to_numpy() == pytest.approx(math.degrees(omega), rel=0.05)


def test_axes_wheels_and_steering_map_to_their_signals():
    t = jittered(0.0, 1.0)
    df = moving(
        t, travel_deg=30.0, pointing_deg=0.0,
        acc_g_x=0.8, acc_g_z=-0.3, wheel_slip_rl=4.0, wheel_slip_rr=6.0, wheel_slip_fl=50.0,
        steer=-0.6,
    )
    df = data.add_signals(data.to_grid(df))

    assert (df["lat_g"] == 0.8).all() and (df["lon_g"] == -0.3).all()
    assert (df["rear_slip"] == 5.0).all()
    assert (df["countersteer"] == -0.6 * data.COUNTERSTEER_SIGN).all()   # slip +30°, so sign(slip) = +1
    assert (df["lock"] == 0.6).all()


@pytest.mark.parametrize("yaw_axis", [1, None])
def test_every_derived_signal_is_causal(monkeypatch, yaw_axis):
    # Change everything after t*: nothing at or before t* may move.
    monkeypatch.setattr(data, "YAW_AXIS", yaw_axis)
    t = jittered(0.0, 4.0)
    rng = np.random.default_rng(0)
    columns = {c: rng.normal(size=len(t)).cumsum() for c in ("vel_x", "vel_z", "heading", "ang_vel_y", "steer")}
    df = packets(t, speed_kmh=60.0, **columns)
    changed = df.copy()
    later = changed["t"] > 2.0
    for c in columns:
        changed.loc[later, c] = rng.normal(size=later.sum()) * 100

    before = data.add_signals(data.to_grid(df))
    after = data.add_signals(data.to_grid(changed))
    past = before["t"] <= 2.0
    pd.testing.assert_frame_equal(before[past], after[past])


# --- load_raw, the cache and the session lists --------------------------------

@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """data/raw, data/processed and splits.json under tmp_path, in place of the repo's."""
    raw, processed, splits = tmp_path / "raw", tmp_path / "processed", tmp_path / "splits.json"
    raw.mkdir()
    monkeypatch.setattr(data, "RAW_DIR", raw)
    monkeypatch.setattr(data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data, "SPLITS_PATH", splits)
    return raw, processed, splits


def write_session(raw, session_id, speed_kmh=50.0):
    moving(jittered(0.0, 1.0), 0.0, 0.0, speed_kmh=speed_kmh).to_csv(raw / f"{session_id}.csv", index=False)


def test_load_raw_leaves_out_a_cut_last_row_and_reads_the_json(dirs):
    raw, _, _ = dirs
    df = packets([0.0, 0.003, 0.006])
    text = df.to_csv(index=False)
    (raw / "s.csv").write_text(text[: text.rindex("\n") - 5])   # the last row loses its tail and newline
    (raw / "s.json").write_text(json.dumps({"car": "s14"}))

    rows, meta = data.load_raw(raw / "s.csv")
    assert rows["packet_id"].tolist() == [0, 1]
    assert list(rows.columns) == COLUMNS
    assert meta == {"car": "s14"}


def test_load_raw_without_a_json_or_with_a_damaged_one_gives_no_meta(dirs):
    raw, _, _ = dirs
    write_session(raw, "s")
    assert data.load_raw(raw / "s.csv")[1] is None
    (raw / "s.json").write_bytes(b"\0\0\0")
    assert data.load_raw(raw / "s.csv")[1] is None


def test_load_raw_refuses_a_session_with_no_rows(dirs):
    raw, _, _ = dirs
    (raw / "off.csv").write_text(",".join(COLUMNS) + "\n")
    with pytest.raises(ValueError, match="no rows"):
        data.load_raw(raw / "off.csv")


def test_load_session_caches_until_rebuild(dirs):
    raw, processed, _ = dirs
    write_session(raw, "s", speed_kmh=50.0)
    built = data.load_session("s")
    assert (processed / "s.parquet").exists()
    pd.testing.assert_frame_equal(data.load_session("s"), built)   # the cache reads back as built

    write_session(raw, "s", speed_kmh=80.0)
    assert (data.load_session("s")["speed_kmh"] == 50.0).all()     # stale until rebuilt
    assert (data.load_session("s", rebuild=True)["speed_kmh"] == 80.0).all()


def test_a_config_change_rebuilds_the_cache(dirs, monkeypatch):
    raw, processed, _ = dirs
    write_session(raw, "s")
    built = data.load_session("s")
    monkeypatch.setattr(data, "COUNTERSTEER_SIGN", -data.COUNTERSTEER_SIGN)

    rebuilt = data.load_session("s")
    assert rebuilt["countersteer"].to_numpy() == pytest.approx(-built["countersteer"].to_numpy())
    assert pd.read_parquet(processed / "s.parquet").attrs["built_with"] == data.built_with()


def test_an_interrupted_cache_write_leaves_no_damaged_cache(dirs, monkeypatch):
    raw, _, _ = dirs
    write_session(raw, "s")

    def interrupted(self, path, **kwargs):
        open(path, "wb").write(b"PAR1 and half a file")
        raise KeyboardInterrupt   # Ctrl+C mid-write

    with monkeypatch.context() as m:
        m.setattr(pd.DataFrame, "to_parquet", interrupted)
        with pytest.raises(KeyboardInterrupt):
            data.load_session("s")
    assert (data.load_session("s")["speed_kmh"] == 50.0).all()


def test_a_cache_from_before_built_with_is_rebuilt(dirs):
    raw, processed, _ = dirs
    write_session(raw, "s", speed_kmh=50.0)
    data.load_session("s")
    old = pd.read_parquet(processed / "s.parquet")
    old.attrs.clear()
    old.assign(speed_kmh=0.0).to_parquet(processed / "s.parquet", index=False)

    assert (data.load_session("s")["speed_kmh"] == 50.0).all()


def test_dev_sessions_leave_out_the_locked_test_sessions(dirs):
    raw, _, splits = dirs
    for s in ("2026-10-12_100000", "2026-10-11_100000", "2026-10-13_100000"):
        write_session(raw, s)
    assert data.list_sessions() == ["2026-10-11_100000", "2026-10-12_100000", "2026-10-13_100000"]
    assert data.dev_sessions() == data.list_sessions()   # nothing locked yet

    splits.write_text(json.dumps({"test": ["2026-10-12_100000"]}))
    assert data.dev_sessions() == ["2026-10-11_100000", "2026-10-13_100000"]


def test_test_sessions_load_only_when_confirmed(dirs):
    raw, _, splits = dirs
    write_session(raw, "a")
    write_session(raw, "b")
    with pytest.raises(RuntimeError, match="no test sessions locked"):
        data.load_test_sessions(confirm=True)

    splits.write_text(json.dumps({"test": ["b"]}))
    with pytest.raises(RuntimeError, match="confirm=True"):
        data.load_test_sessions()
    assert list(data.load_test_sessions(confirm=True)) == ["b"]


def test_test_sessions_are_rebuilt_never_read_from_the_cache(dirs):
    raw, _, splits = dirs
    write_session(raw, "b", speed_kmh=50.0)
    data.load_session("b")                    # cached while still a dev session
    write_session(raw, "b", speed_kmh=80.0)   # a change the cache can't see, like a code change
    splits.write_text(json.dumps({"test": ["b"]}))

    assert (data.load_test_sessions(confirm=True)["b"]["speed_kmh"] == 80.0).all()
