"""
Tests for spinwarn.spotcheck: spins, the seeded pick, video times and the CLI.

Synthetic grid sessions with the few columns the labeler reads, so they run
anywhere without data/raw/.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from spinwarn import spotcheck
from spinwarn.config import GRID_HZ


STEP = 1 / GRID_HZ


def two_spins():
    """14 s on the grid with spins at 5 s and 12 s.

    The first peaks at 120° at speed, then the car crawls at 10 km/h reading
    179° (noise near standstill) and drives off at 8 s, re-arming at 10 s.
    The second peaks higher, at 160°, and never re-arms.
    """
    t = np.arange(round(14.0 * GRID_HZ)) / GRID_HZ
    edge = lambda s: t < s - STEP / 2
    speed = np.select([edge(6.0), edge(8.0)], [50.0, 10.0], 40.0)
    slip = np.select([edge(5.0), edge(6.0), edge(8.0), edge(12.0)], [20.0, 120.0, 179.0, 10.0], 160.0)
    return pd.DataFrame({"t": t, "segment": 0, "speed_kmh": speed, "abs_slip": slip, "gear": 3})


def spin_rows(n):
    return pd.DataFrame({"spin": range(1, n + 1), "t": np.arange(n) * 30.0, "peak_slip": 100.0, "speed_kmh": 50.0})


# --- spins ------------------------------------------------------------------

def test_each_spin_peaks_over_its_own_frames_at_speed_and_reads_its_speed_at_onset():
    rows = spotcheck.spins(two_spins())
    assert rows["spin"].tolist() == [1, 2]
    assert rows["t"].to_numpy() == pytest.approx([5.0, 12.0])
    # Not 179° (the crawl) or 160° (the next spin) for the first.
    assert rows["peak_slip"].tolist() == [120.0, 160.0]
    assert rows["speed_kmh"].tolist() == [50.0, 40.0]


def test_a_session_without_spins_gives_an_empty_table():
    df = two_spins().assign(abs_slip=20.0)
    rows = spotcheck.spins(df)
    assert rows.empty and list(rows.columns) == ["spin", "t", "peak_slip", "speed_kmh"]


# --- pick -------------------------------------------------------------------

def test_pick_takes_every_spin_when_there_are_n_or_fewer():
    rows = spin_rows(5)
    pd.testing.assert_frame_equal(spotcheck.pick(rows, 5), rows)
    pd.testing.assert_frame_equal(spotcheck.pick(rows, 20), rows)


def test_pick_is_seeded_distinct_and_in_time_order():
    rows = spin_rows(30)
    picked = spotcheck.pick(rows, 7)
    assert len(picked) == 7 and picked["spin"].is_unique
    assert picked["t"].is_monotonic_increasing
    pd.testing.assert_frame_equal(spotcheck.pick(rows, 7), picked)


# --- video_time and table -----------------------------------------------------

@pytest.mark.parametrize("t, offset, text", [
    (0.0, 3.2, "00:03.2"),
    (56.76, 3.2, "01:00.0"),    # 59.96 s rounds up into the next minute
    (125.04, 3.2, "02:08.2"),
    (594.7, 4.0, "09:58.7"),
])
def test_video_time(t, offset, text):
    assert spotcheck.video_time(t, offset) == text


def test_table_has_a_line_per_spin_with_an_empty_hit_column():
    lines = spotcheck.table(spotcheck.spins(two_spins()), offset=3.2)
    assert len(lines) == 4
    assert lines[2] == "| 1 | 00:08.2 | 5.0 | 120 | 50 |  |"
    assert lines[3] == "| 2 | 00:15.2 | 12.0 | 160 | 40 |  |"


# --- the CLI ----------------------------------------------------------------

@pytest.mark.parametrize("text, value", [("1", 1), ("20", 20)])
def test_parse_n(text, value):
    assert spotcheck.parse_n(text) == value


@pytest.mark.parametrize("text", ["0", "-3", "2.5", "abc"])
def test_parse_n_refuses(text):
    with pytest.raises(argparse.ArgumentTypeError):
        spotcheck.parse_n(text)


@pytest.mark.parametrize("text, value", [("0", 0.0), ("3.2", 3.2), ("8.8", 8.8)])
def test_parse_offset(text, value):
    assert spotcheck.parse_offset(text) == value


@pytest.mark.parametrize("text", ["-0.5", "nan", "inf", "abc"])
def test_parse_offset_refuses(text):
    with pytest.raises(argparse.ArgumentTypeError):
        spotcheck.parse_offset(text)


@pytest.fixture
def sessions(monkeypatch):
    """Dev session "a", locked test session "b"; loading either gives two_spins()."""
    monkeypatch.setattr(spotcheck, "list_sessions", lambda: ["a", "b"])
    monkeypatch.setattr(spotcheck, "dev_sessions", lambda: ["a"])
    loaded = []
    monkeypatch.setattr(spotcheck, "load_session", lambda s: loaded.append(s) or two_spins())
    return loaded


def test_main_prints_the_picked_onsets_from_a_path_or_an_id(sessions, capsys):
    spotcheck.main(["data/raw/a.csv", "--n", "1", "--video-offset", "3.2"])
    out = capsys.readouterr().out
    assert out.startswith("a: 1 of 2 spin onsets (seed 0), t = 0 at 3.2 s in the video")
    assert out.count("|  |") == 1
    assert sessions == ["a"]


@pytest.mark.parametrize("argv, message", [
    (["b", "--video-offset", "1"], "b is a locked test session"),
    (["c", "--video-offset", "1"], "no session c in"),
    (["a"], "required: --video-offset"),
    (["a", "--video-offset", "-1"], "--video-offset: must be 0 or more seconds, got -1"),
])
def test_main_refuses_before_loading_anything(sessions, capsys, argv, message):
    with pytest.raises(SystemExit):
        spotcheck.main(argv)
    assert message in capsys.readouterr().err
    assert sessions == []
