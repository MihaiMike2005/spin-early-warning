"""
Tests for spinwarn.run: the dataset build, its staleness check and the test lock.

Synthetic grid sessions with every column labels and features read, and the
paths and session lists swapped for tmp_path ones, so they run anywhere
without data/raw/ and never touch the repo's splits.json.
"""

import json

import numpy as np
import pandas as pd
import pytest

from spinwarn import run
from spinwarn.config import GRID_HZ, N_FOLDS
from spinwarn.features import FEATURES, INPUTS, make_features
from spinwarn.labeling import label


STEP = 1 / GRID_HZ


def session(spin_at=5.0, seed=0):
    """12 s at 50 km/h on the grid: 8 s, a 0.5 s gap, 4 s. |slip| is over 90° from spin_at, if given."""
    t = np.concatenate([np.arange(round(8.0 * GRID_HZ)), np.arange(round(8.5 * GRID_HZ), round(12.5 * GRID_HZ))])
    t = t / GRID_HZ
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.normal(size=len(t)).cumsum() for name in INPUTS})
    over = t >= spin_at - STEP / 2 if spin_at is not None else np.zeros(len(t), dtype=bool)
    slip = np.where(over, 120.0, 20.0)
    return df.assign(t=t, segment=(t > 8.0).astype(int), speed_kmh=50.0, gear=3, abs_slip=slip, slip_deg=slip)


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    """Recorded sessions a, b and c, all dev until a test changes "dev"; "loaded" lists the loads.

    a spins once, at 5 s; b and c never do.
    """
    state = {"all": ["a", "b", "c"], "dev": ["a", "b", "c"], "loaded": []}
    monkeypatch.setattr(run, "DATASET_PATH", tmp_path / "processed" / "dataset.parquet")
    monkeypatch.setattr(run, "SPLITS_PATH", tmp_path / "splits.json")
    monkeypatch.setattr(run, "list_sessions", lambda: list(state["all"]))
    monkeypatch.setattr(run, "dev_sessions", lambda: list(state["dev"]))

    def load(session_id):
        state["loaded"].append(session_id)
        return session(spin_at=5.0 if session_id == "a" else None, seed=len(state["loaded"]))

    monkeypatch.setattr(run, "load_session", load)
    return state


def lock(sessions, *test):
    """Lock test as the test sessions, as lock-test would, and leave them out of the dev sessions."""
    run.SPLITS_PATH.write_text(json.dumps({"test": list(test)}))
    sessions["dev"] = [s for s in sessions["all"] if s not in test]


# --- session_rows -------------------------------------------------------------

def test_session_rows_are_one_per_frame_with_labels_onsets_baseline_signals_and_features():
    df = session()
    rows = run.session_rows("a", df)

    assert list(rows.columns) == run.COLUMNS
    assert len(rows) == len(df) and (rows["session"] == "a").all()
    labelled = label(df)
    for name in ["segment", "t", "target", "excluded"]:
        assert (rows[name].to_numpy() == labelled[name].to_numpy()).all()
    assert rows.loc[rows["onset"], "t"].to_numpy() == pytest.approx([5.0])
    assert (rows[run.BASELINE_SIGNALS].to_numpy() == df[run.BASELINE_SIGNALS].to_numpy()).all()
    pd.testing.assert_frame_equal(rows[FEATURES], make_features(df)[FEATURES].reset_index(drop=True))


def test_a_session_without_spins_has_no_onsets():
    rows = run.session_rows("b", session(spin_at=None))
    assert not rows["onset"].any() and not rows["target"].any()


# --- build and load_dataset -----------------------------------------------------

def test_build_writes_the_dev_sessions_and_prints_their_counts(sessions, capsys):
    lock(sessions, "b")
    run.main(["build"])
    out = capsys.readouterr().out

    assert sessions["loaded"] == ["a", "c"]
    dataset = run.load_dataset()
    assert dataset["session"].unique().tolist() == ["a", "c"]
    assert list(dataset.columns) == run.COLUMNS
    # a's 30 frames before its onset are scored: HORIZON_S at 60 Hz.
    assert "a: 0.2 min" in out and "1 spins" in out and "30 positive frames" in out
    assert "total:" in out and "No test sessions locked yet" not in out


def test_build_before_the_lock_says_so_and_nothing_models_on_it(sessions, capsys):
    run.main(["build"])
    assert "No test sessions locked yet" in capsys.readouterr().out
    # build writes every session, for the spin counts to pick by...
    assert pd.read_parquet(run.DATASET_PATH)["session"].unique().tolist() == ["a", "b", "c"]
    # ...but modelling can't read it until the test sessions are locked.
    with pytest.raises(RuntimeError, match="no test sessions locked"):
        run.load_dataset()


def test_a_dataset_built_before_the_lock_is_refused_after_it(sessions):
    run.main(["build"])
    lock(sessions, "b")
    with pytest.raises(RuntimeError, match="other sessions"):
        run.load_dataset()


def test_a_dataset_built_before_a_session_was_copied_in_is_refused(sessions):
    lock(sessions, "b")
    run.main(["build"])
    sessions["all"].append("d")
    sessions["dev"].append("d")
    with pytest.raises(RuntimeError, match="other sessions"):
        run.load_dataset()


def test_a_dataset_built_with_other_constants_is_refused(sessions, monkeypatch):
    lock(sessions, "b")
    run.main(["build"])
    monkeypatch.setattr(run, "HORIZON_S", 1.0)
    with pytest.raises(RuntimeError, match="config.py has changed"):
        run.load_dataset()


def test_no_dataset_says_to_build_it(sessions):
    with pytest.raises(FileNotFoundError, match="run build"):
        run.load_dataset()


# --- lock-test ----------------------------------------------------------------

def test_lock_test_writes_the_test_sessions_once(sessions, capsys):
    sessions["all"] = list("abcdef")
    run.main(["lock-test", "--sessions", "e, b"])
    assert json.loads(run.SPLITS_PATH.read_text()) == {"test": ["b", "e"]}
    assert "locked b, e as the test sessions" in capsys.readouterr().out

    locked = run.SPLITS_PATH.read_text()
    with pytest.raises(SystemExit):
        run.main(["lock-test", "--sessions", "c,d"])
    assert "the test sessions are locked once" in capsys.readouterr().err
    assert run.SPLITS_PATH.read_text() == locked


@pytest.mark.parametrize("recorded, argv, message", [
    ("abcdef", ["--sessions", "a,x"], "no session x in"),
    ("abcdef", ["--sessions", "a,a"], "a session is listed twice"),
    ("abcdef", ["--sessions", ","], "no session ids"),
    ("abcdef", [], "required: --sessions"),
    # Today's 3 sessions, and one short of N_FOLDS after the lock.
    ("abc", ["--sessions", "a,b"], f"only 1 dev sessions would be left, and {N_FOLDS}-fold CV"),
    ("abcde", ["--sessions", "a,b"], f"only 3 dev sessions would be left, and {N_FOLDS}-fold CV"),
])
def test_lock_test_refuses_and_writes_nothing(sessions, capsys, recorded, argv, message):
    sessions["all"] = list(recorded)
    with pytest.raises(SystemExit):
        run.main(["lock-test", *argv])
    assert message in capsys.readouterr().err
    assert not run.SPLITS_PATH.exists()
