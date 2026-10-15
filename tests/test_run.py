"""
Tests for spinwarn.run: the dataset build, its staleness check, the test lock and cv.

Synthetic grid sessions with every column labels and features read, and the
paths and session lists swapped for tmp_path ones, so they run anywhere
without data/raw/ and never touch the repo's splits.json, results/ or
experiments.md.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from spinwarn import cv, evaluate, run
from spinwarn.config import EXPERIMENTS_PATH, GRID_HZ, N_FOLDS
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

    The sessions in "spinning" spin once, at 5 s: a, until a test changes it.
    """
    state = {"all": ["a", "b", "c"], "dev": ["a", "b", "c"], "loaded": [], "spinning": {"a"}}
    monkeypatch.setattr(run, "DATASET_PATH", tmp_path / "processed" / "dataset.parquet")
    monkeypatch.setattr(run, "SPLITS_PATH", tmp_path / "splits.json")
    monkeypatch.setattr(run, "CV_DIR", tmp_path / "cv")
    monkeypatch.setattr(run, "EXPERIMENTS_PATH", tmp_path / "experiments.md")
    monkeypatch.setattr(run, "list_sessions", lambda: list(state["all"]))
    monkeypatch.setattr(run, "dev_sessions", lambda: list(state["dev"]))

    def load(session_id):
        state["loaded"].append(session_id)
        return session(spin_at=5.0 if session_id in state["spinning"] else None, seed=len(state["loaded"]))

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


# --- build --horizon --------------------------------------------------------------

def test_build_at_another_horizon_writes_a_dataset_of_its_own(sessions, capsys):
    lock(sessions, "b")
    run.main(["build", "--horizon", "1"])
    out = capsys.readouterr().out
    path = run.DATASET_PATH.with_name("dataset_h1.parquet")

    assert path.exists() and not run.DATASET_PATH.exists()
    # a's 60 frames before its onset are positive at 1.0 s.
    assert "60 positive frames" in out and f"wrote {path}" in out and "horizon 1 s" in out
    dataset = run.load_dataset(1.0)
    assert dataset.attrs["built_with"] == run.built_with(1.0) and dataset.attrs["built_with"]["horizon"] == 1.0
    a = dataset[dataset["session"] == "a"].reset_index(drop=True)
    labelled = label(session(spin_at=5.0), 1.0)
    assert (a[["target", "excluded"]].to_numpy() == labelled[["target", "excluded"]].to_numpy()).all()
    with pytest.raises(FileNotFoundError, match=r"run `python -m spinwarn.run build` first"):
        run.load_dataset()

    # The main one is built beside it, and neither replaces the other.
    run.main(["build"])
    assert run.load_dataset()["target"].sum() == 30 and run.load_dataset(1.0)["target"].sum() == 60


def test_a_horizon_dataset_out_of_date_says_to_build_it_at_that_horizon(sessions):
    lock(sessions, "b")
    run.main(["build", "--horizon", "1"])
    sessions["dev"].append("d")
    with pytest.raises(RuntimeError, match=r"run `python -m spinwarn.run build --horizon 1` again"):
        run.load_dataset(1.0)


@pytest.mark.parametrize("text", ["0", "-0.5", "nan", "inf", "0.51", "half"])
def test_build_refuses_a_horizon_that_isnt_whole_grid_frames_over_0(sessions, capsys, text):
    with pytest.raises(SystemExit):
        run.main(["build", "--horizon", text])
    assert "argument --horizon" in capsys.readouterr().err
    assert not run.DATASET_PATH.parent.exists()


# --- cv ---------------------------------------------------------------------------

def cv_ready(sessions, *build_args):
    """Six recorded sessions, f locked as test, the other five built: a, b and c spin."""
    sessions["all"] = list("abcdef")
    sessions["spinning"] = {"a", "b", "c"}
    lock(sessions, "f")
    run.main(["build", *build_args])


def last_row(path):
    """The cells of the last row of a Markdown table file."""
    return [cell.strip() for cell in path.read_text().splitlines()[-1].strip("|").split(" | ")]


def test_cv_saves_out_of_fold_scores_and_appends_a_row_to_experiments(sessions, capsys):
    cv_ready(sessions)
    capsys.readouterr()
    run.main(["cv", "--method", "b1", "--note", "first | try"])
    out = capsys.readouterr().out

    (path,) = run.CV_DIR.glob("*.parquet")
    assert path.stem.endswith("_b1")
    oof = pd.read_parquet(path)
    dataset = run.load_dataset()
    assert list(oof.columns) == [*run.OOF_COLUMNS, "fold", "score"]
    assert (oof[run.OOF_COLUMNS] == dataset[run.OOF_COLUMNS]).all().all()
    assert (oof["score"] == dataset["abs_slip"]).all()
    assert (oof["fold"] == cv.fold_ids(dataset)).all() and oof["fold"].nunique() == N_FOLDS
    assert {k: oof.attrs[k] for k in ["run_id", "method", "params", "horizon", "note"]} == {
        "run_id": path.stem, "method": "b1", "params": {}, "horizon": 0.5, "note": "first | try"}
    assert oof.attrs["sessions"] == list("abcde") and oof.attrs["built_with"] == run.built_with(0.5)
    assert oof.attrs["metrics"]["pr_auc"] == pytest.approx(evaluate.pr_auc(dataset, dataset["abs_slip"]))

    assert "PR-AUC" in out and "at 1 FA/min" in out and "at 0.5 FA/min" in out and "3 spins" in out
    lines = run.EXPERIMENTS_PATH.read_text().splitlines()
    assert lines[0] == "| " + " | ".join(run.EXPERIMENT_COLUMNS) + " |" and len(lines) == 3
    date, run_id, method, horizon, features, params, pr_auc, recall_min_lead, recall, note = last_row(
        run.EXPERIMENTS_PATH)
    assert (run_id, method, horizon, features, params) == (f"`{path.stem}`", "b1", "0.5", "`abs_slip`", "`{}`")
    at_1 = oof.attrs["metrics"]["operating_points"][0]
    assert pr_auc == f"{oof.attrs['metrics']['pr_auc']:.3f}"
    assert recall_min_lead == f"{at_1['recall_min_lead']:.2f} ({at_1['caught_min_lead']}/3)"
    assert recall == f"{at_1['recall']:.2f} ({at_1['caught']}/3)"
    assert note == r"first \| try" and date == path.stem[:10]


def test_cv_b2_without_params_picks_theta1_first(sessions, capsys, monkeypatch):
    # The synthetic sessions drift at 20°, so θ1 has to be under it to score anything.
    monkeypatch.setattr(cv, "B2_THETA1_DEG", (10, 15))
    cv_ready(sessions)
    run.main(["cv", "--method", "b2"])
    out = capsys.readouterr().out

    assert "θ1 10°" in out and "θ1 15°" in out and "θ1 = 10°" in out
    (path,) = run.CV_DIR.glob("*.parquet")
    assert pd.read_parquet(path).attrs["params"] == {"theta1": 10}
    row = last_row(run.EXPERIMENTS_PATH)
    assert row[5] == '`{"theta1": 10}`' and row[-1] == "θ1 best of 10–15° by recall @ 1 FA/min, lead ≥ 0.5 s"


def test_cv_b2_with_theta1_scores_at_it(sessions, capsys):
    cv_ready(sessions)
    run.main(["cv", "--method", "b2", "--params", '{"theta1": 15}', "--note", "fixed"])
    out = capsys.readouterr().out

    assert "θ1 =" not in out
    oof = pd.read_parquet(next(run.CV_DIR.glob("*.parquet")))
    dataset = run.load_dataset()
    assert oof.attrs["params"] == {"theta1": 15}
    np.testing.assert_array_equal(oof["score"], np.where(dataset["abs_slip"] > 15, dataset["abs_slip_rate"], -np.inf))
    assert last_row(run.EXPERIMENTS_PATH)[-1] == "fixed"


def test_cv_at_another_horizon_reads_its_dataset(sessions, capsys):
    cv_ready(sessions, "--horizon", "1")
    run.main(["cv", "--method", "b3", "--horizon", "1"])

    (path,) = run.CV_DIR.glob("*.parquet")
    assert path.stem.endswith("_b3_h1") and pd.read_parquet(path).attrs["horizon"] == 1.0
    assert last_row(run.EXPERIMENTS_PATH)[2:5] == ["b3", "1", "`time_to_90`"]


@pytest.mark.parametrize("argv, message", [
    (["--method", "b1", "--params", '{"theta1": 50}'], "b1 takes params [], not ['theta1']"),
    (["--method", "b2", "--params", '{"theta2": 1}'], "b2 takes params ['theta1'], not ['theta2']"),
    (["--method", "b2", "--params", '{"theta1": "50"}'], "b2's theta1 is a number"),
    (["--method", "b2", "--params", "[50]"], "not a JSON object"),
    (["--method", "b2", "--params", "{theta1: 50}"], "not JSON"),
    (["--method", "xgb"], "invalid choice: 'xgb'"),
    (["--params", "{}"], "required: --method"),
    (["--method", "b1", "--horizon", "1"], "run `python -m spinwarn.run build --horizon 1` first"),
])
def test_cv_refuses_and_writes_nothing(sessions, capsys, argv, message):
    cv_ready(sessions)
    with pytest.raises(SystemExit):
        run.main(["cv", *argv])
    assert message in capsys.readouterr().err
    assert not run.CV_DIR.exists() and not run.EXPERIMENTS_PATH.exists()


def test_cv_before_the_lock_is_refused(sessions, capsys):
    run.main(["build"])
    with pytest.raises(SystemExit):
        run.main(["cv", "--method", "b1"])
    assert "no test sessions locked" in capsys.readouterr().err


def test_run_ids_never_clash(sessions):
    now = datetime(2026, 10, 15, 14, 30, 12)
    assert run.new_run_id("b1", 0.5, now) == "2026-10-15_143012_b1"
    assert run.new_run_id("b1", 1.0, now) == "2026-10-15_143012_b1_h1"
    run.CV_DIR.mkdir()
    (run.CV_DIR / "2026-10-15_143012_b1.parquet").touch()
    (run.CV_DIR / "2026-10-15_143012_b1_2.parquet").touch()
    assert run.new_run_id("b1", 0.5, now) == "2026-10-15_143012_b1_3"


def test_the_repos_experiments_md_has_the_columns_cv_writes():
    header = next(line for line in EXPERIMENTS_PATH.read_text().splitlines() if line.startswith("| Date"))
    assert header == "| " + " | ".join(run.EXPERIMENT_COLUMNS) + " |"


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
