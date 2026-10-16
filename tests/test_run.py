"""
Tests for spinwarn.run: the dataset build, its staleness check, the test lock, cv, tune and the final run.

Synthetic grid sessions with every column labels and features read, and the
paths and session lists swapped for tmp_path ones, so they run anywhere
without data/raw/ and never touch the repo's splits.json, results/ or
experiments.md. The final run's test sessions are synthetic too, from a
stand-in for data.load_test_sessions: no test here loads the real ones.
"""

import copy
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.inspection import permutation_importance

from spinwarn import baselines, cv, evaluate, models, run
from spinwarn.config import EXPERIMENTS_PATH, FA_TARGETS, GRID_HZ, N_FOLDS, SEED
from spinwarn.features import FEATURES, INPUTS, make_features
from spinwarn.labeling import find_onsets, label


STEP = 1 / GRID_HZ


def session(spin_at=5.0, seed=0, ramp=False):
    """12 s at 50 km/h on the grid: 8 s, a 0.5 s gap, 4 s. |slip| is over 90° from spin_at, if given.

    With ramp, |slip| climbs from 20° toward 90° over the second before
    spin_at, so B1 sees each spin coming; otherwise it jumps.
    """
    t = np.concatenate([np.arange(round(8.0 * GRID_HZ)), np.arange(round(8.5 * GRID_HZ), round(12.5 * GRID_HZ))])
    t = t / GRID_HZ
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.normal(size=len(t)).cumsum() for name in INPUTS})
    over = t >= spin_at - STEP / 2 if spin_at is not None else np.zeros(len(t), dtype=bool)
    slip = np.where(over, 120.0, 20.0)
    if ramp and spin_at is not None:
        climbing = (t > spin_at - 1.0 + STEP / 2) & ~over
        slip[climbing] = 20.0 + 70.0 * (t[climbing] - (spin_at - 1.0))
    return df.assign(t=t, segment=(t > 8.0).astype(int), speed_kmh=50.0, gear=3, abs_slip=slip, slip_deg=slip)


def locked_session(state, session_id):
    """A locked test session as the stand-in loads it: the same every time it's loaded."""
    spin_at = 5.0 if session_id in state["spinning"] else None
    return session(spin_at=spin_at, seed=100 + state["all"].index(session_id), ramp=state["ramp"])


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    """Recorded sessions a, b and c, all dev until a test changes "dev"; "loaded" lists the loads.

    The sessions in "spinning" spin once, at 5 s: a, until a test changes
    it. With "ramp", every spin is seen coming (session()). The locked test
    sessions, "test", load only through a stand-in for
    data.load_test_sessions, which counts its calls in "test_loads"; their
    kinds are in "kinds".
    """
    return recorded(tmp_path, monkeypatch)


def recorded(tmp_path, monkeypatch):
    """The sessions fixture's state, with run's paths and session loads swapped for tmp_path's and these."""
    state = {"all": ["a", "b", "c"], "dev": ["a", "b", "c"], "test": [], "loaded": [], "spinning": {"a"},
             "ramp": False, "kinds": {}, "test_loads": 0}
    monkeypatch.setattr(run, "DATASET_PATH", tmp_path / "processed" / "dataset.parquet")
    monkeypatch.setattr(run, "SPLITS_PATH", tmp_path / "splits.json")
    monkeypatch.setattr(run, "CV_DIR", tmp_path / "cv")
    monkeypatch.setattr(run, "EXPERIMENTS_PATH", tmp_path / "experiments.md")
    monkeypatch.setattr(run, "CHOSEN_PATH", tmp_path / "results" / "chosen.json")
    monkeypatch.setattr(run, "METRICS_PATH", tmp_path / "results" / "metrics.json")
    monkeypatch.setattr(run, "FINAL_LOCK_PATH", tmp_path / "results" / "final.lock")
    monkeypatch.setattr(run, "list_sessions", lambda: list(state["all"]))
    monkeypatch.setattr(run, "dev_sessions", lambda: list(state["dev"]))

    def load(session_id):
        state["loaded"].append(session_id)
        spin_at = 5.0 if session_id in state["spinning"] else None
        return session(spin_at=spin_at, seed=len(state["loaded"]), ramp=state["ramp"])

    def load_test(confirm=False):
        assert confirm, "only the final run loads the test sessions, with confirm=True"
        state["test_loads"] += 1
        return {s: locked_session(state, s) for s in state["test"]}

    monkeypatch.setattr(run, "load_session", load)
    monkeypatch.setattr(run, "load_test_sessions", load_test)
    monkeypatch.setattr(run, "session_kind", lambda s: state["kinds"].get(s))
    return state


def lock(sessions, *test):
    """Lock test as the test sessions, as lock-test would, and leave them out of the dev sessions."""
    run.SPLITS_PATH.write_text(json.dumps({"test": list(test)}))
    sessions["dev"] = [s for s in sessions["all"] if s not in test]
    sessions["test"] = list(test)


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


def cells(line):
    """The cells of a row of a Markdown table."""
    return [cell.strip() for cell in line.strip("|").split(" | ")]


def last_row(path):
    """The cells of the last row of a Markdown table file."""
    return cells(path.read_text().splitlines()[-1])


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


def test_cv_fits_a_model_per_fold_with_its_params(sessions, capsys):
    cv_ready(sessions)
    run.main(["cv", "--method", "lr", "--params", '{"C": 0.1}'])
    out = capsys.readouterr().out

    (path,) = run.CV_DIR.glob("*.parquet")
    oof = pd.read_parquet(path)
    dataset = run.load_dataset()
    np.testing.assert_allclose(oof["score"], cv.oof_scores(dataset, "lr", {"C": 0.1}))
    assert path.stem.endswith("_lr") and oof.attrs["params"] == {"C": 0.1}
    assert 'lr {"C": 0.1} at horizon 0.5 s' in out and "3 spins" in out
    assert last_row(run.EXPERIMENTS_PATH)[2:6] == ["lr", "0.5", f"`FEATURES` ({len(FEATURES)})", '`{"C": 0.1}`']


@pytest.mark.parametrize("argv, message", [
    (["--method", "b1", "--params", '{"theta1": 50}'], "b1 takes params [], not ['theta1']"),
    (["--method", "b2", "--params", '{"theta2": 1}'], "b2 takes params ['theta1'], not ['theta2']"),
    (["--method", "b2", "--params", '{"theta1": "50"}'], "b2's theta1 is a number"),
    (["--method", "rf", "--params", '{"max_dept": 5}'], "rf takes params ["),
    (["--method", "lr", "--params", '{"C": true}'], "lr's C is a number, a string or null, not True"),
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


def test_cv_with_a_value_scikit_learn_refuses_writes_nothing(sessions):
    cv_ready(sessions)
    with pytest.raises(ValueError, match="'C' parameter"):
        run.main(["cv", "--method", "lr", "--params", '{"C": -1}'])
    assert not run.CV_DIR.exists() and not run.EXPERIMENTS_PATH.exists()


# --- tune -------------------------------------------------------------------------

def chosen(horizon=0.5):
    return json.loads(run.chosen_path(horizon).read_text())


def test_tune_runs_each_config_and_records_the_best_by_pr_auc(sessions, capsys):
    cv_ready(sessions)
    capsys.readouterr()
    run.main(["tune", "--method", "lr", "--n", "15"])
    out = capsys.readouterr().out

    # lr's grid has 4 configs, so --n 15 runs all 4, in models.configs' order.
    runs = sorted(run.CV_DIR.glob("*.parquet"))
    assert len(runs) == 4
    oofs = [pd.read_parquet(path).attrs for path in runs]
    assert [r["params"] for r in oofs] == models.configs("lr", 15)
    rows = [cells(line) for line in run.EXPERIMENTS_PATH.read_text().splitlines()[2:]]
    assert [row[-1] for row in rows] == ["tune 1/4", "tune 2/4", "tune 3/4", "tune 4/4"]
    assert [row[1] for row in rows] == [f"`{r['run_id']}`" for r in oofs]

    best = max(oofs, key=lambda r: r["metrics"]["pr_auc"])
    assert chosen() == {"lr": {"run_id": best["run_id"], "params": best["params"],
                               "chosen_by": "PR-AUC, best of 4 configs", "pr_auc": best["metrics"]["pr_auc"]}}
    assert f"chosen: lr {json.dumps(best['params'])}, run {best['run_id']}" in out
    assert out.count("/4 {") == 4 and "at 0.5 FA/min" in out


def test_tune_tries_n_configs_sampled_with_the_seed(sessions):
    cv_ready(sessions)
    run.main(["tune", "--method", "lr", "--n", "2"])
    params = [pd.read_parquet(path).attrs["params"] for path in sorted(run.CV_DIR.glob("*.parquet"))]
    assert params == models.configs("lr", 2) and len(params) == 2
    assert chosen()["lr"]["chosen_by"] == "PR-AUC, best of 2 configs"


def test_tunes_note_goes_on_every_runs_row_after_its_own(sessions):
    cv_ready(sessions)
    run.main(["tune", "--method", "lr", "--n", "2", "--note", "again | fixed"])
    run.main(["tune", "--method", "b1", "--note", "again"])
    notes = [cells(line)[-1] for line in run.EXPERIMENTS_PATH.read_text().splitlines()[2:]]
    assert notes == [r"tune 1/2; again \| fixed", r"tune 2/2; again \| fixed", "tune; again"]
    assert pd.read_parquet(min(run.CV_DIR.glob("*_lr.parquet"))).attrs["note"] == "tune 1/2; again | fixed"


def test_a_tie_on_pr_auc_goes_to_the_config_tried_first(sessions, monkeypatch):
    # Every config scores alike when the model ignores its params.
    make_model = models.make_model
    monkeypatch.setattr(models, "make_model", lambda name, params: make_model(name, {}))
    cv_ready(sessions)
    run.main(["tune", "--method", "lr", "--n", "3"])
    first = pd.read_parquet(min(run.CV_DIR.glob("*.parquet"))).attrs
    assert chosen()["lr"]["run_id"] == first["run_id"] and chosen()["lr"]["params"] == models.configs("lr", 3)[0]


@pytest.mark.parametrize("method", ["b1", "b3"])
def test_tune_runs_a_baseline_with_nothing_to_tune_once(sessions, capsys, method):
    cv_ready(sessions)
    run.main(["tune", "--method", method, "--n", "15"])
    (path,) = run.CV_DIR.glob("*.parquet")
    assert chosen() == {method: {"run_id": path.stem, "params": {}, "chosen_by": "nothing to tune",
                                 "pr_auc": pd.read_parquet(path).attrs["metrics"]["pr_auc"]}}
    assert last_row(run.EXPERIMENTS_PATH)[-1] == "tune"
    assert "1/1 {}" in capsys.readouterr().out


def test_tune_b2_picks_theta1_by_the_headline_as_cv_does(sessions, capsys, monkeypatch):
    monkeypatch.setattr(cv, "B2_THETA1_DEG", (10, 15))
    cv_ready(sessions)
    run.main(["tune", "--method", "b2"])
    (path,) = run.CV_DIR.glob("*.parquet")
    swept = "θ1 best of 10–15° by recall @ 1 FA/min, lead ≥ 0.5 s"
    assert chosen()["b2"] == {"run_id": path.stem, "params": {"theta1": 10}, "chosen_by": swept,
                              "pr_auc": pd.read_parquet(path).attrs["metrics"]["pr_auc"]}
    assert last_row(run.EXPERIMENTS_PATH)[-1] == f"{swept}; tune"
    assert "θ1 = 10°" in capsys.readouterr().out
    assert swept in run.CHOSEN_PATH.read_text()   # readable, not \u escapes


def test_tune_keeps_the_other_methods_and_replaces_its_own(sessions):
    cv_ready(sessions)
    run.main(["tune", "--method", "lr", "--n", "1"])
    run.main(["tune", "--method", "b3"])
    first_lr = chosen()["lr"]
    assert list(chosen()) == ["b3", "lr"]    # in METHODS order

    run.main(["tune", "--method", "lr", "--n", "2"])
    assert list(chosen()) == ["b3", "lr"] and chosen()["lr"]["chosen_by"] == "PR-AUC, best of 2 configs"
    assert chosen()["lr"]["run_id"] != first_lr["run_id"]


def test_tune_at_another_horizon_records_into_a_file_of_its_own(sessions):
    cv_ready(sessions, "--horizon", "1")
    run.main(["tune", "--method", "b1", "--horizon", "1"])
    assert run.chosen_path(1.0).name == "chosen_h1.json" and not run.CHOSEN_PATH.exists()
    (path,) = run.CV_DIR.glob("*.parquet")
    assert path.stem.endswith("_b1_h1") and chosen(1.0)["b1"]["run_id"] == path.stem


@pytest.mark.parametrize("argv, message", [
    (["--method", "lr", "--n", "0"], "argument --n: 0 configs"),
    (["--method", "lr", "--n", "-3"], "argument --n: -3 configs"),
    (["--method", "lr", "--n", "two"], "not a whole number: 'two'"),
    (["--method", "xgb"], "invalid choice: 'xgb'"),
    (["--n", "3"], "required: --method"),
    (["--method", "b1", "--horizon", "1"], "run `python -m spinwarn.run build --horizon 1` first"),
])
def test_tune_refuses_and_writes_nothing(sessions, capsys, argv, message):
    cv_ready(sessions)
    with pytest.raises(SystemExit):
        run.main(["tune", *argv])
    assert message in capsys.readouterr().err
    assert not run.CV_DIR.exists() and not run.EXPERIMENTS_PATH.exists() and not run.CHOSEN_PATH.exists()


def test_tune_before_the_lock_is_refused(sessions, capsys):
    run.main(["build"])
    with pytest.raises(SystemExit):
        run.main(["tune", "--method", "lr"])
    assert "no test sessions locked" in capsys.readouterr().err


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


# --- final ------------------------------------------------------------------------

def final_ready(sessions, monkeypatch, *horizon_args):
    """Eight recorded sessions, f (natural), g and h (deliberate) locked as test, the rest built and every method tuned.

    Every session but d and e spins once, at 5 s, and is seen coming
    (session()'s ramp). Each model tunes one small config (SMALL_GRIDS).
    """
    for name, grid in SMALL_GRIDS.items():
        monkeypatch.setitem(models.SEARCH, name, grid)
    monkeypatch.setattr(cv, "B2_THETA1_DEG", (10, 15))
    quick_final(monkeypatch)
    sessions["all"] = list("abcdefgh")
    sessions["spinning"] = {"a", "b", "c", "f", "g", "h"}
    sessions["ramp"] = True
    sessions["kinds"] = {"f": "natural", "g": "deliberate", "h": "deliberate"}
    lock(sessions, "f", "g", "h")
    run.main(["build", *horizon_args])
    for method in cv.METHODS:
        run.main(["tune", "--method", method, "--n", "1", *horizon_args])


# One small config per model for the synthetic tune, so a final run on it takes about a second.
SMALL_GRIDS = {"lr": {"C": [0.1]}, "rf": {"n_estimators": [20], "min_samples_leaf": [5]},
               "hgb": {"max_leaf_nodes": [15]}}


def quick_final(monkeypatch):
    """The final run timing N_TIMING = 5 predictions per method and shuffling each feature twice."""
    monkeypatch.setattr(run, "N_TIMING", 5)
    monkeypatch.setattr(run, "N_PERMUTATIONS", 2)


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    """(files, state): final_ready's, built and tuned once for every final test that starts from it."""
    root = tmp_path_factory.mktemp("tuned")
    with pytest.MonkeyPatch.context() as monkeypatch:
        state = recorded(root, monkeypatch)
        final_ready(state, monkeypatch)
    return root, state


@pytest.fixture
def ready(sessions, tmp_path, monkeypatch, tuned):
    """The sessions fixture as final_ready leaves it, from a copy of tuned's files."""
    root, state = tuned
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    sessions.update({key: copy.deepcopy(state[key]) for key in ["all", "dev", "test", "spinning", "ramp", "kinds"]})
    quick_final(monkeypatch)
    return sessions


def final_test_rows(sessions, horizon=0.5):
    """The test sessions' rows as the final run scores them: loaded, then labelled with today's rule."""
    return pd.concat([run.session_rows(s, locked_session(sessions, s), horizon) for s in sessions["test"]],
                     ignore_index=True)


def refitted(method, params, dataset):
    """method fitted on every dev session's training rows, without run.refit."""
    fit_on = dataset.iloc[cv.training_rows(dataset, np.arange(len(dataset)))]
    return models.make_model(method, params).fit(fit_on[FEATURES], fit_on["target"])


def expected_scores(method, params, dataset, test):
    """method's scores on the test rows, a model's refit on every dev session, without run.method_scores."""
    if method in baselines.BASELINES:
        return baselines.score(method, test, params)
    known = test[FEATURES].notna().all(axis=1).to_numpy()
    scores = np.full(len(test), np.nan)
    scores[known] = refitted(method, params, dataset).predict_proba(test.loc[known, FEATURES])[:, 1]
    return scores


def as_written(summary):
    """summary as metrics.json holds it: NaN as null."""
    return {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in summary.items()}


def test_final_scores_every_chosen_run_on_the_test_sessions_at_its_cv_thresholds(ready, monkeypatch, capsys):
    monkeypatch.setattr(run, "MIN_KIND_SPINS", 2)
    capsys.readouterr()
    run.main(["final"])
    out = capsys.readouterr().out

    assert ready["test_loads"] == 1
    assert json.loads(run.FINAL_LOCK_PATH.read_text())["forced"] is False
    metrics = json.loads(run.METRICS_PATH.read_text())
    assert "NaN" not in run.METRICS_PATH.read_text()   # null instead, as strict JSON has it
    assert list(metrics["methods"]) == cv.METHODS and metrics["horizon"] == 0.5 and metrics["forced"] is False
    dataset, test = run.load_dataset(), final_test_rows(ready)
    assert metrics["dev"]["sessions"] == list("abcde") and metrics["built_with"] == run.built_with(0.5)
    # One spin each, found again by today's rule: f natural, g and h deliberate.
    assert {s: (v["kind"], v["spins"]) for s, v in metrics["test"]["sessions"].items()} == {
        "f": ("natural", 1), "g": ("deliberate", 1), "h": ("deliberate", 1)}
    assert all(len(find_onsets(locked_session(ready, s))[0]) == 1 for s in "fgh")
    assert "30 positive frames" in out and "kinds: f natural, g deliberate, h deliberate" in out

    chosen = json.loads(run.CHOSEN_PATH.read_text())
    for method in cv.METHODS:
        result, entry = metrics["methods"][method], chosen[method]
        oof = pd.read_parquet(run.CV_DIR / f"{entry['run_id']}.parquet")
        assert (result["run_id"], result["params"]) == (entry["run_id"], entry["params"])
        assert result["cv"]["pr_auc"] == pytest.approx(entry["pr_auc"])
        assert len(result["cv"]["curve"]["threshold"]) == len(evaluate.recall_fa_curve(oof, oof["score"]))
        scores = expected_scores(method, entry["params"], dataset, test)
        assert result["test"]["pr_auc"] == pytest.approx(evaluate.pr_auc(test, scores))

        for target, point in zip(FA_TARGETS, result["test"]["operating_points"], strict=True):
            # The threshold from the chosen run's out-of-fold scores, never from the test sessions.
            threshold = evaluate.threshold_for_fa(oof, oof["score"], target)
            assert (point["fa_target"], point["threshold"]) == (target, threshold)
            summary = evaluate.summarize(test, scores, threshold)
            assert {k: point[k] for k in summary} == pytest.approx(as_written(summary))
            # Both recalls' intervals, from one catch or miss per test spin.
            spins = evaluate.spin_catches(test, scores, threshold)
            assert point["recall_min_lead_ci"] == list(evaluate.bootstrap_recall(spins["caught_min_lead"]))
            assert point["recall_ci"] == list(evaluate.bootstrap_recall(spins["caught"]))
            assert [(s["session"], s["kind"], s["caught"]) for s in point["spins"]] == list(
                zip("fgh", ["natural", "deliberate", "deliberate"], spins["caught"]))
            for kind, of_kind in [("natural", ["f"]), ("deliberate", ["g", "h"])]:
                rows = test["session"].isin(of_kind).to_numpy()
                by_kind = point["by_kind"][kind]
                expected = evaluate.summarize(test[rows], scores[rows], threshold)
                assert {k: by_kind[k] for k in expected} == pytest.approx(as_written(expected))
                in_kind = spins[spins["session"].isin(of_kind)]
                assert by_kind["recall_ci"] == list(evaluate.bootstrap_recall(in_kind["caught"]))
                assert by_kind["enough"] is (kind == "deliberate")   # 2 spins against 1, at MIN_KIND_SPINS 2
        assert set(result["timing"]) == ({"ms", "ms_n_jobs_1"} if method == "rf" else {"ms"})
        assert all(ms > 0 for ms in result["timing"].values())

    # Every spin climbs for a second first, so B1 at its CV threshold catches all three a second ahead.
    b1 = metrics["methods"]["b1"]["test"]["operating_points"][0]
    assert (b1["caught_min_lead"], b1["false_alarms"], b1["recall_min_lead_ci"]) == (3, 0, [1.0, 1.0])
    assert "b1 {}: PR-AUC" in out and "    natural: recall with ≥ 0.5 s lead 1.00 (1/1) [1.00, 1.00]" in out
    assert "under 2 spins, so for the limitations" in out and f"wrote {run.METRICS_PATH}" in out


def test_permutation_importance_is_the_best_cv_models_on_the_scored_test_frames(ready, monkeypatch, capsys):
    run.main(["final"])
    metrics = json.loads(run.METRICS_PATH.read_text())
    ranked = metrics["importance"]

    # Best by CV PR-AUC, as the Project Plan selects models: the test sessions never pick it.
    chosen = json.loads(run.CHOSEN_PATH.read_text())
    best = max(models.MODELS, key=lambda m: chosen[m]["pr_auc"])
    assert (ranked["method"], ranked["scoring"], ranked["n_repeats"]) == (best, "average_precision", 2)
    assert sorted(f["feature"] for f in ranked["features"]) == sorted(FEATURES)
    means = [f["mean"] for f in ranked["features"]]
    assert means == sorted(means, reverse=True)

    test = final_test_rows(ready)
    scored = test[~test["excluded"]]
    expected = permutation_importance(refitted(best, chosen[best]["params"], run.load_dataset()), scored[FEATURES],
                                      scored["target"], scoring="average_precision", n_repeats=2, random_state=SEED)
    assert {f["feature"]: f["mean"] for f in ranked["features"]} == pytest.approx(
        dict(zip(FEATURES, expected.importances_mean)))
    assert f"permutation importance of {best}" in capsys.readouterr().out


def test_the_final_run_refuses_to_run_again_while_its_lock_exists(ready, monkeypatch, capsys):
    run.main(["final"])
    written, logged = run.METRICS_PATH.read_text(), run.EXPERIMENTS_PATH.read_text()
    capsys.readouterr()

    with pytest.raises(SystemExit):
        run.main(["final"])
    err = capsys.readouterr().err
    assert f"{run.FINAL_LOCK_PATH} exists: a final run loaded the test sessions at" in err
    assert f"its results are in {run.METRICS_PATH}" in err and "--force scores them again" in err
    assert ready["test_loads"] == 1
    assert run.METRICS_PATH.read_text() == written and run.EXPERIMENTS_PATH.read_text() == logged


def test_force_scores_the_test_sessions_again_and_logs_it_in_experiments(ready, monkeypatch):
    run.main(["final", "--force", "--note", "not logged"])   # nothing to force yet: an ordinary run
    first = json.loads(run.FINAL_LOCK_PATH.read_text())["started"]
    rows = run.EXPERIMENTS_PATH.read_text().splitlines()
    assert json.loads(run.METRICS_PATH.read_text())["forced"] is False and "not logged" not in rows[-1]

    run.main(["final", "--force", "--note", "fixed a | bug"])
    assert ready["test_loads"] == 2
    assert json.loads(run.METRICS_PATH.read_text())["forced"] is True
    assert json.loads(run.FINAL_LOCK_PATH.read_text())["forced"] is True
    after = run.EXPERIMENTS_PATH.read_text().splitlines()
    assert after[:-1] == rows
    date, run_id, method, horizon, *rest, note = cells(after[-1])
    assert (run_id, method, horizon, rest) == ("`final`", "every method", "0.5", ["—"] * 5)
    assert note == (f"final run forced: the test sessions scored again (last loaded at {first}), results in "
                    r"metrics.json; fixed a \| bug")


def test_metrics_json_blocks_an_unforced_final_run_with_its_lock_deleted(ready, monkeypatch, capsys):
    run.main(["final"])
    first = json.loads(run.METRICS_PATH.read_text())["started"]
    written, logged = run.METRICS_PATH.read_text(), run.EXPERIMENTS_PATH.read_text()
    run.FINAL_LOCK_PATH.unlink()   # taken for a stale lockfile
    capsys.readouterr()

    with pytest.raises(SystemExit):
        run.main(["final"])
    err = capsys.readouterr().err
    assert f"{run.METRICS_PATH} exists, though final.lock doesn't: a final run loaded the test sessions at {first}" in err
    assert "--force scores them again" in err
    assert ready["test_loads"] == 1 and not run.FINAL_LOCK_PATH.exists()
    assert run.METRICS_PATH.read_text() == written and run.EXPERIMENTS_PATH.read_text() == logged

    # Forced, it runs as any forced run does: logged, with when the results were from.
    run.main(["final", "--force"])
    assert ready["test_loads"] == 2
    assert json.loads(run.METRICS_PATH.read_text())["forced"] is True
    assert json.loads(run.FINAL_LOCK_PATH.read_text())["forced"] is True
    assert f"(last loaded at {first})" in run.EXPERIMENTS_PATH.read_text().splitlines()[-1]


def test_write_json_writes_nan_as_null_and_numpy_scalars_as_python_ones(tmp_path):
    path = tmp_path / "results" / "out.json"
    run.write_json({"a": float("nan"), "b": [np.float64("nan"), np.float32(0.5), np.int64(3)], "c": (np.bool_(True),),
                    "θ1": 25}, path)
    text = path.read_text()
    assert "NaN" not in text and "θ1" in text

    def strict(constant):
        raise ValueError(f"{constant} isn't strict JSON")

    assert json.loads(text, parse_constant=strict) == {"a": None, "b": [None, 0.5, 3], "c": [True], "θ1": 25}


def test_a_final_run_that_stops_partway_keeps_its_lock(ready, monkeypatch, capsys):

    def broken(model, test):
        raise RuntimeError("a bug after the test sessions loaded")

    monkeypatch.setattr(run, "importance", broken)
    with pytest.raises(RuntimeError, match="a bug after"):
        run.main(["final"])
    assert ready["test_loads"] == 1 and run.FINAL_LOCK_PATH.exists() and not run.METRICS_PATH.exists()

    capsys.readouterr()
    with pytest.raises(SystemExit):
        run.main(["final"])
    assert "it never wrote metrics.json, so it stopped partway" in capsys.readouterr().err
    assert ready["test_loads"] == 1


def edit_chosen(change):
    """A change to the repo's chosen.json, written back."""
    def apply(sessions):
        chosen = json.loads(run.CHOSEN_PATH.read_text())
        change(chosen)
        run.CHOSEN_PATH.write_text(json.dumps(chosen))
    return apply


def edit_run(method, change):
    """A change to method's chosen CV run file, written back."""
    def apply(sessions):
        path = run.CV_DIR / f"{json.loads(run.CHOSEN_PATH.read_text())[method]['run_id']}.parquet"
        oof = pd.read_parquet(path)
        change(oof)
        oof.to_parquet(path)
    return apply


def drop_run(method):
    def apply(sessions):
        (run.CV_DIR / f"{json.loads(run.CHOSEN_PATH.read_text())[method]['run_id']}.parquet").unlink()
    return apply


def copy_in(sessions):
    sessions["all"].append("i")
    sessions["dev"].append("i")


@pytest.mark.parametrize("change, message", [
    (lambda sessions: run.CHOSEN_PATH.unlink(),
     "chosen.json: run `python -m spinwarn.run tune --method <method>` for every method (b1, b2, b3, lr, rf, hgb) first"),
    (edit_chosen(lambda chosen: chosen.pop("hgb")), "chosen.json has no run for hgb"),
    (drop_run("rf"), "rf's chosen run. results/cv/ isn't in git, but tune is seeded: tune it again with "
                     "`python -m spinwarn.run tune --method rf`"),
    (edit_chosen(lambda chosen: chosen["lr"].update(params={"C": 12345})), """not lr with chosen.json's {"C": 12345}"""),
    (edit_run("b3", lambda oof: oof.attrs.update(sessions=["a", "b"])), "b3's chosen run scored other sessions"),
    (edit_run("b1", lambda oof: oof.attrs["built_with"].update(GAP_S=0.2)), "b1's chosen run was built with other"),
    (edit_run("hgb", lambda oof: oof.__setitem__("target", ~oof["target"])), "hgb's chosen run scored other rows"),
    (copy_in, "the dataset was built from other sessions"),
])
def test_final_refuses_before_it_loads_the_test_sessions(ready, monkeypatch, capsys, change, message):
    change(ready)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run.main(["final"])
    assert message in capsys.readouterr().err
    assert ready["test_loads"] == 0 and not run.FINAL_LOCK_PATH.exists() and not run.METRICS_PATH.exists()


def test_final_at_another_horizon_reads_its_own_chosen_runs_and_writes_its_own_files(sessions, monkeypatch, capsys):
    final_ready(sessions, monkeypatch, "--horizon", "1")
    capsys.readouterr()
    run.main(["final", "--horizon", "1"])
    out = capsys.readouterr().out

    assert (run.metrics_path(1.0).name, run.lock_path(1.0).name) == ("metrics_h1.json", "final_h1.lock")
    assert run.lock_path(1.0).exists() and not run.FINAL_LOCK_PATH.exists() and not run.METRICS_PATH.exists()
    metrics = json.loads(run.metrics_path(1.0).read_text())
    assert metrics["horizon"] == 1.0 and metrics["built_with"] == run.built_with(1.0)
    chosen = json.loads(run.chosen_path(1.0).read_text())
    assert all(metrics["methods"][m]["run_id"] == chosen[m]["run_id"] for m in cv.METHODS)
    # The test sessions labelled at 1.0 s: 60 target frames before each spin.
    assert "60 positive frames" in out and "labelled at horizon 1 s" in out

    # Each horizon has its own lock: the 0.5 s run isn't refused by 1.0's, only by its missing chosen runs.
    with pytest.raises(SystemExit):
        run.main(["final"])
    assert f"no {run.CHOSEN_PATH}: run" in capsys.readouterr().err and sessions["test_loads"] == 1


def test_final_at_a_horizon_nothing_was_tuned_at_says_to_tune_every_method_there(sessions, capsys):
    cv_ready(sessions)
    with pytest.raises(SystemExit):
        run.main(["final", "--horizon", "1"])
    assert (f"no {run.chosen_path(1.0)}: run `python -m spinwarn.run tune --method <method> --horizon 1` for every "
            "method (b1, b2, b3, lr, rf, hgb) first") in capsys.readouterr().err
    assert sessions["test_loads"] == 0 and not run.lock_path(1.0).exists()


def test_each_kind_is_scored_on_its_own_sessions_and_a_kind_without_spins_has_no_interval(monkeypatch):
    monkeypatch.setattr(run, "MIN_KIND_SPINS", 1)
    test = pd.concat([run.session_rows("f", session(ramp=True)), run.session_rows("g", session(spin_at=None, seed=1))],
                     ignore_index=True)
    # B1 at 20° alarms only on f's climb, from 4.0 s plus a frame.
    point = run.operating_point(test, test["abs_slip"].to_numpy(), 20.0, {"f": "natural", "g": "deliberate"})

    assert (point["n_spins"], point["caught_min_lead"], point["false_alarms"]) == (1, 1, 0)
    assert point["spins"] == [{"session": "f", "t": 5.0, "caught": True, "caught_min_lead": True,
                               "lead": pytest.approx(59 / 60), "kind": "natural"}]
    natural, deliberate = point["by_kind"]["natural"], point["by_kind"]["deliberate"]
    assert natural["n_spins"] == 1 and natural["enough"] and natural["recall_min_lead_ci"] == [1.0, 1.0]
    assert (deliberate["n_spins"], deliberate["alerts"], deliberate["enough"]) == (0, 0, False)
    assert np.isnan(deliberate["recall_ci"]).all() and np.isnan(deliberate["recall_min_lead_ci"]).all()
    assert deliberate["minutes"] == pytest.approx(evaluate.summarize(test[test["session"] == "g"], test.loc[
        test["session"] == "g", "abs_slip"], 20.0)["minutes"])


class Timed:
    """A model with n_jobs, as rf has: it keeps each row it scores, and the n_jobs it scored it at."""

    def __init__(self):
        self.n_jobs, self.calls = -1, []

    def get_params(self):
        return {"n_jobs": self.n_jobs}

    def set_params(self, **params):
        self.n_jobs = params["n_jobs"]
        return self

    def predict_proba(self, X):
        self.calls.append((X, self.n_jobs))
        return np.array([[0.5, 0.5]])


def test_a_model_is_timed_on_one_row_arrays_of_scored_frames_and_on_one_core_too(monkeypatch):
    monkeypatch.setattr(run, "N_TIMING", 7)
    test = run.session_rows("f", session())
    model = Timed()
    timing = run.time_per_prediction("rf", {}, model, test)

    assert set(timing) == {"ms", "ms_n_jobs_1"} and all(ms >= 0 for ms in timing.values())
    assert [n_jobs for _X, n_jobs in model.calls] == [-1] * 7 + [1] * 7 and model.n_jobs == -1
    scored = {tuple(row) for row in test.loc[~test["excluded"], FEATURES].to_numpy()}
    for X, _n_jobs in model.calls:
        assert isinstance(X, np.ndarray) and X.shape == (1, len(FEATURES)) and tuple(X[0]) in scored
    # The same frames on one core as on all of them, picked with the seed.
    assert [X.tolist() for X, _ in model.calls[:7]] == [X.tolist() for X, _ in model.calls[7:]]


def test_a_baseline_is_timed_on_one_row_slices_of_its_signals(monkeypatch):
    monkeypatch.setattr(run, "N_TIMING", 7)
    calls = []
    monkeypatch.setitem(baselines.BASELINES, "b2", lambda rows, theta1: calls.append((list(rows.columns), len(rows),
                                                                                     theta1)))
    timing = run.time_per_prediction("b2", {"theta1": 10}, None, run.session_rows("f", session()))
    assert set(timing) == {"ms"} and calls == [(["abs_slip", "abs_slip_rate"], 1, 10)] * 7
