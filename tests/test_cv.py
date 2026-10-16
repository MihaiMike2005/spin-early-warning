"""
Tests for spinwarn.cv: session-grouped folds, training rows, out-of-fold scores and B2's θ1.

Dataset-like sessions on the GRID_HZ grid with the columns cv reads, and a
stand-in model that shows what each fold fits on, so they run anywhere
without data/raw/.
"""

import numpy as np
import pandas as pd
import pytest

from spinwarn import baselines, cv, evaluate, models
from spinwarn.config import FA_TARGETS, GRID_HZ, HORIZON_S, N_FOLDS, TRAIN_HZ, WINDOW_S
from spinwarn.features import FEATURES


STEP = 1 / GRID_HZ


def during(t, start, end):
    """Grid times in [start, end], with a margin well under a grid step."""
    return (t > start - STEP / 2) & (t < end + STEP / 2)


def session(name, seconds=120.0, onsets=(30.0, 60.0, 90.0), seed=0):
    """A dataset-like session, one segment, driving straight at 20° with random features.

    Each onset has its HORIZON_S of target frames before it, and is followed
    by 2 s spinning. Excluded, as label() would: the spins, the segment's
    first WINDOW_S (where the features are NaN) and its last HORIZON_S.
    """
    t = np.arange(round(seconds * GRID_HZ)) / GRID_HZ
    df = pd.DataFrame({"session": name, "segment": 0, "t": t})
    df["target"] = np.zeros(len(t), dtype=bool)
    df["excluded"] = (t < WINDOW_S - STEP / 2) | (t > seconds - HORIZON_S - STEP / 2)
    for onset in onsets:
        df["target"] |= during(t, onset - HORIZON_S, onset - STEP)
        df["excluded"] |= during(t, onset, onset + 2.0)
    df["onset"] = np.isin(np.rint(t * GRID_HZ), np.rint(np.asarray(onsets) * GRID_HZ))
    df["abs_slip"], df["abs_slip_rate"], df["time_to_90"] = 20.0, 0.0, 5.0
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(len(t), len(FEATURES))), columns=FEATURES)
    features[t < WINDOW_S - STEP / 2] = np.nan
    return pd.concat([df, features], axis=1)


def dataset(n=5, **kwargs):
    """n sessions, s0 to s<n-1>: one more than N_FOLDS, so a fold holds two."""
    return pd.concat([session(f"s{k}", seed=k, **kwargs) for k in range(n)], ignore_index=True)


# --- folds and training rows ----------------------------------------------------

def test_every_session_is_scored_in_one_fold_by_folds_that_never_train_on_it():
    ds = dataset()
    splits = cv.folds(ds)
    assert len(splits) == N_FOLDS
    for train, test in splits:
        assert not set(ds["session"].iloc[train]) & set(ds["session"].iloc[test])
        assert sorted(np.r_[train, test]) == list(range(len(ds)))
    assert sorted(np.concatenate([test for _train, test in splits])) == list(range(len(ds)))
    fold = pd.Series(cv.fold_ids(ds))
    assert fold.groupby(ds["session"]).nunique().eq(1).all() and fold.nunique() == N_FOLDS


def test_a_model_trains_on_the_given_rows_scored_frames_at_train_hz():
    # s1 is scored on [1.0, 10.0) and (12.0, 19.5): its first WINDOW_S, a spin
    # from 10 s to 12 s and its last HORIZON_S are excluded.
    ds = dataset(n=2, seconds=20.0, onsets=(10.0,))
    kept = cv.training_rows(ds, np.flatnonzero(ds["session"] == "s1"))

    assert (ds["session"].iloc[kept] == "s1").all() and not ds["excluded"].iloc[kept].any()
    assert ds["t"].iloc[kept].to_numpy() == pytest.approx(np.r_[np.arange(20, 200), np.arange(241, 390)] / TRAIN_HZ)


# --- oof_scores -------------------------------------------------------------------

@pytest.mark.parametrize("method, params", [("b1", {}), ("b2", {"theta1": 50}), ("b3", {})])
def test_a_baseline_is_scored_out_of_fold_with_nothing_fitted(method, params):
    ds = dataset()
    rng = np.random.default_rng(0)
    ds["abs_slip"] = rng.uniform(0, 100, len(ds))
    ds["abs_slip_rate"] = rng.normal(0, 50, len(ds))
    ds["time_to_90"] = rng.uniform(0, 5, len(ds))
    np.testing.assert_array_equal(cv.oof_scores(ds, method, params), baselines.score(method, ds, params))


class StandIn:
    """A model, as models.make_model makes: remembers what it's fitted on, and scores with its number."""

    def __init__(self, made):
        self.number = len(made)
        made.append(self)

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_proba(self, X):
        p = np.full(len(X), (self.number + 1) / 10)
        return np.c_[1 - p, p]


def test_a_model_is_fitted_per_fold_on_the_other_folds_and_scores_its_own(monkeypatch):
    ds = dataset()
    made = []
    monkeypatch.setattr(cv, "METHODS", [*cv.METHODS, "stand-in"])
    monkeypatch.setattr(models, "make_model", lambda name, params: StandIn(made))
    scores = cv.oof_scores(ds, "stand-in")

    assert len(made) == N_FOLDS
    unknown = ds[FEATURES].isna().any(axis=1).to_numpy()
    for model, (train, test) in zip(made, cv.folds(ds)):
        assert list(model.X.columns) == FEATURES
        assert model.X.index.tolist() == cv.training_rows(ds, train).tolist()
        assert model.y.tolist() == ds["target"].iloc[cv.training_rows(ds, train)].tolist()
        # Every row of its fold, but those with a feature unknown (a segment's first WINDOW_S).
        assert (scores[test][~unknown[test]] == (model.number + 1) / 10).all()
    assert np.isnan(scores[unknown]).all() and not np.isnan(scores[~unknown]).any()


@pytest.mark.parametrize("method, params", [
    ("lr", {"C": 0.1}), ("rf", {"n_estimators": 20}), ("hgb", {"max_leaf_nodes": 15}),
])
def test_each_model_scores_every_frame_with_its_features_known_out_of_fold(method, params):
    ds = dataset(seconds=30.0, onsets=(10.0, 20.0))
    scores = cv.oof_scores(ds, method, params)
    unknown = ds[FEATURES].isna().any(axis=1).to_numpy()
    assert np.isnan(scores[unknown]).all()
    assert ((scores[~unknown] >= 0) & (scores[~unknown] <= 1)).all()
    np.testing.assert_allclose(cv.oof_scores(ds, method, params), scores, rtol=0, atol=1e-12)


def test_the_methods_are_the_baselines_then_the_models():
    assert cv.METHODS == ["b1", "b2", "b3", "lr", "rf", "hgb"]


@pytest.mark.parametrize("method, params, message", [
    ("b2", {"C": 1}, r"b2 takes params \['theta1'\]"),
    ("b1", {}, None),
    ("lr", {"theta1": 50}, r"lr takes params \['C'\], not \['theta1'\]"),
    ("hgb", {}, None),
])
def test_params_are_checked_as_a_baselines_or_a_models(method, params, message):
    if message is None:
        cv.check_params(method, params)
    else:
        with pytest.raises(ValueError, match=message):
            cv.check_params(method, params)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="no method 'xgb'"):
        cv.oof_scores(dataset(n=4, seconds=5.0, onsets=()), "xgb")


# --- metrics ----------------------------------------------------------------------

def test_metrics_are_pr_auc_and_each_fa_targets_operating_point():
    ds = dataset()
    score = np.random.default_rng(0).random(len(ds)) + ds["target"].to_numpy()
    got = cv.metrics(ds, score)

    assert got["pr_auc"] == evaluate.pr_auc(ds, score)
    assert [p["fa_target"] for p in got["operating_points"]] == list(FA_TARGETS)
    for point in got["operating_points"]:
        threshold = evaluate.threshold_for_fa(ds, score, point["fa_target"])
        expected = {"fa_target": point["fa_target"], "threshold": threshold, **evaluate.summarize(ds, score, threshold)}
        assert point == pytest.approx(expected, nan_ok=True)


# --- choose_theta1 ----------------------------------------------------------------

def b2_dataset(bump_deg, bump_rate, bump_times):
    """dataset() with B2's signals: spins it can catch, and bumps no spin follows.

    Before each onset, |slip| is 52° and rising at 50 to 100°/s. Each bump is
    0.3 s at bump_deg and bump_rate. A held 85° drift with |slip| falling at
    50°/s at 115 s gives every θ1 a frame to score.
    """
    ds = dataset()
    t = ds["t"].to_numpy()
    for onset in (30.0, 60.0, 90.0):
        before = during(t, onset - HORIZON_S, onset - STEP)
        ds.loc[before, "abs_slip"] = 52.0
        ds.loc[before, "abs_slip_rate"] = 100 - (onset - t[before]) * 100
    for at in bump_times:
        ds.loc[during(t, at, at + 0.3), ["abs_slip", "abs_slip_rate"]] = bump_deg, bump_rate
    ds.loc[during(t, 115.0, 115.3), ["abs_slip", "abs_slip_rate"]] = 85.0, -50.0
    return ds


# θ1 swept from 30°, over b2_dataset's 20° drifts, whatever config.py sweeps.
THETA1 = tuple(range(30, 81, 5))


@pytest.fixture
def theta1_from_30(monkeypatch):
    monkeypatch.setattr(cv, "B2_THETA1_DEG", THETA1)


def test_b2_keeps_the_theta1_with_the_best_recall_at_1_fa_per_min(theta1_from_30):
    # Bumps at 47° and 200°/s, 4 a minute: under θ1 = 50 they alarm before
    # any spin does, so no threshold within 1 FA/min catches one; over 52°,
    # neither alarms.
    theta1, table = cv.choose_theta1(b2_dataset(47.0, 200.0, (10, 20, 40, 50, 70, 80, 100, 110)))
    assert theta1 == 50
    assert table["theta1"].tolist() == list(THETA1)
    by_theta1 = table.set_index("theta1")
    for recall in ["recall", "recall_min_lead"]:
        assert by_theta1[recall].to_dict() == {th: 1.0 if th == 50 else 0.0 for th in THETA1}


def test_b2_keeps_the_theta1_that_warns_min_lead_ahead_over_one_with_a_better_pr_auc(theta1_from_30):
    # Before each onset |slip| climbs from 36° at HORIZON_S (= MIN_LEAD_S)
    # ahead to 53° on the last frame, so only θ1 under 36° alarms that far
    # ahead. From 40° to 50° every spin is still caught, but later. A bump at
    # 38° and 200°/s per session ranks over every target frame at 30° and
    # 35°, which costs them PR-AUC: picked by recall and PR-AUC, θ1 would be 40.
    ds = b2_dataset(38.0, 200.0, (45,))
    t = ds["t"].to_numpy()
    for onset in (30.0, 60.0, 90.0):
        before = during(t, onset - HORIZON_S, onset - STEP)
        ds.loc[before, "abs_slip"] = 36 + 36 * (t[before] - (onset - HORIZON_S))
    theta1, table = cv.choose_theta1(ds)
    by_theta1 = table.set_index("theta1")

    assert (by_theta1.loc[30:50, "recall"] == 1.0).all()
    assert by_theta1["recall_min_lead"].to_dict() == {th: 1.0 if th <= 35 else 0.0 for th in THETA1}
    assert (by_theta1.loc[30:35, "pr_auc"] < by_theta1.loc[40, "pr_auc"]).all()
    assert theta1 == 30


def test_b2_breaks_a_recall_tie_by_pr_auc_then_by_the_lower_theta1(theta1_from_30):
    # One bump per session at 40° and 75°/s: under θ1 = 40 it alarms, 5 false
    # alarms in 9 min, so both recalls are 1 from 30° to 50°. But it ranks
    # among the target frames, which costs PR-AUC at 30° and 35° only.
    theta1, table = cv.choose_theta1(b2_dataset(40.0, 75.0, (45,)))
    by_theta1 = table.set_index("theta1")
    assert (by_theta1.loc[30:50, ["recall", "recall_min_lead"]] == 1.0).all().all()
    assert (by_theta1.loc[30:35, "pr_auc"] < by_theta1.loc[40, "pr_auc"]).all()
    assert theta1 == 40
