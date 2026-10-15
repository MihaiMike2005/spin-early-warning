"""
Session-grouped cross-validation: out-of-fold scores for every method, baselines included.

oof_scores scores every dev frame with a method that never saw its session:
GroupKFold over N_FOLDS, grouped by session. A model fits on the other
folds' non-excluded frames, thinned to TRAIN_HZ since neighbouring grid
frames are near duplicates, and scores every frame of the held-out fold. A
baseline goes through the same folds with nothing to fit, so every method
is scored the same way.

metrics turns the scores into what run.py cv prints and logs: PR-AUC, and
each FA_TARGETS operating point, with its threshold set on the same scores.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from spinwarn import baselines, models
from spinwarn.baselines import BASELINES
from spinwarn.config import B2_THETA1_DEG, FA_TARGETS, GRID_HZ, N_FOLDS, TRAIN_HZ
from spinwarn.evaluate import pr_auc, summarize, threshold_for_fa
from spinwarn.features import FEATURES


METHODS = list(BASELINES)   # Step 13 adds the models from models.make_model: lr, rf, hgb


def folds(dataset) -> list:
    """[(train, test)]: row positions for each of N_FOLDS folds, every session in one test fold.

    GroupKFold without shuffling, so the same folds every run: it deals the
    sessions out largest first, each to the fold with the fewest frames.
    """
    return list(GroupKFold(N_FOLDS).split(dataset, groups=dataset["session"]))


def fold_ids(dataset) -> np.ndarray:
    """Per row, the fold that scores it."""
    fold = np.empty(len(dataset), dtype=int)
    for k, (_train, test) in enumerate(folds(dataset)):
        fold[test] = k
    return fold


def features_of(method) -> list:
    """The dataset columns a method reads: a baseline's signals, or every feature."""
    return baselines.SIGNALS[method] if method in BASELINES else FEATURES


def training_rows(dataset, rows) -> np.ndarray:
    """The row positions in rows a model fits on: not excluded, and thinned to TRAIN_HZ.

    Thinned to every (GRID_HZ / TRAIN_HZ)th grid frame counted from t = 0,
    so the same frames are kept whatever the segment or fold.
    """
    rows = np.asarray(rows)
    frame = np.rint(dataset["t"].to_numpy()[rows] * GRID_HZ).astype(np.int64)
    keep = frame % round(GRID_HZ / TRAIN_HZ) == 0
    return rows[keep & ~dataset["excluded"].to_numpy(dtype=bool)[rows]]


def fit_predict(model, dataset, train, test) -> np.ndarray:
    """model fitted on train's training_rows, then its spin probability on every row of test.

    A row with a feature unknown gets NaN, since not every model takes one.
    That's only a segment's first WINDOW_S, which label() always excludes;
    on a scored frame evaluate refuses the NaN.
    """
    fit_on = dataset.iloc[training_rows(dataset, train)]
    model.fit(fit_on[FEATURES], fit_on["target"])
    features = dataset[FEATURES].iloc[test]
    known = features.notna().all(axis=1).to_numpy()
    scores = np.full(len(features), np.nan)
    if known.any():
        scores[known] = model.predict_proba(features[known])[:, 1]
    return scores


def oof_scores(dataset, method, params=None) -> np.ndarray:
    """method's out-of-fold score for every row of the dataset, in its order.

    Each fold is scored by the method fitted on the other folds alone: a
    model from models.make_model, fitted by fit_predict. A baseline has
    nothing to fit, so it's just scored, fold by fold all the same.
    """
    if method not in METHODS:
        raise ValueError(f"no method {method!r}: {', '.join(METHODS)}")
    params = params or {}
    scores = np.full(len(dataset), np.nan)
    for train, test in folds(dataset):
        if method in BASELINES:
            scores[test] = baselines.score(method, dataset.iloc[test], params)
        else:
            scores[test] = fit_predict(models.make_model(method, params), dataset, train, test)
    return scores


def metrics(dataset, scores) -> dict:
    """PR-AUC, and each FA_TARGETS operating point: its threshold on these scores, and summarize() there.

    {"pr_auc": ..., "operating_points": [{"fa_target", "threshold", **summarize}, ...]}, in
    FA_TARGETS order, so the first is 1 FA/min.
    """
    points = []
    for target in FA_TARGETS:
        threshold = threshold_for_fa(dataset, scores, target)
        points.append({"fa_target": target, "threshold": threshold, **summarize(dataset, scores, threshold)})
    return {"pr_auc": pr_auc(dataset, scores), "operating_points": points}


def choose_theta1(dataset) -> tuple:
    """(theta1, table): B2's θ1 from B2_THETA1_DEG with the best CV headline recall at 1 FA/min.

    The headline is recall with at least MIN_LEAD_S of lead, the one the
    models are judged by. Ties, likely with a few dozen spins, go to the
    higher recall by the Project Plan's rule, then the higher PR-AUC, then
    the lower θ1. table has a row per θ1: the threshold at 1 FA/min,
    summarize() there, and PR-AUC.
    """
    rows = []
    for theta1 in B2_THETA1_DEG:
        scores = oof_scores(dataset, "b2", {"theta1": theta1})
        threshold = threshold_for_fa(dataset, scores, FA_TARGETS[0])
        rows.append({"theta1": theta1, "threshold": threshold, **summarize(dataset, scores, threshold),
                     "pr_auc": pr_auc(dataset, scores)})
    table = pd.DataFrame(rows)
    # A stable sort keeps θ1's order among ties.
    best = table.sort_values(["recall_min_lead", "recall", "pr_auc"], ascending=False, kind="stable").index[0]
    return B2_THETA1_DEG[best], table
