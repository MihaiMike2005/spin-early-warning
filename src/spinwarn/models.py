"""
The scikit-learn models: logistic regression, random forest, gradient boosting.

make_model builds one, unfitted, with the Project Plan's class weights and
SEED, and any params on top; cv.oof_scores fits a fresh one per fold.
SEARCH is each model's tuning grid. run.py tune tries configs sampled from
it with SEED (configs), each one CV run, and keeps the best by PR-AUC.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from spinwarn.config import SEED
from spinwarn.features import FEATURES


# The Build Plan's grids, by the classifier's parameter names.
SEARCH = {
    "lr": {"C": [0.01, 0.1, 1, 10]},
    "rf": {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 5, 20],
           "max_features": ["sqrt", 0.5]},
    "hgb": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50, 200],
            "l2_regularization": [0, 1]},
}
MODELS = list(SEARCH)

# rear_slip spikes like logcheck's front wheels: median about 4, but up to
# 2.7e8 (Step 9). Trees only split on its order, but a spike would swamp
# StandardScaler's mean and variance, and a spike frame would score at an
# extreme, so logistic regression reads these through a signed log.
SPIKY = ["rear_slip_mean", "rear_slip_slope"]


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_log(X):
    """X with its SPIKY columns as sign × log1p(|x|), the rest as they are; X itself is left alone.

    X is a frame of FEATURES, or an array of rows of them in FEATURES order,
    as a live loop (Step 16) or a single-row timing would pass.
    """
    if isinstance(X, pd.DataFrame):
        return X.assign(**{c: _signed_log(X[c]) for c in SPIKY})
    X = np.array(X, dtype=float)
    spiky = [FEATURES.index(c) for c in SPIKY]
    X[:, spiky] = _signed_log(X[:, spiky])
    return X


def check_params(name, params):
    """ValueError unless name is one of MODELS and params are some of the ones SEARCH tunes, each a JSON scalar.

    The values are scikit-learn's to check: it refuses a bad one at the
    first fit, before run.py writes anything.
    """
    if name not in SEARCH:
        raise ValueError(f"no model {name!r}: {', '.join(MODELS)}")
    unknown = sorted(set(params) - set(SEARCH[name]))
    if unknown:
        raise ValueError(f"{name} takes params {sorted(SEARCH[name])}, not {unknown}")
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str, type(None))):
            raise ValueError(f"{name}'s {key} is a number, a string or null, not {value!r}")


def make_model(name, params=None):
    """An unfitted model, name one of MODELS, with params set on its classifier."""
    params = params or {}
    check_params(name, params)
    if name == "lr":
        return make_pipeline(
            FunctionTransformer(signed_log, feature_names_out="one-to-one"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, **params),
        )
    if name == "rf":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **params)
    # No early stopping: scikit-learn's default turns it on above 10,000
    # rows, as every fold has, and holds out 10% of the training frames to
    # judge it. On Step 13's folds it never stopped before max_iter, so it
    # only cost those frames.
    return HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=SEED,
                                          **params)


def configs(name, n) -> list:
    """n configs from name's SEARCH grid, sampled without replacement with SEED; the whole grid if it has n or fewer."""
    grid = SEARCH[name]
    return list(ParameterSampler(grid, min(n, len(ParameterGrid(grid))), random_state=SEED))
