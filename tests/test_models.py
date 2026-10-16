"""
Tests for spinwarn.models: the three models, their params and the seeded search.

Small random frames with the FEATURES columns, so they run anywhere without data/raw/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from spinwarn import models
from spinwarn.config import SEED
from spinwarn.features import FEATURES


def frames(n=600, seed=0):
    """(X, y): n rows of random FEATURES, positive more often where abs_slip_last is high."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = X["abs_slip_last"] + rng.normal(size=n) > 1.5
    return X, y


# --- make_model -------------------------------------------------------------------

def test_lr_is_a_signed_log_then_scaled_balanced_logistic_regression():
    model = models.make_model("lr", {})
    log, scaler, classifier = (step for _name, step in model.steps)
    assert log.func is models.signed_log and isinstance(scaler, StandardScaler)
    assert isinstance(classifier, LogisticRegression)
    assert (classifier.class_weight, classifier.max_iter) == ("balanced", 1000)


def test_rf_and_hgb_are_balanced_and_seeded():
    rf = models.make_model("rf")
    assert isinstance(rf, RandomForestClassifier)
    assert (rf.class_weight, rf.n_jobs, rf.random_state) == ("balanced_subsample", -1, SEED)
    hgb = models.make_model("hgb")
    assert isinstance(hgb, HistGradientBoostingClassifier)
    assert (hgb.class_weight, hgb.random_state) == ("balanced", SEED)


def test_hgb_fits_on_every_frame_it_is_given_even_over_10000():
    # scikit-learn's default early stopping would hold out 10% of them here.
    X, y = frames(n=12000)
    hgb = models.make_model("hgb").fit(X, y)
    assert not hgb.do_early_stopping_ and hgb.n_iter_ == hgb.max_iter


@pytest.mark.parametrize("name, params", [
    ("lr", {"C": 0.1}),
    ("rf", {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5, "max_features": 0.5}),
    ("hgb", {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 200, "l2_regularization": 1}),
])
def test_params_are_set_on_the_classifier(name, params):
    model = models.make_model(name, params)
    classifier = model[-1] if name == "lr" else model
    assert {k: classifier.get_params()[k] for k in params} == params


@pytest.mark.parametrize("name, params, message", [
    ("lr", {"c": 1}, r"lr takes params \['C'\], not \['c'\]"),
    ("rf", {"max_dept": 5, "n_estimators": 200}, r"not \['max_dept'\]"),
    ("hgb", {"class_weight": None}, r"not \['class_weight'\]"),
    ("lr", {"C": True}, "lr's C is a number, a string or null, not True"),
    ("rf", {"max_depth": [10]}, r"rf's max_depth is a number, a string or null, not \[10\]"),
    ("xgb", {}, "no model 'xgb': lr, rf, hgb"),
])
def test_params_are_some_of_the_ones_the_model_is_tuned_on_each_a_json_scalar(name, params, message):
    with pytest.raises(ValueError, match=message):
        models.make_model(name, params)


@pytest.mark.parametrize("name, params", [
    ("lr", {"C": 0.1}), ("rf", {"n_estimators": 20, "max_depth": 10}), ("hgb", {"max_leaf_nodes": 15}),
])
def test_a_model_scores_the_same_every_fit(name, params):
    X, y = frames()
    first = models.make_model(name, params).fit(X, y).predict_proba(X)[:, 1]
    again = models.make_model(name, params).fit(X, y).predict_proba(X)[:, 1]
    # The same trees every fit, but rf's threads add their votes up in any
    # order, so the last bit of a probability can differ.
    np.testing.assert_allclose(first, again, rtol=0, atol=1e-12)
    assert ((first >= 0) & (first <= 1)).all()


@pytest.mark.parametrize("name, params", [("lr", {}), ("rf", {"n_estimators": 20}), ("hgb", {})])
def test_a_model_fitted_on_a_frame_scores_an_array_of_features_the_same(name, params):
    # As Step 14's single-row timing or Step 16's live loop might pass them.
    X, y = frames()
    model = models.make_model(name, params).fit(X, y)
    with pytest.warns(UserWarning, match="valid feature names"):   # scikit-learn's, for any array
        from_array = model.predict_proba(X.to_numpy())[:, 1]
    np.testing.assert_allclose(from_array, model.predict_proba(X)[:, 1], rtol=0, atol=1e-12)


# --- the signed log -------------------------------------------------------------

def test_the_signed_log_takes_only_the_spiky_columns_and_keeps_their_sign():
    X, _y = frames(n=4)
    X["rear_slip_mean"] = [2.7e8, 4.0, 0.0, 1.0]
    X["rear_slip_slope"] = [-2.7e8, -1.0, 0.0, 3.0]
    before = X.copy()
    logged = models.signed_log(X)

    pd.testing.assert_frame_equal(X, before)   # X itself untouched
    assert logged["rear_slip_mean"].tolist() == pytest.approx([np.log1p(2.7e8), np.log1p(4), 0, np.log1p(1)])
    assert logged["rear_slip_slope"].tolist() == pytest.approx([-np.log1p(2.7e8), -np.log1p(1), 0, np.log1p(3)])
    others = [f for f in FEATURES if f not in models.SPIKY]
    pd.testing.assert_frame_equal(logged[others], X[others])
    assert list(logged.columns) == FEATURES

    # An array of the same rows, in FEATURES order, by position.
    array = X.to_numpy()
    np.testing.assert_array_equal(models.signed_log(array), logged.to_numpy())
    np.testing.assert_array_equal(array, before.to_numpy())   # the array untouched too


def test_rear_slip_spikes_neither_squash_lrs_scaling_nor_reach_an_extreme():
    # rear_slip_mean about 4, with 4% of frames spiking to 1e3-4.5e5, as on
    # the dev set's scored frames. Without the log, the other frames scale to
    # a spread of about 1e-4, and Step 9's 2.7e8 to about 12,000 SDs.
    X, y = frames(n=2000)
    rng = np.random.default_rng(1)
    X["rear_slip_mean"] = np.abs(X["rear_slip_mean"]) * 3 + 1
    spikes = rng.random(len(X)) < 0.04
    X.loc[spikes, "rear_slip_mean"] = 10 ** rng.uniform(3, np.log10(4.5e5), spikes.sum())
    scale = models.make_model("lr").fit(X, y)[:-1]   # the steps before the classifier
    column = FEATURES.index("rear_slip_mean")

    assert scale.transform(X)[~spikes, column].std() > 0.1
    assert abs(scale.transform(X.iloc[:1].assign(rear_slip_mean=2.7e8))[0, column]) < 20


# --- configs --------------------------------------------------------------------

def test_every_search_param_is_one_its_model_takes():
    for name, grid in models.SEARCH.items():
        config = {key: values[-1] for key, values in grid.items()}
        models.make_model(name, config)


def test_configs_are_distinct_from_the_grid_and_the_same_every_time():
    picked = models.configs("hgb", 15)
    assert len(picked) == 15 and picked == models.configs("hgb", 15)
    grid = list(ParameterGrid(models.SEARCH["hgb"]))
    assert all(config in grid for config in picked)
    assert len({tuple(sorted(c.items())) for c in picked}) == 15


def test_configs_are_the_whole_grid_when_it_has_n_or_fewer():
    picked = models.configs("lr", 15)
    assert sorted(c["C"] for c in picked) == [0.01, 0.1, 1, 10]
    assert len(models.configs("rf", 100)) == len(ParameterGrid(models.SEARCH["rf"])) == 36
