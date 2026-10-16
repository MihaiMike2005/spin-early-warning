"""
Tests for spinwarn.evaluate: alerts from scores, and alerts scored against spin onsets.

Hand-made alerts and onsets on the GRID_HZ grid, and small dataset-like
sessions with the columns evaluate reads, so they run anywhere without data/raw/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from spinwarn import evaluate
from spinwarn.config import GRID_HZ, MERGE_GAP_S, MIN_LEAD_S


STEP = 1 / GRID_HZ


def grid(start, end):
    """The grid times in [start, end)."""
    return np.arange(round(start * GRID_HZ), round(end * GRID_HZ)) / GRID_HZ


def during(t, start, end):
    """Grid times in [start, end], with a margin well under a grid step."""
    return (t > start - STEP / 2) & (t < end + STEP / 2)


def session(seconds, onsets=(), excluded=(), name="a", segment=0, start=0.0):
    """A dataset-like grid session of `seconds`: excluded in each [a, b] of `excluded`, scored elsewhere."""
    t = grid(start, start + seconds)
    df = pd.DataFrame({"session": name, "segment": segment, "t": t})
    df["excluded"] = np.zeros(len(t), dtype=bool)
    for a, b in excluded:
        df["excluded"] |= during(t, a, b)
    df["onset"] = np.isin(np.rint(t * GRID_HZ), np.rint(np.asarray(onsets, dtype=float) * GRID_HZ))
    return df


def spikes(df, *spans, height=1.0, score=None):
    """score (0 everywhere if None) set to `height` on the frames in each [a, b] span."""
    score = np.zeros(len(df)) if score is None else score.copy()
    for a, b in spans:
        score[during(df["t"].to_numpy(), a, b)] = height
    return score


def alerts_of(df, score, threshold=0.5):
    return np.array(evaluate.find_alerts(df["t"], score, threshold, df["segment"], df["excluded"]))


# --- match --------------------------------------------------------------------

def test_an_alert_from_1_3_s_before_onset_through_it_is_one_catch_with_1_s_lead():
    caught, lead, false_alarm = evaluate.match([(3.7, 5.5)], [5.0])
    assert caught.tolist() == [True] and lead == pytest.approx([1.0])
    assert false_alarm.tolist() == [False]


def test_an_alert_starting_0_05_s_before_onset_is_too_late_but_not_false():
    caught, lead, false_alarm = evaluate.match([(4.95, 5.5)], [5.0])
    assert caught.tolist() == [False] and np.isnan(lead).all()
    assert false_alarm.tolist() == [False]


@pytest.mark.parametrize("onsets", [[], [3.6]])
def test_an_alert_with_no_onset_up_to_1_s_after_it_is_false(onsets):
    # 3.6 s is 1.1 s after the alert ends, and its credit window starts 0.1 s after.
    caught, _lead, false_alarm = evaluate.match([(2.0, 2.5)], onsets)
    assert false_alarm.tolist() == [True] and not caught.any()


@pytest.mark.parametrize("start, end, caught, lead, false_alarm", [
    (3.0, 4.0, True, 1.0, False),                  # ends exactly 1.0 s before onset
    (3.0, 4.0 - STEP, False, None, True),          # one frame earlier: off before the window
    (4.9, 5.2, True, 0.1, False),                  # starts exactly 0.1 s before onset
    (4.9 + STEP, 5.2, False, None, False),         # one frame later: too late
    (4.4, 4.6, True, 0.6, False),                  # lead under the cap
])
def test_the_credit_window_edges(start, end, caught, lead, false_alarm):
    got_caught, got_lead, got_false = evaluate.match([(start, end)], [5.0])
    assert got_caught.tolist() == [caught] and got_false.tolist() == [false_alarm]
    assert got_lead[0] == pytest.approx(lead) if caught else np.isnan(got_lead[0])


def test_lead_is_from_the_earliest_alert_that_catches_and_neither_alert_is_false():
    caught, lead, false_alarm = evaluate.match([(4.2, 4.3), (4.6, 4.95)], [5.0])
    assert caught.tolist() == [True] and lead == pytest.approx([0.8])
    assert false_alarm.tolist() == [False, False]


# --- find_alerts ----------------------------------------------------------------

@pytest.mark.parametrize("gap, merged", [
    (STEP, True), (0.15, True), (MERGE_GAP_S - STEP, True), (MERGE_GAP_S, False), (0.3, False),
])
def test_alarms_less_than_merge_gap_apart_merge(gap, merged):
    # Alarms on [2.0, 2.5] and [2.5 + gap, 3.0]: gap is the second's start − the first's end.
    df = session(10.0)
    alerts = alerts_of(df, spikes(df, (2.0, 2.5), (2.5 + gap, 3.0)))
    expected = [(2.0, 3.0)] if merged else [(2.0, 2.5), (2.5 + gap, 3.0)]
    assert alerts == pytest.approx(np.array(expected))


def test_alerts_only_exist_on_scored_frames():
    # The first 1 s and [2.0, 2.4] are excluded. With threshold −∞ every
    # scored frame alarms, and an alarm across [2.0, 2.4] can't start in it.
    df = session(10.0, excluded=[(0.0, 1.0 - STEP), (2.0, 2.4)])
    assert alerts_of(df, spikes(df, (1.5, 1.6)), threshold=-np.inf) == pytest.approx(
        np.array([(1.0, 2.0 - STEP), (2.4 + STEP, 10.0 - STEP)]))
    assert alerts_of(df, spikes(df, (2.0, 3.0))) == pytest.approx(np.array([(2.4 + STEP, 3.0)]))


def test_alerts_never_cross_segments():
    # 0.1 s apart, under MERGE_GAP_S, but either side of a gap.
    df = pd.concat([session(6.0, segment=0), session(6.0, segment=1, start=6.1)], ignore_index=True)
    alerts = alerts_of(df, spikes(df, (5.0, 6.0 - STEP), (6.1, 7.0)))
    assert alerts == pytest.approx(np.array([(5.0, 6.0 - STEP), (6.1, 7.0)]))


def test_no_alarm_no_alerts():
    df = session(5.0)
    assert evaluate.find_alerts(df["t"], np.zeros(len(df)), 0.5, df["segment"], df["excluded"]) == []


# --- summarize ------------------------------------------------------------------

def test_an_alert_on_through_a_spin_is_one_catch_with_1_s_lead():
    # The score stays high through the spin, but frames from onset to re-arm
    # are excluded, so the alert ends on the frame before the onset.
    df = session(10.0, onsets=[5.0], excluded=[(0.0, 1.0 - STEP), (5.0, 7.0)])
    summary = evaluate.summarize(df, spikes(df, (3.7, 6.0)), 0.5)
    assert summary["n_spins"] == 1 and summary["caught"] == 1 and summary["recall"] == 1.0
    assert summary["alerts"] == 1 and summary["false_alarms"] == 0 and summary["alert_precision"] == 1.0
    assert summary["median_lead"] == pytest.approx(1.0)


def test_time_in_alarm_is_the_scored_frames_from_each_alerts_first_frame_to_its_last():
    # 12 s, the first 1 s excluded (60 frames) and one frame at 2.55 s.
    # Alarms on [2.0, 2.5] and [2.6, 3.0] merge into one alert, 61 frames,
    # whose below-threshold gap counts as in alarm and whose excluded frame
    # (alarming or not) doesn't. A second alert on [6.0, 6.5] is 31 frames.
    df = session(12.0, excluded=[(0.0, 1.0 - STEP), (2.55, 2.55)])
    score = spikes(df, (2.0, 2.5), (2.6, 3.0), (6.0, 6.5))
    score = spikes(df, (2.55, 2.55), height=9.0, score=score)
    summary = evaluate.summarize(df, score, 0.5)

    scored = 720 - 60 - 1
    assert summary["alerts"] == 2
    assert summary["time_in_alarm"] == pytest.approx((61 - 1 + 31) / scored)
    assert summary["minutes"] == pytest.approx(scored / GRID_HZ / 60)


def test_sessions_pool_by_summing_counts_and_minutes_and_never_match_across_sessions():
    # a's alert catches a's spin at 20 s. b spins at 20 s too, with no alert
    # of its own, and has a false alert at 10 s.
    a = session(60.0, onsets=[20.0], excluded=[(0.0, 1.0 - STEP)], name="a")
    b = session(60.0, onsets=[20.0], excluded=[(0.0, 1.0 - STEP)], name="b")
    df = pd.concat([a, b], ignore_index=True)
    score = np.r_[spikes(a, (19.2, 19.8)), spikes(b, (10.0, 10.5))]
    summary = evaluate.summarize(df, score, 0.5)

    minutes = 2 * (3600 - 60) / GRID_HZ / 60
    assert summary == pytest.approx({
        "n_spins": 2, "caught": 1, "recall": 0.5, "caught_min_lead": 1, "recall_min_lead": 0.5,
        "alerts": 2, "false_alarms": 1, "minutes": minutes, "fa_per_min": 1 / minutes, "alert_precision": 0.5,
        "time_in_alarm": (37 + 31) / (2 * (3600 - 60)), "median_lead": 0.8,
    })


def test_the_headline_counts_a_catch_only_with_at_least_min_lead():
    # Three spins, all caught: by alerts starting exactly MIN_LEAD_S before
    # onset, one frame later, and 0.1 s before.
    df = session(30.0, onsets=[5.0, 15.0, 25.0], excluded=[(0.0, 1.0 - STEP)])
    score = spikes(df, (5.0 - MIN_LEAD_S, 4.9), (15.0 - MIN_LEAD_S + STEP, 14.9), (24.9, 24.95))
    summary = evaluate.summarize(df, score, 0.5)
    assert summary["caught"] == 3 and summary["recall"] == 1.0
    assert summary["caught_min_lead"] == 1 and summary["recall_min_lead"] == pytest.approx(1 / 3)


def test_an_onset_in_the_next_segment_neither_catches_nor_excuses_an_alert():
    # Matched across the gap, [5.3, 5.8] would catch the onset at 6.3 s.
    df = pd.concat([session(6.0, segment=0), session(6.0, onsets=[6.3], segment=1, start=6.2)],
                   ignore_index=True)
    summary = evaluate.summarize(df, spikes(df, (5.3, 5.8)), 0.5)
    assert summary["caught"] == 0 and summary["false_alarms"] == 1


def test_no_alerts_and_no_spins_leave_their_ratios_undefined():
    df = session(10.0)
    summary = evaluate.summarize(df, np.zeros(len(df)), 0.5)
    assert summary["alerts"] == 0 and summary["fa_per_min"] == 0.0 and summary["time_in_alarm"] == 0.0
    for name in ["recall", "recall_min_lead", "alert_precision", "median_lead"]:
        assert np.isnan(summary[name])


def test_a_nan_score_is_refused_on_a_scored_frame_only():
    df = session(10.0, excluded=[(0.0, 1.0 - STEP)])
    score = np.zeros(len(df))
    score[0] = np.nan
    evaluate.summarize(df, score, 0.5)
    score[100] = np.nan
    with pytest.raises(ValueError, match="NaN score"):
        evaluate.summarize(df, score, 0.5)
    with pytest.raises(ValueError, match="scores for"):
        evaluate.summarize(df, np.zeros(len(df) - 1), 0.5)


# --- threshold_for_fa and recall_fa_curve ---------------------------------------

def toy():
    """2 scored minutes: false alarms of height 1 to 6, and a spin caught by an alarm of height 10."""
    df = session(121.0, onsets=[100.0], excluded=[(0.0, 1.0 - STEP)])
    score = np.zeros(len(df))
    for height, at in enumerate([10.0, 25.0, 40.0, 55.0, 70.0, 85.0], start=1):
        score = spikes(df, (at, at + 0.1), height=height, score=score)
    return df, spikes(df, (99.2, 99.5), height=10.0, score=score)


@pytest.mark.parametrize("target, threshold", [(1.0, 4.0), (0.5, 5.0)])
def test_threshold_for_fa_is_the_lowest_threshold_within_the_target(target, threshold):
    # Over 4, two false alarms in 2 min; over 3, three.
    df, score = toy()
    assert evaluate.threshold_for_fa(df, score, target) == threshold
    summary = evaluate.summarize(df, score, threshold)
    assert summary["fa_per_min"] == target and summary["recall"] == 1.0


def test_threshold_for_fa_never_leaves_the_alarm_on():
    # The toy on a floor of 0.1, with its first scored frame at 0. Over 0,
    # every other scored frame alarms: one alert through the spin, so no
    # false alarm, within any target. Over 0.1, the six false alarms: the
    # most of any threshold, so the warning range starts there.
    df, score = toy()
    score = np.where(score > 0, score, 0.1)
    score[60] = 0.0
    curve = evaluate.recall_fa_curve(df, score)
    always_on = curve.set_index("threshold").loc[0.0]
    assert always_on["false_alarms"] == 0 and always_on["time_in_alarm"] > 0.99
    assert evaluate.warning_range(curve)["threshold"].tolist() == [0.1, 1, 2, 3, 4, 5, 6, 10]
    assert evaluate.threshold_for_fa(df, score, 1.0) == 4.0
    assert evaluate.threshold_for_fa(df, score, 10.0) == 0.1


def test_warning_range_takes_a_curve_as_metrics_json_keeps_it():
    # Thresholds out of order, the most false alarms tied at 1 and 3.
    curve = {"threshold": [3, 0, 5, 1, 4], "fa_per_min": [6.0, 1.1, 0.0, 6.0, 0.8]}
    assert evaluate.warning_range(curve)["threshold"].tolist() == [1, 3, 4, 5]


def test_recall_fa_curve_summarizes_every_candidate_threshold():
    df, score = toy()
    curve = evaluate.recall_fa_curve(df, score)

    assert curve["threshold"].tolist() == [0, 1, 2, 3, 4, 5, 6, 10]
    assert curve["false_alarms"].tolist() == [6, 5, 4, 3, 2, 1, 0, 0]
    # The highest score alarms nowhere.
    assert curve["alerts"].iloc[-1] == 0 and curve["recall"].iloc[-1] == 0.0
    row = curve.set_index("threshold").loc[4].to_dict()
    assert row == pytest.approx(evaluate.summarize(df, score, 4.0), nan_ok=True)


# --- pr_auc ---------------------------------------------------------------------

def test_pr_auc_is_average_precision_on_scored_frames_only():
    df = session(10.0, excluded=[(0.0, 1.0 - STEP)])
    rng = np.random.default_rng(0)
    df["target"] = rng.random(len(df)) < 0.1
    score = rng.random(len(df)) + df["target"].to_numpy()
    # Excluded frames would score perfectly, if they counted.
    score[df["excluded"]] = np.where(df.loc[df["excluded"], "target"], 9.0, -9.0)
    scored = ~df["excluded"]

    expected = average_precision_score(df.loc[scored, "target"], score[scored])
    assert evaluate.pr_auc(df, score) == pytest.approx(expected)


def test_pr_auc_takes_minus_infinity_as_the_lowest_score():
    df = session(10.0)
    rng = np.random.default_rng(1)
    df["target"] = rng.random(len(df)) < 0.1
    score = rng.random(len(df)) + df["target"].to_numpy()
    score[::3] = -np.inf
    expected = average_precision_score(df["target"], np.where(np.isinf(score), -1.0, score))
    assert evaluate.pr_auc(df, score) == pytest.approx(expected)


# --- spin_catches ---------------------------------------------------------------

def test_spin_catches_are_summarizes_counts_spin_by_spin():
    # In a: caught with 1.0 s of lead, then with 0.3 s; b's spin is missed.
    a = session(20.0, onsets=[5.0, 15.0], name="a")
    b = session(10.0, onsets=[5.0], name="b")
    df = pd.concat([a, b], ignore_index=True)
    score = np.r_[spikes(a, (3.7, 4.9), (14.7, 14.9)), spikes(b)]
    spins = evaluate.spin_catches(df, score, 0.5)

    assert list(spins.columns) == evaluate.SPIN_COLUMNS
    assert spins["session"].tolist() == ["a", "a", "b"] and spins["t"].tolist() == pytest.approx([5.0, 15.0, 5.0])
    assert spins["caught"].tolist() == [True, True, False]
    assert spins["caught_min_lead"].tolist() == [True, False, False]
    assert spins["lead"].iloc[:2].tolist() == pytest.approx([1.0, 0.3]) and np.isnan(spins["lead"].iloc[2])
    summary = evaluate.summarize(df, score, 0.5)
    assert (summary["n_spins"], summary["caught"], summary["caught_min_lead"]) == (3, 2, 1)


def test_no_spins_no_spin_catches():
    spins = evaluate.spin_catches(session(10.0), np.ones(round(10.0 * GRID_HZ)), 0.5)
    assert spins.empty and list(spins.columns) == evaluate.SPIN_COLUMNS


# --- bootstrap_recall -----------------------------------------------------------

def test_bootstrap_recall_is_the_same_on_every_run_with_the_same_seed():
    caught = [True] * 14 + [False] * 6
    lo, hi = evaluate.bootstrap_recall(caught)
    assert evaluate.bootstrap_recall(caught) == (lo, hi)
    assert lo < 0.7 < hi


@pytest.mark.parametrize("caught, interval", [([True] * 20, (1.0, 1.0)), ([False] * 20, (0.0, 0.0))])
def test_bootstrap_recall_of_all_or_nothing_has_no_width(caught, interval):
    assert evaluate.bootstrap_recall(caught) == interval
