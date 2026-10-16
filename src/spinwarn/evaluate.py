"""
Scores to alerts, and alerts scored: spins caught, false alarms per minute, lead time.

Everything here takes rows of the dataset from run.build (session, segment,
t, excluded, onset, and target for pr_auc), in time order within each
session, with one score per row from any method. Frames scoring above a
threshold are alarms, and alarms less than MERGE_GAP_S apart are one
alert. The Project Plan's rules then score the alerts against the onsets:

- a spin is caught if an alert is active anywhere from CREDIT_EARLY_S to
  CREDIT_LATE_S before its onset; lead time is onset − alert start, capped
  at CREDIT_EARLY_S;
- an alert is a false alarm if no onset falls in (start, end + FA_AFTER_S];
- so an alert starting under CREDIT_LATE_S before an onset is too late:
  neither caught nor false.

The headline recall is stricter: it counts a catch only with at least
MIN_LEAD_S of lead. By the rule above alone, a threshold on |slip| just
under 90° catches nearly every spin 0.1 s ahead, so every method ties.

Alerts and onsets are matched within a segment, as targets are labelled:
what happens across a gap is unknown, and no feature reaches across one.
Grid times are k / GRID_HZ, so they're compared as whole frames, and no
edge hangs on a float landing an ulp either side of it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from spinwarn.config import (
    CREDIT_EARLY_S, CREDIT_LATE_S, FA_AFTER_S, GRID_HZ, MERGE_GAP_S, MIN_LEAD_S, N_THRESHOLDS, SEED,
)


def _frames(times) -> np.ndarray:
    """Grid times as whole frames since t = 0."""
    return np.rint(np.asarray(times, dtype=float) * GRID_HZ).astype(np.int64)


def _runs(frame, alarm) -> tuple:
    """(starts, ends): row positions of each alert's first and last frame, in one segment."""
    rows = np.flatnonzero(alarm)
    if not len(rows):
        return rows, rows
    new = np.diff(frame[rows]) >= round(MERGE_GAP_S * GRID_HZ)
    return rows[np.r_[True, new]], rows[np.r_[new, True]]


def find_alerts(t, score, threshold, segment, excluded) -> list:
    """[(start, end)] of each alert in one session: the grid times of its first and last frame.

    Excluded frames get score −∞ first, so alerts only exist on scored
    frames, and none starts in an excluded one. Alarms (score > threshold)
    less than MERGE_GAP_S apart merge, but never across segments.
    """
    t = np.asarray(t, dtype=float)
    segment = np.asarray(segment)
    alarm = np.where(np.asarray(excluded, dtype=bool), -np.inf, np.asarray(score, dtype=float)) > threshold
    alerts = []
    for s in pd.unique(segment):
        rows = np.flatnonzero(segment == s)
        starts, ends = _runs(_frames(t[rows]), alarm[rows])
        alerts += zip(t[rows[starts]].tolist(), t[rows[ends]].tolist())
    return alerts


def match(alerts, onsets) -> tuple:
    """(caught, lead, false_alarm) for alerts [(start, end)] and onset times in one segment.

    caught and lead are per onset: lead is onset − the start of the earliest
    alert that catches it, capped at CREDIT_EARLY_S, and NaN if none does.
    false_alarm is per alert. An alert that switched on early and stayed on
    through the spin is a catch, not a miss plus a false alarm.
    """
    alerts = np.asarray(alerts, dtype=float).reshape(-1, 2)
    caught, lead, false_alarm = _match_frames(_frames(alerts[:, 0]), _frames(alerts[:, 1]), _frames(onsets))
    return caught, np.where(caught, lead / GRID_HZ, np.nan), false_alarm


def _match_frames(start, end, onset) -> tuple:
    """match, in grid frames; lead means something only where caught."""
    early, late, after = (round(s * GRID_HZ) for s in (CREDIT_EARLY_S, CREDIT_LATE_S, FA_AFTER_S))
    # Alerts × onsets: how long before each onset each alert starts, and ends.
    to_start = onset[None, :] - start[:, None]
    to_end = onset[None, :] - end[:, None]
    # Active somewhere in [onset − early, onset − late]: on by its end, still on at its start.
    credits = (to_start >= late) & (to_end <= early)
    caught = credits.any(axis=0)
    lead = np.where(credits, np.minimum(to_start, early), 0).max(axis=0, initial=0)
    false_alarm = ~((to_start > 0) & (to_end <= after)).any(axis=1)   # no onset in (start, end + after]
    return caught, lead, false_alarm


def _scores(df, score) -> np.ndarray:
    """score as floats, one per row of df; ValueError on a NaN where it's scored, which would never alarm."""
    score = np.asarray(score, dtype=float)
    if len(score) != len(df):
        raise ValueError(f"{len(score)} scores for {len(df)} rows")
    if np.isnan(score[~df["excluded"].to_numpy(dtype=bool)]).any():
        raise ValueError("a scored frame has a NaN score")
    return score


def _segments(df, score) -> list:
    """Each segment of each session in df, ready to score at any threshold.

    Per segment: its (session, segment), its grid frames, the score with −∞
    on excluded frames, the onset frames, and a running count of scored
    frames, for time in alarm.
    """
    excluded = df["excluded"].to_numpy(dtype=bool)
    masked = np.where(excluded, -np.inf, _scores(df, score))
    frame = _frames(df["t"])
    onset = df["onset"].to_numpy(dtype=bool)
    return [
        (key, frame[rows], masked[rows], frame[rows][onset[rows]], np.r_[0, np.cumsum(~excluded[rows])])
        for key, rows in df.groupby(["session", "segment"], sort=False).indices.items()
    ]


def _share(part, whole) -> float:
    return part / whole if whole else float("nan")


def _summary(segments, threshold) -> dict:
    """summarize, from _segments: counts summed over every segment, then the ratios."""
    n_spins = caught = caught_min_lead = alerts = false_alarms = in_alarm = scored = 0
    leads = []
    min_lead = round(MIN_LEAD_S * GRID_HZ)
    for _key, frame, score, onsets, scored_before in segments:
        starts, ends = _runs(frame, score > threshold)
        hit, lead, false_alarm = _match_frames(frame[starts], frame[ends], onsets)
        n_spins += len(onsets)
        caught += int(hit.sum())
        caught_min_lead += int((hit & (lead >= min_lead)).sum())
        leads += (lead[hit] / GRID_HZ).tolist()
        alerts += len(starts)
        false_alarms += int(false_alarm.sum())
        # An alert is active from its first frame to its last, merged gaps
        # included; the excluded frames in between aren't driving it's scored on.
        in_alarm += int((scored_before[ends + 1] - scored_before[starts]).sum())
        scored += int(scored_before[-1])
    minutes = scored / GRID_HZ / 60
    return {
        "n_spins": n_spins,
        "caught": caught,
        "recall": _share(caught, n_spins),
        "caught_min_lead": caught_min_lead,
        "recall_min_lead": _share(caught_min_lead, n_spins),
        "alerts": alerts,
        "false_alarms": false_alarms,
        "minutes": minutes,
        "fa_per_min": _share(false_alarms, minutes),
        "alert_precision": _share(alerts - false_alarms, alerts),
        "time_in_alarm": _share(in_alarm, scored),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
    }


def summarize(df, score, threshold) -> dict:
    """The alerts at threshold, scored, pooled over every session in df.

    n_spins, caught, recall, caught_min_lead and recall_min_lead (the
    headline: caught with at least MIN_LEAD_S of lead), alerts,
    false_alarms, minutes (of scored driving), fa_per_min, alert_precision
    (alerts not false ÷ all), time_in_alarm (share of scored frames with an
    alert active) and median_lead (s, over caught spins). Sessions are
    pooled by summing counts and minutes, not by averaging their ratios.

    Alert precision can't rank methods: nearly every spin gets one alert
    that isn't false, whatever the method, so it comes out close to
    spins ÷ (spins + false alarms), which the false-alarm target sets.
    """
    return _summary(_segments(df, score), threshold)


SPIN_COLUMNS = ["session", "t", "caught", "caught_min_lead", "lead"]


def spin_catches(df, score, threshold) -> pd.DataFrame:
    """One row per spin onset in df, in order, at threshold: what summarize counts, spin by spin.

    session and t (the onset), caught by the Project Plan's rule,
    caught_min_lead (the headline: with at least MIN_LEAD_S of lead), and
    lead (s, NaN if missed). One bool per spin is what bootstrap_recall
    resamples.
    """
    min_lead = round(MIN_LEAD_S * GRID_HZ)
    rows = []
    for (session, _segment), frame, masked, onsets, _scored_before in _segments(df, score):
        starts, ends = _runs(frame, masked > threshold)
        hit, lead, _false_alarm = _match_frames(frame[starts], frame[ends], onsets)
        rows += [
            {"session": session, "t": float(onset / GRID_HZ), "caught": bool(h),
             "caught_min_lead": bool(h and ahead >= min_lead), "lead": float(ahead / GRID_HZ) if h else float("nan")}
            for onset, h, ahead in zip(onsets, hit, lead)
        ]
    return pd.DataFrame(rows, columns=SPIN_COLUMNS)


def _candidates(df, score) -> np.ndarray:
    """Candidate thresholds: N_THRESHOLDS quantiles of the finite scores on scored frames, plus the highest.

    Spaced geometrically in the top tail, from every frame down to one, since
    1 FA/min or so alarms on only a few percent of frames. The highest score
    alarms nowhere, so some candidate always meets any false-alarm target.
    """
    score = _scores(df, score)[~df["excluded"].to_numpy(dtype=bool)]
    finite = score[np.isfinite(score)]
    if not len(finite):
        raise ValueError("no finite score on a scored frame")
    tail = np.geomspace(1, 1 / len(finite), N_THRESHOLDS)
    return np.unique(np.r_[np.quantile(finite, 1 - tail, method="inverted_cdf"), finite.max()])


def recall_fa_curve(df, score) -> pd.DataFrame:
    """summarize at every candidate threshold: one row each, with its threshold, lowest first."""
    segments = _segments(df, score)
    return pd.DataFrame([{"threshold": float(th), **_summary(segments, th)} for th in _candidates(df, score)])


def threshold_for_fa(df, score, target) -> float:
    """The lowest candidate threshold with at most `target` false alarms per minute.

    The lowest, so the one that alerts most within the target. Every
    candidate is scanned rather than bisected, since false alarms don't
    always fall as the threshold rises: one alert can split into two.
    """
    curve = recall_fa_curve(df, score)
    within = curve.loc[curve["fa_per_min"] <= target, "threshold"]
    if within.empty:
        raise ValueError(f"no threshold keeps to {target} false alarms per minute")
    return float(within.min())


def pr_auc(df, score) -> float:
    """Average precision of score for target over the scored frames, pooled over sessions.

    Taken on the scores' ranks: only their order counts, and scikit-learn
    refuses the −∞ that B2 gives frames it never alarms on.
    """
    scored = ~df["excluded"].to_numpy(dtype=bool)
    ranks = pd.Series(_scores(df, score)[scored]).rank(method="min").to_numpy()
    return float(average_precision_score(df["target"].to_numpy(dtype=bool)[scored], ranks))


def bootstrap_recall(caught, n=1000, seed=SEED) -> tuple:
    """(lo, hi), a 95% interval on recall: the spins resampled with replacement n times.

    caught is one bool per spin. With about 20 test spins, one spin is 5
    points of recall, and the interval shows how far that reaches.
    """
    caught = np.asarray(caught, dtype=bool)
    recalls = np.random.default_rng(seed).choice(caught, size=(n, len(caught))).mean(axis=1)
    lo, hi = np.percentile(recalls, [2.5, 97.5])
    return float(lo), float(hi)
