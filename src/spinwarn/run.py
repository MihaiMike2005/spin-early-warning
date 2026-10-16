"""
The command line: build, lock-test, cv, tune, final.

    python -m spinwarn.run build [--horizon 1.0]
    python -m spinwarn.run lock-test --sessions A,B
    python -m spinwarn.run cv --method b1|b2|b3|lr|rf|hgb [--params '{...}'] [--horizon 0.5] [--note "..."]
    python -m spinwarn.run tune --method b1|b2|b3|lr|rf|hgb [--n 15] [--horizon 0.5] [--note "..."]
    python -m spinwarn.run final [--horizon 0.5] [--force] [--note "..."]

build labels every dev session, computes its features and writes them all
into one table, data/processed/dataset.parquet, one row per grid frame. It
prints minutes, spins and positive frames per session: the numbers to pick
the test sessions by. --horizon labels it at another horizon than
HORIZON_S, into a table of its own beside it, e.g. dataset_h1.parquet: the
onset column alone can't relabel the main one, since its excluded already
holds each segment's last HORIZON_S.

lock-test sets the test sessions aside in data/splits.json, once, before
any modelling: load_dataset, which modelling reads the table through,
refuses it until then. From then on everything but the final run leaves
them out, so build again after it.

cv scores every dev frame out of fold with one method (cv.oof_scores),
prints PR-AUC and each FA_TARGETS operating point, saves the scores to
results/cv/<run id>.parquet and appends a row to experiments.md. b2 without
--params picks its θ1 first (cv.choose_theta1). The headline is recall at 1
FA/min with at least MIN_LEAD_S of lead; recall by the Project Plan's rule,
any catch, comes beside it.

tune makes each method's CV run for the final one and records it in
results/chosen.json. A model tries --n configs sampled from its grid with
SEED (models.configs), each one cv run, and keeps the best by PR-AUC, as
the Project Plan picks models. A baseline has nothing to search: b1 and b3
run once, and b2 picks its θ1 by the headline, as cv does. --horizon reads
that horizon's dataset and records into a file of its own, chosen_h1.json
for 1.0.

final scores the test sessions, once, with every method's chosen run.
Each method's thresholds at FA_TARGETS come from its chosen run's
out-of-fold scores, and each model is refit on every dev session; only
then are the test sessions loaded, labelled with today's rule and scored.
On them: both recalls with bootstrap intervals, the same per kind (natural,
deliberate), PR-AUC and time per prediction, and the best model's
permutation importance (best by CV PR-AUC, never by test). It all goes to
results/metrics.json. results/final.lock is written just before the test
sessions load, so even a run that stops partway counts: final refuses to
run while it or metrics.json exists, and --force, which overrides them,
logs a row in experiments.md. --horizon reads that horizon's chosen runs and writes its
own files, metrics_h1.json and final_h1.lock for 1.0.
"""

import argparse
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from spinwarn import baselines, cv, data, models
from spinwarn.baselines import BASELINES
from spinwarn.config import (
    CHOSEN_PATH, CV_DIR, DATASET_PATH, EXPERIMENTS_PATH, FINAL_LOCK_PATH, GRID_HZ, HORIZON_S, METRICS_PATH,
    MIN_KIND_SPINS, MIN_LEAD_S, MIN_SPEED_KMH, N_FOLDS, N_PERMUTATIONS, N_TIMING, N_TUNE, RAW_DIR, REARM_DEG,
    REARM_S, REVERSE_GEAR, SEED, SPIN_HOLD_S, SPLITS_PATH, WINDOW_S,
)
from spinwarn.data import dev_sessions, list_sessions, load_session, load_test_sessions, session_kind
from spinwarn.evaluate import bootstrap_recall, pr_auc, recall_fa_curve, spin_catches, summarize
from spinwarn.features import FEATURES, make_features
from spinwarn.labeling import find_onsets, label


# The signals the baselines score with, kept as they are: B1 abs_slip, B2
# abs_slip and abs_slip_rate, B3 time_to_90.
BASELINE_SIGNALS = list(dict.fromkeys(s for signals in baselines.SIGNALS.values() for s in signals))
# onset marks each spin's first frame, so alerts can be scored from the dataset alone.
COLUMNS = ["session", "segment", "t", "target", "excluded", "onset", *BASELINE_SIGNALS, *FEATURES]
# What a CV run's file keeps beside its scores and folds: enough to score them again.
OOF_COLUMNS = ["session", "segment", "t", "target", "excluded", "onset"]
# experiments.md's table. Run names the run's file in CV_DIR. The headline
# recall comes first, then the Project Plan's, which counts any catch.
EXPERIMENT_COLUMNS = ["Date", "Run", "Method", "Horizon (s)", "Features", "Params", "PR-AUC",
                      f"Recall @ 1 FA/min, lead ≥ {MIN_LEAD_S:g} s", "Recall @ 1 FA/min", "Note"]
# What metrics.json keeps of each method's CV recall-vs-false-alarm curve, for report.py's plot.
CURVE_COLUMNS = ["threshold", "fa_per_min", "recall", "recall_min_lead", "time_in_alarm"]


def at_horizon(path, horizon) -> Path:
    """path itself at HORIZON_S, else the file beside it for horizon: dataset_h1.parquet for 1.0."""
    if horizon == HORIZON_S:
        return path
    return path.with_name(f"{path.stem}_h{horizon:g}{path.suffix}")


def dataset_path(horizon) -> Path:
    """Where build writes the dataset labelled at horizon: DATASET_PATH at HORIZON_S, else beside it."""
    return at_horizon(DATASET_PATH, horizon)


def chosen_path(horizon) -> Path:
    """Where tune records the chosen runs at horizon: CHOSEN_PATH at HORIZON_S, else beside it."""
    return at_horizon(CHOSEN_PATH, horizon)


def metrics_path(horizon) -> Path:
    """Where final writes its results at horizon: METRICS_PATH at HORIZON_S, else beside it."""
    return at_horizon(METRICS_PATH, horizon)


def lock_path(horizon) -> Path:
    """final's run-once lock at horizon: FINAL_LOCK_PATH at HORIZON_S, else beside it."""
    return at_horizon(FINAL_LOCK_PATH, horizon)


def _command(name, horizon) -> str:
    """The run.py command line for name at horizon, for messages."""
    return f"python -m spinwarn.run {name}" + ("" if horizon == HORIZON_S else f" --horizon {horizon:g}")


def built_with(horizon) -> dict:
    """The constants a dataset depends on: the session caches', the labels' and features', and its horizon."""
    return {
        **data.built_with(),
        "MIN_SPEED_KMH": MIN_SPEED_KMH, "REARM_DEG": REARM_DEG, "REARM_S": REARM_S,
        "SPIN_HOLD_S": SPIN_HOLD_S, "REVERSE_GEAR": REVERSE_GEAR, "horizon": horizon,
        "WINDOW_S": WINDOW_S,
    }


def session_rows(session_id, df, horizon=HORIZON_S) -> pd.DataFrame:
    """The dataset's rows for a grid session from load_session: one per frame, in COLUMNS order."""
    onsets, _spinning = find_onsets(df)
    rows = pd.concat([label(df, horizon), make_features(df)], axis=1)
    rows["session"] = session_id
    # Grid times are unique within a session, and the onsets are some of them, exactly.
    rows["onset"] = np.isin(df["t"].to_numpy(), onsets)
    return rows[COLUMNS].reset_index(drop=True)


def build_dataset(sessions, horizon) -> pd.DataFrame:
    """The sessions loaded, labelled at horizon and with features, in one table that remembers what built it."""
    dataset = pd.concat([session_rows(s, load_session(s), horizon) for s in sessions], ignore_index=True)
    dataset.attrs = {"sessions": list(sessions), "built_with": built_with(horizon)}
    return dataset


def write_parquet(df, path):
    """df to path, written aside and renamed into place, as load_session does its cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    df.to_parquet(partial, index=False)
    partial.replace(path)


def load_dataset(horizon=None) -> pd.DataFrame:
    """The table build wrote at horizon (HORIZON_S if None), for modelling: refused before the lock, or if it's out of date.

    Before the lock, nothing is set aside to test on, and the Project Plan
    locks the test sessions before any modelling. build still works then,
    for the spin counts to pick them by.

    Out of date is built from other sessions than today's dev sessions (one
    copied in, or the test sessions locked, since) or with other constants.
    Built before the lock, it still holds the test sessions, and CV would
    train on them. Building again takes seconds.
    """
    horizon = HORIZON_S if horizon is None else horizon
    path = dataset_path(horizon)
    if not path.exists():
        raise FileNotFoundError(f"no {path}: run `{_command('build', horizon)}` first")
    if not SPLITS_PATH.exists():
        raise RuntimeError(f"no test sessions locked in {SPLITS_PATH}: lock them with "
                           "`python -m spinwarn.run lock-test` before any modelling, then build again")
    dataset = pd.read_parquet(path)
    if dataset.attrs.get("sessions") != dev_sessions():
        raise RuntimeError("the dataset was built from other sessions than today's dev sessions: "
                           f"run `{_command('build', horizon)}` again")
    if dataset.attrs.get("built_with") != built_with(horizon):
        raise RuntimeError("config.py has changed since the dataset was built: "
                           f"run `{_command('build', horizon)}` again")
    return dataset


def describe(dataset) -> list:
    """Per session, then in total: minutes, scored minutes, spins and positive (scored target) frames."""
    scored = ~dataset["excluded"]
    by_session = dataset.assign(scored=scored, positive=dataset["target"] & scored).groupby("session", sort=False)
    table = pd.DataFrame({
        "frames": by_session.size(),
        "scored": by_session["scored"].sum(),
        "spins": by_session["onset"].sum(),
        "positive": by_session["positive"].sum(),
    })
    table.loc["total"] = table.sum()
    lines = []
    for session, row in table.iterrows():
        minutes = row.frames / GRID_HZ / 60
        lines.append(f"{session}: {minutes:.1f} min, {row.scored / GRID_HZ / 60:.1f} scored, "
                     f"{row.spins} spins ({row.spins / minutes:.1f}/min), "
                     f"{row.positive} positive frames ({row.positive / row.scored:.1%} of scored)")
    return lines


def parse_sessions(text) -> list:
    """argparse type for --sessions: session ids, comma-separated, none twice."""
    ids = [s.strip() for s in text.split(",") if s.strip()]
    if not ids:
        raise argparse.ArgumentTypeError(f"no session ids in {text!r}")
    if len(set(ids)) < len(ids):
        raise argparse.ArgumentTypeError(f"a session is listed twice: {text}")
    return ids


def parse_horizon(text) -> float:
    """argparse type for --horizon: seconds, over 0, and a whole number of grid frames, as label() counts them."""
    try:
        horizon = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {text!r}") from None
    frames = horizon * GRID_HZ
    if not (math.isfinite(horizon) and horizon > 0 and abs(frames - round(frames)) < 1e-9):
        raise argparse.ArgumentTypeError(f"{text} s isn't a whole number of 1/{GRID_HZ} s grid frames over 0")
    return horizon


def parse_n(text) -> int:
    """argparse type for tune's --n: a whole number of configs, 1 or more."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{n} configs: tune needs at least 1")
    return n


def parse_params(text) -> dict:
    """argparse type for --params: a JSON object."""
    try:
        params = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"not JSON: {error}") from None
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError(f"not a JSON object: {text}")
    return params


def lock_test(test_ids):
    """Lock test_ids as the test sessions in splits.json; ValueError, and nothing written, if it can't.

    Refused if splits.json exists (the test sessions are locked once, before
    any modelling), if an id isn't a recorded session, or if fewer than
    N_FOLDS sessions would be left: CV needs a dev session per fold.
    """
    if SPLITS_PATH.exists():
        raise ValueError(f"{SPLITS_PATH} exists: the test sessions are locked once, before any modelling")
    sessions = list_sessions()
    unknown = [s for s in test_ids if s not in sessions]
    if unknown:
        raise ValueError(f"no session {', '.join(unknown)} in {RAW_DIR}")
    left = len(sessions) - len(test_ids)
    if left < N_FOLDS:
        raise ValueError(f"only {left} dev sessions would be left, and {N_FOLDS}-fold CV needs "
                         f"{N_FOLDS}: record more first")
    # "x" never writes over a file, even one that appeared since the check above.
    with SPLITS_PATH.open("x") as f:
        json.dump({"test": sorted(test_ids)}, f, indent=2)
        f.write("\n")


def new_run_id(method, horizon, now) -> str:
    """<date>_<time>_<method>, with _h<horizon> off HORIZON_S, and _2, _3, ... if CV_DIR has it already."""
    stem = f"{now:%Y-%m-%d_%H%M%S}_{method}" + ("" if horizon == HORIZON_S else f"_h{horizon:g}")
    run_id, n = stem, 1
    while (CV_DIR / f"{run_id}.parquet").exists():
        n += 1
        run_id = f"{stem}_{n}"
    return run_id


def oof_table(dataset, scores, run) -> pd.DataFrame:
    """A CV run's file: OOF_COLUMNS, fold and score per dataset row, and in its attrs the run and what built the dataset."""
    table = dataset[OOF_COLUMNS].assign(fold=cv.fold_ids(dataset), score=scores)
    table.attrs = {**run, "sessions": dataset.attrs["sessions"], "built_with": dataset.attrs["built_with"]}
    return table


def experiment_row(run) -> str:
    """run's row for experiments.md, in EXPERIMENT_COLUMNS order."""
    features = cv.features_of(run["method"])
    at_1 = run["metrics"]["operating_points"][0]
    cells = [
        run["date"], f"`{run['run_id']}`", run["method"], f"{run['horizon']:g}",
        f"`FEATURES` ({len(FEATURES)})" if features == FEATURES else ", ".join(f"`{f}`" for f in features),
        f"`{json.dumps(run['params'])}`", f"{run['metrics']['pr_auc']:.3f}",
        f"{at_1['recall_min_lead']:.2f} ({at_1['caught_min_lead']}/{at_1['n_spins']})",
        f"{at_1['recall']:.2f} ({at_1['caught']}/{at_1['n_spins']})", run["note"],
    ]
    return _table_row(cells)


def _table_row(cells) -> str:
    """A row of experiments.md's table from its cells, in EXPERIMENT_COLUMNS order."""
    # A | would end the cell, and a line break the row.
    return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def append_experiment(row):
    """row onto the end of experiments.md, its table; a new file gets the table's header first."""
    if not EXPERIMENTS_PATH.exists():
        EXPERIMENTS_PATH.write_text("| " + " | ".join(EXPERIMENT_COLUMNS) + " |\n"
                                    + "| --- " * len(EXPERIMENT_COLUMNS) + "|\n")
    with EXPERIMENTS_PATH.open("a") as f:
        f.write(row + "\n")


def report(metrics) -> list:
    """What cv prints: PR-AUC, then each operating point, its headline recall first."""
    lines = [f"PR-AUC {metrics['pr_auc']:.3f}"]
    for p in metrics["operating_points"]:
        lines.append(f"at {p['fa_target']:g} FA/min (threshold {p['threshold']:.4g}): {headline(p)}, "
                     f"{p['fa_per_min']:.2f} FA/min, alert precision {p['alert_precision']:.2f}, "
                     f"time in alarm {p['time_in_alarm']:.1%}, median lead {p['median_lead']:.2f} s")
    return lines


def headline(point) -> str:
    """An operating point's two recalls, the headline first, for one line."""
    return (f"recall with ≥ {MIN_LEAD_S:g} s lead {point['recall_min_lead']:.2f} "
            f"({point['caught_min_lead']}/{point['n_spins']}), "
            f"any lead {point['recall']:.2f} ({point['caught']}/{point['n_spins']})")


def save_run(dataset, method, params, horizon, note) -> dict:
    """One CV run of method with params: its out-of-fold scores saved in CV_DIR, and its row in experiments.md.

    Returns the run: run_id, date, method, params, horizon, note and metrics.
    """
    scores = cv.oof_scores(dataset, method, params)
    now = datetime.now()
    run = {
        "run_id": new_run_id(method, horizon, now), "date": f"{now:%Y-%m-%d}", "method": method,
        "params": params, "horizon": horizon, "note": note, "metrics": cv.metrics(dataset, scores),
    }
    write_parquet(oof_table(dataset, scores, run), CV_DIR / f"{run['run_id']}.parquet")
    append_experiment(experiment_row(run))
    return run


def choose_b2(dataset) -> tuple:
    """(params, note): B2's θ1 from cv.choose_theta1, with its sweep printed, and the note that says how it was picked."""
    theta1, table = cv.choose_theta1(dataset)
    print(f"B2's θ1, by recall with ≥ {MIN_LEAD_S:g} s lead at 1 FA/min (ties by any-lead recall, "
          "then PR-AUC):")
    for row in table.itertuples():
        print(f"  θ1 {row.theta1}°: {headline(row._asdict())}, {row.fa_per_min:.2f} FA/min, PR-AUC {row.pr_auc:.3f}")
    print(f"θ1 = {theta1}°\n")
    return {"theta1": theta1}, (f"θ1 best of {table['theta1'].iloc[0]}–{table['theta1'].iloc[-1]}° "
                                f"by recall @ 1 FA/min, lead ≥ {MIN_LEAD_S:g} s")


def _load_for(horizon, error) -> pd.DataFrame:
    """load_dataset(horizon), or the parser's error saying why not."""
    try:
        return load_dataset(horizon)
    except (FileNotFoundError, RuntimeError) as e:
        error(str(e))


def _scored_on(dataset, horizon) -> str:
    """What a CV run of the dataset covers, for the line before its numbers."""
    spins = int(dataset["onset"].sum())
    minutes = (~dataset["excluded"]).sum() / GRID_HZ / 60
    return (f"horizon {horizon:g} s, {N_FOLDS}-fold CV over {len(dataset.attrs['sessions'])} dev sessions: "
            f"{spins} spins in {minutes:.1f} scored min")


def cross_validate(method, params, horizon, note, error):
    """run.py cv: out-of-fold scores of method on the dataset at horizon, printed, saved and logged.

    error is the cv parser's, for a mistake on the command line.
    """
    # b2 with no params picks its θ1 first; anything else is checked before the dataset loads.
    sweep = method == "b2" and not params
    if not sweep:
        try:
            cv.check_params(method, params)
        except ValueError as e:
            error(str(e))
    dataset = _load_for(horizon, error)

    if sweep:
        params, swept = choose_b2(dataset)
        note = "; ".join(filter(None, [swept, note]))
    run = save_run(dataset, method, params, horizon, note)

    print(f"{method} {json.dumps(params)} at {_scored_on(dataset, horizon)}")
    print("\n".join(report(run["metrics"])))
    print(f"\nwrote {CV_DIR / run['run_id']}.parquet and a row in {EXPERIMENTS_PATH.name}")


def record_chosen(path, method, entry):
    """entry as method's chosen run in the JSON file at path, the other methods' left as they are, in METHODS order."""
    chosen = json.loads(path.read_text()) if path.exists() else {}
    chosen[method] = entry
    chosen = {m: chosen[m] for m in cv.METHODS if m in chosen} | chosen
    write_json(chosen, path)


def _json_ready(value):
    """value with each NumPy scalar as the Python number or bool it holds, and each NaN as None.

    json writes NaN as a bare NaN, which strict JSON (jq, a browser) refuses.
    A missed spin's lead is NaN, and so is a ratio with nothing to divide by.
    """
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and math.isnan(value) else value


def write_json(obj, path):
    """obj to path as JSON, NaN as null, written aside and renamed into place, as write_parquet does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    # ensure_ascii off: b2's θ1 as θ, not θ.
    partial.write_text(json.dumps(_json_ready(obj), indent=2, ensure_ascii=False) + "\n")
    partial.replace(path)


def tune(method, n, horizon, note, error):
    """run.py tune: CV runs of method's configs, each logged as cv logs it, and the chosen one recorded.

    A model tries n configs from models.configs and keeps the best by
    PR-AUC; ties go to the one tried first. b1 and b3 have nothing to tune
    and run once; b2 picks its θ1 by the headline, as cv does. note goes on
    every run's row, after the tune's own. error is the tune parser's.
    """
    dataset = _load_for(horizon, error)
    print(f"tuning {method} at {_scored_on(dataset, horizon)}")
    if method in models.MODELS:
        trials = models.configs(method, n)
        notes = [f"tune {k}/{len(trials)}" for k in range(1, len(trials) + 1)]
        chosen_by = f"PR-AUC, best of {len(trials)} configs"
    elif method == "b2":
        params, swept = choose_b2(dataset)
        trials, notes, chosen_by = [params], [f"{swept}; tune"], swept
    else:
        trials, notes, chosen_by = [{}], ["tune"], "nothing to tune"

    runs = []
    for k, (params, own) in enumerate(zip(trials, notes), start=1):
        run = save_run(dataset, method, params, horizon, "; ".join(filter(None, [own, note])))
        runs.append(run)
        print(f"  {k}/{len(trials)} {json.dumps(params)}: PR-AUC {run['metrics']['pr_auc']:.3f}, at 1 FA/min "
              f"{headline(run['metrics']['operating_points'][0])}  [{run['run_id']}]")
    # max keeps the first of equals: the config tried first.
    best = max(runs, key=lambda r: r["metrics"]["pr_auc"])
    path = chosen_path(horizon)
    record_chosen(path, method, {"run_id": best["run_id"], "params": best["params"], "chosen_by": chosen_by,
                                 "pr_auc": best["metrics"]["pr_auc"]})

    print(f"\nchosen: {method} {json.dumps(best['params'])}, run {best['run_id']}")
    print("\n".join(report(best["metrics"])))
    print(f"\nwrote {len(runs)} run{'s' * (len(runs) > 1)} to {CV_DIR} and {EXPERIMENTS_PATH.name}, "
          f"and {method}'s chosen run to {path}")


def read_chosen(horizon) -> dict:
    """tune's chosen run for every method at horizon, {method: entry} in METHODS order; ValueError if any is missing."""
    path = chosen_path(horizon)
    tune_each = f"`{_command('tune --method <method>', horizon)}` for"
    if not path.exists():
        raise ValueError(f"no {path}: run {tune_each} every method ({', '.join(cv.METHODS)}) first")
    chosen = json.loads(path.read_text())
    missing = [m for m in cv.METHODS if m not in chosen]
    if missing:
        raise ValueError(f"{path} has no run for {', '.join(missing)}: run {tune_each} each first")
    return {m: chosen[m] for m in cv.METHODS}


def chosen_run(method, entry, dataset, horizon) -> pd.DataFrame:
    """method's chosen CV run, its file in CV_DIR; ValueError unless it's the run chosen.json names, on this dataset.

    Its thresholds are set on its scores, so they must score this dataset's
    frames: the same sessions and built_with, as load_dataset checks the
    dataset itself, and the same rows, which a dataset built again after a
    code change can differ in with neither changed.
    """
    path = CV_DIR / f"{entry['run_id']}.parquet"
    again = f"tune it again with `{_command(f'tune --method {method}', horizon)}`"
    if not path.exists():
        raise ValueError(f"no {path}, {method}'s chosen run. results/cv/ isn't in git, but tune is seeded: {again}")
    oof = pd.read_parquet(path)
    run = oof.attrs
    if (run.get("method"), run.get("params")) != (method, entry["params"]):
        raise ValueError(f"{path.name} is {run.get('method')} with {json.dumps(run.get('params'))}, not "
                         f"{method} with chosen.json's {json.dumps(entry['params'])}")
    if run.get("sessions") != dataset.attrs["sessions"]:
        raise ValueError(f"{method}'s chosen run scored other sessions than the dataset's: {again}")
    if run.get("built_with") != dataset.attrs["built_with"]:
        raise ValueError(f"{method}'s chosen run was built with other constants than the dataset: {again}")
    if len(oof) != len(dataset) or not all((oof[c].to_numpy() == dataset[c].to_numpy()).all() for c in OOF_COLUMNS):
        raise ValueError(f"{method}'s chosen run scored other rows than the dataset's, which was built again "
                         f"since: {again}")
    return oof


def cv_results(oof) -> dict:
    """A chosen run's CV numbers from its scores: PR-AUC, each FA_TARGETS point and its threshold, and the curve."""
    scores = oof["score"].to_numpy()
    return {**cv.metrics(oof, scores), "curve": recall_fa_curve(oof, scores)[CURVE_COLUMNS].to_dict("list")}


def refit(method, params, dataset):
    """method fitted on every dev session, as each CV fold was on the others; None for a baseline, which has nothing to fit."""
    if method in BASELINES:
        return None
    return cv.fit(models.make_model(method, params), dataset, np.arange(len(dataset)))


def method_scores(method, params, model, rows) -> np.ndarray:
    """method's score on every row of rows: a baseline's from its signals, a fitted model's from its features."""
    return baselines.score(method, rows, params) if method in BASELINES else cv.predict(model, rows)


def recall_interval(caught) -> list:
    """bootstrap_recall's 95% interval from one bool per spin, as [lo, hi]; NaN, NaN with no spin to resample."""
    caught = np.asarray(caught, dtype=bool)
    return list(bootstrap_recall(caught)) if len(caught) else [float("nan")] * 2


def _intervals(spins) -> dict:
    """Both recalls' intervals from spin_catches rows: the headline's and the Project Plan's."""
    return {"recall_min_lead_ci": recall_interval(spins["caught_min_lead"]), "recall_ci": recall_interval(spins["caught"])}


def operating_point(test, scores, threshold, kinds) -> dict:
    """The test rows scored at a threshold from CV: summarize(), both recalls' intervals, the same per kind, each spin.

    kinds maps each test session to its kind. A kind's recall is enough to
    report on its own with MIN_KIND_SPINS spins or more; with fewer, it's
    for the README's limitations.
    """
    spins = spin_catches(test, scores, threshold)
    spins["kind"] = spins["session"].map(kinds)
    kind_of_row = test["session"].map(kinds).to_numpy()
    by_kind = {}
    for kind in sorted(set(kinds.values())):
        rows = kind_of_row == kind
        of_kind = spins[spins["kind"] == kind]
        by_kind[kind] = {**summarize(test[rows], scores[rows], threshold), **_intervals(of_kind),
                         "enough": len(of_kind) >= MIN_KIND_SPINS}
    return {"threshold": threshold, **summarize(test, scores, threshold), **_intervals(spins),
            "by_kind": by_kind, "spins": spins.to_dict("records")}


def importance(model, test) -> list:
    """Permutation importance of each feature for a fitted model on the scored test frames, highest first.

    [{feature, mean, std}]: how far average precision falls over
    N_PERMUTATIONS shuffles of that feature alone. Features that carry the
    same information share it, so each can look less important than the
    information is.
    """
    scored = test[~test["excluded"].to_numpy(dtype=bool)]
    result = permutation_importance(model, scored[FEATURES], scored["target"], scoring="average_precision",
                                    n_repeats=N_PERMUTATIONS, random_state=SEED)
    order = np.argsort(-result.importances_mean, kind="stable")
    return [{"feature": FEATURES[i], "mean": float(result.importances_mean[i]),
             "std": float(result.importances_std[i])} for i in order]


def median_ms(call, rows) -> float:
    """The median milliseconds call(row) takes, called once per row of rows."""
    times = np.empty(len(rows))
    for k, row in enumerate(rows):
        start = time.perf_counter()
        call(row)
        times[k] = time.perf_counter() - start
    return float(np.median(times) * 1000)


def time_per_prediction(method, params, model, test) -> dict:
    """{"ms": the median ms of one single-row prediction}, over N_TIMING scored test frames picked with SEED.

    A model gets each frame as a 1-row array of FEATURES, as a live loop
    would pass it (a DataFrame's overhead would swamp a small model's), and
    scores it with predict_proba. A baseline gets a 1-row slice of its
    signals. rf predicts on every core (n_jobs=-1), which starts threads for
    every call, so a model with n_jobs is timed on one core too
    ("ms_n_jobs_1"), as a 60 Hz loop would run it.
    """
    scored = np.flatnonzero(~test["excluded"].to_numpy(dtype=bool))
    picks = np.random.default_rng(SEED).choice(scored, size=N_TIMING, replace=len(scored) < N_TIMING)
    if method in BASELINES:
        signals = test[baselines.SIGNALS[method]]
        return {"ms": median_ms(lambda row: BASELINES[method](row, **params), [signals.iloc[[i]] for i in picks])}
    rows = test[FEATURES].to_numpy(dtype=float)[picks, None, :]
    timing = {}
    with warnings.catch_warnings():
        # scikit-learn's, on every array scored by a model fitted on a DataFrame.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        timing["ms"] = median_ms(model.predict_proba, rows)
        if "n_jobs" in model.get_params():
            n_jobs = model.get_params()["n_jobs"]
            model.set_params(n_jobs=1)
            try:
                timing["ms_n_jobs_1"] = median_ms(model.predict_proba, rows)
            finally:
                model.set_params(n_jobs=n_jobs)
    return timing


def _counts(rows) -> dict:
    return {"spins": int(rows["onset"].sum()), "scored_min": float((~rows["excluded"]).sum() / GRID_HZ / 60)}


def _started(path) -> str:
    """When the final run that wrote path, its lock or its results, loaded the test sessions, for messages."""
    try:
        return json.loads(path.read_text())["started"]
    except (ValueError, KeyError, TypeError):
        return f"an unknown time ({path.name} is unreadable)"


def write_lock(path, started, horizon, forced):
    """The run-once lock, saying when the test sessions were loaded. Only a forced run writes over one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if forced else "x") as f:
        json.dump({"started": started.isoformat(timespec="seconds"), "horizon": horizon, "forced": forced}, f,
                  indent=2)
        f.write("\n")


def forced_row(started, horizon, previous, note) -> str:
    """experiments.md's row for a forced final run, so the test sessions scored again stays visible."""
    own = (f"final run forced: the test sessions scored again (last loaded at {previous}), "
           f"results in {metrics_path(horizon).name}")
    return _table_row([f"{started:%Y-%m-%d}", "`final`", "every method", f"{horizon:g}", "—", "—", "—", "—", "—",
                       "; ".join(filter(None, [own, note]))])


def with_intervals(point) -> str:
    """A test operating point's two recalls, each with its bootstrap interval, the headline first."""
    (lo, hi), (any_lo, any_hi) = point["recall_min_lead_ci"], point["recall_ci"]
    return (f"recall with ≥ {MIN_LEAD_S:g} s lead {point['recall_min_lead']:.2f} ({point['caught_min_lead']}/"
            f"{point['n_spins']}) [{lo:.2f}, {hi:.2f}], any lead {point['recall']:.2f} ({point['caught']}/"
            f"{point['n_spins']}) [{any_lo:.2f}, {any_hi:.2f}]")


def final_report(metrics) -> list:
    """What final prints of its results: each method at each CV threshold on the test sessions, then importance."""
    lines = []
    for method, result in metrics["methods"].items():
        timing = ", ".join(f"{ms:.3f} ms" + (" on one core" if key == "ms_n_jobs_1" else "")
                           for key, ms in result["timing"].items())
        lines.append(f"\n{method} {json.dumps(result['params'], ensure_ascii=False)}: PR-AUC "
                     f"{result['cv']['pr_auc']:.3f} on CV, {result['test']['pr_auc']:.3f} on test; {timing} per "
                     "prediction")
        for p in result["test"]["operating_points"]:
            lines.append(f"  at {p['fa_target']:g} FA/min (threshold {p['threshold']:.4g}, from CV): "
                         f"{with_intervals(p)}; {p['fa_per_min']:.2f} FA/min, alert precision "
                         f"{p['alert_precision']:.2f}, time in alarm {p['time_in_alarm']:.1%}, median lead "
                         f"{p['median_lead']:.2f} s")
            for kind, k in p["by_kind"].items():
                few = "" if k["enough"] else f"; under {metrics['min_kind_spins']} spins, so for the limitations"
                lines.append(f"    {kind}: {with_intervals(k)}{few}")
    ranked = metrics["importance"]
    lines.append(f"\npermutation importance of {ranked['method']} (the models' best CV PR-AUC) on the scored test "
                 f"frames: the fall in average precision over {ranked['n_repeats']} shuffles of each feature")
    lines += [f"  {f['feature']}: {f['mean']:.4f} ± {f['std']:.4f}" for f in ranked["features"]]
    return lines


def final(horizon, force, note, error):
    """run.py final: every method's chosen run, its thresholds from CV, scored once on the test sessions.

    Everything the dev set decides comes first: the chosen runs, checked
    against the dataset, their thresholds, each model refit on every dev
    session, and which model is best, by CV PR-AUC. Then the lock, and only
    then the test sessions, labelled with today's rule at horizon. error is
    the final parser's: it stops the run before the lock, so a refused run
    never loads them.
    """
    lock, out = lock_path(horizon), metrics_path(horizon)
    # The results count as a lock too: final.lock reads like a stale lockfile,
    # safe to delete, and without it a rerun would go unforced and unlogged.
    forced = lock.exists() or out.exists()
    again = "They're scored once; --force scores them again and logs it in experiments.md"
    if forced and not force:
        if lock.exists():
            done = f"its results are in {out}" if out.exists() else f"it never wrote {out.name}, so it stopped partway"
            error(f"{lock} exists: a final run loaded the test sessions at {_started(lock)}, and {done}. {again}")
        error(f"{out} exists, though {lock.name} doesn't: a final run loaded the test sessions at {_started(out)}. "
              f"{again}")
    previous = _started(lock if lock.exists() else out) if forced else None
    try:
        chosen = read_chosen(horizon)
    except ValueError as e:
        error(str(e))
    dataset = _load_for(horizon, error)
    try:
        oofs = {m: chosen_run(m, entry, dataset, horizon) for m, entry in chosen.items()}
    except ValueError as e:
        error(str(e))

    print(f"final run: thresholds from each method's chosen CV run at {_scored_on(dataset, horizon)}; models "
          "refit on all of them")
    from_cv = {m: cv_results(oofs[m]) for m in chosen}
    fitted = {m: refit(m, entry["params"], dataset) for m, entry in chosen.items()}
    best = max(models.MODELS, key=lambda m: from_cv[m]["pr_auc"])   # the first of equals

    started = datetime.now()
    write_lock(lock, started, horizon, forced)
    if forced:
        append_experiment(forced_row(started, horizon, previous, note))
    loaded = load_test_sessions(confirm=True)
    # Labelled here, with today's rule: 233636's spins were counted before the reverse fix.
    test = pd.concat([session_rows(s, df, horizon) for s, df in loaded.items()], ignore_index=True)
    kinds = {s: session_kind(s) or "unknown" for s in loaded}
    print(f"\ntest sessions, labelled at horizon {horizon:g} s with today's rule:")
    print("\n".join(describe(test)))
    print("kinds: " + ", ".join(f"{s} {kind}" for s, kind in kinds.items()))

    results = {}
    for method, entry in chosen.items():
        scores = method_scores(method, entry["params"], fitted[method], test)
        results[method] = {
            "run_id": entry["run_id"], "params": entry["params"], "chosen_by": entry["chosen_by"],
            "cv": from_cv[method],
            "test": {"pr_auc": pr_auc(test, scores), "operating_points": [
                {"fa_target": p["fa_target"], **operating_point(test, scores, p["threshold"], kinds)}
                for p in from_cv[method]["operating_points"]
            ]},
            "timing": time_per_prediction(method, entry["params"], fitted[method], test),
        }
    metrics = {
        "started": started.isoformat(timespec="seconds"), "horizon": horizon, "forced": forced,
        "min_lead_s": MIN_LEAD_S, "min_kind_spins": MIN_KIND_SPINS, "built_with": dataset.attrs["built_with"],
        "dev": {"sessions": dataset.attrs["sessions"], **_counts(dataset)},
        "test": {"sessions": {s: {"kind": kinds[s], **_counts(test[test["session"] == s])} for s in loaded},
                 **_counts(test)},
        "methods": results,
        "importance": {"method": best, "chosen_by": "CV PR-AUC, best of the models", "scoring": "average_precision",
                       "n_repeats": N_PERMUTATIONS, "features": importance(fitted[best], test)},
    }
    write_json(metrics, out)

    print("\n".join(final_report(metrics)))
    print(f"\nwrote {out}; {lock} stops it running again")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.run",
        description="Build the dataset, lock the test sessions, run CV and tune, and score the test sessions once.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="label every dev session, compute its features and write "
                                              "data/processed/dataset.parquet")
    build.add_argument("--horizon", type=parse_horizon, default=HORIZON_S,
                       help=f"seconds ahead a spin makes a frame positive (default {HORIZON_S:g}); another "
                            "horizon gets a file of its own, e.g. dataset_h1.parquet for 1.0")
    lock = commands.add_parser("lock-test", help="set the test sessions aside in data/splits.json, "
                                                 "once, before any modelling")
    lock.add_argument("--sessions", type=parse_sessions, required=True,
                      help="the test session ids, comma-separated, e.g. 2026-10-11_194646,2026-10-11_202102")
    cv_parser = commands.add_parser("cv", help="score a method out of fold over the dev sessions, save the "
                                               "scores in results/cv/ and log the run in experiments.md")
    cv_parser.add_argument("--method", choices=cv.METHODS, required=True)
    cv_parser.add_argument("--params", type=parse_params, default={},
                           help="""the method's params as a JSON object, e.g. '{"theta1": 50}' for b2 or """
                                """'{"C": 0.1}' for lr; b2 without them picks the θ1 with the best CV recall """
                                f"at 1 FA/min with at least {MIN_LEAD_S:g} s of lead; a model without them "
                                "takes scikit-learn's defaults")
    cv_parser.add_argument("--horizon", type=parse_horizon, default=HORIZON_S,
                           help=f"the dataset's horizon, as built (default {HORIZON_S:g})")
    cv_parser.add_argument("--note", default="", help="one line for experiments.md")
    tune_parser = commands.add_parser("tune", help="CV-run a method's configs, each saved and logged as cv "
                                                   "does, and record the best in results/chosen.json")
    tune_parser.add_argument("--method", choices=cv.METHODS, required=True)
    tune_parser.add_argument("--n", type=parse_n, default=N_TUNE,
                             help=f"configs a model tries, sampled from its grid with a fixed seed (default "
                                  f"{N_TUNE}; the whole grid if it's smaller); a baseline runs once")
    tune_parser.add_argument("--horizon", type=parse_horizon, default=HORIZON_S,
                             help=f"the dataset's horizon, as built (default {HORIZON_S:g}); another horizon "
                                  "records into a file of its own, e.g. chosen_h1.json for 1.0")
    tune_parser.add_argument("--note", default="", help="one line for every run's row in experiments.md")
    final_parser = commands.add_parser("final", help="score every method's chosen run on the test sessions, "
                                                     "once, into results/metrics.json")
    final_parser.add_argument("--horizon", type=parse_horizon, default=HORIZON_S,
                              help=f"the chosen runs' horizon (default {HORIZON_S:g}); another horizon reads its "
                                   "own chosen runs and writes its own files, e.g. chosen_h1.json, metrics_h1.json "
                                   "and final_h1.lock for 1.0")
    final_parser.add_argument("--force", action="store_true",
                              help="score the test sessions again although results/final.lock or metrics.json "
                                   "says they were; logged in experiments.md")
    final_parser.add_argument("--note", default="", help="why, for experiments.md's row when --force runs it again")
    args = parser.parse_args(argv)

    if args.command == "build":
        sessions = dev_sessions()
        if not sessions:
            parser.error(f"no sessions in {RAW_DIR}")
        dataset = build_dataset(sessions, args.horizon)
        path = dataset_path(args.horizon)
        write_parquet(dataset, path)
        if not SPLITS_PATH.exists():
            print("No test sessions locked yet, so every session is in. Pick 2 with typical spin counts, "
                  "lock them with lock-test, then build again.\n")
        print("\n".join(describe(dataset)))
        print(f"\nwrote {path}: {len(dataset)} frames, {len(FEATURES)} features, "
              f"horizon {args.horizon:g} s")
    elif args.command == "lock-test":
        try:
            lock_test(args.sessions)
        except ValueError as error:
            lock.error(str(error))
        print(f"locked {', '.join(sorted(args.sessions))} as the test sessions in {SPLITS_PATH}.\n"
              "Commit it, then run `python -m spinwarn.run build` again to leave them out of the dataset.")
    elif args.command == "cv":
        cross_validate(args.method, args.params, args.horizon, args.note, cv_parser.error)
    elif args.command == "tune":
        tune(args.method, args.n, args.horizon, args.note, tune_parser.error)
    else:
        final(args.horizon, args.force, args.note, final_parser.error)


if __name__ == "__main__":
    main()
