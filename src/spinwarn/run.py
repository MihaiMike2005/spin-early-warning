"""
The command line: build, lock-test, cv, tune, final.

    python -m spinwarn.run build [--horizon 1.0]
    python -m spinwarn.run lock-test --sessions A,B
    python -m spinwarn.run cv --method b1|b2|b3 [--params '{...}'] [--horizon 0.5] [--note "..."]

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
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from spinwarn import baselines, cv, data
from spinwarn.config import (
    CV_DIR, DATASET_PATH, EXPERIMENTS_PATH, GRID_HZ, HORIZON_S, MIN_LEAD_S, MIN_SPEED_KMH, N_FOLDS,
    RAW_DIR, REARM_DEG, REARM_S, REVERSE_GEAR, SPIN_HOLD_S, SPLITS_PATH, WINDOW_S,
)
from spinwarn.data import dev_sessions, list_sessions, load_session
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


def dataset_path(horizon) -> Path:
    """Where build writes the dataset labelled at horizon: DATASET_PATH at HORIZON_S, else beside it."""
    if horizon == HORIZON_S:
        return DATASET_PATH
    return DATASET_PATH.with_name(f"{DATASET_PATH.stem}_h{horizon:g}{DATASET_PATH.suffix}")


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
        lines.append(f"at {p['fa_target']:g} FA/min (threshold {p['threshold']:.4g}): "
                     f"recall with ≥ {MIN_LEAD_S:g} s lead {p['recall_min_lead']:.2f} "
                     f"({p['caught_min_lead']}/{p['n_spins']}), "
                     f"any lead {p['recall']:.2f} ({p['caught']}/{p['n_spins']}), {p['fa_per_min']:.2f} FA/min, "
                     f"alert precision {p['alert_precision']:.2f}, time in alarm {p['time_in_alarm']:.1%}, "
                     f"median lead {p['median_lead']:.2f} s")
    return lines


def cross_validate(method, params, horizon, note, error):
    """run.py cv: out-of-fold scores of method on the dataset at horizon, printed, saved and logged.

    error is the cv parser's, for a mistake on the command line.
    """
    # b2 with no params picks its θ1 first; anything else has to be complete.
    sweep = method == "b2" and not params
    if not sweep:
        try:
            baselines.check_params(method, params)
        except ValueError as e:
            error(str(e))
    try:
        dataset = load_dataset(horizon)
    except (FileNotFoundError, RuntimeError) as e:
        error(str(e))

    if sweep:
        theta1, table = cv.choose_theta1(dataset)
        print(f"B2's θ1, by recall with ≥ {MIN_LEAD_S:g} s lead at 1 FA/min (ties by any-lead recall, "
              "then PR-AUC):")
        for row in table.itertuples():
            print(f"  θ1 {row.theta1}°: recall with ≥ {MIN_LEAD_S:g} s lead {row.recall_min_lead:.2f} "
                  f"({row.caught_min_lead}/{row.n_spins}), any lead {row.recall:.2f} ({row.caught}/{row.n_spins}), "
                  f"{row.fa_per_min:.2f} FA/min, PR-AUC {row.pr_auc:.3f}")
        print(f"θ1 = {theta1}°\n")
        params = {"theta1": theta1}
        swept = (f"θ1 best of {table['theta1'].iloc[0]}–{table['theta1'].iloc[-1]}° "
                 f"by recall @ 1 FA/min, lead ≥ {MIN_LEAD_S:g} s")
        note = "; ".join(filter(None, [swept, note]))

    scores = cv.oof_scores(dataset, method, params)
    now = datetime.now()
    run = {
        "run_id": new_run_id(method, horizon, now), "date": f"{now:%Y-%m-%d}", "method": method,
        "params": params, "horizon": horizon, "note": note, "metrics": cv.metrics(dataset, scores),
    }
    path = CV_DIR / f"{run['run_id']}.parquet"
    write_parquet(oof_table(dataset, scores, run), path)
    append_experiment(experiment_row(run))

    at_1 = run["metrics"]["operating_points"][0]
    print(f"{method} {json.dumps(params)} at horizon {horizon:g} s, {N_FOLDS}-fold CV over "
          f"{len(dataset.attrs['sessions'])} dev sessions: {at_1['n_spins']} spins in "
          f"{at_1['minutes']:.1f} scored min")
    print("\n".join(report(run["metrics"])))
    print(f"\nwrote {path} and a row in {EXPERIMENTS_PATH.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.run",
        description="Build the dataset, lock the test sessions, run CV, and (later steps) tuning and the final test.",
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
                           help="""the method's params as a JSON object, e.g. '{"theta1": 50}' for b2; """
                                "b2 without them picks the θ1 with the best CV recall at 1 FA/min "
                                f"with at least {MIN_LEAD_S:g} s of lead")
    cv_parser.add_argument("--horizon", type=parse_horizon, default=HORIZON_S,
                           help=f"the dataset's horizon, as built (default {HORIZON_S:g})")
    cv_parser.add_argument("--note", default="", help="one line for experiments.md")
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
    else:
        cross_validate(args.method, args.params, args.horizon, args.note, cv_parser.error)


if __name__ == "__main__":
    main()
