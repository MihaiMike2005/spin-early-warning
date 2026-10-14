"""
The command line: build, lock-test, cv, tune, final.

    python -m spinwarn.run build
    python -m spinwarn.run lock-test --sessions A,B

build labels every dev session, computes its features and writes them all
into one table, data/processed/dataset.parquet, one row per grid frame. It
prints minutes, spins and positive frames per session: the numbers to pick
the test sessions by.

lock-test sets the test sessions aside in data/splits.json, once, before
any modelling: load_dataset, which modelling reads the table through,
refuses it until then. From then on everything but the final run leaves
them out, so build again after it.
"""

import argparse
import json

import numpy as np
import pandas as pd

from spinwarn import data
from spinwarn.config import (
    DATASET_PATH, GRID_HZ, HORIZON_S, MIN_SPEED_KMH, N_FOLDS, RAW_DIR, REARM_DEG, REARM_S,
    REVERSE_GEAR, SPIN_HOLD_S, SPLITS_PATH, WINDOW_S,
)
from spinwarn.data import dev_sessions, list_sessions, load_session
from spinwarn.features import FEATURES, make_features
from spinwarn.labeling import find_onsets, label


# The signals the baselines score with, kept as they are: B1 abs_slip, B2
# abs_slip and abs_slip_rate, B3 time_to_90.
BASELINE_SIGNALS = ["abs_slip", "abs_slip_rate", "time_to_90"]
# onset marks each spin's first frame, so alerts can be scored from the dataset alone.
COLUMNS = ["session", "segment", "t", "target", "excluded", "onset", *BASELINE_SIGNALS, *FEATURES]


def built_with() -> dict:
    """The constants the dataset depends on: the session caches', plus the labels' and features'."""
    return {
        **data.built_with(),
        "MIN_SPEED_KMH": MIN_SPEED_KMH, "REARM_DEG": REARM_DEG, "REARM_S": REARM_S,
        "SPIN_HOLD_S": SPIN_HOLD_S, "REVERSE_GEAR": REVERSE_GEAR, "HORIZON_S": HORIZON_S,
        "WINDOW_S": WINDOW_S,
    }


def session_rows(session_id, df) -> pd.DataFrame:
    """The dataset's rows for a grid session from load_session: one per frame, in COLUMNS order."""
    onsets, _spinning = find_onsets(df)
    rows = pd.concat([label(df), make_features(df)], axis=1)
    rows["session"] = session_id
    # Grid times are unique within a session, and the onsets are some of them, exactly.
    rows["onset"] = np.isin(df["t"].to_numpy(), onsets)
    return rows[COLUMNS].reset_index(drop=True)


def build_dataset(sessions) -> pd.DataFrame:
    """The sessions loaded, labelled and with features, in one table that remembers what built it."""
    dataset = pd.concat([session_rows(s, load_session(s)) for s in sessions], ignore_index=True)
    dataset.attrs = {"sessions": list(sessions), "built_with": built_with()}
    return dataset


def write_dataset(dataset):
    """dataset to DATASET_PATH, written aside and renamed into place, as load_session does its cache."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = DATASET_PATH.with_name(f"{DATASET_PATH.name}.partial")
    dataset.to_parquet(partial, index=False)
    partial.replace(DATASET_PATH)


def load_dataset() -> pd.DataFrame:
    """The table build wrote, for modelling: refused before the lock, or if it's out of date.

    Before the lock, nothing is set aside to test on, and the Project Plan
    locks the test sessions before any modelling. build still works then,
    for the spin counts to pick them by.

    Out of date is built from other sessions than today's dev sessions (one
    copied in, or the test sessions locked, since) or with other constants.
    Built before the lock, it still holds the test sessions, and CV would
    train on them. Building again takes seconds.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"no {DATASET_PATH}: run `python -m spinwarn.run build` first")
    if not SPLITS_PATH.exists():
        raise RuntimeError(f"no test sessions locked in {SPLITS_PATH}: lock them with "
                           "`python -m spinwarn.run lock-test` before any modelling, then build again")
    dataset = pd.read_parquet(DATASET_PATH)
    if dataset.attrs.get("sessions") != dev_sessions():
        raise RuntimeError("the dataset was built from other sessions than today's dev sessions: "
                           "run `python -m spinwarn.run build` again")
    if dataset.attrs.get("built_with") != built_with():
        raise RuntimeError("config.py has changed since the dataset was built: "
                           "run `python -m spinwarn.run build` again")
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.run",
        description="Build the dataset, lock the test sessions, and (later steps) run CV, tuning and the final test.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="label every dev session, compute its features and write "
                                      "data/processed/dataset.parquet")
    lock = commands.add_parser("lock-test", help="set the test sessions aside in data/splits.json, "
                                                 "once, before any modelling")
    lock.add_argument("--sessions", type=parse_sessions, required=True,
                      help="the test session ids, comma-separated, e.g. 2026-10-11_194646,2026-10-11_202102")
    args = parser.parse_args(argv)

    if args.command == "build":
        sessions = dev_sessions()
        if not sessions:
            parser.error(f"no sessions in {RAW_DIR}")
        dataset = build_dataset(sessions)
        write_dataset(dataset)
        if not SPLITS_PATH.exists():
            print("No test sessions locked yet, so every session is in. Pick 2 with typical spin counts, "
                  "lock them with lock-test, then build again.\n")
        print("\n".join(describe(dataset)))
        print(f"\nwrote {DATASET_PATH}: {len(dataset)} frames, {len(FEATURES)} features")
    else:
        try:
            lock_test(args.sessions)
        except ValueError as error:
            lock.error(str(error))
        print(f"locked {', '.join(sorted(args.sessions))} as the test sessions in {SPLITS_PATH}.\n"
              "Commit it, then run `python -m spinwarn.run build` again to leave them out of the dataset.")


if __name__ == "__main__":
    main()
