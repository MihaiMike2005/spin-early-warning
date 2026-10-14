# spin-early-warning

Warns of a spin 0.5 s before it starts, from Assetto Corsa telemetry
recorded with [DriftScore](https://github.com/MihaiMike2005/DriftScoreUI)'s
logger, and compares the model with tuned physics baselines at the same
false-alarm rate. Work in progress: the plans are in `docs/`.

## Setup

Python 3.14, since spinwarn installs driftscore, which requires it. From the
repo root:

```
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install -e "../AssetoCorsa Drift Score"
```

The first install pulls driftscore from GitHub, as CI does. The second swaps
it for the DriftScore checkout next to this repo, so edits there show up
here. Keep that order: `pip install -e '.[dev]'` replaces a local driftscore
with the GitHub one, so after rerunning it, rerun the second line too.

## Test

```
pytest
```

## Data

Raw recordings go in `data/raw/`, which git ignores: the logger's CSV and
JSON for each session, copied from the AC PC. `notes.md` is the session log.
The screen recordings stay outside the repo.

```
python -m spinwarn.data [--rebuild]
```

resamples every dev session onto the 60 Hz grid, adds the derived signals
and caches it in `data/processed/`. Each cached session remembers the
constants that built it and is rebuilt when `config.py` changes them.
`--rebuild` forces it, for what the cache can't see: a change to the code,
or a raw CSV copied over again.

```
python -m spinwarn.labeling
```

prints how many spins the label rule finds in each dev session.

```
python -m spinwarn.spotcheck <session> --n 20 --video-offset 3.2
```

lists seeded-random spin onsets of a dev session as times in its screen
recording, each with its peak |slip| and speed, as a table to paste into
`notes.md` and mark hit or miss on video. The offset is the video time at
which the logger's status line shows t = 0.

`notebooks/01_explore.ipynb` shows each dev session: the raw rate and gaps,
slip angle with the onsets marked, zooms on three onsets, where the drifts
sit relative to 90°, the countersteer sign check behind `COUNTERSTEER_SIGN`,
and every spin with the spot-check commands.

```
python -m spinwarn.run build
```

labels every dev session, computes its features and writes them into one
table, `data/processed/dataset.parquet`, one row per grid frame, then prints
minutes, spins and positive frames per session. Nothing models on it until
the test sessions are locked. A table built from other sessions or other
constants is refused, so build again after copying in a session, locking
the test sessions or changing `config.py`. It can't see a change to the
code, so build again after `python -m spinwarn.data --rebuild` too.

```
python -m spinwarn.run lock-test --sessions A,B
```

sets the test sessions aside in `data/splits.json`, once, before any
modelling: pick 2 with typical spin counts from build's output, commit the
file, and build again. It refuses if the file exists, or if fewer than
`N_FOLDS` sessions would be left for cross-validation.

## Layout

| Path | What it is |
| --- | --- |
| `docs/` | The Project Plan and the Build Plan |
| `data/raw/` | Logger CSV + JSON per session (gitignored) |
| `data/processed/` | Sessions on the 60 Hz grid with derived signals, and the dataset (gitignored) |
| `data/splits.json` | The locked test sessions (committed) |
| `src/spinwarn/config.py` | Every constant, in seconds |
| `src/spinwarn/` | The pipeline: load, label, features, models, evaluation |
| `notebooks/01_explore.ipynb` | Rate and gaps, slip with onsets, the countersteer sign, every spin |
| `notes.md` | Session log: car, track, setup, minutes, spins, kind, video |
| `experiments.md` | One row per CV run |
