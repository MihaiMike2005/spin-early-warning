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

## Layout

| Path | What it is |
| --- | --- |
| `docs/` | The Project Plan and the Build Plan |
| `data/raw/` | Logger CSV + JSON per session (gitignored) |
| `src/spinwarn/config.py` | Every constant, in seconds |
| `src/spinwarn/` | The pipeline: load, label, features, models, evaluation |
| `notes.md` | Session log: car, track, setup, minutes, spins, kind, video |
| `experiments.md` | One row per CV run |
