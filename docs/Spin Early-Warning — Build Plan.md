# Spin Early-Warning — Build Plan

Oct 10, 2026 · @Mihai · companion to *Spin Early-Warning Model — Project Plan*

The Project Plan says what to build and why. This file says what code to write, in what order, so Claude Code can build it one step at a time. Where the two differ on implementation details, this file wins.

## How to use this with Claude Code

- **One step per prompt.** Open Claude Code in the repo the step names and send:

  > Read `Spin Early-Warning Model — Project Plan.md` and `Spin Early-Warning — Build Plan.md`. Implement Step N only. Stop when its "Done when" passes and show me the test output.

- **After each step:** read the diff, run `pytest`, commit. In DriftScore, work on a branch and open one PR per phase so CI runs.
- **Steps marked (you)** are driving or watching video. Claude Code can't do them.
- At Step 5, copy both plan files into `spin-early-warning/docs/` so sessions in that repo can read them.

## Ground rules (both repos)

- **Pure functions apart from I/O.** I/O loops stay thin and untested, like `app.py`; everything else gets a test.
- **One home for constants.** Thresholds, windows and horizons live in module constants (DriftScore) or `config.py` (spinwarn), never inline.
- **Seconds, never frames,** in every parameter and constant.
- **DriftScore stays stdlib-only, on Python 3.14.** Both machines run 3.14 (the AC PC 3.14.6), and so does CI. No new dependencies.
- **spinwarn:** Python 3.14, since it installs driftscore. Dependencies are limited to pandas, numpy, scikit-learn, matplotlib, pyarrow, pytest, jupyter and driftscore.
- **Fixed seeds:** `SEED = 0` everywhere.
- **Test sessions** are loaded only through the final-run path (Step 14).
- **Comments** follow the existing style: explain why, briefly.

## Overview

| Step | Repo | What | Plan day |
| --- | --- | --- | --- |
| 1 | DriftScore | Extend telemetry structs | Sun |
| 2 | DriftScore | Session logger | Sun |
| 3 | DriftScore | Log check | Sun |
| 4 | (you) AC PC | Live check, first recordings | Sun |
| 5 | spin-early-warning | Scaffold, config, CI | Sun/Mon |
| 6 | spin-early-warning | Load sessions, 60 Hz grid, derived signals | Mon |
| 7 | spin-early-warning | Labels | Mon |
| 8 | spin-early-warning + (you) | Explore notebook, spot-check | Mon |
| 9 | spin-early-warning | Features + causality test | Tue |
| 10 | spin-early-warning | Dataset build, lock test sessions | Tue |
| 11 | spin-early-warning | Alert scoring | Wed |
| 12 | spin-early-warning | Baselines + CV harness | Wed |
| 13 | spin-early-warning | Models + tuning | Wed/Thu |
| 14 | spin-early-warning | Final test run | Thu |
| 15 | spin-early-warning | Report + README | Fri |

## Decisions that differ from the Project Plan

- **The logger writes CSV, not Parquet.** DriftScore stays dependency-free, and a CSV flushed every second survives a crash. spinwarn caches to Parquet.
- **The logger records all three axes** of angular velocity and accG. Which axis is yaw or lateral gets decided offline, so a wrong guess never costs a re-recording.
- **`logcheck` (Step 3)** is new: it validates the struct on the AC PC right after a session.
- **Raw packets are resampled to a uniform 60 Hz grid** (Step 6), whatever rate AC actually delivers. Thinning to 20 Hz for training then means every 3rd grid frame.
- **The ML code is a package, `spinwarn`,** under `src/`, in the same layout as DriftScore, instead of loose modules.
- **Python 3.14, not "3.11+".** Both machines run 3.14 (the AC PC 3.14.6), so DriftScore requires it and CI tests only it; spinwarn follows, since it installs driftscore. The Project Plan's reason for 3.11+ still holds: from 3.11, `time.sleep` on Windows uses a high-resolution timer, which the logger's 1 ms poll needs.

## Phase A — DriftScore: telemetry and logger

Branch `feature/session-logger`, one PR at the end of Step 3.

### Step 1 — Extend the telemetry structs

Files: `src/driftscore/telemetry.py`, `tests/test_telemetry.py`

- **`ACPhysics`:** append the fields after `heading`, up to `localAngularVel`, in AC's documented order: `pitch`, `roll`, `cgHeight`, `carDamage[5]`, `numberOfTyresOut` (int), `pitLimiterOn` (int), `abs`, `kersCharge`, `kersInput`, `autoShifterOn` (int), `rideHeight[2]`, `turboBoost`, `ballast`, `airDensity`, `airTemp`, `roadTemp`, `localAngularVel[3]`. The rest are floats. Update the "we stop at heading" comment.
- **`_layout_ = "ms"` on every struct.** Python 3.14 warns that `_pack_` without it is deprecated (an error from 3.19).
- **`ACGraphicsPrefix`** on `acpmf_graphics`: `packetId`, `status`, `session` (all int). Add constants `AC_OFF, AC_REPLAY, AC_LIVE, AC_PAUSE = 0, 1, 2, 3`.
- **`ACStaticPrefix`** on `acpmf_static`: `smVersion[15]`, `acVersion[15]`, `numberOfSessions` (int), `numCars` (int), `carModel[33]`, `track[33]`.
  - Use `ctypes.c_uint16` arrays for the strings, not `c_wchar`. Windows `wchar_t` is 2 bytes, but `c_wchar` is 4 bytes on macOS and Linux, so `c_wchar` would make the layout, and the tests, depend on the OS.
- **`wchars_to_str(arr) -> str`:** decode UTF-16-LE and stop at the first NUL.
- **Generic page access:** `open_page(name, struct)` and `read_page(mm, struct)`. Keep `open_physics` and `read_physics` as thin wrappers so `app.py` keeps working unchanged.

Tests (any OS):

- **Offsets:** `heading` = 208 (unchanged; a regression guard), `localAngularVel` = 296, static `carModel` = 68, `track` = 134. These come from AC's documented layout; Step 4 confirms them on live data. If it doesn't, fix the struct and these numbers together.
- **Round-trip:** `struct.pack_into` a known float at byte 300, then `ACPhysics.from_buffer_copy` reads it back as `localAngularVel[1]`.
- **Decoding:** `wchars_to_str` on "ks_toyota_ae86", followed by a NUL and junk, returns "ks_toyota_ae86".

**Done when:** tests pass locally on 3.14 with no deprecation warning; `python -m driftscore.app` is unchanged.

### Step 2 — Session logger

Files: `src/driftscore/logger.py`, `tests/test_logger.py`, README section

```text
python -m driftscore.logger --kind natural|deliberate [--setup "TC off, 2nd gear"] [--out data/raw] [--poll-ms 1]
```

**Columns** (constant `COLUMNS`, in order):
- `t`, `packet_id`
- `speed_kmh`, `vel_x`, `vel_y`, `vel_z`, `heading`
- `ang_vel_x`, `ang_vel_y`, `ang_vel_z`
- `acc_g_x`, `acc_g_y`, `acc_g_z`
- `steer`, `gas`, `brake`, `gear`, `rpms`
- `wheel_slip_fl`, `wheel_slip_fr`, `wheel_slip_rl`, `wheel_slip_rr`

**Pure parts (tested):**
- `physics_row(phys, t) -> list`: one row in `COLUMNS` order.
- `should_write(packet_id, last_packet_id, status) -> bool`: true only for a new packet while status is `AC_LIVE`.
- `session_paths(out_dir, started) -> (csv_path, json_path)`, named `YYYY-MM-DD_HHMMSS.csv` and `.json`.
- `track_break(wrote, status, rows, left_live) -> (row_is_break, left_live)`: marks the first row after AC left live (pause, replay), so the summary can leave that interval out.
- `session_summary(ts, packet_ids, breaks) -> dict`: rows, live duration, mean rows/s, packetId steps, longest gap, breaks, all with the break intervals left out. Step 3 adds `statuses` and the break times.
- `status_name(status) -> str`: the `ACStatus` name, or the raw int if a live read is out of range.
- `status_line(t, packet_id, rows, rate, status, speed_kmh, slip_deg) -> str`: the live console line, under 80 characters so a narrow window won't wrap it.
- `parse_poll_ms(text) -> float`: the `--poll-ms` type, 0.1 to 1000, so a typo stops the logger before it creates any file.

**Loop (untested, like `app.py`):**

- **Start:** `t0 = time.perf_counter()`. Read the static page once and write the JSON: car, track, AC and shared-memory versions, date, start time, kind, setup, poll interval, Python version, platform.
- **Each poll:** read graphics status and physics, write the row if `should_write`, and note with `track_break` whether it is the first row after a pause or replay. Flush once a second.
- **Status line, about every 0.25 s:** elapsed `t`, packetId, rows written, rows/s over the last second, status name, speed, slip angle. With the console in the screen recording, this is what maps video time to telemetry time.
- **Ctrl+C:** close the CSV and rewrite the JSON with a summary: rows, live duration, mean rows/s, counts of packetId steps (1, 2, 3+, reset), longest gap, breaks.
- **Slip angle** is computed for the status line only, not written to the file.

Tests:
- `physics_row` on a hand-built `ACPhysics`: right column count and order, values carried through.
- `should_write` for a duplicate id, paused, replay, and a new live packet.
- `session_paths` naming.
- `track_break` for a live row, a duplicate packet, pause/replay/off with and without earlier rows, and the first row after a resume.
- `session_summary` leaving break intervals out of every number.
- `status_name` for a known status and an out-of-range int.
- `status_line` under 80 characters with the widest realistic values.
- `parse_poll_ms` accepting 0.1–1000 and refusing 0, below 0.1, negatives, nan, inf, too large and non-numbers; `main` exiting on a bad value before it connects.

README: add a "Recording a session" section with the routine. **Before:** car on track, start the recording, start the logger. **During:** drive 10–20 min. **After:** Ctrl+C, write the `notes.md` line, copy the files to the Mac.

**Done when:** tests pass in CI.

### Step 3 — Log check

Files: `src/driftscore/logcheck.py`, `tests/test_logcheck.py`, plus a logger change (`logger.py`, `tests/test_logger.py`) and README

```text
python -m driftscore.logcheck data/raw/2026-10-11_143000.csv
```

Stdlib only, so it runs on the AC PC right after a session and catches a broken struct before you record more. It reads the CSV and, if there is one, the `.json` of the same name.

The first real session (`2026-10-11_120846`) changed three checks from the first draft. The car slides most of the time, so speed × yaw rate and d(speed)/dt gave r 0.78 and 0.40 for the lateral and longitudinal axes. Front wheel slip spikes to ~5e7 at big angles, which swamps a mean.

**Logger change:** the JSON summary keeps `breaks`, the count, and adds `break_intervals`. Each entry has `start_s` and `end_s`, the `t` of the rows either side, rounded like the CSV's `t` (one constant, `T_DECIMALS`) so logcheck matches them exactly. It also has `statuses`, the non-live AC statuses seen in between, by name, each once, in the order first seen: any of `PAUSE`, `REPLAY` and `OFF` (or the raw number, if a read is out of range), e.g. `["PAUSE"]` or `["PAUSE", "REPLAY"]`. `break_statuses(seen, status, left_live)` collects them; `session_summary(ts, packet_ids, breaks, statuses)` pairs them with the breaks strictly, so a mismatch raises instead of dropping a break.

It prints:

- **Rate:** rows/s, AC's packet rate (packetId change ÷ time, over intervals of 0.1 s or less), the histogram of packetId steps and the share of steps of 2+.
- **Gaps and breaks:** every gap over 0.1 s and every break the JSON lists, however short, with its times and a label.
  - A listed break is labelled with its statuses.
  - Otherwise, a gap is a pause if packetId moved less than half of what the rate predicts (AC freezes packetId while paused), and a logging gap if not.
  - Without the JSON, or with an older one that has no `break_intervals`, the packetId test decides alone.
  - Where the JSON and the packetId test disagree, a note says so, both ways: a listed break where packetId kept counting, and an unlisted gap where it froze ("AC stalled"). Neither is a CHECK reason.
- **Yaw axis:** correlation of each `ang_vel` axis with d(unwrapped `heading`)/dt. Prints the best axis, r and sign.
- **Lateral and longitudinal axes:** correlation of each `acc_g` axis with d(velocity)/dt turned into the car frame by heading h, ÷ 9.81: nose = (−sin h, cos h), side = (cos h, sin h) in x/z.
- **Steering range:** min and max of `steer`.
- **Wheel order:** median wheel slip per wheel while `gas` > 0.8, |slip angle| > 20° and speed ≥ `SLIP_SPEED_FLOOR_KMH` (5 km/h; below it the slip angle is noise). The two rear wheels should be highest.
- **Verdict:** `OK`, or `CHECK` with every reason.

Derivatives are central differences over `DERIV_SPAN_S` = 0.05 s, not adjacent rows: row-to-row on the poll time drops yaw r from 0.999 to 0.98. They never cross a gap over 0.1 s or a listed break. Listed breaks are left out of every number, as the logger leaves them out of its summary. Logging gaps are left out too, since they get their own report, but the logger keeps them: with a logging gap, the JSON counts one more step per gap than logcheck, and its rows/s is a little lower.

**CHECK when** (thresholds are module constants):
- the best yaw |r| is below 0.95, or the best lateral or longitudinal |r| is below 0.9;
- `steer` goes outside [−1.05, 1.05], or doesn't reach both −0.5 and +0.5;
- either rear median isn't above both front medians, or there's under 1 s of qualifying driving to judge from;
- there's any logging gap over 0.1 s;
- more than 10% of packetId steps are 2 or more (the poll is too slow to keep every packet).

**Damaged or missing files:** a last row cut off mid-write (the logger was stopped without Ctrl+C) is left out with a note, and isn't a CHECK reason. A JSON without a summary gets its own note, and so does a damaged one (empty, cut off or zero bytes, from a logger stopped while rewriting it): both fall back on the packetId test. A missing file, or anything but a `.csv` (the `.json`, or the bare stem tab completion stops at), stops with an argparse error.

**Pure parts (tested):** `unwrap`, `derivative`, `pearson`, `best_axis -> (index, r, sign)`, `car_frame`, `accel_refs`, `step_stats`, `break_rows`, gap classification (`classify_gap`, `find_gaps`), `wheel_medians`, `verdict`, `read_columns`, `read_meta`; in the logger, `break_statuses`. `pearson` is written by hand because `statistics.correlation` raises on a constant input; this returns 0, so a dead axis can't crash the check. Opening the files and printing stay in an untested `report()`, like `app.py`.

Tests on synthetic data:
- A sine-wave heading with `ang_vel_y = −d(heading)/dt + noise` gives axis 1, sign −1.
- Unwrapping across ±π leaves no spike in the derivative.
- Rotation: straight-ahead acceleration at heading 0 and at π/2 gives lon = a, lat = 0; along the side gives lat = a, lon = 0.
- The derivative never spans a gap or a listed break.
- Without break times, a 2 s gap with a packetId step of 1 is a pause, and a 0.2 s gap with a step of about 0.2 × rate is a logging gap. With them, a listed REPLAY where packetId kept counting is labelled REPLAY with a note, an unlisted frozen gap gets the "AC stalled" note, and a 23 ms break is still listed and left out of the step counts. A logging gap's step is left out, though the logger counts it.
- One huge front-wheel spike doesn't change the wheel order.
- `verdict` is OK when every check passes, and each failure gives its own reason.
- A cut last row is left out, wherever the cut falls. A damaged JSON reads as None. `main` refuses a missing file and anything but a `.csv`.
- Logger: `break_statuses` for pause, replay, pause then replay, and off; break times equal the CSV's `t` exactly; unpaired breaks and statuses raise.

README: logcheck in the Layout table, and as step 2 of the "After" routine: run it on the PC right after Ctrl+C, before copying the files. The JSON paragraph says each break is listed with its times and kind.

**On `2026-10-11_120846`** (recorded with the old 5 ms poll): AC rate 333.3/s; one 2.07 s pause and no logging gaps; yaw `ang_vel_y` sign −, lateral `acc_g_x` sign +, longitudinal `acc_g_z` sign +, all |r| ≥ 0.998; steer −0.98 to 1.00; rear medians 5.77 and 5.75 against fronts 0.48 and 0.46. Verdict CHECK for one reason: 84% of packetId steps are 2+, from the old poll.

**Done when:** tests pass; open the Phase A PR and get CI green.

### Step 4 — Live check (you, on the AC PC)

- Pull the branch, `pip install -e .` with Python 3.14.
- Log 2 minutes of mixed driving (straights, both drift directions, one spin), then run `logcheck`.
- Write down the rate, yaw axis and sign, lateral and longitudinal axes and signs, steering range and wheel order. They go into `config.py` in Step 5.
- **No axis above |r| = 0.9:** the struct is off. Paste the `logcheck` output to Claude Code to fix Step 1. The fallback is yaw rate from heading, which Step 6 supports.
- Merge the PR, then record sessions as the Project Plan says.

**Done when:** `logcheck` says OK, Phase A is merged, and 3 sessions are recorded.

**Results (2026-10-11):** done. `logcheck` said OK on all four recordings made with the 1 ms poll, PR #2 is merged, and 3 sessions are recorded.

- **Axes,** the same on all four recordings, on both tracks: AC rate 333.3/s; yaw `ang_vel_y`, r −1.000; lateral `acc_g_x`, r +0.977 to +0.999; longitudinal `acc_g_z`, r +0.974 to +0.999; steer −1.000 to 1.000; rear wheel slip medians about 5.6, fronts about 0.4.
- **For `config.py` in Step 5:** `YAW_AXIS, YAW_SIGN = 1, 1`, `LAT_G_AXIS, LAT_G_SIGN = 0, 1`, `LON_G_AXIS, LON_G_SIGN = 2, 1`.
  - `YAW_SIGN` is +1, not the "sign −" that `logcheck` prints. `logcheck` correlates against +d(heading)/dt, so its − means `ang_vel_y` = −d(heading)/dt, which is already Step 6's fallback. +1 keeps both ways of getting yaw rate the same sign.
- **Sessions:** the S14 (`hush_nissan_silvia_s14`) on `rmi_gpk_sepang` with TC off, the track for every session this week. They're in `/Users/mihaistefanita/Claude Projects/Drift Spin Project/driftsessions/`: CSV and JSON in `raw/`, screen recordings in `videos/`, and `notes.md`. Step 5 moves `raw/` into `data/raw/` and `notes.md` to the repo root, both only after `.gitignore` exists. The videos stay outside the repo.

  | Session | Kind | Live min | Spins (driver's count) |
  | --- | --- | --- | --- |
  | `2026-10-11_194646` | natural | 14.5 | ~40 |
  | `2026-10-11_200553` | deliberate; started as natural by mistake, JSON `kind` fixed by hand | 8.2 | 20+ |
  | `2026-10-11_202102` | natural | 10.0 | 30+ |

  That's about 2.7 spins a minute, well above the Project Plan's 0.5–1. Step 7's onset count gives the real number.
- **Not in the dataset:** the Brooklyn Park test runs. `120846` was recorded on the old 5 ms poll, `191720` is the live check, and `191522` and `191538` have 0 rows (AC status OFF). `191720` and its video are a candidate for `data/sample/`.
- **Video offset:** the logger's status line is readable in the screen recording. On `191720`, t = 0 fell at about 8.8 s into the video. Step 8 reads each session's offset the same way.
- **Notes, harmless:** "AC stalled" can follow a pause. In `194646` at 552.6 s, AC's physics hitched for 0.11 s right after a 0.7 s pause. That gap is over `GAP_S`, so Step 6 cuts there. A ~20 ms `PAUSE` right at the start of a session is common (`191720`, `200553`, `202102`): AC pauses itself whenever it loses focus, and the logger's PowerShell window takes it at the start and again at Ctrl+C. It falls in a segment's first `WINDOW_S`, which Step 7 excludes anyway.
- **Pauses are never mid-drift:** you pause only with the car off a drift. Every pause on Sunday came at 6 km/h or less, apart from 552 s in `194646`: rolling straight at 24 km/h in neutral, after which it came back stopped (a reset from the pause menu).

## Phase B — spin-early-warning (Mac)

### Step 5 — Scaffold

Location: `/Users/mihaistefanita/Claude Projects/Drift Spin Project/spin-early-warning`, next to the DriftScore folder and `driftsessions/`.

- `git init`, plus a GitHub repo `spin-early-warning`.
- **`pyproject.toml`** in DriftScore's style:
  - Package `spinwarn` under `src/`, `requires-python = ">=3.14"`: driftscore needs it.
  - Dependencies: pandas, numpy, scikit-learn, matplotlib, pyarrow, and `driftscore @ git+https://github.com/MihaiMike2005/DriftScoreUI.git`. Dev extras: pytest, jupyter.
  - **Locally,** run `pip install -e "../AssetoCorsa Drift Score"` after `pip install -e '.[dev]'`, so DriftScore edits show up. Not before: `pip install -e '.[dev]'` replaces a local driftscore with the GitHub one, so rerun the local install after it.
  - **In CI,** the git install only works while DriftScoreUI is public. It has to be public for the CV link anyway.
- **CI:** copy DriftScore's workflow, Python `["3.14"]`.
- **`.gitignore`:** DriftScore's entries plus `data/raw/`, `data/processed/`, `results/cv/`.

```text
spin-early-warning/
├── docs/                 both plan files
├── data/
│   ├── raw/              gitignored: logger CSV + JSON
│   ├── processed/        gitignored: cached tables
│   ├── sample/           one short session, committed
│   └── splits.json       committed: the locked test sessions
├── notebooks/01_explore.ipynb
├── results/              metrics.json, tables, figures (results/cv/ gitignored)
├── src/spinwarn/
│   ├── config.py         every constant
│   ├── data.py           load, 60 Hz grid, derived signals, splits
│   ├── labeling.py
│   ├── features.py
│   ├── evaluate.py       alerts and scoring
│   ├── baselines.py
│   ├── models.py
│   ├── cv.py
│   ├── run.py            CLI: build, lock-test, cv, tune, final
│   ├── spotcheck.py
│   └── report.py
├── tests/
├── notes.md
├── experiments.md
└── README.md
```

`config.py`:

```python
GRID_HZ = 60              # raw packets are resampled onto this uniform grid
GAP_S = 0.1               # a longer gap starts a new segment
MIN_SPEED_KMH = 20
SPIN_DEG, REARM_DEG, REARM_S = 90, 45, 2.0
SPIN_HOLD_S = 0.0         # raise to 0.25 if the spot-check flags intended high angles
HORIZON_S = 0.5           # 1.0 for the secondary run
WINDOW_S = 1.0
RATE_WINDOW_S = 0.1       # for d|slip|/dt
TTS_CAP_S = 5.0           # time to 90° when |slip| isn't growing
MERGE_GAP_S = 0.2
CREDIT_EARLY_S, CREDIT_LATE_S, FA_AFTER_S = 1.0, 0.1, 1.0
TRAIN_HZ = 20
FA_TARGETS = (1.0, 0.5)
N_FOLDS = 4
SEED = 0
REVERSE_GEAR = 0          # AC gear: 0 = R, 1 = N, 2 = 1st

# From Step 4's logcheck:
YAW_AXIS, YAW_SIGN = None, 1        # YAW_AXIS None -> yaw rate from heading
LAT_G_AXIS, LAT_G_SIGN = None, 1
LON_G_AXIS, LON_G_SIGN = None, 1
# From Step 8's notebook (held drifts should read positive):
COUNTERSTEER_SIGN = 1
```

**Done when:** `pip install -e '.[dev]'` works, a smoke test that imports every module passes, and CI is green.

### Step 6 — Load and prepare sessions (`data.py`)

- **`load_raw(csv_path) -> (DataFrame, meta)`.**
- **`to_grid(df, hz)`:** a uniform grid where each point takes the last packet at or before it. That's a backward as-of join: causal, and the same thing a live 60 Hz loop would see. No grid points inside gaps longer than `GAP_S`; the `segment` id increments at each gap.
- **`add_signals(df)`:**
  - `slip_deg` via `driftscore.signals.slip_angle_deg` row by row, plus `abs_slip`.
  - `abs_slip_rate`: causal change of `abs_slip` over `RATE_WINDOW_S`.
  - `yaw_rate` (deg/s): the configured axis × sign, or −d(unwrapped heading)/dt when `YAW_AXIS` is None. Also `yaw_acc`.
  - `lat_g` and `lon_g`.
  - `rear_slip`: mean of the RL and RR wheel slip.
  - `countersteer` = steer × sign(slip) × `COUNTERSTEER_SIGN`, and `lock` = |steer|.
  - `time_to_90` = (90 − `abs_slip`) ÷ `abs_slip_rate` when the rate is positive, otherwise `TTS_CAP_S`; clipped to [0, `TTS_CAP_S`].
- **`load_session(session_id)`:** the three steps above, cached to `data/processed/<id>.parquet`. The cache rebuilds itself after a config change; `--rebuild` forces it after a code change.
- **Session lists:** `list_sessions()` and `dev_sessions()` from `splits.json`. `load_test_sessions(confirm=False)` raises unless `confirm=True`.

Tests:
- **`to_grid`:** uneven timestamps give an even grid; a 0.3 s gap gives two segments with no points inside the gap.
- **`time_to_90`:** a non-positive rate gives the cap; 60° at 120°/s gives 0.25 s.
- **Slip angle:** a straight-line trace reads 0 (proves the driftscore wiring).

**Done when:** tests pass, and Sunday's sessions load and plot.

**Results (2026-10-12):** done. `python -m spinwarn.data` loads Sunday's three sessions in about 1 s.

- **On the grid:** `194646` 52,180 frames, 14.5 min, segments 0, 2 and 3; `200553` 29,583, 8.2 min, one segment; `202102` 35,937, 10.0 min, one segment. Segment ids can skip: in `194646`, segment 1 is the single packet between the 0.7 s pause and the 0.11 s stall, and gets no grid point. Segment 3 is the 0.9 s after the final pause, so Step 7 excludes all of it.
- **Choices the plan left open:**
  - Slip is zeroed below driftscore's `SLIP_SPEED_FLOOR_KMH` (5 km/h), as `app.py` does.
  - Every derived rate (d|slip|/dt, `yaw_acc`, yaw rate from heading) is a backward difference over `RATE_WINDOW_S` inside the segment: each grid point's packet is up to ~4.5 ms old, too jittery for one 1/60 s step.
  - `splits.json` is `{"test": [ids]}`; Step 10's `lock-test` writes that.
  - Each cached session stores the constants that built it (`built_with()`, in the Parquet metadata) and rebuilds itself when they change. `--rebuild`, on `python -m spinwarn.data` (dev sessions only), forces it for what the cache can't see: a code change here or in driftscore, or a raw CSV copied over again. The cache is written aside and renamed into place, so a write cut short can't leave a damaged one.
  - `load_test_sessions` always rebuilds from the raw CSVs, never from the cache: `--rebuild` doesn't reach test sessions, so their cache could predate a code change. Step 14 needs nothing extra.
  - Paths (`RAW_DIR`, `PROCESSED_DIR`, `SPLITS_PATH`) are in `config.py`.
- **Yaw fallback:** −d(heading)/dt agrees with `ang_vel_y` on all three sessions, r 0.995, ratio 0.999.
- **For Step 8:** with `COUNTERSTEER_SIGN = 1`, held drifts (|slip| 20–70°, over 30 km/h) read negative 98–99% of the time, median −0.56, so it should become −1. The cache rebuilds itself when it changes.
- **For Step 7, a preview** from a scratch version of the rule: about 36 onsets (14, 12, 10), not the ~90 of the driver's counts. Even every rise past 90° at any speed is only 43. 26 of the 36 end under 10 km/h, and none is a quick swing past 90° and back: one spends under 0.25 s over 90°. Reverse entries and big saves peak at 60–90° (58 of them) and aren't counted. At ~1.1 spins a minute, 100+ needs about another hour of driving; the deliberate session gave ~1.5 a minute.

### Step 7 — Labels (`labeling.py`)

- **`find_onsets(df) -> (onset_times, spinning_mask)`:** a per-segment state machine.
  - **Spin:** while armed, a spin starts at the first frame with speed ≥ `MIN_SPEED_KMH` and `abs_slip` > `SPIN_DEG`, held for `SPIN_HOLD_S`. Onset is that first frame. The car is then disarmed.
  - **Re-arm:** while disarmed, it re-arms after `REARM_S` of continuous speed ≥ `MIN_SPEED_KMH` with `abs_slip` < `REARM_DEG`.
  - **`spinning`** covers onset up to re-arm.
- **`label(df, horizon) -> df` with `target` and `excluded`:**
  - **target:** a spin onset falls in (t, t + horizon].
  - **excluded:** any of these:
    - speed below `MIN_SPEED_KMH`;
    - in reverse (`gear == REVERSE_GEAR`);
    - spinning;
    - the first `WINDOW_S` of a segment (incomplete window);
    - the last `horizon` of a segment (its future is unknown).

Tests:
- **The Project Plan's cases:**
  - A crossing at 5.0 s gives one onset, with positives exactly in [4.5, 5.0).
  - A second crossing before re-arm is not a new spin.
  - A spin, a stop and a drive-off re-arm only after 2 s of moving under 45°.
- **Hold:** with `SPIN_HOLD_S = 0.25`, a 0.1 s spike over 90° is not a spin.
- **Segment edges:** the first and last frames of a segment come out excluded.

**Done when:** tests pass; the onset count per session is printed.

**Results (2026-10-12):** done. `python -m spinwarn.labeling` finds 36 onsets in 32.7 min: `194646` 14, `200553` 12, `202102` 10, the same as Step 6's preview. At 1.1 a minute, 100+ still needs about another hour of driving.

- **Choices the plan left open:**
  - Nothing in reverse gear starts a spin. Reversing reads |slip| near 180°, so backing up at 20 km/h or more would count as one, and every frame before it is excluded as reverse, so no model could catch it. On Sunday's sessions every reverse stretch falls inside a spin already (one reached 20.3 km/h, in `200553`), so this changes nothing yet. Since Step 10, reverse gear disarms the car instead, which also covers a shift out of reverse while still rolling backward: see Step 10's results.
  - The session starts armed, and armed or not carries across a gap: a pause freezes the car, so a spin running through one stays one spin. The hold and the re-arm start over after a gap. No gap in Sunday's sessions falls inside a spin, and you never pause mid-drift, so only one of AC's own short stalls ever could.
  - Edges are compared in grid frames, not float seconds, so none hangs on an ulp. Held for `SPIN_HOLD_S` means over 90° from the onset frame through the frame `SPIN_HOLD_S` later; re-arm is the frame `REARM_S` after the run's first.
  - `target` counts only onsets in the same segment. Frames across a gap are excluded anyway.
  - The first `WINDOW_S` is t − segment start < 1 s, so the first scored frame is the first one with Step 9's 1 s slope.
- **On Sunday's sessions:** 11–18% of frames are spinning and 13–22% excluded in all. No onset falls in a segment's first `WINDOW_S`, and all 30 frames before every onset are scored. `202102` ends still spinning: its last spin, at 594.7 s, never re-arms.

### Step 8 — Explore and spot-check

`notebooks/01_explore.ipynb`, per session:
- the real raw rate and the gaps;
- slip angle over time with onsets marked;
- zooms on 3 onsets showing slip, yaw rate, steer and gas;
- a histogram of |slip|, to see where your intended drifts sit relative to 90°;
- the countersteer sign check, which sets `COUNTERSTEER_SIGN`.

`spotcheck.py`:

```text
python -m spinwarn.spotcheck <session> --n 20 --video-offset 3.2
```

- **Output:** 20 seeded-random onsets as video timestamps (mm:ss.s), each with its peak |slip| and speed.
- **Video offset:** the video time at which the logger's status line shows `t = 0`. Read it off the recording.
  - Pauses don't move it. The logger's `t` keeps counting through a pause, AC's own at the start included, and so does the recording, so one offset should hold for the whole session. Read it once more after a pause, such as 552 s in `194646`, to confirm.

(You) Watch each onset and mark it hit or miss in `notes.md`. Under 18/20, set `SPIN_HOLD_S = 0.25` and rerun. Also watch 5 minutes and count missed spins.

**Done when:** the label rule is validated and the spin count is known.

**Results (2026-10-12):** done. The spot-check is 20/20, no spin was missed in 5 minutes of video, and the rule finds 36 spins in 32.7 min, 1.1 a minute. `SPIN_HOLD_S` stays 0.

- **Spot-check** (in `notes.md`): video offsets 3.89, 7.4 and 4.65 s, read off the status line. 18/20 as first marked: two 360s in `200553` were marked n, one deliberate and one ending in a wall. Marked again by whether the car got past sideways, as the rule asks, both are hits; spin 12's wall impact (15 g longitudinal) comes 1.1 s after its onset. Edge cases for the README's limitations: that wall, and `194646` spin 14, which peaks at 95° and is driven out of. Even with both as misses it's 18/20.
- **Missed spins:** none in `202102` from 0:00 to 5:00, where the rule finds 5.
- **360s are spins (your call):** a 360 is a show-off move, not held drifting, so it counts as a spin like any rotation past 90°, deliberate or not. Intent is the session's `kind`: Step 14 reports recall on natural and deliberate spins separately if both have enough, otherwise it goes under Limitations, as the Project Plan says.

- **Rate and gaps:** AC 333.3–333.6 packets/s, 0.01–0.02% of packetId steps 2+. The gaps are Step 4's: in `194646`, the 0.68 s pause at 551.9 s, the 0.11 s stall after it and the 3.0 s pause at the end; the ~20 ms pause at the start of the other two.
- **`COUNTERSTEER_SIGN = −1`:** steer × sign(slip) reads negative on 97–99% of held-drift frames (|slip| 20–70°, over 30 km/h, not spinning) in every session, median −0.55 to −0.56. The caches rebuilt themselves.
- **Where the drifts sit:** of the frames at speed that aren't spinning, the median |slip| is 17–21°; 11–16% are over 45°, 1.1–1.6% over 75° and 0.2–0.4% over 85°. They thin out steadily up to 90°, with no bump just under it.
- **Every spin:** 36 onsets, peak |slip| 91–180°. 27 of them slow to 10 km/h or less before re-arming.
- **`SPIN_HOLD_S = 0.25` would drop 4 of the 36:** `194646` spins 3 and 6, `200553` 9, `202102` 10. None is a quick swing past 90° at speed: each starts at 20–25 km/h and falls under `MIN_SPEED_KMH` within 0.25 s, ending at 0–7 km/h. So the hold would drop slow spins, not intended high angles. On video all 4 are real spins: `194646` 3 and 6 in the random 20, the other two watched as extras.
- **Choices the plan left open:**
  - A spin's peak |slip| is over its frames from onset to re-arm at `MIN_SPEED_KMH` or more, where slip angle isn't noise. Its speed is the speed at onset.
  - `spotcheck` takes one session, since each has its own offset. The notebook splits the 20 in proportion to each session's spins: 8, 7 and 5. Each pick is sorted by time and numbered as in the notebook, and prints as a table with an empty Hit column for `notes.md`.
  - `spotcheck` refuses a locked test session, like everything but the final run.
- **The offset method:** step the paused video frame by frame to where its clock ticks over to a whole second, and subtract the `t` the status line shows on that frame. The status line updates every 0.25 s, so it's good to about that. A spin spotted on video outside the sample, at 07:55 in `194646`, is the rule's spin 7 at 07:55.1.
- **For later steps:** at 1.1 spins a minute, 100+ still needs about another hour of driving.

### Step 9 — Features (`features.py`)

- **`make_features(df) -> DataFrame`:** the Project Plan's feature table plus `time_to_90` and `countersteer`. Computed per segment (`groupby('segment')`):
  - **Windows:** `rolling('1s')` on the time index, never centred.
  - **Slopes:** (x_t − x_{t−1 s}) ÷ 1 s via `shift(GRID_HZ)` inside the segment. That is exact on the uniform grid.
- **`FEATURES`:** the list of feature column names, used everywhere else.

Tests:
- **Causality:** build a synthetic session and compute features. Replace every value after t* with noise, recompute, and features at t ≤ t* must be identical.
- **Segment isolation:** features in the first frames of segment 2 don't change when segment 1's values do.

**Done when:** tests pass.

**Results (2026-10-12):** done. 26 features; Sunday's three sessions take 0.04 s in all, with no NaN on any scored frame.

- **Choices the plan left open:**
  - "Last, mean, max of absolute value, slope" is all four on `abs_slip`, so left and right drifts read the same.
  - **No feature knows left from right.** Yaw rate, yaw acceleration and lateral G are taken relative to the slide, frame by frame, as `<signal>_rel`: × sign(slip), the way `countersteer` is built from steer. Steering's last and mean are `countersteer`'s, plus `lock_last` for how close to full lock, so `steer` itself isn't a feature. A drift and its mirror image give the same features (tested, on both yaw paths; the test fails on the signed versions).
    - **Why:** signed, a left and a right drift read opposite, which a linear model can't use. The only extra the signed versions carry is the direction, and 64% of Sunday's held-drift frames go one way, so that's mostly Sepang's corners, which the trees would learn instead of spin physics.
    - **On Sunday's scored frames,** r with the target: yaw rate +0.09 signed, −0.18 relative; lateral G +0.07 and −0.08; steer +0.03, `countersteer` +0.08. In held drifts (|slip| 20–70°, over 30 km/h), `yaw_rate_rel` reads negative on 82% of frames and `lat_g_rel` on 96%.
    - Computed in `features.py` from `slip_deg`, so `data.py` and the cache don't change.
  - Names are `<signal>_<stat>`, so the `_last` features don't clash with the raw signals Step 10 keeps for the baselines.
  - Windows in a segment's first `WINDOW_S` are partial and its slopes NaN; `label()` excludes those frames.
  - The time index is `pd.to_timedelta(t)`. pandas rounds it to the nanosecond, so 1 s back lands exactly on a grid frame, and every window holds 60 frames. The test checks this at 552 s; a scratch check found it true for every frame of an hour.
- **Causality is tested twice:** on `make_features` alone, and from raw packets through `to_grid` and `add_signals`, on both yaw paths, so a leak in the grid join or a rate would fail it too.
- **For Step 13:** `rear_slip` spikes like the fronts in logcheck: median about 4, but up to 2.7e8. A 1 s mean catches a spike (over 1,000) on 4–5% of scored frames, reaching 5e6. The slope reads the spike itself: under 1% of scored frames, but up to 2.7e8. Trees cope, but logistic regression's StandardScaler won't. A signed log (sign × log1p of the absolute value) on both, in its pipeline, would fix it; a rolling median would fix only the mean.
- **Only with `YAW_AXIS = None`:** yaw rate from heading is unknown for a segment's first `RATE_WINDOW_S`, so `yaw_rate_rel_slope` is NaN on the first 0.1 s of scored frames. Sunday's sessions have none, since those frames are excluded there anyway. The configured axis has no such gap.

### Step 10 — Dataset build and test lock

- **`python -m spinwarn.run build`:** loads every session, labels it and computes features into `data/processed/dataset.parquet`. One row per grid frame: session, segment, t, target, excluded, features, and the raw signals the baselines need. Prints minutes, spins and positive frames per session.
- **`python -m spinwarn.run lock-test --sessions A,B`:** writes `splits.json` and refuses if it already exists.
  - (You) pick 2 main-car sessions with typical spin counts before any modelling, then commit `splits.json`.

**Done when:** the dataset builds and `splits.json` is committed.

**Results (2026-10-14, lock 2026-10-15):** done. On Sunday's three, `build` gives 117,700 frames in 32.7 min (27.1 scored), 36 spins and 1,080 positive frames (36 × 30, 1.1% of scored frames), with no NaN in any feature or baseline signal on a scored frame.

- **Choices the plan left open:**
  - The dataset adds an `onset` column, true on each spin's first frame, so Step 11 scores alerts from the dataset alone.
  - `build` reads dev sessions only, so the test sessions never reach it, as the ground rules ask. Built before the lock, it holds every session.
  - The dataset remembers the sessions and constants that built it. `load_dataset()` refuses it once they change: a session copied in, the lock, or a `config.py` change. Built before the lock, it still holds the test sessions, and CV would train on them. Steps 12–14 read it through `load_dataset()`.
  - `load_dataset()` also refuses any dataset until the test sessions are locked, so no modelling can start before the lock, as the Project Plan asks. `build` still works before it, for the spin counts to pick by.
  - `lock-test` refuses unless at least `N_FOLDS` sessions are left, since GroupKFold needs a session per fold. Today's 3 are refused; 6 sessions with 2 locked leave exactly 4. It writes the ids sorted, and never over an existing file.
- **The lock (2026-10-15):** `2026-10-11_202102` (natural, 10.0 min, 10 spins) and `2026-10-14_233636` (deliberate, 10.2 min, 17 spins), committed in `9cf85dc`. One natural and one deliberate, so Step 14 can report recall on each; one from each recording day; each close to the usual spin rate for its kind. The two biggest sessions (`194646`, `225818`) and Oct 14's two sparse natural runs (`223131` 1 spin, `224308` 4) stay in the dev set for training.
  - **Not what this step first said:** it said 2 of the new sessions, not Sunday's. But Oct 14's natural runs are too sparse to test on, so the natural test session had to come from Sunday. Step 9 weighed the relative features partly by their correlation with the target on all three Sunday sessions, `202102` among them, so it isn't fully unseen. That goes under the README's limitations.
  - `233636`'s 17 is the labeler's count from before the reverse fix below. The final run relabels both test sessions with the current rule, and nothing looks at them before it.
- **The reverse fix (2026-10-15):** the dev set had one false spin, `225818` at 603.97 s. Backing up at 20.6 km/h, the driver shifted out of reverse while the car still rolled backward, so the next frame was over 90° at speed and out of reverse gear, and Step 7's rule counted an onset. All 30 frames before it were excluded (reverse, slow), so no method could ever have caught it. Found by checking that every onset has its 30 scored frames, as Step 7 found on Sunday's sessions: this was the only one without.
  - Reverse gear now disarms the car the way a spin does. Its frames up to the re-arm are excluded (`reversing`, beside `spinning`), and a reverse inside a spin is part of the spin and restarts its re-arm. `spinning` still means onset to re-arm, so `spotcheck` and the notebook are unchanged.
  - Rebuilt, it changes nothing else: that one onset and its 30 target frames go, and no scored frame changes in any dev session.
- **The dev set after the lock and the fix:**

  | Session | Kind | Min | Scored min | Spins | Positive frames |
  | --- | --- | --- | --- | --- | --- |
  | `2026-10-11_194646` | natural | 14.5 | 12.0 | 14 | 420 |
  | `2026-10-11_200553` | deliberate | 8.2 | 6.4 | 12 | 360 |
  | `2026-10-14_223131` | natural | 6.5 | 6.2 | 1 | 30 |
  | `2026-10-14_224308` | natural | 11.6 | 10.9 | 4 | 120 |
  | `2026-10-14_225818` | deliberate | 13.9 | 10.7 | 27 | 810 |
  | **Total** | | 54.7 | 46.3 | 58 | 1,740 (1.0% of scored) |

  Every spin has all 30 frames before it scored. With the test sessions' 27, that's 85 spins over 7 sessions, short of the Project Plan's 100+ over about 10; recording stopped there. Step 14's interval on test recall will be wide.
- **For Step 12:** `target` and `excluded` are built at `HORIZON_S`. The `onset` column gives `target` at any horizon, but `excluded` already holds each segment's last 0.5 s, so `--horizon 1.0` needs a dataset of its own: e.g. `build --horizon` writing a second file, with the horizon in its `built_with`.

### Step 11 — Alert scoring (`evaluate.py`)

- **`find_alerts(t, score, threshold, segment) -> [(start, end)]`:**
  - Excluded frames get score −∞ first, so alerts only exist on scored frames.
  - Runs closer than `MERGE_GAP_S` merge.
  - Alerts never cross segments.
- **`match(alerts, onsets)`:** applies the Project Plan rules.
  - **Caught:** an alert is active anywhere in [onset − 1.0, onset − 0.1]. Lead time = onset − alert start, capped at 1.0 s.
  - **False alarm:** no onset in (start, end + 1.0].
  - **Too late:** an alert starting under 0.1 s before onset is neither caught nor false.
- **`summarize(...) -> dict`:** n_spins, caught, recall, false_alarms, minutes, fa_per_min, alert_precision, time_in_alarm, median_lead. Pools sessions by summing counts and minutes.
- **`threshold_for_fa(df, score, target)`:** scans candidate thresholds (score quantiles) and returns the lowest one with fa_per_min ≤ target.
- **`recall_fa_curve(df, score)`.**
- **`bootstrap_recall(caught, n=1000, seed=SEED) -> (lo, hi)`.**
- **`pr_auc(df, score)`:** `average_precision_score` on scored frames.

Tests:
- An alert starting 1.3 s before onset and lasting through it: 1 catch, lead 1.0 s, 0 false alarms.
- An alert starting 0.05 s before onset: too late, not false.
- An alert with no onset: false.
- Alerts 0.15 s apart merge; alerts 0.3 s apart don't.
- Time in alarm on a hand-made case.
- `threshold_for_fa` on a toy score.
- `bootstrap_recall` is identical across two runs with the same seed.

**Done when:** tests pass.

**Results (2026-10-14):** done. 32 tests. On Sunday's dataset, B1's recall-vs-FA curve and its 1 FA/min threshold take 0.05 s over 167 candidate thresholds, with 36 spins in 27.1 scored minutes, as Step 10 counts.

- **Choices the plan left open:**
  - `find_alerts` also takes `excluded`, since the −∞ has to come from somewhere. An alarm is score > threshold, strictly, so a −∞ frame never alarms, even at threshold −∞.
  - An alert's (start, end) are the times of its first and last alarm frame. Alarms are "apart" by the next one's first frame minus the last one's last frame: exactly `MERGE_GAP_S` doesn't merge.
  - Time in alarm counts every scored frame from an alert's first frame to its last, so the short dips a merge bridges count as in alarm.
  - Alerts and onsets are matched within a segment, as `target` is labelled: an onset just after a gap neither catches an alert before it nor excuses it as a false alarm.
  - Times are compared as whole grid frames, as in `labeling.py`, so no credit-window edge depends on float rounding. Both edges are inclusive: an alert ending exactly 1.0 s before onset, or starting exactly 0.1 s before, is a catch.
  - On real data an alert can't stay on through a spin: frames from the onset to the re-arm are excluded. So it ends on the frame before the onset, which is still a catch with 1.0 s lead and not a false alarm (tested).
  - `summarize(df, score, threshold)` also returns `alerts`, the denominator of alert precision. A ratio with nothing to divide by (no spins, no alerts, nothing caught) is NaN.
  - Candidate thresholds are `N_THRESHOLDS` = 200 (new in `config.py`) quantiles of the finite scores on scored frames, spaced geometrically in the top tail, where 1 FA/min lies, plus the highest score, which alarms nowhere. `threshold_for_fa` scans them all rather than bisecting: false alarms don't always fall as the threshold rises, since one alert can split into two.
  - `pr_auc` scores the ranks, which changes nothing but lets B2's −∞ through scikit-learn.
  - A NaN score on a scored frame raises: it would never alarm, and would pass for a model that just never fires.
- **For Step 14:** `bootstrap_recall` takes one bool per test spin, and `summarize` returns only counts. Step 14 needs a way to get the per-spin catches out: `match`, segment by segment, already returns them.

### Step 12 — Baselines and CV harness

**`baselines.py`:**
- B1: score = `abs_slip`.
- B2: for each θ1 in 30–80° (step 5), score = `abs_slip_rate` where `abs_slip` > θ1, else −∞. Keep the θ1 with the best CV recall at 1 FA/min.
- B3: score = −`time_to_90`.

**`cv.py`, `oof_scores(dataset, method, params)`:**
- GroupKFold(`N_FOLDS`) over dev sessions, grouped by session.
- Fit on non-excluded frames thinned to `TRAIN_HZ`; predict every frame of the held-out fold.
- Returns scores aligned to the dataset. Baselines go through the same function with no fitting, so every method is treated the same way.

**`run.py cv`:**

```text
python -m spinwarn.run cv --method b1|b2|b3|lr|rf|hgb [--params '{...}'] [--horizon 0.5] [--note "..."]
```

- Prints PR-AUC, plus recall, alert precision, time in alarm and median lead at 1 and 0.5 FA/min.
- Saves out-of-fold scores to `results/cv/<run_id>.parquet`.
- Appends a row to `experiments.md`: date, method, features, params, PR-AUC, recall @ 1 FA/min, note.

**Done when:** B1, B2 and B3 rows exist in `experiments.md`.

### Step 13 — Models and tuning

- **`models.py`, `make_model(name, params)`:**
  - **lr:** `Pipeline(StandardScaler, LogisticRegression(class_weight='balanced', max_iter=1000))`.
  - **rf:** `RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)`.
  - **hgb:** `HistGradientBoostingClassifier(class_weight='balanced', random_state=SEED)`.
- **`python -m spinwarn.run tune --method hgb --n 15`:** a seeded random search. Each config is one `cv` run; the best config by PR-AUC goes into `results/chosen.json`.
  - **hgb:** `learning_rate` {0.03, 0.1}, `max_leaf_nodes` {15, 31, 63}, `min_samples_leaf` {20, 50, 200}, `l2_regularization` {0, 1}.
  - **rf:** `n_estimators` {200, 400}, `max_depth` {None, 10, 20}, `min_samples_leaf` {1, 5, 20}, `max_features` {"sqrt", 0.5}.
  - **lr:** `C` {0.01, 0.1, 1, 10}.

**Done when:** `chosen.json` has a config for every method and `experiments.md` has every run.

### Step 14 — Final test run

`python -m spinwarn.run final` runs once, for every method:

1. Thresholds at 1 and 0.5 FA/min from the chosen CV run's out-of-fold scores.
2. Refit on all dev sessions, then score the test sessions via `load_test_sessions(confirm=True)`.
3. Bootstrap intervals on test recall.
4. Permutation importance for the best model, on the test sessions, scored by average precision.
5. Time per prediction: median ms over 1,000 single-row `predict_proba` calls.
6. Write `results/metrics.json` and `results/final.lock`.

- **Run-once guard:** it refuses to run again while the lock exists. `--force` overrides, and the forced run is logged in `experiments.md` so it stays visible.
- **Secondary horizon:** `--horizon 1.0` writes `metrics_h1.json` with its own lock.

**Done when:** `metrics.json` exists and the test sessions were scored once.

### Step 15 — Report and README

- **`python -m spinwarn.report`:** reads `metrics.json` and writes:
  - `results/results_table.md`;
  - `results/recall_vs_fa.png` (all methods, CV);
  - `results/importance.png`;
  - the README section between `<!-- results:start -->` and `<!-- results:end -->`, rewritten. That's the "one command regenerates the table".
- **README:** follow the Project Plan's outline. You write the story; Claude Code checks commands and numbers.
- **`pip freeze > requirements.txt`** for exact reproduction. Commit `data/sample/`.
- **Raw data (optional):** zip the raw CSVs as a GitHub release asset so the full pipeline can be rerun.
- **Make the repo public.**

**Done when:** from a fresh clone, `pytest` passes and `python -m spinwarn.report` regenerates the table, and the README's reproduce commands run end to end with the raw data.

## Phase C — week 2 (optional)

- **Step 16 — Live warning in DriftScore:**
  - Incremental features over a 1 s ring buffer on a 60 Hz loop, using the same grid rule as Step 6.
  - **Test:** replay a recorded session through the live code; the result must match `features.py` (`allclose`).
  - **Model:** export with joblib. scikit-learn becomes an optional `[live]` extra, so DriftScore's core stays stdlib-only.
  - **Budget:** features plus prediction under 16.7 ms.
- **Step 17 — Extensions:** IsolationForest, the unseen car and the 1.0 s horizon, as new `--method` values and flags on `run.py`.
