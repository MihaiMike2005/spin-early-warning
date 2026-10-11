# Spin Early-Warning Model — Project Plan

Oct 10, 2026 · @Mihai

## Goal and definition of done

Build a model that warns of a spin 0.5 s before it starts, from your live telemetry, and beat tuned physics baselines (a slip-angle threshold and time to 90°) at the same false-alarm rate. The results go on the CV, and the application goes out by Oct 17, 2026 (Sun 18 Oct at the latest).

**Done when:**

- [ ] Public repo with a README: headline result, results table, two plots, how to reproduce
- [ ] Best model vs. physics baselines reported on held-out sessions, at the same false-alarm rate, with bootstrap intervals
- [ ] Tests pass for the labeler, feature causality and alert scoring
- [ ] CV entry filled with real numbers, application sent

**Stack:** Python, pandas, NumPy, scikit-learn, matplotlib, pytest. Log on the PC running Assetto Corsa (Python 3.11+ there, see Data collection); train anywhere.

It is Team 2's first task in miniature (failure prediction from sensor data) and covers the 3rd-year duties: compare models against a baseline, tune them, document the experiments.

## Spec

Each frame gets its label from the next 0.5 s and its features from the previous 1 s; the model learns to map one to the other.

**Labels**

- **Spin:** |slip angle| passes 90° while speed is above 20 km/h. Onset = the first such frame.
- **Re-arm:** a new spin counts only after |slip angle| has stayed under 45° for 2 s while above 20 km/h. Slip angle is noise near standstill, so a stopped car never re-arms by itself.
- **Target:** frame t is positive if a spin onset falls in (t, t + 0.5 s\]; otherwise negative.
- **Excluded:** frames from onset to re-arm (already spinning), under 20 km/h, in reverse, paused or in replay. They are neither trained on nor scored.
- **Horizon:** 0.5 s is the main result; rerun at 1.0 s as a secondary one.

Compute slip angle offline with your already-tested function, so labels and features share the same math.

Define every window, horizon and gap in seconds, never in frames. The logging rate is measured on day 1, not assumed (see Data collection).

**Features** — over the window (t − 1 s, t\], backward-looking only:

| Signal | Window features |
| --- | --- |
| Slip angle | last, mean, max of absolute value, slope |
| Slip-angle rate (deg/s) | last, max |
| Time to 90° | last: seconds for the absolute slip angle to reach 90° at its current rate, capped at 5 s (the B3 formula, so the model starts from the strongest baseline) |
| Yaw rate | last, mean, slope; yaw acceleration (last) |
| Lateral and longitudinal G | last, mean |
| Steering | last, mean; counter-steer: steering direction vs. slip direction, how close to full lock |
| Throttle, brake | last, mean |
| Speed | last, slope |
| Rear wheel slip | mean, slope |

Use time-based windows (pandas `rolling('1s')` on a time index, never `center=True`): frame jitter can't skew them and no future data leaks in. A slope is (last value − first value in the window) ÷ elapsed time, not a fitted line: much faster, and good enough. Yaw rate comes from `localAngularVel`, or from the derivative of unwrapped heading if the struct extension fails (see Data collection).

**Baselines**

- **B1:** alarm when |slip angle| exceeds θ, with θ tuned on training sessions.
- **B2:** |slip angle| above θ1 and slip-angle rate above θ2.
- **B3 (strongest):** time to 90°, the seconds |slip angle| needs to reach 90° at its current rate: (90° − |slip|) ÷ d|slip|/dt. Alarm when it drops below τ, tuned on training sessions. It is the drift version of time-to-collision from driver-assistance systems, the first thing an automotive engineer will ask about, and probably the one to beat. Beating B3 is worth more than beating B1 or B2.

**Models (scikit-learn)**

1. Logistic regression in a Pipeline with StandardScaler, `class_weight='balanced'`
2. Random forest, `class_weight='balanced_subsample'`
3. HistGradientBoostingClassifier, `class_weight='balanced'`

Train on frames thinned to about 20 Hz, since neighbouring frames are near-duplicates; score alerts at the full logged rate.

## Evaluation protocol

Split by session and score alarms the way a maintenance system is scored: spins caught in time, false alarms per minute, warning lead time.

1. **Lock 2 test sessions** before any modelling. Score them once, at the very end.
2. **Tune on the rest** with session-grouped cross-validation (GroupKFold, 4 folds, group = session). Model-selection metric: average precision (PR-AUC) on frames.
3. **Scores to alerts:** frames above the threshold are alarms; alarms less than 0.2 s apart merge into one alert.
4. **Score the alerts:**
   - A spin is **caught** if an alert is active at any moment between 1.0 s and 0.1 s before onset. Lead time = onset − alert start, capped at 1.0 s. An alert that switched on early and stayed on through the spin is a catch, not a miss plus a false alarm.
   - An alert is a **false alarm** if no spin onset falls between its start and 1.0 s after its end. An alert that starts less than 0.1 s before onset is too late: neither caught nor false. Alerts that start in excluded frames are ignored.
   - False alarms per minute = false alerts ÷ minutes of scored driving.
   - Alert precision = alerts that are not false ÷ all alerts.
   - Time in alarm = share of scored driving with an alert active, so a model that is always on can't hide behind a low false-alarm count.
5. **Operating point:** on the CV predictions, set each method's threshold to about 1 false alarm/min, and a second one at 0.5/min. Report test-session recall, alert precision, time in alarm and median lead time at each. Expect about 0.5–1 spin per minute of driving, so at 1 FA/min false alarms can outnumber spins: say so in the README before a reader works it out.
6. **Also report:** a recall-vs-false-alarm curve for all methods, frame PR-AUC, time per prediction, permutation importance for the best model, and a bootstrap 95% interval on test recall (resample the test spins 1,000 times; about 10 lines of code).

&#91;embedded content: scoring windows around a spin onset\]

Training labels look 0.5 s ahead; scoring credits any alert that is active in the wider 1.0 to 0.1 s window before onset.

**Results table** (fill in):

| Method | PR-AUC (CV) | Spins caught @ 1 FA/min (test, 95% CI) | Alert precision | Time in alarm (%) | Median lead time (s) | Time per prediction (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| B1: slip threshold |  |  |  |  |  |  |
| B2: slip + rate |  |  |  |  |  |  |
| B3: time to 90° |  |  |  |  |  |  |
| Logistic regression |  |  |  |  |  |  |
| Random forest |  |  |  |  |  |  |
| Gradient boosting |  |  |  |  |  |  |

With about 20 test spins, one spin is about 5 points of recall. Say so in the README, show the bootstrap interval, and don't over-read small gaps.

## Data collection

Target: 100+ spins over about 10 sessions, logged raw at the measured rate, with the label rule checked against your screen recordings.

**What to log** — one row per new physics packet. The current `ACPhysics` struct stops at `heading`, so the struct work below comes first.

- Timestamp (`time.perf_counter()`) and packetId; drop rows with a repeated packetId
- Speed, world velocity, heading
- `localAngularVel` for yaw rate: about 20 fields after `heading` in AC's physics struct, so extend `ACPhysics` field by field and find the vertical-axis index. One wrong field shifts everything after it and reads garbage, so check it against the derivative of unwrapped heading on a recorded session. If it fights you, use the heading derivative as yaw rate and move on.
- accG (check which index is lateral)
- Steering, throttle, brake, gear
- Wheel slip for all four wheels (check the wheel order)
- Graphics page (`acpmf_graphics`): status, via a short prefix struct (`packetId`, `status`), the same trick as the physics one. Log only while live, never paused or in replay.
- Static page (`acpmf_static`): car and track, via a short prefix struct; saved with date and setup notes in one JSON per session

**Sample rate:** today's "60 Hz" comes from `time.sleep(1/60)` in the app loop, not from AC. On Windows with Python older than 3.11, that sleep can round up to about 31 ms, so you'd log at about 30 Hz. Use Python 3.11+ on the AC PC, poll every few ms and keep only new packetIds. On day 1, measure packets per second and the packetId step between rows. Everything downstream is in seconds, so the rate only needs to be steady and known.

One Parquet (or CSV) file per session, e.g. `data/raw/2026-10-11_s01.parquet` plus its `.json`.

**Sessions**

- 10–20 minutes each; short is fine. All on one car and track this week; the second car moves to week 2.
- Mix normal drifting with deliberate over-rotation. Sessions that only push teach the model that aggressive means spin.
- Too few spins? Less grip or more power, one change at a time.
- Cut a segment wherever the stream breaks (pause, reset, gap over 100 ms). No window may span a break.
- Keep `notes.md`: date, car, track, setup, minutes, spin count, and whether the session was **deliberate** (trying to spin) or **natural** (holding drifts at the limit). Planned spins look different from failed saves; report recall on each if both have enough spins, otherwise list it under Limitations.

**Label check**

- Start the screen recording and the logger together, so telemetry time maps to video time. If you can, put the logger's console (elapsed time, packetId) in the recorded scene too, so the mapping is exact.
- Spot-check 20 detected onsets on video and record the hit rate (e.g. 19/20) for the README.
- Watch 5 minutes of footage and count spins the rule missed.
- Intentional high angles flagged as spins? Require more than 90° for 0.25 s, then re-run.

## Schedule

Data by Tue 13, results by Thu 15, write-up Fri 16; Sat 17 is the buffer and the send day.

&#91;embedded content: project week · 4 workstreams, 4 checkpoints\]

Recording and the pipeline run side by side through Tue 13; everything after that works on data you already have.

### Sun 11 Oct — logger and labels (\~4 h)

Work in this order. The logger blocks recording, and recorded sessions are the scarcest resource; the labeler blocks nothing and can be written on the Mac.

- [ ] In DriftScore: extend `ACPhysics` to `localAngularVel`, add the graphics and static prefix structs, and add `driftscore/logger.py`: fields above, live-only, deduplicated
- [ ] First recording: measure packets per second and the packetId step; check yaw rate against the heading derivative
- [ ] Record 3 sessions with screen recording on
- [ ] `spin-early-warning` repo, venv, `requirements.txt`: pandas, numpy, scikit-learn, matplotlib, pytest, pyarrow, plus `driftscore` from GitHub
- [ ] `labeling.py`: onsets, re-arm, targets, exclusions, plus tests on a synthetic trace. If it slips, do it first thing Monday.

Done when: 3 sessions load in pandas, the sample rate is known, yaw rate is checked, the labeler finds spins, tests pass.

### Mon 12 Oct — check the labels (\~2–3 h)

- [ ] Notebook: real sample rate, gaps, slip-angle plots with onsets marked
- [ ] Spot-check 20 onsets on video; tighten the rule if needed
- [ ] Record 2–3 sessions

Done when: label rule validated, spin count known.

### Tue 13 Oct — dataset (\~2–3 h)

- [ ] Record until 100+ spins over about 10 sessions, all on one car
- [ ] `features.py`: causal 1 s windows, time to 90°, targets, exclusions, plus the causality test
- [ ] Pick and lock the 2 test sessions

Done when: one features table covers all sessions and the test sessions are set aside.

### Wed 14 Oct — baselines and first model (\~3 h)

- [ ] `evaluate.py`: alerts, caught spins, false alarms/min, alert precision, time in alarm, lead time, plus tests
- [ ] Baselines B1, B2 and B3 on CV
- [ ] Logistic regression on CV; start `experiments.md`

Done when: the first rows of the results table exist.

### Thu 15 Oct — models and final test (\~3 h)

- [ ] Random forest and gradient boosting; 10–20 tuning configs each, ranked by PR-AUC
- [ ] Set every method's threshold to about 1 and 0.5 false alarms/min on CV
- [ ] Run the test sessions once; bootstrap intervals; permutation importance

Done when: the final results table and plots are saved in `results/`.

### Fri 16 Oct — write-up (\~2–3 h)

- [ ] README per the outline below; one command regenerates the table
- [ ] Fill the CV entry and Skills line; update the CV PDF
- [ ] Draft the motivation note and send it to me for a pass; name the team it targets (failure prediction from sensor data) so the fit is obvious

Done when: repo public, CV final.

### Sat 17 Oct — buffer and send

- [ ] Fix whatever slipped; final read of README and CV
- [ ] Send CV and note to adumitr9@ford.com.tr (Sun 18 at the latest)

## Repo, tests and README

The work splits by where it runs. The logger lives in DriftScore: it extends `telemetry.py`, runs on the Windows PC, and CI already covers it. The ML lives in a separate repo, `spin-early-warning`, which reads the Parquet files (copied over from the AC PC) and installs `driftscore` from GitHub for `slip_angle_deg`: one source of truth, no copied reader. Raw data stays out of git.

```text
spin-early-warning/
├── README.md
├── requirements.txt
├── experiments.md      one row per run
├── notes.md            session log
├── data/               gitignored; one sample session committed
├── src/
│   ├── labeling.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/01_explore.ipynb
├── results/            figures and metrics.json
└── tests/
```

```text
DriftScore/             existing repo, additions only
└── src/driftscore/
    ├── telemetry.py    ACPhysics extended to localAngularVel; graphics and static prefix structs
    └── logger.py       session logger: live-only, deduplicated, one Parquet + JSON per session
```

**Tests** — your pytest habit is a differentiator here:

- Labeler: a synthetic slip trace crossing 90° at t = 5.0 s gives one onset at 5.0 s and positives exactly in [4.5, 5.0); a second crossing before re-arm is not a new spin. A spin, a stop and a drive-off re-arm only after 2 s of moving under 45°.
- Causality: change every value after time t; the features at t must not change.
- Scoring: hand-made alerts and onsets give a known recall, false-alarm rate, alert precision, time in alarm and lead time. Include an alert that starts 1.3 s before onset and stays on through it: one catch with 1.0 s lead time, no false alarm.

**Experiment log:** one row per run in `experiments.md` — date, method, features, parameters, PR-AUC, spins caught at 1 FA/min, one-line note. Fix random seeds.

**README outline**

1. One-line summary with the headline number
2. Problem: early warning of loss of control from 60 Hz vehicle telemetry, framed like failure prediction (rare events, alarms, lead time)
3. Data: sessions, hours, spins, sample rate, car, track; the label rule and its spot-check hit rate
4. Method: causal 1 s windows, physics baselines (including time to 90°), models, session-level split
5. Results: the table with bootstrap intervals, the recall-vs-false-alarm plot, the feature-importance plot
6. Limitations: sim data, one driver, one car, deliberate vs. natural spins, small test set, label edge cases
7. Next steps: live warning, anomaly detector, unseen car
8. Reproduce: setup and commands

## Risks and cut list

The real risks are the telemetry plumbing on day 1, too few spins, and a model that ties the baseline; none of them stops you from sending.

| Risk | Early sign | Response |
| --- | --- | --- |
| Struct extension reads garbage | Yaw rate doesn't match the heading derivative | Use the heading derivative as yaw rate; move on |
| Logging rate too low | Far fewer packets/s than expected on day 1 | Python 3.11+ on the AC PC; poll every few ms; keep only new packetIds |
| Too few spins | Under 40 by Mon night | Less grip or more power; spin-attempt sessions; accept 60–80 and say so |
| Rule flags intended high angles | Spot-check under 18/20 | Require over 90° for 0.25 s, or add a speed-drop condition |
| Model ties the baseline | CV recall within a few points of B3 | Report it; show where the model wins (early warnings while slip rate is still low, lead time) |
| Leakage | Near-perfect CV scores | Check the session split, the causality test, the untouched test set |
| Behind schedule | No features table by Wed | Use the cut list |

**Already cut:** the second car (moved to week 2).

**Cut, in this order:** random forest → tuning (use defaults) → permutation importance → B2 → the 0.5 FA/min operating point.

**Never cut:** the session-level split, the B1 and B3 baselines, the README results table.

## After sending: week 2

None of this is needed to apply; it is interview material for their integration month.

- **Live warning:** run the model inside the DriftScoreUI loop; the whole step must fit the 16.7 ms frame budget at 60 Hz.
- **Anomaly detector:** IsolationForest trained only on non-spin driving vs. the supervised model — how a factory works when failures are rare and unlabeled.
- **Unseen car:** record car B, train on car A, test on car B — does it transfer to a new machine?
- **1.0 s horizon:** how much recall a twice-as-early warning costs.
- **Short:** the live warning on screen — “AI calls my spin before it happens”.

## CV entry

Fill N, H, R, X, Y, Z and K from the results; put the entry first in Projects with the repo link.

**Spin Early-Warning Model from Vehicle Telemetry** · Python, scikit-learn, pandas · Oct 2026 · GitHub

- Built a dataset of N spin events from H h of R Hz vehicle telemetry (Assetto Corsa physics), labeled automatically with a physical rule and spot-checked against video (K/20 correct).
- Compared logistic regression, random forest and gradient boosting on causal 1 s windows to predict spins 0.5 s ahead, using session-level cross-validation and held-out test sessions.
- Best model caught X% of test spins at 1 false alarm/min vs. Y% for the best tuned physics baseline, with Z s median warning lead time.

**Skills line:** AI/ML: scikit-learn, pandas, NumPy (time-series features, model evaluation)

## Concepts to explain in the exam and interview

This project practises the ML basics a technical exam usually covers; be able to explain each in two sentences.

- Class imbalance and class weights
- Precision, recall, PR-AUC, and why PR-AUC is more informative than ROC-AUC for rare events
- Data leakage: random vs. grouped splits, causal features
- Train/validation/test discipline, and why the test set is used once
- Why a model only means something against a baseline, and why a physics one (time to 90°, like time-to-collision) is the fair one to beat
- Choosing an operating point: recall vs. false alarms
- Overfitting, and tuning with cross-validation
- Permutation importance: what it measures and where it misleads
- Why not a neural network here: about 100 events, tabular features, interpretability; and when you would switch
- Bootstrap confidence intervals, and why 20 test spins can't separate small gaps
