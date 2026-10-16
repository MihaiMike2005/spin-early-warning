"""
Every constant spinwarn uses, in one place.

Times are in seconds, never frames, so nothing depends on the logging rate.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]    # the repo: this file is src/spinwarn/config.py
RAW_DIR = ROOT / "data" / "raw"               # logger CSV + JSON, copied from the AC PC
PROCESSED_DIR = ROOT / "data" / "processed"   # cached tables, rebuilt from RAW_DIR
DATASET_PATH = PROCESSED_DIR / "dataset.parquet"   # every dev session, labelled, with features
SPLITS_PATH = ROOT / "data" / "splits.json"   # the locked test sessions
CV_DIR = ROOT / "results" / "cv"              # out-of-fold scores, one file per CV run
EXPERIMENTS_PATH = ROOT / "experiments.md"    # one row per CV run
CHOSEN_PATH = ROOT / "results" / "chosen.json"   # each method's tuned CV run, for the final run
METRICS_PATH = ROOT / "results" / "metrics.json"   # the final run's results, for report.py
FINAL_LOCK_PATH = ROOT / "results" / "final.lock"  # written as the final run loads the test sessions

GRID_HZ = 60             # raw packets are resampled onto this uniform grid
GAP_S = 0.1               # a longer gap starts a new segment
MIN_SPEED_KMH = 20
SPIN_DEG, REARM_DEG, REARM_S = 90, 45, 2.0
SPIN_HOLD_S = 0.0         # raise to 0.25 if the spot-check flags intended high angles
HORIZON_S = 0.5           # 1.0 for the secondary run
WINDOW_S = 1.0
RATE_WINDOW_S = 0.1       # every derived rate: d|slip|/dt, yaw_acc, yaw rate from heading
TTS_CAP_S = 5.0           # time to 90° when |slip| isn't growing
MERGE_GAP_S = 0.2
CREDIT_EARLY_S, CREDIT_LATE_S, FA_AFTER_S = 1.0, 0.1, 1.0
MIN_LEAD_S = 0.5          # the headline recall counts a catch only this far ahead: HORIZON_S
TRAIN_HZ = 20
FA_TARGETS = (1.0, 0.5)
N_THRESHOLDS = 200        # candidate alert thresholds: score quantiles, finest in the top tail
B2_THETA1_DEG = tuple(range(10, 81, 5))   # B2's |slip| gate: CV keeps the best of 10-80°
N_FOLDS = 4
N_TUNE = 15               # configs run.py tune tries per model: the Project Plan's 10-20
MIN_KIND_SPINS = 10       # test spins a kind (natural, deliberate) needs for its own recall: 1 spin = 10 points
N_PERMUTATIONS = 10       # shuffles per feature for the best model's permutation importance
N_TIMING = 1000           # single-row predictions timed per method
SEED = 0
REVERSE_GEAR = 0          # AC gear: 0 = R, 1 = N, 2 = 1st

# From Step 4's logcheck, the same on all four 1 ms recordings (|r| >= 0.974).
# YAW_SIGN is +1 although logcheck prints "sign -": it correlates against
# +d(heading)/dt, so ang_vel_y = -d(heading)/dt, which is already the
# heading fallback. +1 keeps both ways of getting yaw rate the same sign.
YAW_AXIS, YAW_SIGN = 1, 1           # ang_vel_y; YAW_AXIS None -> yaw rate from heading
LAT_G_AXIS, LAT_G_SIGN = 0, 1       # acc_g_x
LON_G_AXIS, LON_G_SIGN = 2, 1       # acc_g_z
# From Step 8's notebook (held drifts should read positive): steer × sign(slip)
# reads negative on 97-99% of held-drift frames in every session, median -0.56.
COUNTERSTEER_SIGN = -1
