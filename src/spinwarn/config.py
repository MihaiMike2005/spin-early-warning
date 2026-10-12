"""
Every constant spinwarn uses, in one place.

Times are in seconds, never frames, so nothing depends on the logging rate.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]    # the repo: this file is src/spinwarn/config.py
RAW_DIR = ROOT / "data" / "raw"               # logger CSV + JSON, copied from the AC PC
PROCESSED_DIR = ROOT / "data" / "processed"   # cached tables, rebuilt from RAW_DIR
SPLITS_PATH = ROOT / "data" / "splits.json"   # the locked test sessions

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
TRAIN_HZ = 20
FA_TARGETS = (1.0, 0.5)
N_FOLDS = 4
SEED = 0
REVERSE_GEAR = 0          # AC gear: 0 = R, 1 = N, 2 = 1st

# From Step 4's logcheck, the same on all four 1 ms recordings (|r| >= 0.974).
# YAW_SIGN is +1 although logcheck prints "sign -": it correlates against
# +d(heading)/dt, so ang_vel_y = -d(heading)/dt, which is already the
# heading fallback. +1 keeps both ways of getting yaw rate the same sign.
YAW_AXIS, YAW_SIGN = 1, 1           # ang_vel_y; YAW_AXIS None -> yaw rate from heading
LAT_G_AXIS, LAT_G_SIGN = 0, 1       # acc_g_x
LON_G_AXIS, LON_G_SIGN = 2, 1       # acc_g_z
# From Step 8's notebook (held drifts should read positive):
COUNTERSTEER_SIGN = 1
