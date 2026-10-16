# Results

Written by `python -m spinwarn.report` from `metrics.json`. The README's results section has the same tables, and the plots.

From `results/metrics.json`: the final run of 2026-10-16 at the 0.5 s horizon, which scored the test sessions once. The test sessions hold 27 spins in 16.4 scored min: `2026-10-11_202102` (natural, 10 spins), `2026-10-14_233636` (deliberate, 17 spins). Each method's thresholds were set on CV, out of fold over 5 dev sessions (58 spins in 46.3 scored min), and each model was refit on all of them. Recall counts a spin caught with at least 0.5 s of lead (the headline) or with any lead from 0.1 s (the Project Plan's rule). Intervals are bootstrap 95%, the test spins resampled 1,000 times; one test spin is 3.7 points of recall.

## At the 1 FA/min threshold

Each method at its threshold for 1 false alarm a minute: the lowest that kept to it on CV. FA/min on test is the rate it gave on the test sessions.

| Method | Recall, lead ≥ 0.5 s | Recall, any lead | FA/min on test | Alert precision | Time in alarm | Median lead (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B1: \|slip\| | 0.33 (9/27) [0.15, 0.52] | 1.00 (27/27) [1.00, 1.00] | 1.10 | 0.60 | 2.7% | 0.28 |
| B2: slip rate, \|slip\| > 25° | 0.59 (16/27) [0.41, 0.78] | 0.63 (17/27) [0.44, 0.78] | 0.67 | 0.61 | 1.3% | 0.67 |
| B3: time to 90° | 0.93 (25/27) [0.81, 1.00] | 1.00 (27/27) [1.00, 1.00] | 1.22 | 0.59 | 2.6% | 0.55 |
| Logistic regression | 0.89 (24/27) [0.78, 1.00] | 1.00 (27/27) [1.00, 1.00] | 0.98 | 0.63 | 2.9% | 0.65 |
| Random forest | 0.93 (25/27) [0.81, 1.00] | 1.00 (27/27) [1.00, 1.00] | 0.91 | 0.64 | 2.7% | 0.62 |
| Gradient boosting | 0.93 (25/27) [0.81, 1.00] | 1.00 (27/27) [1.00, 1.00] | 1.40 | 0.54 | 2.6% | 0.62 |

## At the 0.5 FA/min threshold

Each method at its threshold for 0.5 false alarms a minute: the lowest that kept to it on CV. FA/min on test is the rate it gave on the test sessions.

| Method | Recall, lead ≥ 0.5 s | Recall, any lead | FA/min on test | Alert precision | Time in alarm | Median lead (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B1: \|slip\| | 0.19 (5/27) [0.04, 0.33] | 0.96 (26/27) [0.89, 1.00] | 0.67 | 0.71 | 1.6% | 0.23 |
| B2: slip rate, \|slip\| > 25° | 0.56 (15/27) [0.37, 0.74] | 0.59 (16/27) [0.41, 0.74] | 0.30 | 0.76 | 1.0% | 0.64 |
| B3: time to 90° | 0.19 (5/27) [0.04, 0.33] | 1.00 (27/27) [1.00, 1.00] | 0.85 | 0.67 | 1.7% | 0.40 |
| Logistic regression | 0.74 (20/27) [0.59, 0.89] | 1.00 (27/27) [1.00, 1.00] | 0.79 | 0.68 | 2.3% | 0.53 |
| Random forest | 0.78 (21/27) [0.63, 0.93] | 1.00 (27/27) [1.00, 1.00] | 0.79 | 0.68 | 2.0% | 0.57 |
| Gradient boosting | 0.78 (21/27) [0.59, 0.93] | 1.00 (27/27) [1.00, 1.00] | 0.61 | 0.74 | 2.0% | 0.55 |

## Natural and deliberate spins

The headline per kind. FA/min is the rate in that kind's test sessions. A kind with under 10 spins says so.

| Method | Spins | @ 1 FA/min: recall, lead ≥ 0.5 s | FA/min | @ 0.5 FA/min: recall, lead ≥ 0.5 s | FA/min |
| --- | --- | --- | --- | --- | --- |
| B1: \|slip\| | deliberate (17) | 0.35 (6/17) [0.12, 0.59] | 0.91 | 0.18 (3/17) [0.00, 0.35] | 0.65 |
|  | natural (10) | 0.30 (3/10) [0.00, 0.60] | 1.26 | 0.20 (2/10) [0.00, 0.50] | 0.69 |
| B2: slip rate, \|slip\| > 25° | deliberate (17) | 0.65 (11/17) [0.41, 0.88] | 0.91 | 0.65 (11/17) [0.41, 0.88] | 0.52 |
|  | natural (10) | 0.50 (5/10) [0.20, 0.80] | 0.46 | 0.40 (4/10) [0.10, 0.70] | 0.11 |
| B3: time to 90° | deliberate (17) | 0.94 (16/17) [0.82, 1.00] | 1.30 | 0.18 (3/17) [0.00, 0.35] | 0.91 |
|  | natural (10) | 0.90 (9/10) [0.70, 1.00] | 1.15 | 0.20 (2/10) [0.00, 0.50] | 0.80 |
| Logistic regression | deliberate (17) | 0.94 (16/17) [0.82, 1.00] | 1.04 | 0.82 (14/17) [0.65, 1.00] | 1.04 |
|  | natural (10) | 0.80 (8/10) [0.50, 1.00] | 0.92 | 0.60 (6/10) [0.30, 0.90] | 0.57 |
| Random forest | deliberate (17) | 1.00 (17/17) [1.00, 1.00] | 1.04 | 0.82 (14/17) [0.65, 1.00] | 0.91 |
|  | natural (10) | 0.80 (8/10) [0.50, 1.00] | 0.80 | 0.70 (7/10) [0.40, 1.00] | 0.69 |
| Gradient boosting | deliberate (17) | 1.00 (17/17) [1.00, 1.00] | 1.17 | 0.82 (14/17) [0.65, 1.00] | 0.91 |
|  | natural (10) | 0.80 (8/10) [0.50, 1.00] | 1.61 | 0.70 (7/10) [0.40, 0.90] | 0.34 |

## Each model against B3 and B2, spin by spin

On the same 27 test spins, by the headline: the spins one catches with at least 0.5 s of lead and the other doesn't, and the difference in recall with a paired bootstrap 95% interval (both methods on the same 1,000 resamples of the spins). Each method runs at its own threshold from CV, so at its own false-alarm rate on test, as in the tables above.

| Model | Against | @ 1 FA/min: model only / baseline only | Difference in recall | @ 0.5 FA/min: model only / baseline only | Difference in recall |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | B3: time to 90° | 2 / 3 | −0.04 [−0.19, 0.11] | 16 / 1 | +0.56 [0.33, 0.74] |
| Random forest | B3: time to 90° | 2 / 2 | +0.00 [−0.15, 0.15] | 16 / 0 | +0.59 [0.41, 0.78] |
| Gradient boosting | B3: time to 90° | 2 / 2 | +0.00 [−0.15, 0.15] | 17 / 1 | +0.59 [0.37, 0.81] |
| Logistic regression | B2: slip rate, \|slip\| > 25° | 9 / 1 | +0.30 [0.11, 0.48] | 6 / 1 | +0.19 [0.04, 0.37] |
| Random forest | B2: slip rate, \|slip\| > 25° | 9 / 0 | +0.33 [0.15, 0.52] | 8 / 2 | +0.22 [0.00, 0.44] |
| Gradient boosting | B2: slip rate, \|slip\| > 25° | 9 / 0 | +0.33 [0.15, 0.52] | 8 / 2 | +0.22 [0.00, 0.44] |

## Configs, PR-AUC and time per prediction

PR-AUC is average precision on the scored frames: out of fold on CV, and the refit model on test. Time per prediction is the median of single-row predictions on scored test frames, timed by the final run.

| Method | Params | Chosen by | PR-AUC, CV → test | ms per prediction |
| --- | --- | --- | --- | --- |
| B1: \|slip\| | — | nothing to tune | 0.440 → 0.410 | 0.00512 |
| B2: slip rate, \|slip\| > 25° | `{"theta1": 25}` | θ1 best of 10–80° by recall @ 1 FA/min, lead ≥ 0.5 s | 0.535 → 0.485 | 0.0112 |
| B3: time to 90° | — | nothing to tune | 0.838 → 0.778 | 0.00538 |
| Logistic regression | `{"C": 0.01}` | PR-AUC, best of 4 configs | 0.873 → 0.835 | 0.0752 |
| Random forest | `{"n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.5, "max_depth": null}` | PR-AUC, best of 15 configs | 0.865 → 0.939 | 13.2 (3.35 on one core) |
| Gradient boosting | `{"min_samples_leaf": 50, "max_leaf_nodes": 15, "learning_rate": 0.1, "l2_regularization": 1}` | PR-AUC, best of 15 configs | 0.904 → 0.933 | 1.53 |
