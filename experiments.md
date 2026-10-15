# Experiments

One row per run, appended by `python -m spinwarn.run cv`. Run names the file in `results/cv/` (gitignored) that holds the run's out-of-fold scores, folds and metrics. The headline is recall at 1 FA/min counting only catches with at least 0.5 s of lead; the column after it is the Project Plan's rule, any catch (0.1 s or more ahead).

| Date | Run | Method | Horizon (s) | Features | Params | PR-AUC | Recall @ 1 FA/min, lead ≥ 0.5 s | Recall @ 1 FA/min | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-15 | `2026-10-15_145051_b1` | b1 | 0.5 | `abs_slip` | `{}` | 0.440 | 0.17 (10/58) | 1.00 (58/58) | Step 12 baseline |
| 2026-10-15 | `2026-10-15_145053_b2` | b2 | 0.5 | `abs_slip`, `abs_slip_rate` | `{"theta1": 30}` | 0.561 | 0.69 (40/58) | 0.81 (47/58) | θ1 best of 30–80° by recall @ 1 FA/min, lead ≥ 0.5 s; Step 12 baseline |
| 2026-10-15 | `2026-10-15_145054_b3` | b3 | 0.5 | `time_to_90` | `{}` | 0.838 | 0.78 (45/58) | 0.98 (57/58) | Step 12 baseline |
| 2026-10-15 | `2026-10-15_145240_b2` | b2 | 0.5 | `abs_slip`, `abs_slip_rate` | `{"theta1": 25}` | 0.535 | 0.72 (42/58) | 0.81 (47/58) | θ1 best of 10–80° by recall @ 1 FA/min, lead ≥ 0.5 s; Step 12 baseline; θ1 grid widened from 30–80°, whose best was its edge |
