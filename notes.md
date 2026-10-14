 # Session log

One line per session. Spins are the driver's rough count; Step 7's labeler gives the real number. "t = 0 in video" is the video time at which the logger's status line shows t = 0, read off the recording: the video second the clock ticks over to, minus the t shown on that frame. For a new session, the file names give it to about a second (the logger's start minus the video's start) until it's read.

CSV and JSON are in `data/raw/`. The videos stay outside the repo, in `../driftsessions/videos/`.

| Session | Date | Car | Track | Setup | Live min | Spins | Kind | Video | t = 0 in video | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-10-11_194646` | 2026-10-11 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 14.5 | ~40 | natural | `2026-10-11 19-46-42.mp4` | 3.89 s | 0.7 s pause at 552 s, then AC stalled for 0.11 s |
| `2026-10-11_200553` | 2026-10-11 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 8.2 | 20+ | deliberate | `2026-10-11 20-05-46.mp4` | 7.4 s | Logged as natural by mistake; JSON `kind` fixed by hand |
| `2026-10-11_202102` | 2026-10-11 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 10.0 | 30+ | natural | `2026-10-11 20-20-57.mp4` | 4.65 s | |
| `2026-10-14_223131` | 2026-10-14 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 6.5 | 1 (labeler) | natural | none | | Video deleted by mistake, so this one can't be spot-checked |
| `2026-10-14_224308` | 2026-10-14 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 11.6 | 4 (labeler) | natural | `2026-10-14 22-43-05.mp4` | ~3 s | |
| `2026-10-14_225818` | 2026-10-14 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 13.9 | 28 (labeler) | deliberate | `2026-10-14 22-58-14.mp4` | ~4 s | |
| `2026-10-14_233636` | 2026-10-14 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 10.2 | 17 (labeler) | deliberate | `2026-10-14 23-36-28.mp4` | ~8 s | |

## Label spot-check (Step 8)

20 seeded-random onsets, split by each session's spins (from `notebooks/01_explore.ipynb`), with each session's offset from the table above. Watch each from about 3 s before its Video time to 3 s after, and mark Hit? y (a real spin) or n (a held or saved drift, or nothing). Under 18/20, see the Build Plan's Step 8 results before setting `SPIN_HOLD_S = 0.25`.

`python -m spinwarn.spotcheck 2026-10-11_194646 --n 8 --video-offset 3.89`

| Spin | Video | t (s) | Peak slip (°) | Speed (km/h) | Hit? |
| --- | --- | --- | --- | --- | --- |
| 1 | 00:35.1 | 31.2 | 180 | 59 | y |
| 3 | 01:51.4 | 107.5 | 91 | 22 | y |
| 4 | 02:46.4 | 162.5 | 134 | 46 | y |
| 5 | 03:56.5 | 232.7 | 147 | 67 | y |
| 6 | 07:06.7 | 422.9 | 95 | 24 | y |
| 8 | 08:35.0 | 511.1 | 95 | 25 | y |
| 13 | 12:25.8 | 741.9 | 110 | 59 | y |
| 14 | 13:48.5 | 824.6 | 95 | 35 | y: past sideways, into the grass, then drove on |

`python -m spinwarn.spotcheck 2026-10-11_200553 --n 7 --video-offset 7.4`

| Spin | Video | t (s) | Peak slip (°) | Speed (km/h) | Hit? |
| --- | --- | --- | --- | --- | --- |
| 1 | 00:38.3 | 30.9 | 179 | 66 | y: a 360 |
| 3 | 01:13.2 | 65.8 | 180 | 53 | y: a 360, into the grass |
| 4 | 01:35.2 | 87.8 | 179 | 46 | y |
| 5 | 02:15.6 | 128.2 | 175 | 44 | y |
| 6 | 02:53.6 | 166.2 | 91 | 37 | y |
| 8 | 04:41.9 | 274.5 | 117 | 48 | y |
| 12 | 07:27.6 | 440.2 | 179 | 95 | y: past 180°, then hit the wall at −157°, 1.1 s after onset |

`python -m spinwarn.spotcheck 2026-10-11_202102 --n 5 --video-offset 4.65`

| Spin | Video | t (s) | Peak slip (°) | Speed (km/h) | Hit? |
| --- | --- | --- | --- | --- | --- |
| 3 | 02:54.3 | 169.7 | 94 | 51 | y |
| 4 | 03:24.3 | 199.7 | 119 | 66 | y |
| 5 | 04:14.5 | 249.8 | 116 | 32 | y |
| 6 | 05:58.0 | 353.3 | 110 | 28 | y |
| 8 | 08:34.2 | 509.5 | 180 | 74 | y |

Hits: 20 / 20

As first marked, it was 18/20: `200553` spins 1 and 12 were n because they were 360s, deliberate or ending in a wall. Marked again by whether the car got past sideways, they're hits. Telemetry puts spin 12's wall impact (15 g longitudinal) 1.1 s after its onset. Two hits are edge cases for the README's limitations: spin 12 ends in a wall, and `194646` spin 14 peaks at 95° and is driven out of. Even counting both as misses it's 18/20, so the rule stands with `SPIN_HOLD_S = 0.0`.

Extras, outside the 20: the two spins `SPIN_HOLD_S = 0.25` would drop that the random pick leaves out.

| Session | Spin | Video | t (s) | Peak slip (°) | Speed (km/h) | Hit? |
| --- | --- | --- | --- | --- | --- | --- |
| `200553` | 9 | 05:53.5 | 346.1 | 101 | 24 | y |
| `202102` | 10 | 09:59.4 | 594.7 | 125 | 21 | y |

Missed spins in 5 minutes of video, `202102` from 00:00 to 05:00. The rule's onsets in that stretch are at 00:32.0, 02:33.0, 02:54.3, 03:24.3 and 04:14.5; a clear spin on video more than a second or two from all of them is a miss.

Missed: 0

## Test runs, not in the dataset

Both on Brooklyn Park. Their CSV and JSON are in `../driftsessions/test-runs/`, not `data/raw/`, which would put them in the dataset. Their videos are in `../driftsessions/videos/`.

- `2026-10-11_120846`: first test, on the old 5 ms poll. Video `2026-10-11 12-09-46.mp4`.
- `2026-10-11_191720`: the Step 4 live check. Video `2026-10-11 19-17-11.mp4`, t = 0 at 8.8 s (read off the status line). A candidate for `data/sample/`.
