# Session log

One line per session. Spins are the driver's rough count; Step 7's labeler gives the real number. "t = 0 in video" is the logger's start minus the video's start, both from file names, so it's only good to about a second: Step 8 reads the exact offset off the status line in the video.

CSV and JSON are in `data/raw/`. The videos stay outside the repo, in `../driftsessions/videos/`.

| Session | Date | Car | Track | Setup | Live min | Spins | Kind | Video | t = 0 in video | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-10-11_194646` | 2026-10-11 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 14.5 | ~40 | natural | `2026-10-11 19-46-42.mp4` | ~4 s | 0.7 s pause at 552 s, then AC stalled for 0.11 s |
| `2026-10-11_200553` | 2026-10-11 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 8.2 | 20+ | deliberate | `2026-10-11 20-05-46.mp4` | ~7 s | Logged as natural by mistake; JSON `kind` fixed by hand |
| `2026-10-11_202102` | 2026-10-11 | hush_nissan_silvia_s14 | rmi_gpk_sepang | TC off | 10.0 | 30+ | natural | `2026-10-11 20-20-57.mp4` | ~5 s | |

## Test runs, not in the dataset

Both on Brooklyn Park. Their CSV and JSON are in `~/Downloads/raw 2`; only their videos are in `../driftsessions/videos/`.

- `2026-10-11_120846`: first test, on the old 5 ms poll. Video `2026-10-11 12-09-46.mp4`.
- `2026-10-11_191720`: the Step 4 live check. Video `2026-10-11 19-17-11.mp4`, t = 0 at 8.8 s (read off the status line). A candidate for `data/sample/`.
