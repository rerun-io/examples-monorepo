# MASt3R-SLAM Bottleneck Analysis

This note is the current optimization brief for MASt3R-SLAM. It summarizes the latest end-to-end profiling on the current async-logging codepath and is meant to answer a simple question: what is still taking time in the real pipeline?

## Pipeline Overview

MASt3R-SLAM runs as two cooperating processes plus one logger thread:

```text
Frontend / tracker                    Backend / optimizer
┌──────────────────────────────┐      ┌──────────────────────────────┐
│ every frame:                 │      │ every new keyframe:          │
│ 1. read + resize image       │      │ 1. retrieval query           │
│ 2. create frame state        │      │ 2. symmetric MASt3R matching │
│ 3. run tracking inference    │ ───▶ │ 3. add factor-graph edges    │
│ 4. estimate current pose     │      │ 4. run global optimization   │
│ 5. enqueue log snapshots     │      │ 5. publish updated poses     │
└──────────────────────────────┘      └──────────────────────────────┘
                 │
                 ▼
         Async logger thread
         ┌──────────────────────────────┐
         │ 1. consume events            │
         │ 2. JPEG / pointmap work      │
         │ 3. log current frame         │
         │ 4. log keyframe/map updates  │
         └──────────────────────────────┘
```

The frontend must keep up with incoming frames. The backend only wakes up when a new keyframe is queued. The logger thread is off the frontend critical path, but it can still be a major consumer of total runtime and viewer throughput.

## What We Measured

The current measurements were collected with a temporary profiling harness that:

- monkeypatched timing counters into the frontend, backend, and async logger
- captured `cProfile` dumps for the frontend process and backend subprocess
- wrote all artifacts to `/tmp/mast3r_profile/out`
- left the checked-in code untouched

The four main runs were:

| Run | Dataset | Config | Wall time | Frames | Keyframes | FPS |
|---|---|---|---:|---:|---:|---:|
| `fast-apt` | `normal-apt-tour.mp4` | `fast.yaml` | `82.35 s` | `504` | `88` | `6.12` |
| `base-apt` | `normal-apt-tour.mp4` | `base.yaml` | `360.49 s` | `2523` | `97` | `7.00` |
| `fast-livingroom` | `livingroom-tour.mp4` | `fast.yaml` | `51.31 s` | `361` | `51` | `7.03` |
| `base-livingroom` | `livingroom-tour.mp4` | `base.yaml` | `222.14 s` | `1807` | `45` | `8.13` |

Two important interpretation notes:

- Frontend, backend, and logger overlap in time, so their percentages do not add to 100%.
- The backend spends a lot of time asleep, which means it is often waiting for the frontend rather than setting end-to-end throughput.

## Normalized Timing Table

This table is the most useful high-level summary:

| Run | Tracking s/frame | Snapshot ms/frame | Logger ms/frame | Backend known ms/keyframe |
|---|---:|---:|---:|---:|
| `fast-apt` | `0.074` | `7.56` | `99.60` | `375.45` |
| `base-apt` | `0.088` | `8.82` | `129.03` | `726.48` |
| `fast-livingroom` | `0.066` | `4.64` | `93.30` | `294.88` |
| `base-livingroom` | `0.075` | `5.15` | `111.53` | `397.87` |

Interpretation:

- Frontend snapshot creation is relatively cheap.
- Frontend tracking is consistently expensive.
- Logger work is still very large, especially on `base`.
- Backend work per keyframe is substantial, but because keyframes are sparse the backend is not always the end-to-end limiter.

## Current Bottlenecks

### 1. Async logging is still a major bottleneck

The async logger removed the old synchronous blocking on the tracking thread, but it did **not** make logging cheap overall.

Evidence:

- Logger work is still about `93-129 ms/frame` across the four measured runs.
- On `base-apt`, logger cumulative time is `325.53 s` versus `222.66 s` in tracking.
- On `base-livingroom`, logger cumulative time is `201.53 s` versus `136.35 s` in tracking.

The important correction to the old conclusion is:

- logging is no longer synchronous on the frontend hot path
- but logging is still one of the largest consumers of total work in the system

### 2. The biggest logger hotspot is current-frame image/map logging

The dominant logger cost is not backend camera pose refresh. It is the repeated current-frame path:

- current-frame camera logging
- pointmap/depth/confidence conversion
- current-frame and last-keyframe image logging

Measured logger breakdown:

| Run | Current-frame handler | Map-update handler | Pointmap/confidence | Camera relog |
|---|---:|---:|---:|---:|
| `fast-apt` | `26.85 s` | `8.11 s` | `6.52 s` | `3.12 s` |
| `base-apt` | `198.01 s` | `18.44 s` | `83.01 s` | `6.25 s` |
| `fast-livingroom` | `19.97 s` | `4.34 s` | `4.58 s` | `1.37 s` |
| `base-livingroom` | `121.44 s` | `7.82 s` | `58.94 s` | `2.83 s` |

This matters because it narrows the optimization target:

- lightweight backend camera refresh was the right fix for pose-update overhead
- but it is not the main remaining logging cost
- the current-frame path is the main logging hotspot now

### 3. Frontend tracking inference is the other persistent top bottleneck

Across all runs, the hottest frontend stack is:

1. `tracker.py:51(track)`
2. `mast3r_utils.py:412(mast3r_match_asymmetric)`
3. `mast3r_utils.py:365(mast3r_asymmetric_inference)`
4. decoder / positional-embedding heavy model code

Evidence:

- `fast-apt`: tracking totals `37.17 s`
- `base-apt`: tracking totals `222.66 s`
- `fast-livingroom`: tracking totals `23.84 s`
- `base-livingroom`: tracking totals `136.35 s`

This is steady, unavoidable compute in the current design and remains one of the biggest places to win time if model-side changes are acceptable.

### 4. Backend work matters, but often overlaps rather than dominating

The backend’s measured active work per keyframe is still real:

| Run | Retrieval | Add factors | GN solve |
|---|---:|---:|---:|
| `fast-apt` | `1.56 s` | `14.76 s` | `16.72 s` |
| `base-apt` | `2.32 s` | `19.67 s` | `48.48 s` |
| `fast-livingroom` | `0.90 s` | `8.09 s` | `6.05 s` |
| `base-livingroom` | `1.04 s` | `8.40 s` | `8.46 s` |

But the backend also spends a lot of cumulative time in `time.sleep`:

- `fast-apt`: `35.57 s`
- `base-apt`: `261.50 s`
- `fast-livingroom`: `23.41 s`
- `base-livingroom`: `183.23 s`

That means the backend is frequently idle, waiting for new keyframes from the frontend. So:

- backend cost is important
- backend cost is not always the throughput limiter
- on many runs, frontend + logger work are the stronger practical bottlenecks

### 5. Backend scaling is still most visible on `base-apt`

`base-apt` is still the clearest backend-stress case:

- `solve_GN_rays`: `48.48 s`
- `add_factors`: `19.67 s`
- `torch.cuda.synchronize`: about `40.25 s` inside the backend profile

So the older intuition about backend scaling was not wrong, but it was incomplete. The current picture is:

- backend optimization is expensive on harder `base` runs
- logger and frontend costs are still large enough that backend is not the whole story

## What Has Already Changed

Several useful changes are already in the repo:

- Async logging moved expensive Rerun work off the tracking thread.
- Backend camera refresh now uses lightweight pose-only refresh events instead of full dirty-keyframe snapshots.
- Dirty keyframe pose updates are used instead of relogging all keyframe poses every frame.
- The latest temporary profiler harness measured frontend, logger, and backend separately without modifying tracked files.

## Updated Conclusions

The old summary "`logging is no longer a bottleneck`" is too strong and should be considered outdated.

The current picture is:

- Async logging was a major win for frontend responsiveness.
- Logging is still one of the largest total costs in the system.
- The main remaining logging hotspot is the current-frame path, especially pointmap/depth/confidence work.
- Frontend tracking inference is the other consistent top bottleneck.
- Backend pair construction and Gauss-Newton are still expensive, especially on `base-apt`, but the backend is often idle and therefore not always the end-to-end limiter.

## Recommended Next Steps

1. Optimize current-frame logging first.
   The highest-value targets are:
   - current image logging
   - current pointmap/depth/confidence logging
   - last-keyframe 2D-panel relogging

2. Keep backend camera refresh lightweight.
   The pose-only refresh path was the right direction and should not be regressed back to full dirty-keyframe replay.

3. Then focus on frontend tracking inference.
   The main candidates are:
   - decoder-side optimization
   - AMP / BF16 where safe
   - reducing expensive per-frame model work

4. After that, optimize backend scaling on the hardest runs.
   The main candidates are:
   - reduce pair volume
   - reduce solve frequency
   - local/sliding-window optimization
   - investigate backend synchronization overhead on `base-apt`

## Bottom Line

If you are looking at MASt3R-SLAM performance today, the most accurate short summary is:

- async logging fixed the worst synchronous frontend blockage
- the logger is still expensive, especially for current-frame map visualization
- frontend tracking inference is still a major cost
- backend optimization matters, but it is not the only bottleneck and is often not the immediate throughput limiter
