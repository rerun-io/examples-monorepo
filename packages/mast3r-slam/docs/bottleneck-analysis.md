# MASt3R-SLAM Bottleneck Analysis

This note is the current optimization brief for MASt3R-SLAM. It is written for someone new to the repo and focuses on what we have measured, what is already known to be slow, and what to do next.

## Pipeline Overview

MASt3R-SLAM runs as two cooperating processes:

```
Frontend / tracker                    Backend / optimizer
┌──────────────────────────────┐      ┌──────────────────────────────┐
│ every frame:                 │      │ every new keyframe:          │
│ 1. read + resize image       │      │ 1. retrieval query           │
│ 2. create frame state        │      │ 2. symmetric MASt3R matching │
│ 3. run tracking inference    │ ───▶ │ 3. add factor-graph edges    │
│ 4. estimate current pose     │      │ 4. run global optimization    │
│ 5. log to Rerun              │      │ 5. publish updated poses      │
└──────────────────────────────┘      └──────────────────────────────┘
```

The frontend is responsible for keeping up with incoming frames. The backend is responsible for heavier map-building work that happens when a frame becomes a keyframe.

## What We Measured

The benchmark artifacts live under [packages/mast3r-slam/benchmark](/var/tmp/vibe-kanban/worktrees/f711-mast3r-slam-infe/examples-monorepo/packages/mast3r-slam/benchmark). The most useful runs are:

| Run | FPS | Avg frame time | Avg logging time |
|---|---:|---:|---:|
| `fast-full` | 5.51 | 181.45 ms | 56.54 ms |
| `fast-no-logging-full` | 7.88 | 126.98 ms | 0.00 ms |
| `base-full` | 5.51 | 181.51 ms | 54.72 ms |
| `base-no-logging-full` | 8.51 | 117.53 ms | 0.00 ms |

The headline result is simple:

- Logging is a major source of slowdown in the default logged path.
- Disabling logging recovers about `+2.36 FPS` on `fast` and `+3.00 FPS` on `base`.
- Once logging is removed, the frontend becomes much more stable.
- After that, the main remaining bottleneck on `base` is backend scaling, especially MASt3R pair construction and global optimization.

## Current Bottlenecks

### 1. Rerun logging overhead in the default configuration

This is the main avoidable bottleneck in the current user-facing pipeline.

Why we know this:

- `fast` improved from `5.51 FPS` to `7.88 FPS` when logging was disabled.
- `base` improved from `5.51 FPS` to `8.51 FPS` when logging was disabled.
- The logged runs spend roughly `55 ms/frame` in logging alone.

What is still expensive in the current logging path:

- current-camera RGB, pointmap, depth, and confidence are produced every frame
- camera path is rebuilt from full history every frame
- factor-graph edge geometry is rebuilt from the full edge set
- `last_keyframe` image and maps are relogged every frame
- array conversion and image compression happen synchronously on the tracking thread

One important update: keyframe pose logging is already better than it used to be. The code now uses dirty keyframe pose updates instead of relogging every keyframe transform every frame. That helped, but it was not enough to remove the main logging cost.

### 2. Backend MASt3R pair construction

The backend still spends most of its time building new factors for incoming keyframes. In practice, this is dominated by decoder work plus dense matching.

Evidence:

- `fast-no-logging-full`: backend `add_factors` averages `139.13 ms`
- `base-no-logging-full`: backend `add_factors` averages `181.44 ms`

This is the main backend bottleneck once logging is out of the way.

### 3. Backend global optimization growth

On longer runs, backend optimization time grows with the map.

Evidence:

- `base-no-logging-full`: backend `global_opt` averages `167.24 ms`
- `base-full`: backend task time grows by about `4.48 ms` per keyframe

At `base` resolution, global optimization is nearly as expensive as pair construction. That means backend scaling, not just frontend logging, becomes a real problem on long sequences.

### 4. Tracking cost

Tracking is not the main source of the large logged-vs-no-logging gap, but it is still a meaningful steady-state cost.

Evidence:

- `fast-no-logging-full`: avg tracking time `82.58 ms`
- `base-no-logging-full`: avg tracking time `90.06 ms`

This matters, but it is not the first thing to optimize while the logged path is still paying large synchronous visualization costs.

## What Has Already Changed

Several useful changes are already in the repo:

- Deep frontend/backend benchmark instrumentation was added.
- Full benchmark outputs were committed for `fast`, `base`, and no-logging variants.
- A `--disable-logging` benchmark mode was added so logging cost can be isolated directly.
- Keyframe pose logging was improved to use dirty updates rather than relogging every keyframe pose each frame.

These changes were enough to prove that logging overhead was real and to quantify its impact.

## Logging Optimization Plan

The next logging change should preserve the same viewer experience while removing logging from the tracking critical path.

The current direction is:

1. Move logging to a dedicated worker process.
2. Have that worker initialize Rerun through the existing `RerunTyroConfig`.
3. Keep the frontend focused on producing tracking results, not compressing and sending visualization data.
4. Use Rerun's latest-at and partial-update model correctly:
   - log static camera/pinhole state once
   - send only changed transforms or changed components for dynamic entities
   - stop rebuilding full-history entities every frame
5. Keep blueprint handling static instead of refreshing it as the map grows.

This should address the main current issue: logging is still synchronous and still does too much repeated work on the frontend thread.

## Recommended Next Steps

1. Optimize the logging path first.
   Use a worker process, reuse `RerunTyroConfig`, and remove unnecessary full-history relogs.

2. Rerun the full logged benchmarks.
   The goal is to confirm that the logged path gets close to the no-logging baseline.

3. After logging is fixed, focus on backend pair construction.
   The first candidates are decoder-side improvements such as AMP/BF16, `torch.compile`, and reducing backend pair volume.

4. Then reduce backend optimization scaling.
   The main candidates are sliding-window or local-subgraph optimization and solving less often.

## Bottom Line

If you are looking at MASt3R-SLAM performance for the first time, the current picture is:

- With normal logging enabled, logging is a major bottleneck.
- Without logging, the system is much faster and more stable.
- After logging is addressed, the next major problem is backend scaling, especially decoder-heavy pair construction and global optimization on larger maps.
