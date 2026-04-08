# MASt3R-SLAM Logging Handoff

Goal: reimplement the logging refactor cleanly on top of newer upstream changes. The code from this branch was intentionally removed.

## Problem

Current bottleneck is Rerun logging in the hot path.

- Frontend used to do synchronous per-frame logging.
- Logger work scaled with keyframe count.
- This materially slowed `fast.yaml`.

The local note in [bottleneck-analysis.md](/var/tmp/vibe-kanban/worktrees/3258-mast3r-slam-mult/examples-monorepo/packages/mast3r-slam/docs/bottleneck-analysis.md) was directionally correct: logging should be decoupled from tracking.

## Recommended Architecture

Use explicit event-driven logging:

1. Frontend process
   - dataset I/O
   - tracking
   - current-frame snapshots
   - new keyframe static snapshots
2. Backend process
   - retrieval / relocalization
   - factor-graph updates
   - optimized keyframe pose updates
   - backend text logs
3. Logger process or worker
   - the only owner of scene/text Rerun logging
   - consumes immutable CPU snapshots/events

Do not have the logger poll `SharedStates` / `SharedKeyframes` under locks.

## Event Model That Worked

These event shapes were enough:

- `LogSessionInit`
- `LogCurrentFrame`
- `LogKeyframeStatic`
- `LogKeyframePoseUpdate`
- `LogEdgesUpdate`
- `LogText`
- `LogTerminate`

Important rules:

- Only send CPU-serializable data.
  Why: it keeps the logger isolated from CUDA IPC, shared-memory lifetime, and lock ownership issues.
- Do not send live `Frame` objects or shared tensors.
  Why: the logger should consume immutable snapshots, not reach back into live frontend/backend state.
- Only current-frame updates may be droppable/coalesced.
  Why: they are transient UI updates, so dropping stale ones trades freshness for throughput safely.
- Structural updates must not be dropped.
  Why: keyframe creation, pose updates, and edges define map state; if they are lost the recording becomes inconsistent.

## Rerun-Specific Findings

Use partial-update semantics:

- log keyframe image / pointmap / confidence once
- later log only pose updates
- log edges only when changed
- log current-frame entities every frame

Keep existing entity paths stable:

- `/world/current_camera`
- `/world/path`
- `/world/edges`
- keyframe entities
- dense pointmap / depth / confidence entities

`send_columns` is worth exploring for current-frame image-like data where the producer already has the full batch or an obvious sequence to ship. It seems like the most promising place to use it is the "current" image/depth/confidence stream, not structural state like keyframes, poses, or edges. I would still keep the primary architecture event-driven and use `send_columns` as a targeted optimization once correctness is stable.

## Critical Gradio Constraint

Gradio is not the same as the CLI path.

Important historical context: at commit `2500a5fee4fbd2bd8d26e693e0430bfc991fecf2` (`Extract shared SLAM pipeline generator for CLI + Gradio parity`), the new [architecture.md](/var/tmp/vibe-kanban/worktrees/3258-mast3r-slam-mult/examples-monorepo/packages/mast3r-slam/docs/architecture.md) and Gradio code still used `@rr.thread_local_stream(...)` plus `rr.binary_stream()`. So this was not just a quirk of my experimental branch.

That said, do not assume current upstream still does this. Re-check the active Gradio recording setup first. The core constraint is stable either way: the logger must write into the same recording the embedded viewer is reading from.

This was the exact failure mode seen during testing:

- async logger thread started
- logger handled `LogSessionInit`
- `rr.send_blueprint(...)` failed with:
  - `ValueError: No application id found. You must call rerun.init before sending a blueprint.`

Root cause:

- the threaded logger had no bound recording
- in the tested Gradio architecture, the logger thread had to write into the same recording used by the request’s `binary_stream`

If a thread-based logger is used for Gradio, it must explicitly inherit/bind the active recording before logging. A separate process is likely simpler for CLI, but Gradio needs special handling around recording ownership.

## Measured Results

Benchmark strategy: compare the pre-refactor commit against the experimental refactor commit with the exact same command, dataset, and config. That is a reasonable way to isolate whether the logging change moved end-to-end runtime, even before deeper profiling.

CLI benchmark command:

```bash
pixi run -e mast3r-slam --frozen python tools/mast3r_slam_inference.py \
  --dataset data/normal-apt-tour.mp4 \
  --img-size 224 \
  --config config/fast.yaml \
  --rr-config.serve
```

Observed wall time:

- before refactor (`c9cd81b`): `105.68s`
- async-logging version (`4629289`): `62.77s`

Directional conclusion: decoupling logging produced a large CLI speedup, about `1.68x`.

## Gradio Bench Notes

Prepared example used in browser:

- `normal-apt-tour.mp4`
- likely worth changing the default `gr.Examples` config to subsample every `4` frames rather than `1` for routine Gradio testing; the full-rate example is useful, but expensive

Playwright flow:

1. launch `tools/gradio-app.py`
2. open the app with `playwright-cli`
3. remove the stuck Rerun loading overlay if needed:
   - remove `[data-testid="status-tracker"]`
4. click the prepared example button
5. click `Run MASt3R-SLAM`

Observed behavior:

- old code path (`c9cd81b`) completed successfully in about `14m 32s`
- initial async-logging Gradio version failed about `11s` after clicking Run due to the recording-init issue above
- after binding the logger thread to the active recording, Gradio progressed normally again

## Validation To Repeat After Reimplementation

1. CLI regression / structure
   - ensure entity paths above are still present
2. CLI performance
   - rerun the `fast.yaml` benchmark command
3. Gradio smoke test
   - use the prepared example and click through with `playwright-cli`
   - confirm it processes frames instead of entering `Error`
4. Gradio full run
   - let one run finish end-to-end

## Suggested Reimplementation Strategy

- Rebuild this on top of upstream, not by reviving the removed code.
- Start with CLI event-driven logging first.
- Then handle Gradio as a separate recording-ownership problem.
- Keep the implementation narrow:
  - no shared-state polling in the logger
  - no broad performance tuning outside logging
  - preserve existing Rerun paths and viewer layout
