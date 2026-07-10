# gsplat-rust-renderer Goal Status

Last updated: 2026-07-10T08:30:00-07:00

## Success criteria

- [x] **FPS:** at least 30 FPS with a continuously moving `EyeControls3D` camera at 1920x1080, verified with `GSPLAT_FPS_PROBE=1` on lego, hotdog, chair, drums, ficus, materials, mic, and ship.
- [x] **Quality:** standalone `gsplat-render` output matches each checkpoint's published full-200-image PSNR/SSIM. Checkpoint prediction renders must first validate the evaluator.
- [x] **Brush:** a pixi-managed osx-arm64 Brush trainer exports PLY snapshots, and this package replays retained snapshots on an `iterations` timeline with rerun-sdk 0.34.1 into an RRD no larger than about 2 GB.
- [x] All required Rust, Python, and GPU regression gates are green.
- [x] Pixel and numeric evidence is published on `/tmp/fleet-artifacts/gsplat-goals.html`.

## Session 4 refinements

- [x] Audit Brush v0.3.0 CLI and pinned source: pure-trainer stdout provides eval PSNR/SSIM and refine splat counts, but discards train loss.
- [x] Add a minimal pinned-source patch that emits the already-computed loss every 50 iterations without enabling Brush's embedded Rerun; build it project-locally through Pixi with `cargo --locked` and delete the 2.3 GB build tree afterward.
- [x] TDD incremental metric parsing, static JPEG camera images, the rich replay default, and final-log synchronization.
- [x] Pass a real 100-step Metal smoke: 100 training cameras, four metric series, two splat snapshots, a 6.4 MiB RRD, and full-frame saved-RRD pixel validation.
- [x] Complete a fresh 30,000-step Lego run and replace the Brush RRD/screenshots/progression video with rich evidence.
- [x] Regenerate all eight pretrained orbit videos at 0.2 rad/s with measured wall-clock playback pacing and inspect sampled frames.
- [x] Run the complete gate set and update final evidence.
- [x] Commit locally without pushing.

## Session 5 refinements

- [x] Audit pinned Brush v0.3.0's eval logging and current Brush blueprint source (`3b809857...`) before changing the replay.
- [x] TDD the 7,000-step default, eight-snapshot retention, canonical metric paths, JPEG eval GT/render logging, and race-safe complete-eval cleanup.
- [x] Pass a real 100-step Metal smoke with Brush-native eval output, all 100 temporary eval PNGs consumed/deleted, a 2.94 MB RRD, and full-frame saved-RRD pixel validation.
- [x] Complete the fresh 7,000-step Lego run and replace the Brush RRD/screenshots/progression video.
- [x] Run the complete gate set, refresh final evidence, and commit locally without pushing.

## Current evidence

### FPS

The unoptimized release viewer was measured from a fresh process per scene with `GSPLAT_FPS_PROBE=1`, a 1920x1080 headless window, collapsed panels, and the Brush visualization's fixed `EyeControls3D` 0.2-rad/s orbital spin. Values are the 300-sample prepare-period EMA:

| Scene | Splats | EMA ms | Baseline FPS |
|---|---:|---:|---:|
| lego | 325,000 | 84.15 | 11.9 |
| hotdog | 150,000 | 68.37 | 14.6 |
| chair | 270,000 | 78.77 | 12.7 |
| drums | 350,000 | 84.12 | 11.9 |
| ficus | 300,000 | 77.62 | 12.9 |
| materials | 290,000 | 81.57 | 12.3 |
| mic | 320,000 | 80.50 | 12.4 |
| ship | 330,000 | 87.20 | 11.5 |

Probe logs and the JSON summary are under `/tmp/fleet-artifacts/gsplat-goals/fps-baseline/`. Lego's live intersection demand was about 0.67–0.70M entries while its capacity-sized tile sort dispatched over 16.8M slots, making capacity-sized radix work the first measured optimization target.

Optimization 1 makes the tile radix count/scatter dispatch over GPU-authored live-intersection workgroup dimensions. Lego improves from 11.9 to 12.7 FPS (+6.7%). The complete 1,600-frame standalone output hash guard remains bit-exact for all eight scenes; results are in `/tmp/fleet-artifacts/gsplat-goals/hash-guard-indirect.json`, and rendered frames were deleted immediately after hashing.

Optimization 2 reduces the raster shader's workgroup-shared splat batch from 256 to 64 entries. A 16x16 output tile still uses 256 pixel lanes and processes splats in the identical sorted order, but the smaller shared allocation permits substantially more Metal occupancy. The final clean release viewer measured:

| Scene | Baseline FPS | Final EMA ms | Final FPS | Speedup |
|---|---:|---:|---:|---:|
| lego | 11.9 | 29.55 | 33.8 | 2.84x |
| hotdog | 14.6 | 14.33 | 69.8 | 4.78x |
| chair | 12.7 | 24.46 | 40.9 | 3.22x |
| drums | 11.9 | 29.11 | 34.4 | 2.89x |
| ficus | 12.9 | 22.32 | 44.8 | 3.47x |
| materials | 12.3 | 26.63 | 37.5 | 3.05x |
| mic | 12.4 | 26.89 | 37.2 | 3.00x |
| ship | 11.5 | 33.09 | 30.2 | 2.63x |

Ship, the 300-frame floor, sustained 31.1 FPS over a separate 1,200-frame probe. The full 1,600-frame hash guard remains bit-exact after both optimizations (`hash-guard-batch64.json`). A fresh full PSNR/SSIM run after the performance milestone reproduced the standalone metrics below, and each scene's render batch was deleted immediately after evaluation.

### Quality

The CPU-side evaluator now pairs images by relative path, rejects incomplete splits, averages per-image metrics, and compares against `results.json`. Its alpha compositing and float64 SSIM behavior match nerfbaselines' uint8 + dm-pix conventions. The standalone no-Rerun binary now has an all-frame output mode that reuses one GPU/resource setup, with a Python/Tyro harness and Pixi tasks for one or all scenes.

| Scene | Images | Measured PSNR | Published PSNR | Delta | Measured SSIM | Published SSIM | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| lego | 200 | 35.74852097 | 35.74852 | +0.00000097 | 0.984148929 | 0.98415 | -0.000001071 |
| hotdog | 200 | 37.33804952 | 37.33805 | -0.00000048 | 0.986084722 | 0.98608 | +0.000004722 |
| chair | 200 | 36.31699061 | 36.31699 | +0.00000061 | 0.988609547 | 0.98861 | -0.000000453 |
| drums | 200 | 26.21745431 | 26.21745 | +0.00000431 | 0.954457522 | 0.95446 | -0.000002478 |
| ficus | 200 | 34.87592108 | 34.87592 | +0.00000108 | 0.987546316 | 0.98755 | -0.000003684 |
| materials | 200 | 30.56295960 | 30.56296 | -0.00000040 | 0.964319325 | 0.96432 | -0.000000675 |
| mic | 200 | 37.27579022 | 37.27579 | +0.00000022 | 0.993795700 | 0.99380 | -0.000004300 |
| ship | 200 | 30.75679554 | 30.75680 | -0.00000446 | 0.907153705 | 0.90715 | +0.000003705 |

All eight checkpoint-prediction reference splits pass at a 5e-6 absolute tolerance. The Apple M4 Metal standalone renderer completed all 1,600 images and matches the published CUDA results within cross-backend, 8-bit quantization tolerance:

| Scene | Images | Standalone PSNR | Published PSNR | Delta | Standalone SSIM | Published SSIM | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| lego | 200 | 35.79878165 | 35.74852 | +0.05026165 | 0.984169675 | 0.98415 | +0.000019675 |
| hotdog | 200 | 37.38059470 | 37.33805 | +0.04254470 | 0.986055113 | 0.98608 | -0.000024887 |
| chair | 200 | 36.34425761 | 36.31699 | +0.02726761 | 0.988596524 | 0.98861 | -0.000013476 |
| drums | 200 | 26.22850613 | 26.21745 | +0.01105613 | 0.954485765 | 0.95446 | +0.000025765 |
| ficus | 200 | 34.90838691 | 34.87592 | +0.03246691 | 0.987553852 | 0.98755 | +0.000003852 |
| materials | 200 | 30.56799199 | 30.56296 | +0.00503199 | 0.964311388 | 0.96432 | -0.000008612 |
| mic | 200 | 37.28985409 | 37.27579 | +0.01406409 | 0.993778632 | 0.99380 | -0.000021368 |
| ship | 200 | 30.78432717 | 30.75680 | +0.02752717 | 0.907167719 | 0.90715 | +0.000017719 |

The standalone PSNR delta is at most 0.0503 dB and SSIM delta at most 2.58e-5. Direct pixel comparison establishes why these aggregate values do not round identically across the original CUDA checkpoint renderer and Apple Metal: at least 99.9921% of RGB channels match within one 8-bit LSB on every scene, with a +0.056 to +0.209 LSB mean signed bias. This is a cross-backend quantization-level match, not a metric implementation error. The exact baseline output hashes are saved with the evidence artifact and will guard every performance edit against pixel drift.

### Brush

The session-5 replay defaults to 7,000 steps. It logs every training camera as a static frustum under `world/cameras/train_###` with a 160-pixel JPEG-quality-80 GT plane, keeps the white-background 0.2-rad/s spinning Scene dominant, places Brush's rendered eval image beside its ground truth, and shows loss / PSNR / SSIM / splat-count curves together. All panels are collapsed.

The official Brush v0.3.0 Apple Silicon binary is installed project-locally by a Pixi task after SHA-256 verification (`65b263...a1048c`; upstream commit `3edecbb2...d486`). The metrics binary uses the same pinned source plus a one-file patch that only prints the already-computed loss every 50 steps. The trainer omits both `--with-viewer` and `--rerun-enabled`, so Brush's embedded Rerun 0.24 never creates a recording.

Pinned v0.3.0's `VisualizeTools` and eval-export source were audited alongside current Brush's `send_default_blueprint` at `3b809857...e0`. The 0.34.1 replay mirrors the canonical `eval/view_0/{ground_truth,render}`, `loss/total`, `psnr/eval`, `ssim/eval`, and `splats/num_splats` paths. The rendered images come directly from Brush's built-in `--eval-save-to-disk` path, not from the standalone renderer. Brush writes all 100 eval views; the logger waits for the complete directory and a stable aggregate byte count across two scans, JPEG-logs the tracked view, then removes the PNG batch.

The fresh default Metal run completed 7,000 steps in 202 seconds. It produced 140 PLY exports at 50-step cadence, 140 loss samples, 14 eval PSNR/SSIM samples and Brush-native renders at 500-step cadence, 34 refine-count events, and a 12,936-splat final model. The logger retained exactly eight full-geometry splat snapshots (50, 1,000, 2,000, …, 7,000) with higher-order SH only at 7,000. Thirty thousand steps remain reachable by explicitly matching Brush's `--total-steps 30000` with the replay's `--total-iters 30000`; that schedule retains 31 snapshots.

The RRD is 6,238,559 bytes (5.95 MiB), 0.31% of the 2 GB cap. The conservative one-million-splat estimate for the default eight snapshots is 552 MB. `rerun rrd stats` confirms eight `world/splats` chunks, 14 `eval/view_0/render` chunks, one static eval GT, all 100 camera pinholes/JPEG images, both `iterations` and `step` timelines, one recording plus its matching-app blueprint, final-only SH, and serialized `EyeControls3D:spin_speed`. The run directory retains only the RRD, final PLY, trainer log, and successful completion sentinel.

Pixel validation used both required paths: a full-frame 1600x900 `ViewerClient` capture after reloading the saved RRD, and a 90-frame 1280x720 `viewer-mcp` timeline sweep over iterations 50–7,000. First/mid/final frames visibly show the Brush render converging toward GT while splats, camera ring, and all four curves evolve. `ffprobe` confirms 90 frames, 15 fps, and a six-second clip; sampled start/middle/end frames were individually inspected. The RRD, screenshot, frames, and video are under `/tmp/fleet-artifacts/gsplat-goals/`.

All eight pretrained scene clips were regenerated at the real 0.2-rad/s blueprint spin. Each 90-frame video was re-encoded from the captured frame dump at `90 / measured_capture_seconds`, making encoded duration equal wall-clock capture duration: 24.45–34.76 seconds, with scene-specific rates of 2.589–3.681 fps. `ffprobe` confirms 90 frames and exact measured duration for every clip. Start/middle/end contact sheets were visually inspected for all eight scenes; no blank or stale sequences were accepted. Pacing metadata is preserved as `/tmp/fleet-artifacts/gsplat-goals/orbit-pacing.json`.

### Assets

Data and pretrained checkpoint archives are downloaded and extracted for all eight scenes under worktree-local `data/nerfbaselines/{data,pretrained}`: lego, hotdog, chair, drums, ficus, materials, mic, and ship.

### Gates

The complete session-5 pre-commit gate set is green as of 2026-07-10 08:26 PDT: Rust fmt, Clippy with warnings denied, 12 Rust tests, Ruff, Pyrefly with zero errors, Vulture, all 36 Python tests, and the Metal `tools/relog_check.py` GPU regression. The re-log check captured full-frame headless screenshots and observed the expected red → green → blue replacement sequence.

## Constraints and decisions

- Preserve the `Gaussians3D` wire contract and rendered pixels.
- Keep all Rerun pins at 0.34.1; do not fork or patch Rerun.
- Optimize only this package's `src/` and `shader/` renderer code.
- Use pixi only; use `--frozen` unless this package's own dependencies intentionally change.
- Use full-frame `ViewerClient` captures with collapsed panels and timeline sweep video for pixel evidence; never rely on logs alone.
- Keep at least about 2 GiB free: stream bulk artifacts per scene and remove only regenerable build/render intermediates after extracting evidence.

## Next action

All original criteria and the session-5 7K/Brush-blueprint refinements are complete. Hand off the refreshed evidence URL and local commit to the orchestrator; do not push from this machine.
