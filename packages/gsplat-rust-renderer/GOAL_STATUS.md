# gsplat-rust-renderer Goal Status

Last updated: 2026-07-09T23:13:35-07:00

## Success criteria

- [ ] **FPS:** at least 30 FPS with a continuously moving `EyeControls3D` camera at 1920x1080, verified with `GSPLAT_FPS_PROBE=1` on lego, hotdog, chair, drums, ficus, materials, mic, and ship.
- [ ] **Quality:** standalone `gsplat-render` output matches each checkpoint's published full-200-image PSNR/SSIM. Checkpoint prediction renders must first validate the evaluator.
- [ ] **Brush:** a pixi-managed osx-arm64 Brush trainer exports PLY snapshots, and this package replays retained snapshots on an `iterations` timeline with rerun-sdk 0.34.1 into an RRD no larger than about 2 GB.
- [ ] All required Rust, Python, and GPU regression gates are green.
- [ ] Pixel and numeric evidence is published on `/tmp/fleet-artifacts/gsplat-goals.html`.

## Current evidence

### FPS

No measurement has been reproduced in this worktree yet. The mission-provided lego baseline is approximately 15.6 FPS for 325k splats; treat it as context, not completed evidence. A single standalone Lego frame attempt loaded all 325,000 splats, then failed at adapter discovery with `metal found no adapters`, confirming the documented workspace sandbox limitation. Do not retry GPU/viewer work until the run is relaunched with adapter access.

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

All eight checkpoint-prediction reference splits pass at a 5e-6 absolute tolerance. Standalone renderer output remains GPU-blocked, so criterion 2 is not complete yet; the all-frame renderer and orchestration are ready to run immediately after relaunch.

### Brush

The official Brush v0.3.0 Apple Silicon binary is installed project-locally by a Pixi task after SHA-256 verification (`65b263...a1048c`; upstream commit `3edecbb2...d486`). The trainer task omits both opt-in `--with-viewer` and `--rerun-enabled`, exports PLY every 50 iterations, and never uses Brush's embedded Rerun 0.24 recording.

The default replay path uses this workspace's rerun-sdk 0.34.1, logs the first export + exact 1,000-step boundaries + final (31 snapshots maximum), retains higher-order SH only in the final snapshot, uses the plural `iterations` timeline, and saves a collapsed-panel spinning-eye blueprint. The conservative one-million-splat estimate is 1.82 GB; Lego's 325k cap is below 600 MB.

A GPU-free smoke replay of two 325k-splat checkpoints produced `/tmp/brush-replay-smoke.rrd`: 55.4 MiB compressed, one recording + one blueprint, `iterations` and `step` timelines, two full-geometry `world/splats` chunks, final-only SH, and serialized `EyeControls3D:spin_speed`. Actual Brush training and the progression video remain GPU-blocked.

### Assets

Data and pretrained checkpoint archives are downloaded and extracted for all eight scenes under worktree-local `data/nerfbaselines/{data,pretrained}`: lego, hotdog, chair, drums, ficus, materials, mic, and ship.

### Gates

The complete non-GPU pre-commit gate set is green: Rust fmt, Clippy with warnings denied, 11 Rust tests, Ruff, Pyrefly, Vulture, and all 28 Python tests. The Python suite includes the Lego/Hotdog 200-image integration guards and Brush retention/timeline/budget behavior. The required `tools/relog_check.py` GPU regression remains sandbox-blocked by the confirmed lack of a Metal adapter and was not retried.

## Constraints and decisions

- Preserve the `Gaussians3D` wire contract and rendered pixels.
- Keep all Rerun pins at 0.34.1; do not fork or patch Rerun.
- Optimize only this package's `src/` and `shader/` renderer code.
- Use pixi only; use `--frozen` unless this package's own dependencies intentionally change.
- Use full-frame `ViewerClient` captures with collapsed panels and timeline sweep video for pixel evidence; never rely on logs alone.
- Current execution is workspace-sandboxed without GPU adapter/socket/process-control access. Record one failed GPU attempt if needed, then continue CPU-verifiable work until relaunched with access.

## Next action

The CPU-verifiable implementation is checkpointed in local commits `4acb55a` (full-split quality guard) and `82a05a2` (bounded Brush replay), with Pixi orchestration and this status included in the following workflow commit.

On GPU-enabled relaunch: run standalone all-scene evaluation, Brush training/replay/video, moving-camera FPS baselines, then guarded renderer optimization.
