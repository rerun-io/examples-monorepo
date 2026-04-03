# Testing & Evaluation

## Overview

The gsplat-rust-renderer uses a standalone GPU renderer (`gsplat-render`) for headless evaluation of Gaussian splat rendering quality. This enables automated correctness testing against ground truth images without needing a running Rerun viewer.

## Test Data

### NeRF Synthetic Scenes

Downloaded from `pablovela5620/nerf-synthetic-mirror` on HuggingFace:

| Scene | Splats | Train views | Test views | Image size |
|-------|--------|------------|------------|------------|
| Chair | 39,901 | 100 | 200 | 800×800 |
| Hotdog | 18,678 | 100 | 200 | 800×800 |

Each scene provides:
- `transforms_train.json` / `transforms_test.json` / `transforms_val.json` — camera parameters
- `train/` / `test/` / `val/` — RGBA PNG images
- Pre-trained PLYs in `trained/` (Brush 30K steps)

### Download Commands

```bash
pixi run --frozen -e gsplat-rust-renderer _download-nerf-chair
pixi run --frozen -e gsplat-rust-renderer _download-nerf-hotdog
pixi run --frozen -e gsplat-rust-renderer _download-trained-plys
```

## Rendering

### Standalone GPU Render

```bash
pixi run --frozen -e gsplat-rust-renderer render -- \
  --ply data/trained/chair.ply \
  --camera data/nerf-synthetic/chair/transforms_test.json \
  --frame 0 \
  --output /tmp/chair_test_0.png \
  --width 800 --height 800 \
  --background 1,1,1
```

Background `1,1,1` (white) matches the NeRF Synthetic convention where RGBA images are composited over white.

### Benchmark FPS

```bash
pixi run --frozen -e gsplat-rust-renderer render -- \
  --ply data/trained/chair.ply \
  --camera data/nerf-synthetic/chair/transforms_test.json \
  --frame 0 \
  --width 800 --height 800 \
  --benchmark --num-frames 20
```

Includes a warmup frame for GPU pipeline compilation.

## Metrics

Computed via `gsplat_rust_renderer.metrics`:

```python
from gsplat_rust_renderer.metrics import compute_metrics
from pathlib import Path

metrics = compute_metrics(
    Path("/tmp/chair_test_0.png"),
    Path("data/nerf-synthetic/chair/test/r_0.png"),
)
print(f"PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}")
```

All metrics apply 8-bit roundtrip quantization (`round(x * 255) / 255`) to match Brush's eval convention.

## Current Baseline (GPU Renderer)

Chair + hotdog, 4 test views each (frames 0, 50, 100, 150):

| Scene | PSNR (dB) | SSIM | LPIPS | Render time |
|-------|-----------|------|-------|-------------|
| Chair (40K splats) | 29.05 | 0.9479 | 0.0612 | 7.4ms avg (135 FPS) |
| Hotdog (19K splats) | 26.65 | 0.9387 | 0.1175 | 7.5ms avg |

Hardware: NVIDIA GeForce RTX 5090 (Vulkan), 800×800 resolution.

Reference: Brush MipNeRF-360 averages: ~29.7 PSNR, ~0.886 SSIM, ~0.196 LPIPS.

## Brush as Oracle

[Brush](https://github.com/ArthurBrussee/brush) (v0.3.0) is installed via conda from `prefix.dev/ai-demos`. The binary is called `brush_app`.

```bash
# Train
pixi run --frozen -e gsplat-rust-renderer -- brush_app data/nerf-synthetic/chair \
  --total-steps 30000 --export-every 30000 --export-path data/trained --export-name chair.ply

# View in Brush
pixi run --frozen -e gsplat-rust-renderer -- brush_app data/trained/chair.ply --with-viewer
```

### Known Brush Conventions

- Renders with **black** background; NeRF Synthetic ground truth uses **white** (alpha over white)
- `--eval-split-every N` holds out every Nth **training** image — NOT the test set
- Brush eval frame `r_0.png` does NOT correspond to test set `r_0.png` (different cameras)
- Our standalone renderer solves this by rendering at exact test set camera positions

## Running Tests

```bash
# Rust unit tests (13 tests: SH, camera, GPU types, etc.)
pixi run --frozen -e gsplat-rust-renderer rust-test

# Python tests (13 tests: metrics, PLY loading, import)
pixi run --frozen -e gsplat-rust-renderer-dev tests

# Lint
pixi run --frozen -e gsplat-rust-renderer-dev lint
pixi run --frozen -e gsplat-rust-renderer clippy
```

## Full Evaluation Script

```python
from pathlib import Path
from gsplat_rust_renderer.metrics import compute_metrics

scenes = ["chair", "hotdog"]
frames = [0, 50, 100, 150]

for scene in scenes:
    print(f"\n=== {scene.upper()} ===")
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for frame in frames:
        rendered = Path(f"/tmp/gsplat_eval/{scene}_gpu_{frame}.png")
        gt = Path(f"data/nerf-synthetic/{scene}/test/r_{frame}.png")
        m = compute_metrics(rendered, gt)
        print(f"  Frame {frame:3d}: PSNR={m['psnr']:6.2f}, SSIM={m['ssim']:.4f}, LPIPS={m['lpips']:.4f}")
        for k in totals:
            totals[k] += m[k]
    n = len(frames)
    print(f"  Average:   PSNR={totals['psnr']/n:6.2f}, SSIM={totals['ssim']/n:.4f}, LPIPS={totals['lpips']/n:.4f}")
```
