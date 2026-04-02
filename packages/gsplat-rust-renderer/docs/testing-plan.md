# Testing Plan: Decouple Renderer + Correctness & Performance Testing

## Problem

The GPU rendering pipeline is tightly coupled to Rerun's `re_renderer` trait system. You can't render a single frame without a full Rerun viewer running. This makes it impossible to:
- Test rendering correctness against ground truth images
- Benchmark render performance in isolation
- Run automated regression tests in CI

## What We Have (Done)

### Data
- **8 NeRF Synthetic scenes** on HuggingFace: `pablovela5620/nerf-synthetic-mirror`
  - Raw scenes: `chair/`, `drums/`, `ficus/`, `hotdog/`, `lego/`, `materials/`, `mic/`, `ship/`
  - Brush-trained PLYs (30K steps): `trained/chair.ply`, `trained/lego.ply`, etc.
  - Download individual scenes: `hf download pablovela5620/nerf-synthetic-mirror --include 'lego/*' --repo-type dataset`
  - Download trained PLYs: `hf download pablovela5620/nerf-synthetic-mirror --include 'trained/lego.ply' --repo-type dataset`

### Python metrics module (complete)
- `gsplat_rust_renderer/metrics.py` — PSNR, SSIM, `load_image_rgb()`, `compute_metrics()`
- 13 passing tests including 6 metrics tests
- Matches standard 3DGS evaluation conventions (8-bit roundtrip, Gaussian-windowed SSIM)

### Brush as oracle
- `brush_app` (v0.3.0) installed via conda from `prefix.dev/ai-demos`
- Binary is called `brush_app` not `brush` (conda package naming)
- Can train: `brush_app data/nerf_synthetic/lego --total-steps 30000 --export-every 30000 --export-path data/trained --export-name lego.ply`
- Can view: `brush_app data/trained/lego.ply --with-viewer`
- Can eval: `brush_app data/nerf_synthetic/lego --eval-split-every 10 --eval-every 1000 --eval-save-to-disk --export-path data/eval/lego`

### Learnings from Phase 1
- Brush `--eval-split-every N` holds out every Nth **training** image for eval — NOT the test set
- Brush eval frame `r_0.png` does NOT correspond to test set `r_0.png` (different camera angles)
- Brush renders with **black** background; NeRF Synthetic ground truth uses **white** (alpha over white)
- Proper baselines require rendering at the exact test set camera positions, which requires the standalone renderer

## What's Next: Phase 2 — Standalone Render CLI

### Goal
Build a `gsplat-render` CLI binary that renders a PLY at a camera position from a NeRF transforms JSON, producing a PNG — all without Rerun.

### Approach: Incremental (CLI first, refactor later)
Instead of extracting the existing GPU pipeline from `gaussian_renderer.rs` (3280 lines, deeply coupled to `re_renderer`), build the CLI as a new binary in the same crate that:
1. Shares existing types (`RenderGaussianCloud`, `CameraApproximation`, etc.) — these are already Rerun-free
2. Shares existing math functions (`project_gaussian_to_ndc`, `evaluate_sh_rgb`, etc.)
3. Has its own CPU software rasterizer (matching `gaussian_splat.wgsl` logic)
4. Later, extract the shared GPU pipeline into `gsplat_core.rs` and have both the Rerun viewer and CLI use it

### New files to write

#### 1. `src/ply_loader.rs` — Rust PLY parser
```rust
pub(crate) fn load_ply(path: &Path) -> Result<RenderGaussianCloud, anyhow::Error>
```
Parses standard 3DGS PLY format. Same fields as `gsplat_rust_renderer/gaussians3d.py`:
- `x/y/z` → centers
- `rot_0/1/2/3` → quaternions (wxyz in PLY → xyzw in our code)
- `scale_0/1/2` → scales (`exp()` + clamp >= 1e-6)
- `opacity` → sigmoid + clamp [0,1]
- `f_dc_0/1/2` → SH DC → base RGB via `SH_C0 * dc + 0.5`
- `f_rest_*` → higher-order SH (channel-major)
- Fallback: `red/green/blue` or `r/g/b` vertex colors

New dep: `ply-rs = "0.1"`

#### 2. `src/nerf_camera.rs` — NeRF JSON camera loader
```rust
pub(crate) fn load_cameras(path: &Path) -> Result<Vec<CameraApproximation>, anyhow::Error>
pub(crate) fn load_camera(path: &Path, frame: usize, width: u32, height: u32) -> Result<CameraApproximation, anyhow::Error>
```
Parses `transforms_test.json`:
```json
{
  "camera_angle_x": 0.6911,
  "frames": [{"file_path": "./test/r_0", "transform_matrix": [[4x4]]}]
}
```
Converts c2w → w2c (invert the 4x4), builds perspective projection from `camera_angle_x` + aspect ratio.

New deps: `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`

#### 3. `src/render_cli.rs` — Standalone render binary
```bash
gsplat-render \
  --ply data/trained/lego.ply \
  --camera data/nerf_synthetic/nerf_synthetic/lego/transforms_test.json \
  --frame 0 \
  --output /tmp/render_lego_0.png \
  --width 800 --height 800
```

Uses the CPU fallback path for the initial version:
1. Load PLY → `RenderGaussianCloud`
2. Load camera from JSON
3. Run `rebuild_visible_indices()` (frustum cull + sort)
4. For each visible splat: `project_gaussian_to_ndc()` + `evaluate_sh_rgb()` + `build_prepared_splat()`
5. Software rasterize (Gaussian evaluation per pixel, back-to-front alpha blending)
6. Save PNG

New deps: `clap = { version = "4", features = ["derive"] }`, `image = "0.25"`

New binary target in `Cargo.toml`:
```toml
[[bin]]
name = "gsplat-rust-renderer"
path = "src/main.rs"

[[bin]]
name = "gsplat-render"
path = "src/render_cli.rs"
```

### Existing code to reuse (all `pub(crate)` in `gaussian_visualizer.rs`)
- `RenderGaussianCloud`, `RenderShCoefficients` — packed cloud data
- `CameraApproximation` — camera parameters
- `SortedSplatIndex`, `ProjectedGaussian`, `PreparedSplat` — intermediate types
- `rebuild_visible_indices()` — CPU frustum culling + depth sort
- `project_gaussian_to_ndc()` — CPU Gaussian projection
- `build_prepared_splat()` — combines projection + color
- `evaluate_sh_rgb()` — spherical harmonics evaluation
- `normalize_quat_or_identity()` — quaternion normalization
- Constants: `SH_C0`, `MAX_SPLATS_RENDERED`, `SIGMA_COVERAGE`, etc.

### Software rasterizer algorithm (matches `gaussian_splat.wgsl`)
```
for each pixel (x, y):
    pixel_ndc = (x / width * 2 - 1, y / height * 2 - 1)
    transmittance = 1.0
    color = (0, 0, 0)

    for each splat (back-to-front):
        delta = pixel_ndc - splat.center_ndc
        inv_cov = mat2(splat.inv_cov_xx, splat.inv_cov_xy, splat.inv_cov_xy, splat.inv_cov_yy)
        mahalanobis = dot(delta, inv_cov * delta)
        if mahalanobis > 9.0: continue  // beyond 3σ
        alpha = exp(-0.5 * mahalanobis) * splat.opacity
        if alpha < 1/255: continue
        color += splat.color * alpha * transmittance
        transmittance *= (1.0 - alpha)
        if transmittance < 1e-4: break  // early termination

    output[x, y] = (color, 1.0 - transmittance)
```

## Phase 3: Correctness Tests

Once the standalone CLI exists:

```python
# tests/test_render_quality.py
@pytest.mark.parametrize("scene", ["lego", "hotdog", "chair"])
def test_psnr_above_threshold(scene):
    """Rendered images should achieve reasonable PSNR against ground truth."""
    for frame_idx in range(10):
        subprocess.run(["gsplat-render", "--ply", f"data/trained/{scene}.ply",
                        "--camera", f"data/.../transforms_test.json",
                        "--frame", str(frame_idx), "--output", f"/tmp/{scene}_{frame_idx}.png"])
        rendered = load_image_rgb(f"/tmp/{scene}_{frame_idx}.png")
        gt = load_image_rgb(f"data/.../test/r_{frame_idx}.png")
        assert psnr(rendered, gt) > 20.0  # reasonable threshold for 30K-step training
```

## Phase 4: Performance Benchmarks

```bash
gsplat-render --ply data/trained/lego.ply --camera ... --benchmark --frames 10
# Output: avg 4.2ms/frame, 42401 splats, 38000 visible
```

## Re_renderer Coupling Points (for future GPU pipeline extraction)

When we eventually extract the GPU pipeline into `gsplat_core.rs`, these are the 6 coupling points to replace:

| Coupling | re_renderer type | Raw wgpu replacement |
|----------|-----------------|---------------------|
| Device/Queue | `RenderContext` | `wgpu::Device` + `wgpu::Queue` |
| Pipeline handles | `GpuBindGroupLayoutHandle`, `GpuRenderPipelineHandle` | `Arc<wgpu::BindGroupLayout>`, `wgpu::RenderPipeline` |
| Draw system | `DrawPhase`, `DrawDataDrawable`, `DrawInstruction` | Direct render pass calls |
| Vertex format | `VertexBufferLayout` | `wgpu::VertexBufferLayout` directly |
| Shader loading | `FileSystem` | `include_str!()` macros |
| Buffer padding | `PaddingRow` | Manual `[f32; 4]` arrays |

The compute dispatch (7-stage pipeline, lines 2819-3105 in `gaussian_renderer.rs`) only needs `device`, `queue`, and a `CommandEncoder` — minimal coupling.

## Quick Reference

```bash
# Train a scene with Brush
pixi run --frozen -e gsplat-rust-renderer -- brush_app data/nerf_synthetic/nerf_synthetic/lego \
  --total-steps 30000 --export-every 30000 --export-path data/trained --export-name lego.ply

# View in Brush's native viewer
pixi run --frozen -e gsplat-rust-renderer -- brush_app data/trained/lego.ply --with-viewer

# View in our Rerun viewer
pixi run --frozen -e gsplat-rust-renderer viewer  # terminal 1
pixi run --frozen -e gsplat-rust-renderer log-ply  # terminal 2

# Run Python tests
pixi run --frozen -e gsplat-rust-renderer-dev tests

# Run Python lint
pixi run --frozen -e gsplat-rust-renderer-dev lint
```
