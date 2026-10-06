# Architecture

| Crate | Responsibility |
|---|---|
| `gsplat-core` | Raw-wgpu projection, sorting, rasterization; native array conversion; shared Scene and independent ViewState |
| `gsplat-viewer` | Rerun queries, runtime blueprint defaults, GPU cache, camera and composite |
| `gsplat-render` | PLY and camera input, PNG output, persistent renderer |
| `gsplat-bench` | Shared cameras/settings, parity, archetype loss and synchronized timing |
| `gsplat-eval` | Image metrics |

The core does not depend on Rerun. Its [API notes](../crates/gsplat-core/README.md)
and the [viewer usage](../crates/gsplat-viewer/README.md) cover the public entry points.
The root is a virtual Cargo workspace. The legacy core and `ours-old` backend
were removed after baseline collection.

## Native data and runtime selection

Python `splats_from_ply` returns `rr.GaussianSplats3D`. It follows Rerun 0.38.1's
Rust PLY conversion rules: f32 exp/sigmoid, normalized xyzw quaternions,
clamped RGB/alpha rounded with +0.5, and coefficient-major 15×3 f16 SH.
Short optional component arrays repeat their last value. Core conversion widens
SH, derives DC from RGBA8, and applies inverse opacity/scale activations.
Missing SH uses degree zero; requested degrees are capped at three.

Both native and compute visualizers remain registered. A public ViewContextSystem
checks the active blueprint's `ActiveVisualizers` component. If no explicit
selection exists, it replaces the native instruction type with compute, preserves
other visualizers, and writes the instruction plus active IDs through public
blueprint APIs. Compute skips heuristic-only instructions until that write applies,
so the first frame does not draw both. Explicit native, compute, or empty selections
are left alone. Runtime writes belong to this viewer's active blueprint, not the
recording. No Rerun crate is patched or vendored.

`ComputeGaussianSplats3D:render_mode` is a Text property on the visualizer
instruction. A separate mode query keeps it out of the scene upload signature.
`default` and `mip` select the per-view core mode. Python rejects an explicit
`--render-mode` without `--compute`; stock viewers do not use this property.

## GPU lifetime and composition

One signature-keyed, store-owned cache holds uploaded Scenes and small bounds/count
metadata. The signature hashes resolved native component row IDs and mappings,
plus actual blueprint values where Rerun clears the source row ID. It excludes
unrelated view defaults such as the eye and background. CPU float arrays exist only during conversion/upload. Camera and mode
changes reuse the scene. Each view/instance has a ViewState and its own target;
pipelines live for the device lifetime. Unused views expire after one unused frame; shared scenes allow one extra
frame for blueprint activation. Relogged attributes change the signature. `GSPLAT_UPLOAD_PROBE=1` prints
one count line per upload for integration checks.

Initial intersections are capped at 1,048,576. GPU feedback grows capacity and
requests another frame on overflow. Pending state comes from the core; errors
invalidate the last render and are reported per instruction. Resize retains the
previous composite until feedback confirms a completed replacement.

The optional `TextureDepth` target adds alpha-weighted expected positive camera
depth, using existing sorted depth keys. Normal float/packed/texture targets keep
their original raster path. The viewer unpremultiplies display RGB, converts to
linear, re-premultiplies, and outputs reverse-Z fragment depth against native
opaque content. Expected depth gives soft edges a continuous representative
depth; it is an approximation for mixtures straddling an opaque surface.

Each uploaded scene stores a conservative AABB (centers plus three maximum-axis
scales). Instance transforms map its eight corners. Public EyeControls3D fallback
providers combine these bounds with native geometry for default/reset framing.
Rerun's private per-entity bounds/picking output remains unavailable: compute
picking, hover, outlines and focus-entity bounds remain gaps. The previous-frame
eye limitation and cross-entity ordering are listed in the viewer README.

## Device and validation

Headed eframe and headless kittest share Rerun's adapter selector and core device
helpers: full `adapter.limits()`, required `SUBGROUP`, and optional `TIMESTAMP_QUERY`.
Startup reports the adapter name when subgroup/compute requirements are missing.
Texture limits are never reduced below stock Rerun's adapter limits.

Unit tests cover conversion and selection. GPU tests cover optional expected
depth and tiny transforms. Viewer tests poll fresh screenshots until pixels settle;
black and white fixed-eye pairs check color, and native/compute recordings check
selection. `probe` is an opt-in feature for complete-frame timing including GPU
completion; screenshot readback occurs after the samples.

## Training recordings

`crates/gsplat-train` drives Brush's pinned `create_process_with_device` stream.
The local observer adapter exposes the existing loss/LR/refine values without
changing training math, loading, config merge, evaluation or exports. It emits
step events every five steps and at the final step. Refine statistics come
from refine events, not sampled step events. The temporary adapter and complete
patch live in `vendor/brush-process`; remove it when Brush exposes these stats.

The trainer captures tensor handles from the current splat slot before advancing
the stream. A separate logging thread reads them asynchronously, folds Brush's
minimum-scale filter as its exporter does, and logs native `GaussianSplats3D`.
Scales use exp, WXYZ rotations become normalized XYZW, DC/opacity become RGBA8,
and higher SH uses coefficient-major 15-by-3 f16 values. Degree 4 is truncated
only at this logging boundary. Snapshots retain step 50, every 1,000 steps, and
the final step by default; only the final snapshot carries higher SH.
`--snapshot-first` changes the first retained step. Brush's
`--rerun-log-splats-every` changes the periodic cadence. These and
`--rerun-log-train-stats-every` accept positive multiples of five; the latter
keeps Brush's 50-step default. Console progress is independent at 100 steps.
`--rerun-max-img-size` controls eval thumbnails (default 512); camera thumbnails
are capped at 256. Images use JPEG quality 85. The legacy `--rerun-enabled` and
unsupported distribution flag are rejected explicitly, including merged config.
Brush's own Rerun logger stays disabled.

All dynamic data uses `iterations`. Scalars include loss, step milliseconds,
splat counts, learning rates, refine statistics, sampled GPU memory and Brush's
EvalResult PSNR/SSIM. Cameras use Pinhole + Transform3D and small JPEG thumbnails;
distorted camera models are explicitly documented as pinhole approximations.
Four fixed eval views are rendered only at Brush's eval steps, at thumbnail
resolution on the logging thread. They use a black background; GT is
premultiplied with Brush's packed-image conversion. All snapshot and scalar
readbacks run asynchronously on that thread. A logging failure emits a warning
and does not abort training; sink flushing has a two-second timeout. A live
sink must connect during a bounded check before training; an unavailable sink
is disabled with a warning to avoid the SDK's unconnected queue backpressure.

The Rust blueprint places a 3D scene and four eval pairs above metric tabs.
`--video` adds a spinning eye and a flat Loss/PSNR/SSIM/Splats row. Brush's
estimated axis is a model-rotation target from -Y. Apply that rotation's inverse
to -Y for the eye up and nearest world ViewCoordinates, so the spin follows the
unrotated scene. The default has no visualizer override and opens in
stock Rerun. `--compute-visualizer` explicitly selects `ComputeGaussianSplats3D`
for a compatible custom viewer; it cannot be combined with stock `--spawn`.
`--connect` and `--save`
fan out through Rerun's sinks, so live and saved recordings contain the same data.

