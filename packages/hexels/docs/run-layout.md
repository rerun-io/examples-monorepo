# Contract: run outputs and the comparison protocol

The code that implements this contract is `hexels_bench` (`layout`, `methods`, `cameras`, `metrics`,
`paper`). `<runs>` is `Roots.runs`: `$HEXELS_WORK/runs` by default (`~/hexels-work/runs` when unset), or
`--roots.runs`.

## Directory layout

```
<runs>/<dataset>/<scene>/<method>/
  model.ply          # Brush: 3DGS PLY. Hexels: hexel PLY (hexel-ply.md). WGSL re-renders link the source run's model
  cameras.json       # held-out cameras, see below
  renders/test/<image stem>.png   # 16-bit RGB PNG, one per held-out view, same names as GT
  train.json         # see below (runs we trained only)
  metrics.json       # see below
  fps.json           # WGSL runs: `gsplat render --fps` report (adapter, backend, trials, median FPS)
  brush-summary.json # Brush runs: the trainer's --summary document
  published_metrics.json, upstream_float_metrics.json, upstream-eval/   # published CUDA runs, see below
  log.txt            # stdout/stderr of the run
```

`<dataset>` is `mipnerf360`, `dtu`, `tnt`. `<scene>` uses the upstream names (`garden`, `scan24`,
`Truck`). 2D runs use `<dataset>` = `imagegs` and `<scene>` = image stem (`anime-1_2k`).

`<method>` values (`hexels_bench.methods.METHODS` is the authority):

| method | what |
|---|---|
| `brush-7k` | Brush 3DGS, 7,000 steps, our protocol |
| `hexels-v1-7k`, `hexels-v3-7k` | upstream Hexels trained by us, 7,000 steps |
| `hexels-v{1,2,3}-30k-pub-cuda` | published 30k fixed-sfm checkpoint, rendered by upstream CUDA |
| `hexels-v{1,2,3}-30k-pub-wgsl` | same checkpoint, rendered by our WGSL renderer |
| `brush-7k-wgsl` | Brush PLY re-rendered by our WGSL core (should equal brush-7k) |
| `brush-smoke` | 30-step check of the Brush protocol path, outside the benchmark |
| 2D: `hexels-2d`, `imagegs-2d`, `hexels-2d-pub-cuda`, `hexels-2d-pub-wgsl` | 2D fits |

## Protocol (from the paper and upstream docs/experiments.md)

- Split: registered cameras ordered lexically by image name; every 8th camera starting at index 0 is
  held out. Identical cameras and images for every method. Where upstream ships a camera contract
  (`cameras/nvs/<dataset>/<scene>/contract.json` on HF), the held-out set is exactly the contract's.
- Resolution: Mip-NeRF 360 `images_4` outdoor (bicycle, flowers, garden, stump, treehill), `images_2`
  indoor (room, counter, kitchen, bonsai); DTU resolution factor 2; T&T `images_2` (factor 1 on it).
- Initialization: the scene's COLMAP SfM points. Primitive cap: the SfM point count (Table 3 rule).
  Hexels `fixed` keeps the initial population. Brush may densify up to the cap.
- Steps: 7,000 for our runs. Published checkpoints are 30,000.
- Scenes: new compute runs on the representative scenes (`layout.REPRESENTATIVE_SCENES`); tables compare
  them with the published per-scene numbers, not with the paper's dataset means.
- Metrics: float RGB clamped once to [0,1]; PSNR, SSIM, LPIPS-VGG (inputs mapped to [-1,1]). One
  shared scorer for every method (`hexels_bench.metrics`), reading the 16-bit PNG renders and the GT images.
  The published-checkpoint check additionally runs the upstream direct-float evaluator, and both
  numbers are kept (they are different conditions).
- Train time: wall time of the optimisation loop only (exclude dataset load and final export);
  record both loop time and total time.
- Peak VRAM (one sampler for every trainer, `hexels_bench.gpu_monitor.run_measured`):
  the trainer runs as a direct child; a thread samples NVML every 5 ms and sums `usedGpuMemory` of the
  trainer's pid over all GPUs (the process's own allocations: CUDA context + torch caching allocator, or
  Brush's wgpu/Vulkan buffers; never other processes). Peak = max of the samples taken up to the end of
  training and export, reported in GiB (2^30 bytes). Cutoff: Brush trains, evaluates, exports and then renders
  its test views in one process, so its peak stops at `training_end_unix_seconds` (brush-summary.json); the
  upstream Hexels trainer runs with --skip_postprocess and the 2D fits render nothing afterwards, so their
  whole process counts. Null (never 0) when NVML is unavailable.
- FPS: 2 warm-up passes, then 4 trials of 10 passes over all test views, GPU-synchronized, median
  per scene, mean over scenes. Exclude loading, readback and metrics.
- Deploy MB: primitive count x FP32 bytes per primitive (Hexels V1 80 B, V2 116 B, V3 260 B; 3DGS
  59 floats = 236 B with SH3) / 1e6, plus the actual PLY file size.

## train.json

```json
{"method": "...", "dataset": "...", "scene": "...", "steps": 7000, "loop_seconds": 0.0,
 "total_seconds": 0.0, "peak_vram_gib": 0.0, "primitives": 0, "sfm_points": 0,
 "gpu": "<NVML GPU name>", "host": "<host name>", "source_sha": "...", "command": "..."}
```

## metrics.json

```json
{"method": "...", "dataset": "...", "scene": "...", "views": 0, "psnr": 0.0, "ssim": 0.0,
 "lpips_vgg": 0.0, "fps": null, "deploy_mb": 0.0, "ply_mb": 0.0, "scorer": "shared-png16|upstream-float",
 "per_view": [{"name": "...", "psnr": 0.0, "ssim": 0.0, "lpips_vgg": 0.0}]}
```

Unscored numbers are `null`, never NaN.

## cameras.json (written beside `renders/`; read by `gsplat render --camera`)

It copies the upstream camera contract records
(`hexels-pretrained/cameras/nvs/<dataset>/<scene>/contract.json`, `geometry.test[]`) so the writer
does no conversion. One record per held-out view, in render order:

```json
{"format": "hexels-cameras-1", "dataset": "dtu", "scene": "scan24", "split": "test",
 "background": [0.0, 0.0, 0.0],
 "cameras": [{"name": "0000", "width": 777, "height": 581,
              "fov_x": 0.5248904435803646, "fov_y": 0.397701717462152,
              "world_view_transform": [[...4 floats...], [...], [...], [...]],
              "camera_center": [0.0, 0.0, 0.0]}]}
```

- `name`: GT image stem; the render is `renders/test/<name>.png`.
- `world_view_transform`: exactly upstream's 2DGS/3DGS tensor (row-vector convention: a world point
  `[x y z 1] @ M` gives the OpenCV camera point, x right, y down, z forward; `M` is the transpose of the
  usual world-to-camera matrix).
- Intrinsics follow upstream (2DGS `getProjectionMatrix`): `fx = width / (2 tan(fov_x/2))`,
  `fy = height / (2 tan(fov_y/2))`, principal point at the image centre (upstream ignores COLMAP cx/cy).
- `background`: the RGB the upstream render used (`white_background` false -> black).
- The reader also accepts the upstream `contract.json` itself (it takes `geometry.test`).

## Shared ground truth and published reference

- `<runs>/<dataset>/<scene>/gt/test/<name>.png`: the held-out targets as 16-bit RGB PNG, one per camera in
  `cameras.json`. They are bit-exactly the float32 tensors upstream trains and evaluates against
  (2DGS loadCam: each RGB band resized on its own with PIL's default filter, alpha dropped, /255).
  The writer (`cameras.write_targets`) stages them beside the directory and moves them into place only
  after they match the camera contract's `ordered_target_tensors_sha256`. k/255 is stored as k*257, so
  the 16-bit values are lossless. Every method of a scene is scored against this one directory; the
  scorer writes it when the scene has none yet.
- `published_metrics.json` beside a `hexels-v*-30k-pub-cuda` run: the authors' direct-float scores
  from the HF `evaluation.json` of that checkpoint, in the metrics.json schema with
  `scorer = "upstream-float"`. These are published numbers, not re-measured here.

## Surface outputs

```
<runs>/<dataset>/<scene>/<method>/surface/          # method's own depth: median (Hexels) or expected (Brush)
<runs>/<dataset>/<scene>/<method>/surface-<mode>/   # a depth-mode override (e.g. Hexels expected depth)
  cameras.json          # hexels-cameras-1, split "all" (all-view surface runs) or "train" (NVS runs)
  surface.rrd           # gsplat render --surface: per view on timeline `view`, world/camera/<method>/{rgb,
                        #   median_depth, expected_depth (f32 DepthImage, meter 1), alpha, normal (0.5 n + 0.5)}
  renders/<name>.png    # the same views as 8-bit PNG
  fuse.ply fuse_post.ply  # Open3D TSDF (upstream settings), post = largest cluster (upstream _post_process_mesh)
  fusion.json           # depth source/mode, alpha cut, TSDF settings, views, valid depth fraction, triangles
  depth_parity.json     # CUDA method only: per-pixel median depth and alpha vs the WGSL recording
  dtu_eval/ | tnt_eval/ # unchanged 2DGS evaluator outputs
  surface_metrics.json  # {dataset, scene, method, evaluator, cd_mm, acc_mm, comp_mm, precision, recall, f1, tau}; null when unscored
<runs>/<dataset>/<scene>/<scene>-surface.rrd        # viewer recording: meshes, depth/normal maps, cameras, DTU reference, metrics
```

Surface methods: `hexels-v3-surf-30k-pub-wgsl` / `hexels-v3-surf-30k-pub-cuda` (the paper's surface checkpoint
`hexels_v3-sh3_fixed-mvs_normal-and-distortion`, all views, depth from WGSL or from upstream CUDA),
`hexels-v3-surf-30k` (our run of the paper's surface recipe), and any NVS method (`brush-7k`, `hexels-v{1,3}-7k`,
`hexels-v{1,3}-30k-pub-wgsl`) fused over its training views. Note: the paper's surface checkpoints are a different
model from the NVS `hexels-v3-30k-pub-*` runs, hence the `surf` methods.

## Viewer outputs

```
<runs>/<dataset>/<scene>/
  orbit.json                      # 300 gsplat-cli CameraSpecs, 1920x1080, one closed orbit (gsplat speed --path, gsplat render --camera)
  viewer[/<tag>]/
    <dataset>-<scene>-compare.rrd         # Brush | Hexels side by side + metrics table (custom viewer: compute; stock Rerun: native splats + hexel centres)
    <dataset>-<scene>-probe-{brush,hexels}.rrd   # one-view recordings for gsplat probe (timeline fps 60: one pose per frame)
    probe-{brush,hexels}[-r1|-r2].json    # gsplat probe reports (best median of r1/r2 copied to probe-<layout>.json)
    shots/                                # custom-f<frame>.png, stock-f000.png, spin-spin.mp4 (+ frames)
```

- Orbit: centre = least-squares closest point to every contract camera's optical axis (training + test), up = minus the
  mean camera y axis, radius and elevation = median horizontal distance and height of the cameras; pose 0 faces camera 0;
  fov_y = median test fov_y; square pixels; principal point at the image centre. Recorded on timeline `frame` (0..299)
  as `/world/orbit_cam` (Transform3D + transparent zero-width Pinhole), tracked by both 3D views.
- Captures whose cameras leave an azimuth gap wider than 90 degrees (DTU) get a cosine-eased sweep over the covered arc
  and back (still 300 poses, closed loop) with the up axis taken from the camera x axes.
- Entities: `/world/brush` (GaussianSplats3D via Rerun's PLY import), `/world/hexels` (Hexels3D + Points3D fallback),
  `/world/cameras/test/<name>` (test frustums), `/metrics/table` (one row per metric on timeline `row`, one column per model).
