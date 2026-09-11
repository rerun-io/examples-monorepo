# slam-rs

<p align="center">
  <img src="media/slam-rs-github.gif" alt="slam-rs replaying a Monado SLAM Dataset clip in Rerun: the estimated trajectory against ground truth, the landmarks, the rig, and the tracked keypoints on the camera frames" width="800" />
</p>

Visual-inertial odometry with a Rust core and support for multiple cameras,
a CPU frontend and a
portable GPU frontend through CubeCL and wgpu. Python owns the plumbing —
catalog feed, decode, evaluation and Rerun logging — and talks to the core
through a PyO3 extension module, so the whole pipeline runs from Python:
`_core.Vio` consumes IMU samples and framesets and reports a pose, and
`tools/apps/replay.py --stage vio` draws the estimate against the ground truth
on the same frames. `fast` is the default profile; `reference` selects the
unmodified dataset configuration. Accuracy is ATE against catalog ground truth.
Each lane/profile is compared with its measured baseline in
`gate.toml`. MIO10 GPU fast scores about 1.55 cm on the RTX 5090.
The same code runs on `linux-64`, `linux-aarch64`, and macOS `osx-arm64`.

Design notes — the module-by-module account of the estimator, the full Python API,
the reference set, the gates and every recorded decision:
[docs/design-notes.md](docs/design-notes.md).

## Run it

The pixi tasks run from anywhere in the repository; the `python tools/...`
commands below run from `packages/slam-rs`. Install the environment and build
the core:

```bash
pixi install -e slam-rs-dev
pixi run -e slam-rs-dev --frozen slam-rs-build   # cargo build + install _core.so in place
```

Then replay a segment. `--stage vio` is the whole pipeline; `--stage input` (the
default) logs only what the estimator is fed, and `--stage frontend` runs the
optical flow over it and draws its keypoints:

```bash
cd packages/slam-rs
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio   # the smoke segment, in a viewer
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --rr-config.headless --rr-config.save data/replay-vio.rrd
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --segment <segment-id>       # another catalog segment
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --rrd base.rrd --gt-rrd gt.rrd   # a recording of your own
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --catalog rerun+http://dgx-spark:9988 --segment <any-segment-id>   # override the default catalog URL
```

In a shell without `DISPLAY`, pass `--rr-config.headless` or the spawned viewer
wedges the recording stream. A long segment still wants `--max-framesets`.

The GPU frontend is the off-by-default `gpu-wgpu` cargo feature, through
CubeCL and wgpu (Vulkan / Metal / DX12). It writes the in-place
`slam_rs/_core.so`; `--gpu` selects that frontend and `--profile fast` the
speed profile:

```bash
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-build   # a core whose `--gpu` is wgpu
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --gpu                 # the default fast profile
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --gpu --profile reference  # the unmodified dataset config
```

Design notes: [the GPU lane](docs/design-notes.md#the-gpu-lane), and
[the portable lane, and the two silent failures](docs/design-notes.md#the-portable-lane-and-the-two-silent-failures).

On macOS everything above runs from the mac lane's environment, which is where
that platform's `slam-rs` features are solved: `-e slam-rs-osx-dev` in place of
`-e slam-rs-dev`.

## Two profiles

Each dataset names its configuration under `configs/`. A profile is a flat
overlay in `configs/profiles/<name>.json`. `reference` is empty; `fast` is the
default in every tracking tool.
`fast` is three keys:

```json
{"config.vio_max_iterations": 7, "port.redetect_survivor_ratio": 0.85, "port.frame_update_max_iterations": 5}
```

Keys under `port.` select additional scheduling options. Their zero defaults
retain per-frame detection and joint optimization. Unknown configuration or
overlay keys raise `KeyError`, so a typo cannot silently change a run. The fast profile changes
two things about the schedule and nothing about the arithmetic:

- **Detection on demand.** The reference profile detects every frameset. The fast profile detects only when camera 0 holds fewer than 85 % of
  the keypoints the last detecting frameset ended with, which is cuVSLAM's
  schedule.
- **The window is solved at keyframes.** The reference profile solves the
  whole sliding window every frameset. The fast profile does so at keyframes only; between
  them it solves the newest pose, velocity and biases against the held landmarks
  and the IMU factor, up to six trials with the inclusive iteration cap, and falls back to the joint solve when
  that update declines. `VioSnapshot.frame_update` says which one ran.

The gate compares each clip with the baseline for the selected lane and
profile. It allows at most 10% more ground-truth RMSE. The same-host speed
clause allows at most 10% more median tracker time. Baselines record the core
digest, host, frameset count, and measurement date.

Every tracking tool takes `--profile reference|fast`: `replay.py`,
`bench_track.py`, `fleet_check.py`, `robocap_fleet.py`, and `robocap_probe.py`. In code,
`slam_rs.reference.profiled_config_text(path, "fast")` returns the overlaid JSON.

Design notes: [D74](docs/design-notes.md#d74--speed-profile) the profile,
[D75](docs/design-notes.md#d75--redetect-on-demand-the-fast-profile-detects-when-camera-0-has-lost-tracks) detection on demand,
[D76](docs/design-notes.md#d76--the-fast-profile-solves-the-window-at-keyframes-and-the-newest-state-alone-between-them) the keyframe-gated solve.

## How data gets in and out

The library reads one thing: a recording in the dataforge rig schema. One base
`.rrd` per sequence carries the rig calibration, one video stream per camera and
the IMU stream. Ground truth and results are separate layer files that stack onto
the same entity paths.

The catalog is the dataset source for replay and fleet checks. The gate
stores segment selectors, sensor models, tiers, hold-outs and measured baselines;
it stores no copied capture facts or layer fingerprints. Any msd-index / msd-g2 / msd-odyssey segment replays from the catalog.

- Catalog replay: `tools/apps/replay.py --stage vio --segment <segment-id>`.
  `--catalog` overrides the gate URL.
- Local replay: `tools/apps/replay.py --stage vio --rrd base.rrd [--gt-rrd gt.rrd]`.
  This explicit file pair runs through an in-process server.

The outputs are an estimated trajectory CSV on the absolute device clock and
an optional Rerun recording with the estimate, ground truth, landmarks, and
window. Without a ground-truth layer, replay prints "ground truth absent, not scored".

RoboCap s15 has no ground truth. Its catalog replay is compared with
`tests/fixtures/robocap_s15_trajectory.csv`, recorded with our GPU fast core.
This is a regression measurement, reported without an accuracy gate.
`--reference-csv` selects another regression trajectory.

In code the contract is three calls: `Calibration` is the rig, `Vio.push_imu`
takes one IMU sample, `Vio.track` takes one synchronized frameset of `uint8`
images. The feed is the only adapter between the recording and those calls.

## Python API

```python
from pathlib import Path

from slam_rs import _core
from slam_rs.reference import profiled_config_text

calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)  # the feed's dataclasses
config = _core.VioConfig.from_json(Path("configs/msdmi_config.json").read_text())  # the dataset configuration
config = _core.VioConfig.from_json(profiled_config_text(Path("configs/msdmi_config.json"), "fast"))  # or with the overlay

vio = _core.Vio(calibration, config, threads=1, gpu=False)   # gpu=True runs the frontend on this host's GPU
vio.push_imu_batch(t_ns, gyro, accel)     # int64[n], float64[n, 3], float64[n, 3], uncalibrated
result = vio.track(t_ns, [left, right])   # uint8[h, w] per camera
result.status, result.world_from_rig      # VioStatus, [tx ty tz qx qy qz qw]

snapshot = vio.snapshot()   # None until a frameset has measured: the window, the landmarks, the LM numbers, the stage timers
frame = vio.flow_frame()    # the keypoints of the last accepted frameset, or None
```

The frontend alone is driven the same way, for a run that needs no backend:

```python
flow = _core.OpticalFlow(calibration, config, threads=1)
frame = flow.process(t_ns, [left, right])
frame.ids(0), frame.positions(0), frame.transforms(0), frame.occupancy(0)
```

One `track` call runs the whole pipeline for one frameset in the calling
thread (Offline mode, D17), so every result is final and a repeat run over the
same input is bit-identical. `VioStatus` has two states: `NeedMoreImu` until
IMU coverage extends past the frameset, `Tracking` otherwise; `replay.py` holds
such a frameset and pushes it again once its samples arrive.

Design notes — the accessors field by field, every refusal and its ceiling, and
`Calibration.from_catalog`'s rules: [Python API](docs/design-notes.md#python-api).

## What exists

### Layout

| Path | What it is |
|---|---|
| `crates/slam-rs` | The core (`slam_rs` lib). No Python, no Rerun; the GPU frontend is its `gpu-wgpu` feature. |
| `crates/slam-rs-py` | PyO3 `cdylib` built in place as `slam_rs/_core.so`. |
| `crates/slam-rs-cli` | `slam-rs` binary: a placeholder. `version` is the only subcommand that does anything; a replay runs through the Python tools. |
| `slam_rs/` | The Python package: stubs, Tyro entry points under `apis/`. |
| `tools/` | Thin CLI shims over `slam_rs/apis/`. |
| `gate.toml` | Gate schema 10: sensor models, tiers, hold-outs, and lane baselines. |
| `configs/` | Dataset VIO configurations and the `profiles/` overlays. |

`Cargo.lock` is committed. `cargo` never runs during `pixi lock` or
`pixi install`: the build is an explicit, cached pixi task.

### Core modules

| Module | What it is |
|---|---|
| `lie` | `So3`/`Se3` over any `f32`/`f64` scalar: SO(3) operations through kornia-algebra, the adjoint, four SO(3) Jacobians and the decoupled SE(3) pair. |
| `types` | `TimeCamId`, `KeypointId`/`LandmarkId`, `AbsOrderMap`, the three pose states and the two fixed-linearization wrappers. |
| `config` | `VioConfig`, read straight from `configs/*_config.json`, plus the `port.*` overlay keys the profiles set. |
| `calib` | `Calibration`: extrinsics, the six shipped camera models, the 9- and 12-parameter IMU bias calibrations. |
| `camera` | `pinhole`, `kb4` and `pinhole-radtan8` with 4-D homogeneous `project`/`unproject` and their analytic Jacobians. |
| `image` | `ImageU16`: an owned flat 16-bit frame with an explicit row stride, and bilinear sampling, central-difference gradients and interpolation bounds. |
| `pyramid` | The `PyramidBuilder` stage seam, `PyramidU16`, and `CpuPyramidBuilder`, with a separable integer Gaussian filter and one final rounding. |
| `landmark` | `StereographicParam`, the three-parameter `Landmark`, and `LandmarkDatabase` with a reproducible iteration order (D31). |
| `ba_base` | `BundleAdjustmentBase`: the window state maps, Huber-weighted `compute_error`, the reprojection residual and its three Jacobians, DLT `triangulate`. |
| `imu` | Preintegration: midpoint propagation, the covariance and bias-Jacobian recurrences, the 9-vector residual, gravity initialisation. |
| `frontend` | The optical-flow frontend: `patterns`, `se2`, `ldlt`, `patch`, `tracker`, `detect`, `flow` (`FrameToFrameOpticalFlow`, with detection on demand) and `parallel`. |
| `gpu` | The CubeCL frontend behind `gpu-wgpu`: the kernels, the per-cell corner selection, the patch and track stages, the `ReadRelay` that lets one stage's download carry another's buffers, and the seam counters. |
| `linearize` | The square-root linearization: `LandmarkBlock` and `LinearizationAbsQR`, which produce `H`, `b`, `Q2Jp`, `Q2r` and `l_diff`. |
| `marg` | Square-root marginalization: `MargHelper`'s rank-revealing flat Householder QR and `marginalizeHelperSqrtToSqrt`. |
| `qr` | In-place nalgebra reflections and Givens rotations, using column-major storage and reusable scratch. |
| `estimator` | The Offline sliding-window driver: `process_frame`, `schedule` (the keyframe vote and the keep/marginalize sets), the Levenberg-Marquardt `optimize` over reused scratch, and `frame_update`, the fast profile's between-keyframes solve. |

Design notes — each module's role and numerical contracts
([Core modules](docs/design-notes.md#core-modules)), and the four stages where an ulp or a rank
decision is load-bearing: [the frontend](docs/design-notes.md#the-frontend-and-the-one-thing-that-is-not-bit-parity),
[the landmark stage](docs/design-notes.md#the-landmark-stage-and-where-an-ulp-is-load-bearing),
[marginalization](docs/design-notes.md#marginalization-and-where-a-rank-decision-is-load-bearing),
[the damping machinery](docs/design-notes.md#the-damping-machinery-the-shipped-vio-never-uses).

## Accuracy and speed

The following tables record earlier profile comparisons. Current gate baselines
are stored in `gate.toml`.

Latency is the synchronous `Vio.track` call, one CPU core, decode excluded,
median over the clip after the first 60 framesets. ATE is RMSE against ground
truth after rigid alignment. On the RTX 5090 through Vulkan, wgpu frontend,
three interleaved rounds each:

| clip | cameras | length | reference: ms / cm | fast: ms / cm | cuVSLAM: ms / cm |
|---|---:|---:|---|---|---|
| `MIO10_short_2_panorama` (the smoke segment) | 2 | 7.6 s | 5.12 / 1.50 | 1.38 / 1.55 | 1.20 / 4.00 |
| `MIO11_short_3_backandforth` | 2 | 11 s | 4.73 / 2.47 | 1.35 / 2.76 | 1.04 / 2.54 |
| `MIO07_mapping_easy` | 2 | 76 s | 5.7 / 2.08 | 1.39 / 2.10 | 1.00 / 1.77 |
| `MGO07_mapping_easy` | 4 | 53 s | 10.2 / 2.29 | 2.10 / 2.37 | — |

cuVSLAM is NVIDIA's tracker in its offline Inertial mode on the same frames; its
mode for the four-camera rig runs without the IMU and is not comparable, so that
cell is blank. `MIO11` is the one clip of the four where the fast profile misses
its band, by 0.04 cm.

Over the whole catalog — 64 recordings, 316 minutes of video, one pass per
profile — the fast profile is inside its 10 % band on 51, more accurate than the
reference on 32, and loses no frameset on any; its tracker call is 2.0x
(msd-index), 2.35x (msd-g2) and 2.7x (msd-odyssey) shorter at the median.
Replaying the 156 minutes of msd-index end to end on one core, decode included,
takes 63 minutes on the fast profile against 81 on the reference. The ten-clip
gate's hardest clip, `MIO14_moving_props`, reads 9.72 cm on the reference (D71)
and 6.37 cm on the fast profile.

The fleet's fast-profile tracker medians below are milliseconds for
`MIO10` / `MIO07` / `MGO07`. Ratios compare GPU with CPU on the same host.
The 5090 values are the reference rows in `gate.toml`; GB10 and M4 use the
median of three matched runs per lane from S36. These are different measurement
sessions, not a cross-machine timing budget.

| device | backend | GPU vs CPU fast medians, ms | CPU/GPU | ≥1.2x, accuracy in band |
|---|---|---|---|---|
| RTX 5090, x86-64 | Vulkan | 2.02 / 2.70 / 3.08 vs 5.60 / 5.94 / 9.76 | 2.77x / 2.20x / 3.17x | pass; ten-clip ratios span 2.0–3.2x |
| GB10 (Spark), aarch64 | Vulkan | 2.88 / 3.07 / 4.75 vs 5.35 / 5.42 / 9.00 | 1.86x / 1.76x / 1.90x | pass |
| Apple M4 (Mac mini) | Metal | 4.59 / 4.76 / 5.97 vs 4.93 / 5.06 / 8.48 | 1.07x / 1.06x / 1.42x | MIO10 and MIO07 miss; MGO07 passes |
| RTX 3060, x86-64 | Vulkan | not re-run since the S32 tip; box needs a driver reboot | — | not measured |

The Metal lane's two sleeps are fixed: wgpu 30 replaces the HAL's 1 ms
completion polling, and our CubeCL patch parks and wakes the idle device
worker. The upload copy fix also ships. The Mac's two-camera clips still need
about 0.5 ms less tracker time to meet the 1.2x margin. MIO14's unchanged Mac
ATE is checked against its own host row. Passing that regression gate does not
establish the GPU/CPU speed margin.

See [S36 — the Metal lane's two sleeps](docs/design-notes.md#s36--the-metal-lanes-two-sleeps)
for the reports, measured budget, rejected experiments and remaining work.
The [S32 fleet table](docs/design-notes.md#the-fast-profile-across-the-fleet)
remains as historical evidence.

## Tests and gates

```bash
pixi run -e slam-rs-dev --frozen slam-rs-build      # cargo build + install _core.so in place
pixi run -e slam-rs-dev --frozen tests              # pytest (depends on the build)
pixi run -e slam-rs-dev --frozen lint               # ruff
pixi run -e slam-rs-dev --frozen typecheck          # pyrefly
pixi run -e slam-rs-dev --frozen deadcode           # vulture
pixi run -e slam-rs-dev --frozen slam-rs-clippy     # cargo clippy -D warnings
pixi run -e slam-rs-dev --frozen slam-rs-rust-test  # cargo test --workspace (default features)
pixi run -e slam-rs-dev --frozen slam-rs-version    # print the core version
```

The wgpu lane's gates run from the base feature on every Linux platform the package
declares, and from `slam-rs-osx-dev` on the Mac, where Metal is the backend:

```bash
pixi run -e slam-rs-dev     --frozen slam-rs-wgpu-clippy  # the portable lane compiles and is warning-clean, tests included
pixi run -e slam-rs-dev     --frozen slam-rs-wgpu-test    # workspace tests with wgpu; five nonempty GPU binary checks
pixi run -e slam-rs-dev     --frozen slam-rs-wgpu-doc     # rustdoc with warnings denied
pixi run -e slam-rs-osx-dev --frozen slam-rs-wgpu-doc     # the same strict docs through the Mac lane
pixi run -e slam-rs-osx-dev --frozen slam-rs-wgpu-test    # workspace tests with wgpu through Metal
```

Fast tests use synthetic inputs and Hypothesis properties; they finish in
seconds and need no recordings. Tests that read recordings are marked `slow`
and use the catalog. They check layer fingerprints, rig geometry, decode
consistency, exported clocks, and the smoke gate.

The ground-truth gate requires zero lost framesets, at least
`MIN_ASSOCIATED_POSES` estimate poses associated with ground truth, finite
measurements, and an estimated extent no greater than `DIVERGENCE_FACTOR`
times the truth's extent. The gate prefers a baseline for the host, lane and profile; if absent, it uses
the first row for that lane/profile. With that baseline, RMSE must be at most
`1.10 * baseline.gt_rmse_cm`. A missing lane/profile baseline prints "no baseline"
and leaves accuracy ungated; tracking and validity clauses still apply.

The gate measures the median over all accepted tracker calls. It must be at most `1.10 * baseline.median_tracker_ms` only on the
baseline host with the matching lane/profile. Elsewhere the row prints the
measurement and "speed not gated on this host".

```bash
cd packages/slam-rs
pixi run -e slam-rs-dev --frozen pytest -q
pixi run -e slam-rs-dev --frozen pytest -q -m slow -k "catalog or gate or robocap"
pixi run -e slam-rs-dev --frozen python tools/apps/fleet_check.py --gpu --tier release
pixi run -e slam-rs-dev --frozen python tools/apps/fleet_check.py --tier smoke
pixi run -e slam-rs-dev --frozen python tools/apps/robocap_probe.py --gpu --rr-config.headless
```

The tiers are `smoke` (MIO10, MGO09), `release` (MIO07, MGO07, MIO14), and
`listed` (the other five). The two hold-out flags remain excluded from tuning.
The historical design notes are retained separately in `docs/design-notes.md`.

### Patched dependencies

`slam-rs-patch-deps` verifies the CubeCL archive SHA256, applies the checked-in
channel park patch into `target/patch/`, and checks the prepared files on reuse.
A process lock makes concurrent preparation safe. Cargo consumes that tree
through `[patch.crates-io]`; the Pixi Cargo tasks prepare it first.

Bare Cargo and rust-analyzer need this once per fresh checkout:

```bash
pixi run -e slam-rs-dev --frozen slam-rs-patch-deps
# macOS: use -e slam-rs-osx-dev
```

`slam-rs-patch-test` resolves the prepared crate as a standalone package and
runs offline. Fill each Cargo home's cache once from the package directory:

```bash
pixi run -e slam-rs-dev --frozen cargo fetch --locked --manifest-path target/patch/cubecl-common-0.11.0-pre.3/Cargo.toml
pixi run -e slam-rs-dev --frozen slam-rs-patch-test
```

Use `slam-rs-osx-dev` on macOS. A missing `test-log` offline error means that
cache is incomplete. The version-bump runbook is beside `[patch.crates-io]`
in `Cargo.toml`.

## What is next

Not in this branch, in the order they are likely to matter:

- **Metal.** Close the two-camera 1.2x gap. Measure persistent staging and
  in-place `ComputeClient::write` uploads, then the cold read hand-off. A
  SIMD-local KLT reduction redesign is a later option; changed reduction order
  must pass accuracy gates on every lane. Four KLT iterations and a simple
  32-thread mapping were tested and rejected.
- **A second core.** One core was the rule for this branch. The between-keyframes
  solve and the frontend's host work are independent enough to overlap.
- **The Python seam.** 0.14 ms a frameset between the feed and `Vio.track`,
  fixed across clips: a tenth of a fast `MIO10` call.
- **More datasets.** Camera-only datasets (Assembly101, HO-Cap, the WildCap sets)
  need a vision-only estimator alongside the VIO. Aria recordings need
  the fisheye624 camera model.
- **Results as a catalog layer.** One layer per segment with the estimated poses, the
  landmarks and the keypoints on the base recording's entity paths, registered beside
  the ground truth, so a run is browsed in the viewer, not in a CSV.
- **Less code.** With ground-truth accuracy checks the Lie groups and the camera
  models could move further into kornia-rs. S34 already uses nalgebra for QR,
  the damped solve and SVD; the ground-truth gate checks further replacements.

`gate.toml` beside this README holds the rigs’ sensor noise models that the catalog does not carry, gate tiers, hold-outs, decode paths, and measured lane/profile baselines. It also holds RoboCap rig and clock rules plus its one regression trajectory path. Camera geometry and capture facts come from the catalog.
