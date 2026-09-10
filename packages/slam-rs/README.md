# slam-rs

<p align="center">
  <img src="media/slam-rs-github.gif" alt="slam-rs replaying a Monado SLAM Dataset clip in Rerun: the estimated trajectory against ground truth and the basalt C++ run, the landmarks, the rig, and the tracked keypoints on the camera frames" width="800" />
</p>

Visual-inertial odometry with a Rust core. The estimator is a port of the
basalt VIO fork: pure Rust, N-camera from the start, with a CPU frontend and a
portable GPU frontend through CubeCL and wgpu. Python owns the plumbing —
catalog feed, decode, evaluation and Rerun logging — and talks to the core
through a PyO3 extension module, so the whole pipeline runs from Python:
`_core.Vio` consumes IMU samples and framesets and reports a pose, and
`tools/apps/replay.py --stage vio` draws the estimate against the ground truth
on the same frames. Two config profiles run on one code path. `reference`
reproduces the C++ fork's estimator byte for byte; `fast` keeps its error within
10 % of the reference's on each clip and makes the tracker call two to four times
shorter. On the smoke segment the reference reads 1.50 cm from ground truth,
where the C++ itself reads 1.43 cm; the fast profile reads 1.55 cm in 1.38 ms a
frameset on an RTX 5090. The same code runs unchanged on `linux-64`,
`linux-aarch64` and macOS `osx-arm64`.

Design notes — the module-by-module account of the port, the full Python API,
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
optical flow over it and draws its keypoints beside the C++ fork's:

```bash
cd packages/slam-rs
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio   # the smoke segment, in a viewer
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --rr-config.headless --rr-config.save data/replay-vio.rrd
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --segment <segment-id>       # another reference segment
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --rrd base.rrd --gt-rrd gt.rrd   # a recording of your own
```

In a shell without `DISPLAY`, pass `--rr-config.headless` or the spawned viewer
wedges the recording stream. A long segment still wants `--max-framesets`.

The GPU frontend is the off-by-default `gpu-wgpu` cargo feature, through
CubeCL and wgpu (Vulkan / Metal / DX12). It writes the in-place
`slam_rs/_core.so`; `--gpu` selects that frontend and `--profile fast` the
speed profile:

```bash
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-build   # a core whose `--gpu` is wgpu
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --gpu                 # the reference profile
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py --stage vio --gpu --profile fast  # the fast one
```

Design notes: [the GPU lane](docs/design-notes.md#the-gpu-lane), and
[the portable lane, and the two silent failures](docs/design-notes.md#the-portable-lane-and-the-two-silent-failures).

On macOS everything above runs from the mac lane's environment, which is where
that platform's `slam-rs` features are solved: `-e slam-rs-osx-dev` in place of
`-e slam-rs-dev`.

## Two profiles

The configs under `configs/` are the files the C++ reference runs read, key for
key, and a test keeps them that way. A profile is a flat overlay in
`configs/profiles/<name>.json` applied on top of one. `reference` is empty.
`fast` is three keys:

```json
{"config.vio_max_iterations": 7, "port.redetect_survivor_ratio": 0.85, "port.frame_update_max_iterations": 5}
```

A key spelled `port.` is a knob basalt has no field for; absent, it reproduces
basalt's behaviour. A key that is neither a basalt field nor a listed port key
is a `KeyError`, so a typo cannot silently change a run. The fast profile changes
two things about the schedule and nothing about the arithmetic:

- **Detection on demand.** basalt tops up every empty grid cell on every
  frameset. The fast profile detects only when camera 0 holds fewer than 85 % of
  the keypoints the last detecting frameset ended with, which is cuVSLAM's
  schedule.
- **The window is solved at keyframes.** basalt re-solves the whole sliding
  window on every frameset. The fast profile does so at keyframes only; between
  them it solves the newest pose, velocity and biases against the held landmarks
  and the IMU factor, five steps at most, and falls back to the joint solve when
  that update declines. `VioSnapshot.frame_update` says which one ran.

The gate is per clip: ATE against ground truth within 1.1x the reference
profile's, and zero lost framesets. Fast trajectories are not byte-identical to
reference ones, and not byte-identical across GPU vendors either. Reference
trajectories are, on every Vulkan device measured; Metal differs in the last bit.

Every tracking tool takes `--profile reference|fast`: `replay.py`,
`bench_track.py`, `fleet_check.py` and `robocap_fleet.py`. In code,
`slam_rs.reference.profiled_config_text(path, "fast")` returns the overlaid JSON.

Design notes: [D74](docs/design-notes.md#d74--speed-profile) the profile,
[D75](docs/design-notes.md#d75--redetect-on-demand-the-fast-profile-detects-when-camera-0-has-lost-tracks) detection on demand,
[D76](docs/design-notes.md#d76--the-fast-profile-solves-the-window-at-keyframes-and-the-newest-state-alone-between-them) the keyframe-gated solve.

## How data gets in and out

The library reads one thing: a recording in the dataforge rig schema. One base
`.rrd` per sequence carries the rig calibration, one video stream per camera and
the IMU stream. Ground truth and results are separate layer files that stack onto
the same entity paths.

Raw data, whatever its files look like (an EuRoC folder, a ROS bag, a vendor
SDK), is converted into that recording once, with a dataforge ingester. After
that every tool reads the same bytes: the viewer, this estimator, the C++
reference.

- One sequence: `tools/apps/replay.py --stage vio --rrd base.rrd [--gt-rrd gt.rrd]`.
  The feed serves the files from an in-process server; no catalog server is
  needed.
- Many sequences: register the base and layer files on a catalog server and
  address them by segment id through the reference manifest.

The estimate comes back the same way: a trajectory CSV, and a Rerun recording
that layers the estimated poses, the landmarks and the window onto the input,
beside the ground truth and, where the reference set has one, the C++ run.

In code the contract is three calls: `Calibration` is the rig, `Vio.push_imu`
takes one IMU sample, `Vio.track` takes one synchronized frameset of `uint8`
images. The feed is the only adapter between the recording and those calls.

## Python API

```python
from pathlib import Path

from slam_rs import _core
from slam_rs.reference import profiled_config_text

calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)  # the feed's dataclasses
config = _core.VioConfig.from_json(Path("configs/msdmi_config.json").read_text())  # the file the C++ ran
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

One `track` call is basalt's whole pipeline for one frameset in the calling
thread (Offline mode, D17), so every result is final and a repeat run over the
same input is bit-identical. `VioStatus` has two states: `NeedMoreImu` where
basalt would block on its IMU queue, `Tracking` otherwise; `replay.py` holds
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
| `reference_segments.toml` | The frozen reference set. |
| `configs/` | The basalt VIO configs the reference runs used, vendored from the fork, and the `profiles/` overlays. |
| `tests/reference/` | Checked-in basalt C++ trajectories the gate tests reproduce. |
| `slam_rs/reference_bundle.py` | Resolves the two long-tier artifacts kept out of git. |

`Cargo.lock` is committed. `cargo` never runs during `pixi lock` or
`pixi install`: the build is an explicit, cached pixi task.

### Core modules

| Module | What it is |
|---|---|
| `lie` | `So3`/`Se3` over any `f32`/`f64` scalar: Sophus's `exp`/`log`, the adjoint, basalt's four SO(3) Jacobians and the decoupled SE(3) pair. |
| `types` | `TimeCamId`, `KeypointId`/`LandmarkId`, `AbsOrderMap`, the three pose states and the two fixed-linearization wrappers. |
| `config` | basalt's `VioConfig`, read straight from `configs/*_config.json`, plus the `port.*` overlay keys the profiles set. |
| `calib` | basalt's `Calibration`: extrinsics, the six shipped camera models, the 9- and 12-parameter IMU bias calibrations. |
| `camera` | `pinhole`, `kb4` and `pinhole-radtan8` with basalt's 4-D homogeneous `project`/`unproject` and their analytic Jacobians. |
| `image` | `ImageU16`: an owned flat 16-bit frame with an explicit row stride, and `interp`/`interp_grad`/`in_bounds` from `image.h`. |
| `pyramid` | The `PyramidBuilder` stage seam, `PyramidU16`, and `CpuPyramidBuilder`, whose `subsample` is bit-exact with `image_pyr.h`. |
| `landmark` | `StereographicParam`, the three-parameter `Landmark`, and `LandmarkDatabase` with a reproducible iteration order (D31). |
| `ba_base` | `BundleAdjustmentBase`: the window state maps, Huber-weighted `compute_error`, the reprojection residual and its three Jacobians, DLT `triangulate`. |
| `imu` | Preintegration: basalt's midpoint propagation, the covariance and bias-Jacobian recurrences, the 9-vector residual, gravity initialisation. |
| `frontend` | The optical-flow frontend: `patterns`, `se2`, `ldlt`, `patch`, `tracker`, `detect`, `flow` (`FrameToFrameOpticalFlow`, with detection on demand) and `parallel`. |
| `gpu` | The CubeCL frontend behind `gpu-wgpu`: the kernels, the per-cell corner selection, the patch and track stages, the `ReadRelay` that lets one stage's download carry another's buffers, and the seam counters. |
| `linearize` | The square-root linearization: `LandmarkBlock` and `LinearizationAbsQR`, which produce `H`, `b`, `Q2Jp`, `Q2r` and `l_diff`. |
| `marg` | Square-root marginalization: `MargHelper`'s rank-revealing flat Householder QR and `marginalizeHelperSqrtToSqrt`. |
| `eigen` | The Eigen ports in one place — `qr`, `ldlt`, `svd`, `blas` — each reproducing Eigen's **operation order** rather than only its result (D44). |
| `estimator` | The Offline sliding-window driver: `process_frame`, `schedule` (the keyframe vote and the keep/marginalize sets), the Levenberg-Marquardt `optimize` over reused scratch, and `frame_update`, the fast profile's between-keyframes solve. |

Design notes — what each module reproduces, quoted against the C++ it comes from
([Core modules](docs/design-notes.md#core-modules)), and the four stages where an ulp or a rank
decision is load-bearing: [the frontend](docs/design-notes.md#the-frontend-and-the-one-thing-that-is-not-bit-parity),
[the landmark stage](docs/design-notes.md#the-landmark-stage-and-where-an-ulp-is-load-bearing),
[marginalization](docs/design-notes.md#marginalization-and-where-a-rank-decision-is-load-bearing),
[the damping machinery](docs/design-notes.md#the-damping-machinery-the-shipped-vio-never-uses).

## Accuracy and speed

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

The same tip on the fleet, fast profile, tracker median in ms for
`MIO10` / `MIO07` / `MGO07`, one unpinned pass with decode in the same process:

| device | backend | fast, ms | fast over reference | trajectory against the 5090 |
|---|---|---|---|---|
| RTX 5090, x86-64 | Vulkan | 2.0 / 2.7 / 3.1 | 2.0x / 1.8x / 2.2x | the baseline |
| GB10 (Spark), aarch64 | Vulkan | 2.9 / 3.2 / 4.8 | 1.85x / 1.9x / 2.1x | byte-identical, both profiles |
| RTX 3060, x86-64 | Vulkan | 14.7 / 14.9 / 20.2 | 1.4x / 1.5x / 1.7x | inside the band, last-bit drift |
| Apple M4 (Mac mini) | Metal | 18.8 / 18.6 / 21.2 | 1.2x / 1.2x / 1.2x | inside the band, within 0.04 cm |

Every row tracks every frameset. The 3060 stayed at its idle clock for the run,
so its numbers are that operating point, not the card's. The Mac is correct and
slow for a measured reason: a synchronising read costs 7 ms of host time on
Metal against 0.12 ms on the 5090, and the frontend makes two a frameset. The
Pi 5 and the RK3588 cap have not run this tip.

Design notes — the precision band (D60), the portability table and the fleet
numbers with the Metal diagnosis:
[the portable lane](docs/design-notes.md#the-portable-lane-and-the-two-silent-failures),
[where it runs](docs/design-notes.md#where-the-portable-lane-runs),
[the fast profile across the fleet](docs/design-notes.md#the-fast-profile-across-the-fleet).

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

The oracle lanes are the fixtures the C++ fork itself produced — pyramid, camera,
IMU, landmark, linearization and marginalization, under
`crates/slam-rs/tests/fixtures/`, plus the frontend's keypoint parity in
`crates/slam-rs/tests/flow_parity.rs` — and they run inside `slam-rs-rust-test`.
`crates/slam-rs/tests/vio_oracle.rs` pins the vendored configs to the ones the
C++ runs used and the reference profile's LM trail to the C++'s;
`tests/test_config_profiles.py` covers the overlays and their typo check.

The smoke digest is checked in: `frames.sha256` and a copy of `gt.csv` under
`tests/reference/msd/<segment>/` for the smoke pair, so that gate runs with no
NAS and no catalog. The rest of the slow lane reads a reference `.rrd` from the
NAS or queries the catalog, and skips when neither is reachable:

```bash
cd packages/slam-rs
pytest -m slow -q                                             # NAS + catalog
pytest -m slow -q -s tests/test_v2_gate.py                    # the iteration set
SLAM_RS_V2_ALL=1 pytest -m slow -q -s tests/test_v2_gate.py   # all ten, whole, shortest first
SLAM_RS_V2_WINDOW_S=5 SLAM_RS_V2_ALL=1 pytest -m slow -q -s tests/test_v2_gate.py   # all ten, first 5 s each
```

Design notes — the [reference set](docs/design-notes.md#the-reference-set) and its three tiers, the
[gate policy](docs/design-notes.md#the-basalt-c-reference-and-the-gate-policy) per segment,
[two clocks](docs/design-notes.md#two-clocks-converted-once), the
[feed, the metrics and the replay tool](docs/design-notes.md#the-feed-the-metrics-and-the-replay-tool)
with its entity trees, every clause of [the V2 gate](docs/design-notes.md#the-v2-gate), and what
[the fast and slow lanes](docs/design-notes.md#tests) each cover.

## What is next

Not in this branch, in the order they are likely to matter:

- **Metal.** The M4 runs the lane correctly and pays 7 ms per synchronising read
  and ten times the 5090's device time per kernel. The fix is a lower-latency
  completion path in the read routine, gated to Metal, and per-adapter workgroup
  shapes chosen at start-up; the 5090 A/B harness stays the gate, so that path
  is untouched.
- **Four cameras on the fast profile.** The schedule knobs were tuned on
  two-camera clips. `MGO09_short_1_updown`, 3 s long, is the one ten-clip miss
  (0.98 cm against a 0.85 cm band); the 53 s `MGO07` passes by 0.08 cm.
- **A second core.** One core was the rule for this branch. The between-keyframes
  solve and the frontend's host work are independent enough to overlap.
- **Run provenance.** Every tool's output should carry the profile name and the
  overlay's digest, so an archived number says which profile produced it.
- **The Python seam.** 0.14 ms a frameset between the feed and `Vio.track`,
  fixed across clips: a tenth of a fast `MIO10` call.
- **More datasets.** Camera-only datasets (Assembly101, HO-Cap, the WildCap sets)
  need basalt's vision-only estimator ported beside the VIO. Aria recordings need
  the fisheye624 camera model.
- **Results as a catalog layer.** One layer per segment with the estimated poses, the
  landmarks and the keypoints on the base recording's entity paths, registered beside
  the ground truth, so a run is browsed in the viewer, not in a CSV.
- **A catalog URL on the command line.** The feed reads the catalog already; the replay
  tool only reaches it through manifest entries.
- **Less code.** Under the tolerance requirement (D60, D64) the Lie groups and the camera
  models could come from kornia-rs and the Eigen-order QR, LDLT and SVD from nalgebra;
  the ten-clip gate decides. About 2,800 lines.
