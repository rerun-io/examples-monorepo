# slam-rs

Visual-inertial odometry with a Rust core. The estimator is a port of the
basalt VIO fork: pure Rust, CPU first, N-camera from the start. Python owns the
plumbing — catalog feed, decode, evaluation and Rerun logging — and talks to the
core through a PyO3 extension module.

## Core modules

The core is being filled in stage by stage, bottom up. What is in it today:

| Module | What it is |
|---|---|
| `lie` | `So3`/`Se3` over any `f32`/`f64` scalar: Sophus's `exp`/`log`, the adjoint, basalt's four SO(3) Jacobians and their inverses, the decoupled SE(3) pair, and the left-multiplied pose increment the estimator runs on. |
| `types` | `TimeCamId`, `KeypointId`/`LandmarkId`, `AbsOrderMap`, `PoseState`/`PoseVelState`/`PoseVelBiasState` and the two fixed-linearization wrappers. |
| `config` | basalt's `VioConfig`, read straight from `data/**/*_config.json`. |
| `calib` | basalt's `Calibration`: extrinsics, the six shipped camera models, the 9- and 12-parameter IMU bias calibrations, plus a constructor that takes what the Python catalog feed reports. |
| `camera` | `pinhole`, `kb4` and `pinhole-radtan8` with basalt's 4-D homogeneous `project`/`unproject` and their analytic Jacobians (2x4 point, 2xN parameter, 4x2 and 4xN for unprojection), the `rpmax` and `z >= epsilonSqrt` domain checks, and a `CameraEnum` that dispatches without a vtable. `ds`, `eucm` and `ucm` parse but are rejected here. |
| `image` | `ImageU16`: an owned flat 16-bit frame with an explicit row stride, the stride-aware `u8 << 8` widening basalt's readers do, and `interp`/`interp_grad`/`in_bounds` reproduced from `image.h` in the same arithmetic order. |
| `pyramid` | The `PyramidBuilder` stage seam with an associated `Pyramid` type that lends nothing (geometry plus a copy into the caller's buffer), `PyramidU16` (one flat buffer per level, not basalt's packed mipmap) and `CpuPyramidBuilder`, whose `subsample` is bit-exact with `image_pyr.h:99-140`. |
| `landmark` | `StereographicParam` (`project`/`unproject` and both Jacobians), the three-parameter `Landmark` with its backup pair, and `LandmarkDatabase`: the host->target->landmark adjacency, the `min_num_obs = 2` sweep and `remove_keyframes`. Landmarks live in one id-sorted `Vec` behind a `BTreeMap` index rather than a per-landmark hash map, and every map is a `BTreeMap`, so iteration order is reproducible (D31). |
| `ba_base` | `BundleAdjustmentBase`: the two window state maps, `get_pose_state_with_lin`, basalt's Huber-weighted `compute_error` with optional outlier collection, `compute_projections`, `compute_delta`, `backup`/`restore`, the reprojection residual and its three Jacobians from `ba_utils.h`, `computeRelPose`, and DLT `triangulate` over a ported Eigen `JacobiSVD`. |
| `imu` | Preintegration: `IntegratedImuMeasurement<S>` with basalt's midpoint propagation, covariance and bias-Jacobian recurrences, the 9-vector residual and its Jacobians, the LDLT square-root inverse covariance, the between-frames accumulation loop, gravity initialisation, and the 15-row IMU block the estimator whitens. |
| `frontend` | The optical-flow frontend: `patterns` (Pattern52/51 from `patterns.h`; the other two are unreachable on every shipped config), `se2` (`AffineCompact2` and `Sophus::SE2::exp`), `ldlt` (Eigen's pivoted LDLT at 3x3), `patch` (the streaming inverse-compositional patch build), `tracker` (`PatchSoA`, `FlowTransforms`, the `SourcePatches`/`PatchTracker` stage traits and `CpuPatchTracker`), `detect` (basalt's centred cell grid over kornia-rs's FAST plus OpenCV's suppression), `flow` (`FrameToFrameOpticalFlow`, generic over the builder and tracker) and `parallel` (the explicit thread budget). |
| `linearize` | The square-root linearization: `LandmarkBlock` (basalt's `[ J_p \| pad \| J_l \| r ]` buffer, the layout arithmetic of `landmark_block_abs_dynamic.hpp:83-96`, the Huber-weighted residual rows, three Householder reflections, back-substitution with its exact model cost change) and `LinearizationAbsQR`, which owns the blocks, the IMU blocks and the marginalization prior and produces `H`, `b`, `Q2Jp`, `Q2r` and `l_diff`. No damping and no Jacobian scaling: the fork comments every call site out and the port carries none of it (D34, D68). Eigen's `makeHouseholder` and `applyHouseholderOnTheLeft` are ported coefficient for coefficient rather than delegated to nalgebra's equivalents (D44). |
| `marg` | Square-root marginalization: `MargHelper`'s rank-revealing flat Householder QR, `marginalizeHelperSqrtToSqrt` — the one routine of the three the shipped path reaches — plus the `marginalize()` mechanics of `sqrt_keypoint_vio.cpp:896-1178` given an explicit keep/marginalize schedule. The two squared-form routines, the complete orthogonal decomposition they inverted the marginalized block with, and `checkMargNullspace`/`checkEigenvalues` are **not** ported: `SqrtKeypointVio::new` refuses `vio_sqrt_marg` off, so nothing on any shipped config reaches them (D68). |
| `eigen` | The Eigen ports, in one place, each reproducing Eigen's **operation order** rather than only its result: `qr` (`makeHouseholder`, `applyHouseholderOnTheLeft`, `makeGivens` and the `Redux.h` traversals), `ldlt` (the pivoted LDLT at dynamic size, D41, which the LM step solves through), `svd` (the Jacobi SVD the DLT triangulation needs) and `blas` (the `gemv` associations the other three call). Not "numerics utilities" to be swapped for nalgebra's (D44): the last bit reaches a rank test, a finite check and a triangulation gate. |

Two conventions in `ba_base` are basalt deviating from its own papers, and the
port keeps **both** halves of each. The reprojection residual is `pi(...) - z`,
the flip of Paper 1 Eq. (8) (`ba_utils.h:117`), which the estimator compensates
for by negating the increment (`sqrt_keypoint_vio.cpp:1450`); and the Huber
weight is taken on the raw pixel residual, before the `1/sigma` scaling
(`ba_base.cpp:179-182`), so the shipped 1.0 px threshold against a 0.5 px sigma
is an effective 2 sigma. `compute_error` is sequential in this stage, written as
a fixed-order fold over per-host-frame partials so the `threads` config field can
later turn it into a `par_chunks` without changing the sum. The linearizer's four
reductions are a different matter: they replace
`tbb::parallel_deterministic_reduce`, whose order is a balanced join tree rather
than a fold - see the `linearize` module.

Every convention is quoted against the C++ it comes from, file and line, in the
doc comments. `crates/slam-rs/tests/fixtures/` holds the shipped basalt
calibration JSON the parsers are tested against, unmodified - the VIO configs
they run with are the package's own `configs/`, read from there - plus four
fixtures produced by the C++ fork itself: `pyramid/`, the first frame of the
smoke reference segment as a PGM next to the four pyramid levels the fork builds
from it, which the pyramid is checked against byte for byte;
`camera_oracle.json`, what basalt's camera headers return for ten cameras and
thirty points each in **both** precisions - pixel, bearing, and in double also
both projection Jacobians and the unprojection Jacobian - plus six probe pixels
handed straight to `unproject`, one of them singular; `imu/imu_oracle.json`,
the delta state, covariance, bias Jacobians, Eigen LDLT and square-root inverse
covariance of seven preintegration runs, plus what
`Quaternion::FromTwoVectors` returns for ten accelerometer readings;
`flow/`, three 960x960 frameset pairs as PGMs beside the keypoints the C++
frontend produced from eight of them; `linearize/linearize_oracle.json`, four
small visual-odometry problems (two and three frames, two cameras, three to six
landmarks with two to four observations each, two of them carrying a
marginalization prior with both of their first frames frozen at their
linearization point) with, per landmark block, the layout numbers, the
linearization error, the whole `storage` buffer after `linearizeLandmark` and
after `performQR` (the damped and undamped buffers are in the file too, read by
nothing since D68 dropped the damping stack), the `Q2Jp`/`Q2r` and `JtJ`/`Jtr`
exports, and
what `backSubstitute` leaves behind - plus, per problem, what
`LinearizationAbsQR` returns through its public interface: the error, the dense
`H` and `b`, the stacked `Q2Jp`/`Q2r` and the total `l_diff`; and
`lmdb/lmdb_oracle.json`, the
stereographic chart with both its Jacobians at twelve points, `linearizePoint`'s
residual, `d_res_d_xi`, `d_res_d_p` and `proj` for five configurations of each of
the two shipped reference cameras, ten `triangulate` cases with four of them
placed on basalt's `0 < inv_dist < 3` acceptance gate, 32 more per precision
sitting on that gate (half of them chosen because the summation order alone
decides them), 32 probes of `head<3>().squaredNorm()` per precision, and 14
residuals through the Huber-weighted cost, five of them above the threshold.

The camera port reproduces every double to 1e-15 relative (1e-12 for
unprojections, which run a Newton iteration) and every float **exactly**; the IMU
port reproduces every double to 1e-14, and to 1e-7 through the whitening, which
inverts the covariance; the landmark port reproduces the chart and the residual
to 1e-12 in double and **exactly** in float, triangulation bit for bit on nine of
the ten double cases, and Eigen's three-coefficient reduction order and the
Huber-weighted cost of one observation bit for bit in both precisions. The
linearization port reproduces **every** coefficient of all four problems - the
QR'd block, `H`, `b`, `Q2Jp`, `Q2r`, the landmark increments and `l_diff` - to
`6.3e-16` relative in double and `1.4e-6` in float, measured against the array's
own scale. All six generators live on the fork's `slam-rs-reference` branch, as
`tools/dump_pyramid.cpp`, `tools/camera_oracle.cpp`, `tools/imu_oracle.cpp`,
`tools/dump_flow.cpp`, `tools/lmdb_oracle.cpp` and `tools/linearize_oracle.cpp`;
the monorepo never compiles C++.

The IMU fixture earns its keep on one run: the covariance after a single sample
with a still gyroscope and accelerometer is rank deficient, and what basalt does
with it is decided entirely by `Eigen::LDLT`. Eigen pivots on the *un-updated*
diagonal, eliminates velocity first and leaves the position pivots at `-1.6e-27`,
which basalt's `vectorD()[i] < numeric_limits::min()` test zeroes. A textbook
pivoted LDLT eliminates position first, leaves a tiny *positive* pivot, and puts
an information weight of `6.2e26` on a direction the measurement says nothing
about.

One thing the camera port inherits and the frontend does live with:
`unproject` runs a fixed three (kb4) or five (radtan8) Newton steps, and on wide
calibrations that is not always enough. Inside basalt's own
`optical_flow_image_safe_radius` nine of the ten shipped cameras invert to 1e-11;
msd-g2 cam2 is off by 0.12 in bearing *inside* that radius, and RoboCap cam1
outside it returns a bearing pointing backwards. basalt's C++ returns the same
numbers to the last figure, so this is a property of the algorithm, not of the
port; `crates/slam-rs/tests/camera_jacobians.rs` pins all three cases.

### The frontend, and the one thing that is not bit-parity

Everything on the tracking path is the C++'s arithmetic in the C++'s order, and
it shows: seeded with basalt's own keypoints on the smoke segment's first eight
framesets, the port's tracker puts **697 of 697** of them within half a pixel of
where the C++ put the same id one frame later, the worst of them 0.0003 px away.
That gate lives in `crates/slam-rs/tests/flow_parity.rs` and runs off two small
fixtures: three 960x960 frameset pairs as PGMs (the exact bytes the Python feed
decodes) and the eight per-frameset JSON dumps the fork's `tools/dump_flow.cpp`
produced from them.

The detector is the one place the port cannot be bit-identical (decision D09,
trap 2), but it is close. basalt runs `cv::FAST` on each 8-bit cell; the port
runs kornia-rs's FAST over basalt's own centred cell geometry, and at arc length
9 kornia returns the same canonical FAST-9 `cornerScore` OpenCV computes. The
grid, the threshold ladder, the per-cell budget, the safe radius, the masks, the
edge margin **and OpenCV's non-maximum suppression** — strictly greater than all
eight neighbours, so a tie kills both sides — are all reproduced. What is left is
the cell walk itself and an unstable `std::sort`. Measured against the C++ dump:
**95.5 to 96.9% of the C++'s keypoints have a port keypoint within one pixel**,
the two agree on camera 0's keypoint count on **seven of the eight** framesets,
their occupancy grids agree on **87 of 87** occupied cells, and every one of the
171 corners both sides picked at the same pixel carries the **same integer
`cornerScore`** — OpenCV's `max(a0, -b0) - 1`, which is one less than the value
kornia returns. The gate still seeds the tracker rather than diffing keypoint
sets, because "close" is not "equal".

The frontend is generic over its two stages. `PyramidBuilder` and `PatchTracker`
(with `SourcePatches` beside it) carry associated pyramid and patch types, and
`FrameToFrameOpticalFlow<P, B, T>` defaults them to the CPU pair — so a CubeCL
backend arrives through `with_backends` and no public signature here names a
concrete pyramid, and `FlowResult` has a public writing surface (`reset`,
`set_track`, `parts_mut`, `finish`) that the CPU tracker itself publishes
through, so a second backend can too. Every per-patch buffer is
structure-of-arrays with the patch index fast-varying, including the 2x3 warps,
which live as six flat coefficient arrays; the patch build streams straight into
those arrays, and at `-O3` its largest local allocation measures 36 bytes against
a 344-byte stack frame.

A frame the frontend refuses is as if it never happened: the whole call runs
against a staging pyramid set over a snapshot of the keypoint state, and the two
pyramid sets swap and the clock advances only after tracking, the cell counts and
the add/match/filter passes have all succeeded. Snapshot and restore are
**allocation-free** — every type in the chain writes `clone_from` by hand, since
the derived one replaces the buffers instead of overwriting them — and
`crates/slam-rs/tests/frame_allocations.rs` counts that with a global allocator
rather than claiming it. A whole frame is not allocation-free: on a 200x200
scene it reaches the allocator a few hundred times, all of it inside kornia's
FAST, which allocates one `Vec` per image row per cell per rung of the threshold
ladder. A frame that finds no keypoints at all costs more, which is what pins the
cost there rather than on anything the port owns.

Three deviations are recorded in the source. `E[i]` is computed from `T_c0_ci`
per camera rather than reusing the cam0-cam1 matrix everywhere, which is
upstream's bug (`optical_flow.h:207-213`); `FrontendOptions::epipolar_per_camera`
switches it back for a C++-parity run and makes no difference at all on a
two-camera rig. Image bounds come from each camera's own resolution rather than
camera 0's, because the msd-g2 recordings are stored rotated — while the
detection grid is per camera, as the C++'s is, and the occupancy matrix keeps
camera 0's shape, as the C++'s does. And `FrontendOptions::max_keypoints` is a
budget basalt has no equivalent of: the port's buffers are preallocated, so
detection and matching stop adding once a camera is full rather than producing a
frame the tracker cannot carry.

### The landmark stage, and where an ulp is load-bearing

Three ulp-level findings came out of the landmark fixture, and all three changed
code outside the stage. `So3 * Vector3` now sums Sophus's three terms in
Sophus's order (`so3.hpp:408-417`) rather than nalgebra's association, which was
an ulp off in triangulation; `compute_error` scales the residual **row** before
the dot product, as C++'s left-associative `*` does (`ba_base.cpp:182`), which
was 4e-6 off in `f32` on a 10-pixel residual; and every `head<3>().norm()` on
the residual path sums in Eigen's order, which is **not the same in the two
precisions**.

That last one is `LieScalar::eigen_redux3`. Eigen picks a reduction strategy by
comparing `find_best_packet<Scalar, 3>` against the three coefficients
(`Redux.h:29-67`): in `f64`, `Packet2d` holds two of them, so one packet is
reduced and the remainder folded in — `(a + b) + c`; in `f32`, `Packet4f` is
wider than the whole expression, the aligned part is empty, and the scalar
unroller's `Length / 2` split runs instead — `a + (b + c)`. Using one order for
both is an ulp, and the ulp is load-bearing at each end: in `f32` it moved the
inverse depth of a landmark at `inv_dist = 1e-7`, and in `f64` it made a
landmark exactly 1/3 m away normalise to `2.9999999999999996` instead of `3.0`,
which basalt's `inv_dist < 3` gate **accepts** where the C++ rejects. Sixteen of
the 32 boundary cases in the fixture are there because the summation order
decides them on its own.

The packet widths are fixed by the fork's own build flags — a trailing
`-march=nocona` overrides the earlier `-march=native`, so the reference binary
is SSE3 on every host and this is a property of basalt, not of the machine.

All three follow decision D44's rule: an elementary operation whose rounding can
reach a threshold comparison is ported in Eigen's or Sophus's operation order,
not delegated to nalgebra's equivalent.

The DLT's 4x4 SVD is a step-for-step port of Eigen's `JacobiSVD`, not a call into
nalgebra's: a 4x4 with `ComputeFullV` takes Eigen's square path, so the whole
algorithm is the scaling, the sweep of 2x2 real Jacobi rotations and the final
sort, with no QR preconditioner. It is worth porting because basalt gates
landmark acceptance on `0 < inv_dist < 3`, where a borderline point either exists
or does not.


### Marginalization, and where a rank decision is load-bearing

`marginalizeHelperSqrtToSqrt` is one flat, rank-revealing Householder QR over
the stacked `[J_marg | J_keep]`, columns permuted **marginalized first**
(`marg_helper.cpp:259-273`) so the sweep eliminates what is leaving before it
reaches what stays; the prior is then the rows between the marginalized rank and
the total rank (`:320-323`). A column whose reflector produces
`|beta| <= sqrt(epsilon)` is zeroed and does **not** advance the rank
(`:301-310`) — an absolute threshold, not a relative one against the largest
pivot, so it depends on the units the problem is scaled in. That is basalt's
choice and reproducing its decision is what keeps two runs on the same
trajectory. `tests/fixtures/marg/marg_oracle.json` carries three cases that are
the same matrix apart from a single spike placed **exactly on** `sqrt(epsilon)`,
one ulp below and one ulp above: the first two are rejected and produce a
bit-identical reduced system, the third is accepted and produces a different
one, in both precisions.

**Which reduction the column norms take is part of that decision, and it is not
the same at every call site.** `makeHouseholder` needs `tail.squaredNorm()`, and
`Redux.h` picks the traversal from the *C++* matrix's storage order. The
landmark block's `storage` is `Eigen::RowMajor`, so its columns have an inner
stride of `num_cols`, carry no `PacketAccessBit`, and fold left to right
(`Redux.h:236-244`). basalt's marginalization matrices are plain
`Eigen::Matrix<Scalar, Dynamic, Dynamic>` — column-major — so a column segment
is contiguous and `LinearVectorizedTraversal` runs instead
(`Redux.h:274-325`): two packet accumulators, `predux` to pair the lanes
(`(a₀+a₂) + (a₁+a₃)` for a `Packet4f`), then a scalar tail. `alignedStart` is
always zero and that is a property of the expression, not of the address —
`squaredNorm()` reduces a `CwiseUnaryOp` (`Dot.h:24`) whose flags keep only
`RowMajorBit`, so `DenseCoeffsBase.h:533` returns zero whatever the matrix's
base pointer is. The fork's `tools/marg_norm_probe.cpp` reproduces Eigen bit for
bit on 7,486 of 7,486 shapes per precision with that emulation, on 5,869 (`f64`)
and 4,832 (`f32`) with a pointer-derived offset, and on 2,627 and 2,897 with the
sequential fold; a left fold over a packet's lanes in place of `predux`'s
pairing reproduces 5,572 in `f32`, and in `f64` a packet holds two lanes, so
there the two orders coincide. Using the sequential fold in the flat QR and in
`ColPivHouseholderQR` flips `|beta| > sqrt(epsilon)` and `rank()` on valid
inputs — in *opposite directions* in the two precisions on the same `9x2`
problem — which is why the traversal is a parameter of `make_householder` and
named at every call site.

`MargHelper`'s other two routines decide rank a second way, and the port
carries neither. They invert the marginalized block with Eigen's complete
orthogonal decomposition (`marg_helper.cpp:99-100`, which basalt reached after
trying and rejecting `ldlt`, `fullPivLu`, `colPivHouseholderQr` and a Jacobi-SVD
pseudo-inverse — the last one with a "DO NOT USE!!!") and take the square root
of the reduced Hessian through `Eigen::LDLT`. Both were ported, COD and all, and
D68 removed them: they are the squared form basalt keeps behind `vio_sqrt_marg`
as its 2019 baseline, `SqrtKeypointVio::new` refuses that flag off, and no
shipped config reaches either. `marg_oracle.json` still carries their outputs as
the C++ emitted them; nothing reads those entries. Eigen's pivoted LDLT is still
ported at dynamic size, for the reason D41 gave and for the one live consumer
left: `LDLT::solve` is where the LM increment comes from, and the increment
reaches a threshold comparison.

Finally, one shape makes basalt read out of range. When the marginalized block
consumes every row of rank, `total_rank == marg_rank == rows` and `:320-323` asks
for a block whose first row is one past the end; the C++ aborts on Eigen's block
assertion in a debug build and reads past the allocation in a release one, where
there is no defined result — a probe returned `H = [7, 2.42e-322]` with a garbage
residual. The port returns a zero row instead, which is the answer the
arithmetic gives — nothing is left to constrain the kept variables — and stays a
disclosed deviation rather than a reproduction. The oracle carries the case with
the flat QR skipped on the C++ side.


## Layout

| Path | What it is |
|---|---|
| `crates/slam-rs` | The core (`slam_rs` lib). No Python, no Rerun, no GPU. |
| `crates/slam-rs-py` | PyO3 `cdylib` built in place as `slam_rs/_core.so`. |
| `crates/slam-rs-cli` | `slam-rs` binary: a placeholder. `version` is the only subcommand that does anything; a replay runs through the Python tools. |
| `slam_rs/` | The Python package: stubs, Tyro entry points under `apis/`. |
| `tools/` | Thin CLI shims over `slam_rs/apis/`. |

`Cargo.lock` is committed. `cargo` never runs during `pixi lock` or
`pixi install`: the build is an explicit, cached pixi task.

## Build and gates

```bash
pixi run -e slam-rs-dev --frozen slam-rs-build      # cargo build + install _core.so in place
pixi run -e slam-rs-dev --frozen tests              # pytest (depends on the build)
pixi run -e slam-rs-dev --frozen lint               # ruff
pixi run -e slam-rs-dev --frozen typecheck          # pyrefly
pixi run -e slam-rs-dev --frozen deadcode           # vulture
pixi run -e slam-rs-dev --frozen slam-rs-clippy     # cargo clippy -D warnings
pixi run -e slam-rs-dev --frozen slam-rs-rust-test  # cargo test --workspace
pixi run -e slam-rs-dev --frozen slam-rs-version    # print the core version
```

## Python API

The frontend alone is driven the same way, for a run that needs no backend:

```python
flow = _core.OpticalFlow(calibration, config, threads=1)
frame = flow.process(t_ns, [left, right])
frame.ids(0)         # int64[n], ascending
frame.positions(0)   # float32[n, 2] pixels
frame.transforms(0)  # float32[n, 2, 3], [[m00, m01, tx], [m10, m11, ty]]
frame.occupancy(0)   # int32[rows, columns] over camera 0's detection grid
frame.num_new(0), frame.num_tracks(0)
flow.t_ns              # int | None: the last accepted frameset, None before the first
```

Any `int64` is a timestamp, negative ones included: the frontend's clock is an
`Option<i64>` rather than basalt's `t_ns = -1` sentinel (`optical_flow.h:172`),
which read every negative timestamp as "no previous frame" and so restarted
tracking on each one. Framesets must still arrive strictly in order, and a
refused frameset leaves the frontend exactly as the last accepted one did.
