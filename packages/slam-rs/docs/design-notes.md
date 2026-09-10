# slam-rs design notes

The sections the [README](../README.md) cuts to stay short: the port's
module-by-module account, the numbers each stage is pinned against, the two GPU
lanes, the full Python API, the reference set and the gates. Every repo path in
this file is relative to `packages/slam-rs/`.

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
| `estimator` | The Offline sliding-window driver: `process_frame` (cover, initialise, measure, optimise, marginalize), `schedule` (basalt's keyframe vote, the lazy keyframe budget and the keep/marginalize sets `marg` is given) and `optimize` (the Levenberg-Marquardt loop, with the per-frame `lambda` reset, the `lambda · diag(H)` damping and the shared 7-iteration budget). |

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

### The damping machinery the shipped VIO never uses

`optimize()` calls exactly four things on the linearizer: `linearizeProblem`,
`performQR`, `get_dense_H_b` and `backSubstitute`
(`sqrt_keypoint_vio.cpp:1297`, `:1320`, `:1393`, `:1454`). Everything else is
commented out - the Jacobian scaling at `:1307-1317` and `:1461-1463`, the pose
damping at `:1361-1365`, the landmark damping at `:1373-1377` - which matches the
ICCV 2021 paper's own statement that the Givens damping stack is not used in the
sliding-window VIO. Levenberg-Marquardt damping enters through
`H.diagonal() * lambda` in the dense solve instead (`:1415-1417`).

The port carries none of it (D68). The five methods it used to implement -
`set_pose_damping`, `set_landmark_damping`, `scale_jl_cols`, `scale_jp_cols`,
`get_jp_diag2` - together with the six-Givens stack and its LIFO un-apply, were
reachable from no driver and pinned only against the C++ fixture, because a live
run never reaches them. `backSubstitute`'s own `setLandmarkDamping(0)`
(`landmark_block_abs_dynamic.hpp:310`) has nothing to undo here: `storage`
starts zeroed, both QR paths stop at `num_rows - 3`, so the three damping rows
are provably still zero when the model cost change is computed.

One consequence of that same code path changes what `l_diff` means. With the
optimal landmark increment substituted in, the first three rows of `Q^T J inc`
are `-Q1^T r` — so the *updated residual* `Q^T J inc + Q^T r` is zero there,
which is what "the landmarks move to their own optimum" means — and

```text
l_diff = 0.5 * sum ||Q1^T r||^2  -  inc^T b  -  0.5 inc^T H inc
```

The first term does not depend on the pose increment at all, so basalt's
`l_diff` is **nonnegative at `inc = 0`**, and zero exactly when the eliminated
residual `Q1^T r` already is. A port that dropped the constant would make every
Levenberg-Marquardt gain ratio wrong in the same direction, which still
converges, only worse.

None of that is still to come: the sliding-window driver, its keyframe and
marginalization schedule and this Levenberg-Marquardt loop are the `estimator`
module (`schedule.rs`, `optimize.rs`), and every reference number in this README
is what they produce. What is not ported is realtime mode — basalt's two threads
joined by bounded queues. Offline mode runs the frontend and then the estimator
to completion in the calling thread, which is what makes a repeat run over the
same input bit-identical (D17).

## The GPU lane

The CubeCL frontend is the off-by-default `gpu-wgpu` cargo feature. The wgpu
runtime (Vulkan / Metal / DX12) is the only GPU lane. Its tasks run from
`slam-rs`/`slam-rs-dev` on Linux and `slam-rs-osx`/`slam-rs-osx-dev` on macOS.
The default build remains CPU-only.

```bash
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-clippy     # the portable lane compiles and is warning-clean, tests included
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-test       # the same kernels, on this host's GPU
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-build      # a core whose `--gpu` is wgpu
```

`slam-rs-clippy` lints the default features; `slam-rs-wgpu-clippy` checks the
GPU code and its tests with warnings denied.

On macOS the same three tasks run from the mac lane's environment, which is
where that platform's `slam-rs` features are solved, and Metal is the backend
`AutoGraphicsApi` picks there:

```bash
pixi run -e slam-rs-osx-dev --frozen slam-rs-wgpu-test   # the same kernels through Metal
```

## The portable lane, and the two silent failures

`gpu-wgpu` builds the same kernels through `cubecl-wgpu`, which is what the Mac,
the Spark, the Pi 5 and the cap run. cubecl-wgpu picks its shader compiler at
run time from the adapter's backend — Vulkan takes SPIR-V, Metal takes MSL,
anything else takes WGSL — but which compilers are *built* is a cargo feature,
and the two the fleet needs cannot both be on for one platform. So the choice is
per **target**, not per feature of ours:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
cubecl-wgpu = { workspace = true, optional = true, features = ["msl"] }

[target.'cfg(not(target_os = "macos"))'.dependencies]
cubecl-wgpu = { workspace = true, optional = true, features = ["spirv"] }
```

One `--features gpu-wgpu` line therefore works everywhere, and a Mac build
cannot silently forget the Metal compiler. **WGSL is not a lane**: its compiler
has no 16- or 8-bit element type — `U16 is not a valid WgpuElement` — and the
pyramid is `u16` while the candidate image is `u8`, so on that path every kernel
here produces nothing.

`gpu-wgpu` also turns on **`cubecl-wgpu/exclusive-memory-only`**, one buffer per
handle rather than slices of a shared page. That is correctness, not tuning:
cubecl-wgpu 0.10 aligns its pool to the adapter's
`min_uniform_buffer_offset_alignment` and then binds the slices as *storage*
buffers, so an adapter whose `min_storage_buffer_offset_alignment` is the larger
of the two rejects every slice past the first — the RK3588's Mali G610 asks for
64 and got 32. It costs nothing here (fifteen long-lived allocations per frame)
and drops the pool plateau from 40.00 MiB to 11.97, which is the direction the
8 GB targets want.

That failure and the missing-CUDA-install one are the same shape and both are
**silent**: the panic happens on cubecl's own worker thread, the launch reports
success, and every read comes back as a buffer of zeros. Neither is visible in
`client.properties()`, which describes the device rather than the compiler.
So `gpu::probe_storage` runs first on every GPU backend: it writes a known
pattern of all four widths, copies it **on the device**, reads it back, and
refuses the runtime if it does not survive. Microseconds once, and it is what a
fleet machine fails on instead of producing a trajectory out of zeros.

Measured on this host (RTX 5090, MIO07/1500, three interleaved rounds): the
portable lane runs at **7.153 ms** against CUDA's 6.005 and the CPU lane's
9.384 — **1.31x** over the CPU, 19 % behind CUDA — and holds **+271 MiB** of
device memory over idle against CUDA's +667. Every per-kernel tolerance test
passes on it, with the pyramid and the corner scan bit-exact; whole-clip ATE is
2.085 cm on MIO07 and 2.295 cm on MGO07, the same numbers as the other two
lanes.

**Historical D66 result: not gate-clean.** Over the ten reference clips whole, the wgpu lane was
inside the C++'s own precision band on nine and **read 11.98 cm against an
allowed 10.63 on `MIO14_moving_props`** — a D60 failure on one of the ten,
written up rather than smoothed over. At that point, every
tolerance test passed on Vulkan, the pyramid and the corner scan were bit-exact,
and the worst lane-to-lane tracked position over the fixture was 3.1e-5 px.
MIO14 is the 410 s clip the accuracy-band pass identified as chaotic and D60 was built around —
the C++'s own two precisions differ by 2.3 cm on it — so the reading was that
the band might not hold a third backend there. That was an untested explanation,
not grounds to widen the gate.

**Resolution, D71 (2026-09-09).** The finite predicate was faulty, but fixing it
leaves all 22,117 MIO14 poses byte-identical. Native sine error near zero is
amplified by the SE(2) translation factor's division by theta. The small-angle sine
polynomial with native cosine gives **9.72 cm versus GT** (sin+cos: **9.48 cm**),
below the unchanged **10.63 cm limit**, with every frameset tracked. The nine other historical reference results
are not a new ten-clip gate run; the targeted regression evidence is in D71.

## Where the portable lane runs

Measured device by device in the portability run, the two smoke
clips and the eleven per-kernel tolerance tests on each:

| device | driver → compiler | tolerance suite | the lane |
|---|---|---|---|
| RTX 5090 (this host) | Vulkan → SPIR-V | 11/11 | works, 1.1x faster than its CPU lane on the smoke clips |
| RTX 3060 (Ubuntu 22.04 desktop) | Vulkan → SPIR-V | 11/11 | works, 1.08x faster on the four-camera clip, a wash on the two-camera one |
| Apple M4 (Mac mini) | Metal → MSL | 11/11 | works, **2.3x slower** than the M4's CPU lane |
| NVIDIA GB10 (Spark) | Vulkan → SPIR-V | 11/11 | works, 1.08x faster |
| Mali G610 (RoboCap cap, RK3588) | Vulkan 1.3.276 → SPIR-V | 11/11 | works, **2.2x slower** than the cap's CPU lane |
| VideoCore VII (Pi 5) | Mesa v3dv → SPIR-V | **6/11** | refused: the driver is wrong on sub-word storage |
| VideoCore VII (Pi 5) | Mesa lavapipe (software) → SPIR-V | 11/11, patch build bit-exact | works, 2.2x slower — a CPU rasteriser |
| Maxwell (Jetson Nano, L4T 32.7) | Vulkan 1.2 → SPIR-V | **9/11** | refused: the two per-patch cube kernels SIGSEGV in the driver |

Accuracy is the same on every device that runs it, and the same as the CPU
lane's: 0.31 cm against the basalt C++ trajectory, 0.77 and 1.50 cm against
ground truth on the two clips. **On a shared-memory SoC the GPU lane is slower
than the CPU port**, so it is a portability result there, not a speed one.
How much either GPU lane pays is the host's business as much as the card's: on
the Ubuntu desktop, where a mid-range RTX 3060 sits beside a 2017 Zen 1 CPU, the
per-frameset frontend is **2.06x** faster on CUDA and 1.19x on the portable lane
against that CPU port — the widest gap measured, on a card slower than this
host's.

Do not run `vulkaninfo` on the Pi 5: it hangs in uninterruptible sleep and
wedges the box's I/O. The tolerance suite is the probe.

## Python API

```python
from pathlib import Path

from slam_rs import _core

calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)  # the feed's dataclasses
config = _core.VioConfig.from_json(Path("configs/msdmi_config.json").read_text())  # the file the C++ ran
config.optical_flow_image_safe_radius = 472.0    # settable per device, though the shipped file carries it

vio = _core.Vio(calibration, config, threads=1)
vio.push_imu_batch(t_ns, gyro, accel)     # int64[n], float64[n, 3], float64[n, 3], uncalibrated
result = vio.track(t_ns, [left, right])   # uint8[h, w] per camera
result.status, result.world_from_rig      # VioStatus, [tx ty tz qx qy qz qw]
```

One `track` call is basalt's whole pipeline for one frameset — the frontend's
own preintegration and pose prediction, `processFrame`, then the estimator's
`measure` — in the calling thread (Offline mode, D17), so every result is final
and a repeat run over the same input is bit-identical.

`VioStatus` has two states. basalt's estimator initialises inside the same
`process_frame` that measures, so a measured frameset always has a state and an
uncovered one never does: `NeedMoreImu` where basalt would block on its IMU
queue, `Tracking` otherwise. `NeedMoreImu` moves nothing — not the frontend,
neither IMU buffer, not the estimator — so the frameset is pushed again once its
samples arrive and tracks as it would have with them all along (D17). A caller
that drops it instead loses the frame; `replay.py` holds it and retries.

Two accessors carry what a Rerun rung draws, both copies rather than views:

```python
snapshot = vio.snapshot()       # None until a frameset has measured
snapshot.window_t_ns            # int64[n], the 15-dof states then the pose blocks
snapshot.window_poses           # float64[n, 7], [tx ty tz qx qy qz qw]
snapshot.window_keyframe        # bool[n]: a keyframe, and window_long_term for a long-term one
snapshot.kf_ids, snapshot.marginalized                   # the keyframes as ids, plus what left
snapshot.landmark_ids, snapshot.landmark_positions       # int64[p], float64[p, 3] world
snapshot.landmark_hosts         # int64[p], the hosting keyframe's timestamp
snapshot.lm_iterations, snapshot.lm_lambda, snapshot.num_observations
snapshot.lm_error_before, snapshot.lm_error_after
snapshot.timings_ms             # the six estimator stages, milliseconds

frame = vio.flow_frame()        # the keypoints of the last accepted frameset, or None
```

`snapshot()` describes the last frameset that **measured**, which
`snapshot.t_ns` names; on a `NeedMoreImu` frameset it is the previous one.

Images are copied in and the GIL is released around the core call, so a decoder
thread keeps running. Wrong dtype, rank, shape or memory layout raises
`ValueError`; IMU samples must be strictly increasing in time.

**Every refusal is an exception, not a panic.** A Rust panic crosses PyO3 as
`pyo3_runtime.PanicException`, which derives from `BaseException` and so walks
straight through an `except Exception` handler, and a panic on a rayon worker
inside the released-GIL region aborts the process outright (decision D32). So
every value that sizes a buffer, bounds a loop or spawns a thread — the
calibration included, since it shapes the occupancy grid — is checked in the
core before it is used. The refusal is a `ValueError`, or the `TypeError`,
`OverflowError` or `IndexError` PyO3 itself raises for an object of the wrong
type, an integer outside the parameter's own type, or a camera past the end of
the rig:

| what | ceiling or rule | why the core cannot just try it |
|---|---|---|
| `max_keypoints` | `tracker::MAX_CAPACITY` = 1,048,576 | every per-patch buffer is preallocated from it; `Vec::with_capacity(2**63)` panics with `capacity overflow` |
| `threads` | `parallel::MAX_THREADS` = 1024 | rayon spawns exactly what it is asked for, so 100,000 workers wedge the machine rather than erroring |
| `optical_flow_levels` | at most 23 reductions, i.e. `tracker::MAX_LEVELS` = 24 stored levels | it multiplies every buffer, each sized with `levels + 1`; a `Vec` whose bytes do not exist **aborts** instead of unwinding |
| `optical_flow_detection_min_threshold` | at least 1 | the detector halves the FAST threshold until it drops below this, and zero halves to zero for ever — `keypoints.cpp:162,187` has the same non-terminating loop, so basalt hangs on it too |
| `optical_flow_detection_max_threshold` | at least `min_threshold` | otherwise the ladder never runs and the detector can never add a keypoint |
| frameset image size | exactly the calibration's, per camera | the camera model, the detection grid and the occupancy matrix are all the calibrated geometry |
| the calibrated resolution over `optical_flow_detection_grid_size` | `detect::MAX_CELLS` = 1,048,576 cells per camera | the occupancy counts are one `i32` per cell per camera, so a calibration is a memory request too: a one-pixel grid over a 4,294,967,294-pixel-square frame asked for 2^64 counts and `vec![0; rows * columns]` panicked with `capacity overflow` with no image in sight |

`tests/test_frontend_boundary.py` walks the whole surface against hostile
integers, objects and arrays and fails on anything that is not one of the four
exceptions above. That is a walk over the surface, not a proof about every
object a caller could construct.

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

`Calibration.from_catalog` reads `slam_rs.catalog_feed.CameraCalib` and
`ImuCalib` attribute by attribute and hands them to `Calibration::from_catalog_parts`,
so the catalog-to-basalt rules — the rotation-matrix check, the model names, the
isotropic noise densities — are not written a second time in Python. One of
basalt's own files is read by `Calibration.from_json` or `VioConfig.from_json`,
which is what the frontend's constructor then takes.

## The reference set

`reference_segments.toml` freezes ten Monado SLAM Dataset segments — five
two-camera `msd-index` (KB4 fisheye, 54 Hz) and five four-camera `msd-g2`
(radtan8, 30 Hz) — in three tiers: **smoke** on every commit, **accuracy** per
pull request, **long** nightly. A `[[dataset]]` block per catalog dataset pins the
rig geometry and names the basalt VIO config its segments run with; a `[robocap]`
section adds the two RoboCap sessions, 15 (1,588 framesets) and 21 (4,648), which
have no ground truth and are gated against basalt's own output instead. Only
session 15 carries a reference wall, measured on the cap itself.

Four things are frozen because the catalog cannot carry them and each one moves
the numbers: the IMU noise densities and update rate, the camera-to-IMU time
offset (0 for MSD, 14,902,432 ns for RoboCap), the decode path
(`cpu_gray8_dav1d_1thread`, worth about 5 cm of ATE against NVDEC RGB), and the
VIO config, vendored under `configs/` — basalt's constructor defaults are not its
shipped files, and `vio_marg_lost_landmarks` alone was worth up to 12 cm (C72),
so `slam_rs.reference.flow_config` reads the dataset's file and asserts the
manifest's image safe radius against it. `slam_rs.reference`'s module docstring
carries the rest of the account, including where the V2 tolerances live and why.

```python
from slam_rs.reference import load_manifest

manifest = load_manifest()
segment = manifest.in_tier("smoke")[0]
```

### The basalt C++ reference and the gate policy

`tests/reference/msd/<segment>/` holds what the basalt C++ fork produced on each
segment: `run.json` for all ten (fork commit, deterministic settings, the VIO
config and calibration actually pushed, timings and the ATE against `gt.csv`),
`basalt_traj.csv` for the eight smoke and accuracy segments, and `frames.sha256`
plus a copy of `gt.csv` for the smoke pair so its gate runs with no NAS and no
catalog. The two long-tier trajectories are 3.4 MB and 4.8 MB and stay out of
git; `slam_rs.reference_bundle` resolves them from `SLAM_RS_REFERENCE_DIR` (or
`data/reference/`) and the tests skip with a message naming the variable.

Each segment carries a `gate_policy`, because basalt is not equally good
everywhere:

| policy | segments | why |
|---|---|---|
| `tight` | MIO10, MGO09, MIO07, MGO07 | basalt scores 0.8-2.4 cm; a regression is unambiguous |
| `standard` | MIO04, MGO14, MIO14, MIPT03 | 8-38 cm, stable; gate relative to basalt's own number, not an absolute threshold |
| `no_divergence` | MGO01, MGO13 | basalt is near failure: 43 cm and 78 cm here, 68 cm for the C++ binary on the raw files, and **18-32 cm of spread between two legitimate decode paths of the same estimator** — on MGO01 the ordering even flips. Only "kept tracking, did not diverge" is measurable. |

`slam_rs.trajectory.ate` reproduces all ten published C++ figures exactly, and a
re-decode through `catalog_feed` reproduces the C++ run's per-camera pixel
digests frame for frame (824 of 824 on MIO10, 428 of 428 on MGO09). That second
result is the load-bearing one: it means an A/B between the two estimators
measures the estimator, not the decoder.

### Two clocks, converted once

The catalog indexes a segment on `video_time`, which is **relative** to
`capture.start_time_ns`, while every basalt CSV including the `gt.csv` sidecars is
on the **absolute** device clock; on the Index smoke segment the two differ by
10,433,867,587,166 ns, so a trajectory exported on the wrong clock associates with
nothing at all. The feed works in `video_time` throughout and
`trajectory.shift_clock` converts once, at the CSV boundary.

## The feed, the metrics and the replay tool

`slam_rs.catalog_feed` turns one segment into calibration and grayscale
framesets, reading a catalog URL or local `.rrd` files served in process (no
catalog server needed); its module docstring states the decisions that silently
change the numbers — the pinned `gray8` dav1d path, the one-query windows whose
edges land on frames that are keyframes in every camera, the rig shape and the
two clocks. `slam_rs.trajectory` reads and writes basalt's CSV form (w-first,
integer nanoseconds) and reports the rigid-aligned ATE the gate is written
against, in `golden_compare.py`'s own arithmetic rather than the shared
`simplecv` helper, whose variance floor would reject a stationary rig that the
fork passes.

```bash
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py \
    --rr-config.headless --rr-config.save data/replay-smoke.rrd
```

Both `--rr-config.headless` and `--rr-config.save` are honoured; in a shell
without `DISPLAY`, pass `--rr-config.headless` or the spawned viewer wedges the
recording stream.

### `--stage frontend`, and the C++ overlay

`--stage input` (the default) logs what the estimator is fed. `--stage frontend`
runs the optical flow over the same framesets and logs what it produced, under
the dataset's own entity tree so nothing needs a second coordinate convention:

| entity | what |
|---|---|
| `/world/rig_00/cam_MM/pinhole/image` | the frame the frontend tracked, **full resolution** (JPEG), because the keypoints are in its pixels |
| `.../keypoints` | `Points2D`, 2 px, one stable colour per track id from a hash of the id |
| `.../trails` | `LineStrips2D`, the last ten positions of every live track, in the track's own colour |
| `.../cells` | `Boxes2D` over the occupied cells of basalt's centred detection grid |
| `.../keypoints_cpp` | what the C++ fork's `dump_flow.cpp` produced for the same frameset, in one contrasting magenta |
| `/stats/frontend/...` | `num_tracks` and `num_new` per camera, and `frontend_ms` |

The overlay is the parity claim made visible, and it is only ever drawn on the
recording it came from: the eight committed dumps under
`crates/slam-rs/tests/fixtures/flow/dumps/` name their segment in
`dumps/source.json`, and `slam_rs.frontend_log`'s module docstring says why a
`video_time` timestamp is not an association and what a directory from another
recording, another rig or no `source.json` gets instead. On the smoke segment the
port hands out 175 keypoint ids over the first eight framesets where the C++
hands out 174, and every magenta ring in the viewer carries a coloured port dot
at its centre bar a handful — the detector gap the flow gate measures.

A blueprint is sent with the recording: one 2D view per camera plus the counters,
panels collapsed. The whole 412-frameset smoke segment is 34.5 MiB of `.rrd` and
takes 17.5 s, of which 29.5 ms per frameset is the frontend itself.

```bash
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py \
    --stage frontend --rr-config.headless --rr-config.save data/replay-frontend.rrd
```

### `--stage vio`, and the three trajectories

`--stage vio` runs the whole pipeline and draws what the estimator decided, under
the same tree:

| entity | what |
|---|---|
| `/world/runs/slam_rs/trajectory` | the estimate so far, one green `LineStrips3D` |
| `/world/runs/gt/trajectory` | the ground truth up to the cursor, near-white |
| `/world/runs/basalt_cpp/trajectory` | the C++ reference up to the cursor, orange |
| `/world/runs/slam_rs/rig` (+ `/cam_MM`) | the estimated rig's current pose, as `Pinhole` frusta from the calibration |
| `/world/runs/slam_rs/window` | one frustum wireframe per window frame, blue for a keyframe, yellow for a long-term one, grey for a pose block |
| `/world/runs/slam_rs/marginalized` | the frames the last marginalization removed, the same wireframes faded |
| `/world/runs/slam_rs/landmarks` | `Points3D` in the world frame, coloured by the keyframe that hosts them |
| `/world/rig_00/cam_MM/pinhole/keypoints` | the estimator's own frontend output, in the frontend rung's palette |
| `/stats/vio/...` | landmark, observation and keyframe counts, LM iterations, lambda and the error before and after, the six `stage_ms/*`, `track_ms`, and `ate_cm/{gt,cpp}` |

The three trajectories do not start in one frame — basalt initialises its world
at the identity with gravity along z, the ground truth is in the capture rig's
own frame — so the run and the C++ reference carry the rigid alignment onto the
ground truth as a `Transform3D`, refreshed every 30 framesets, and a run visibly
settles into place over its first second. All three are drawn only up to the
cursor and thinned to the frameset cadence: 1.04 MB each over the 412-frameset
smoke segment, against the 17.20 MB of that recording's 54.13 MB that re-logging
the 917 Hz ground truth whole cost, and about 100 MB each over a 4,000-frameset
clip, so a long segment still wants `--max-framesets`. `slam_rs.vio_log`'s module
docstring carries the reasoning, the visible time range that makes a per-frameset
segment render as a path, and why the window is wireframes and the rig is not.

```bash
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py \
    --stage vio --rr-config.headless --rr-config.save data/replay-vio.rrd
```

## The V2 gate

`tests/test_v2_gate.py` is the milestone (D14, D35, D36, D58). Per gated clip,
driving `_core.Vio` and the feed directly with nothing logged: every frameset
resolved, a refusal for want of IMU held and tracked again once the samples
arrive (D17); at most 2 cm of ATE RMSE against the basalt C++ trajectory fed the
same decoded pixels, and only where the C++ meets that against itself, which is
clips under `PATH_BOUND_MAX_CLIP_S` = 100 seconds of replayed footage — on the
410-second `MIO14` its own two precisions are 4.24 cm apart; against the `gt.csv`
sidecar, inside **the C++'s own precision band** (`rmse_cm` and `rmse_cm_f64`,
the same code on the same pixels with `use-double` flipped) or within
`GT_BAND_RATIO` = 1.2 of the band's worst member, whichever is looser, which is
the second alone since the ratio is above one — the band is 0.00007 cm wide on
`MIO10` and 2.3 cm wide on `MIO14`, so "inside the band" on its own would gate
the tight clips on rounding; and speed, the replay's own feed loop (decode plus
`track`, nothing logged, the loop the C++ recorded as `run.feed_wall_time_s`)
within 1.2x the C++ single-thread wall for the same footage (D58), never left
off. The clauses, the association convention, the `no_divergence` pair's
exception and the "every named clip is asserted" rule (C56) are stated once in
the test's own module docstring.

The tolerances live in `slam_rs/reference.py` (`ATE_VS_CPP_CM`,
`PATH_BOUND_MAX_CLIP_S`, `GT_BAND_RATIO`, `SPEED_TOLERANCE`,
`DIVERGENCE_FACTOR`), not in the test: they are the milestone's verdict, and the accuracy-band pass
measured what a meaningful band is (D60). Every row prints what it was judged on:
`tracked, vs C++ <cm> (bound 2 cm | no bound, <n> s clip), vs GT <cm> (band [f32,
f64], allowed <cm>), wall, C++ wall, ratio`.

The lanes are D59's iteration rule. The default is the **iteration set** — MIO10
whole plus the first ten seconds of one two-camera and one four-camera clip,
about 1,650 framesets — because finding out at the end of a ten-clip run that
everything failed is the way not to iterate. Either lane runs **shortest clip
first** and asserts each clip as soon as it is measured, so the first clip that
misses stops the run with its own row printed and is the one that gets fixed.

```bash
cd packages/slam-rs
pytest -m slow -q -s tests/test_v2_gate.py                    # the iteration set
SLAM_RS_V2_ALL=1 pytest -m slow -q -s tests/test_v2_gate.py   # all ten, whole, shortest first
SLAM_RS_V2_WINDOW_S=5 SLAM_RS_V2_ALL=1 pytest -m slow -q -s tests/test_v2_gate.py   # all ten, first 5 s each
```

`SLAM_RS_V2_WINDOW_S` cuts every clip to its first N seconds and recomputes the
C++'s own ground-truth error over exactly that span, so a windowed run is gated
against the budget it actually had.

## Tests

`pytest -q` deselects the `slow` marker and runs in about two seconds on
synthetic inputs. The slow tests read a reference `.rrd` from the NAS or query the
catalog, and skip when neither is reachable:

```bash
pixi run -e slam-rs-dev --frozen tests   # fast
cd packages/slam-rs && pytest -m slow -q # NAS + catalog
```

## D70 — one GPU runtime: the CUDA lane is removed; wgpu is the GPU lane

Decision, 2026-09-09: use wgpu as the only GPU runtime. Remove the CUDA cargo
feature, runtime, dependencies, Pixi environments and tasks. Keep `gpu-core`,
`gpu-wgpu` and the shared kernels unchanged. The default build is CPU-only.

Earlier CUDA measurements and failure accounts below and above are historical;
the CUDA lane was removed on 2026-09-09. D64's tolerance requirement still
applies. At removal, D66's MIO14 moving-props exception was unchanged: 11.98 cm
against 10.63 cm allowed. D71 records its later correction.

## D71 — wgpu finite check and small-angle trigonometry

Decision, 2026-09-09: classify GPU floats by exponent bits, and use a degree-9 sine
Taylor polynomial for `|theta| <= 0.5` in the SE(2) update. Cosine stays native;
sine is native outside that interval. Keep the CPU lane unchanged.

The device probe proved `value * 0 == 0` accepts uploaded and device-generated
NaN and both infinities. CubeCL 0.10's constant-operand optimizer replaces a
multiply by constant zero with zero. The replacement tests whether the exponent
bits are all ones; both patch validity and increment validity use the same helper
as the permanent device regression. This is a real defect, but correcting it
leaves the full MIO14 trajectory byte-identical: **11.982561 cm versus GT** and
**4.551881 cm versus C++**.

The next probe measured native sin error up to 1.86e-7 (257 ULP) over ±0.001.
An absolute bound is insufficient here: above the 1e-5 small-angle cutoff, the
SE(2) translation factor divides normalized sin by theta. The sine polynomial stays
within one ULP on the regression grid; native cosine stays within two ULP.
The sine truncation error on ±0.5 is below f32 rounding. The
normalization and SE(2) translation formulas stay unchanged.

The original sin+cos change brings MIO14 to **9.482019 cm versus GT**, below **10.633937 cm**,
and **6.665221 cm versus C++**. All **22,117/22,117** framesets track, with no
retries, in **437.9 s (50.5 fps)**. The reproduced CPU run reads 8.734170 cm
versus GT in 499.4 s. The long-clip gate uses the GT band; the short-clip C++ path
bound does not apply to MIO14. S29-G's sine-only ablation gives **9.716469 cm versus
GT** and **3.076043 cm versus C++**, with all 22,117 framesets tracked; the cosine
polynomial is unnecessary for this measured gate and was removed. No gate threshold was changed.

Localization found finite differences before any track-set change: the original
lane first exceeds a 0.001 px position gap at frame 50, camera 0, keypoint 478;
IDs first differ at frame 149. No nonfinite or outside-image output appears in
the 1,100-frame reduced run. Landmark and observation counts initially match,
while LM cost differs from frame 4. Later newly allocated IDs need not identify
the same physical point across lanes. The scalar correction closes the measured
miss; it does not make every frontend value closer to CPU.

All **20 GPU kernel tests** and GPU all-target Clippy pass. The MIO10 CPU bench
CSV is byte-identical. The four GPU bench trajectories change from pose 4, with
maximum position shifts of 0.29–2.35 mm and GT ATE changes below 0.0011 cm:

| Clip | Before GT ATE (cm) | After GT ATE (cm) |
|---|---:|---:|
| MIO10 | 1.503795 | 1.503822 |
| MIO11 | 2.495883 | 2.496897 |
| MGO10 | 0.881272 | 0.881638 |
| MGO11 | 2.244357 | 2.244077 |

MIO10 also passes its 1.713388 cm GT limit. The other three bench clips have no
precision-band entry in the ten-clip manifest. These checks establish the local
MIO14 correction and short-clip regression behavior, not a fresh all-device or
ten-clip gate. The GPU stays opt-in and the default build stays CPU-only.

## D73 — the estimator's per-frame scratch is the estimator's, and its hot loops walk columns

Decision, 2026-09-10: hold the Levenberg-Marquardt loop's buffers on
`SqrtKeypointVio` instead of allocating them per inner step, and write the three
hot inner loops of the backend so that the coefficient they walk is the
contiguous one. No arithmetic changes: every sum keeps its order, every product
keeps its operand order, and the MIO10 trajectory stays byte-identical to the
CPU lane's.

**What the estimator was spending.** On MIO10 the `measure` stage was 2.443 ms of
a 5.117 ms call (optimize 2.236, of which linearize 0.741 and solver 1.182,
marginalize 0.147) on a window of 7 keyframes and 3 states — 87 pose parameters,
~55 landmark blocks, seven inner LM steps on the median frame. That is a few
MFLOP. Three things ate it, in this order:

1. **Row-major loops over column-major storage.** `nalgebra`'s `DMatrix` is
   column-major, and three loops indexed it with the column innermost:
   `apply_householder_on_the_left_block`'s rank-1 update (`eigen/qr.rs`), the
   `O(n³)` trailing update of Eigen's LDLT sweep (`eigen/ldlt.rs:338`), and both
   of them at a stride of `nrows`. Interchanging them is free where the
   operation is elementwise (the Householder update) and needs one accumulator
   per output row where it is a reduction (the LDLT), which keeps each
   coefficient's additions in their original order because the reduced index
   becomes the outer loop. Worth 0.152 ms and 0.164 ms on MIO10, the two largest
   single gains of the pass.
2. **A hot loop the vectoriser refused.** `LandmarkBlock::add_dense_h_b_over`
   was 31% of `optimize`'s self time and compiled to scalar `mulss`/`addss`: its
   accumulator was a slice of runtime length, reached through the same
   `&mut DenseHbScratch` as the row it multiplies, so the compiler had neither a
   trip count nor a disjointness proof. A fixed `[S; 8]` local accumulator over
   a row buffer padded to whole lanes gives it both. Worth 0.115 ms.
3. **Per-step allocation.** `get_dense_h_b` took a fresh `opt_size`-square
   accumulator, a fresh subtree partial per recursion depth and a fresh leaf
   transpose on every call; `damped_solve` cloned the reduced system per damping
   attempt and `EigenLdlt::new` allocated two more workspaces with it. Measured
   with a counting allocator: **126.8 allocator calls per LM step**, 1,531 per
   frameset. They are now a `DenseHbWorkspace`, an `EigenLdlt` and an increment
   the estimator owns and resets, at 59.5 and 1,133.

**Why pooling is not arithmetic.** The reduction resets every subtree buffer
before a leaf writes it, and that reset restores exactly `+0.0` over the
columns that were written; the accumulator it hands back is zeroed whole rather
than by recorded column, because the IMU blocks, the prior and the
fixed-keyframe pinning all write into it without recording anything. Eigen's
LDLT is an in-place factorization, so the damped copy is the buffer the sweep
consumes — writing it in place is what Eigen does, not a shortcut.

**What did not pay, measured.** Pooling the landmark blocks across frames —
~275 allocations and 182 kB of zeroing a frame — **regressed** MIO10 by
0.074 ms and was dropped; a window whose landmark set shifts hands each pooled
block to a different landmark, whose row count often differs, so the storage is
replaced anyway and the shape test and the `Vec` rebuilds are what is left. The
same reasoning retires the marginalization's permutation copy: `marginalize` is
0.103 ms a frame after the Householder fix, no marginalization symbol appears in
the top twenty of the native profile, and its copy is ~7 µs.

**What is left, and where it is.** After the pass, `optimize` is 1.92 ms and its
self time is `add_dense_h_b_over` 23%, `apply_householder_on_the_left_block`
19%, the damped solve 17%, `linearize_problem` 12% and the dense reduction's
joins and resets 8.5%. The Householder is the next lever and it needs the change
this pass would not make: `LandmarkBlock::storage` is column-major here where
basalt's is `Eigen::RowMajor`, so the reflection's long dimension — 92 columns —
is the strided one and vectorising over its 3-to-5-row short dimension is most
of what it can do. Making the block row-major would give both the dot product
and the outer product a 92-long contiguous inner loop, and it would match the
layout the port already models in `ColumnRedux::Strided`. It touches every
reader of `storage` and is not a local change.

**Gate.** MIO10, three interleaved rounds against `44cbdb0f`: median
5.140 → 4.733 ms, `measure` 2.451 → 2.108, ATE vs ground truth 1.504 cm
unchanged, zero lost framesets, and every candidate trajectory and state digest
byte-identical to the baseline's. `tests/frame_allocations.rs` gates the
allocation half: zero allocator calls on a warm dense reduction, and a
slope-and-total bound per frameset that the pre-pooling code fails.
## D72 — the detector picks each grid cell's corner on the device

Decision, 2026-09-09: on the GPU lane, pick one corner per detection grid cell in
a CubeCL kernel and download one packed key per cell, instead of downloading the
candidate image and its bitmask and walking them on the host. Keep the band walk
as the reference and as the fallback. The CPU lane is unchanged.

The band path downloaded 0.92 MB of candidate image plus 0.115 MB of bitmask per
camera — **2.07 MB per two-camera frameset** on MIO10 — and then did all of the
selection on the host: a row band per cell row and rung, a column filter, OpenCV's
non-maximum suppression per cell, a sort and three gates. The three FAST kernels
were 0.04 ms of that; the stage was **1.45 ms**.

Only the **last rung the ladder visits** decides the winner, and that is exact
rather than an approximation. That rung is `max_threshold` halved until the next
halving would fall under `max(min_threshold, 1)`, which for the shipped 40/5
configs is 5 but for 40/6 is 10 — not the configured minimum, which a halving
ladder need never reach. `threshold_rungs` is the one place it is computed, and
the cell walk steps through the same iterator, so the two cannot drift; handing
the device `min_threshold` instead would let it admit a corner scoring between
the two that the walk never sees. A candidate at rung `t` is `kept > t`;
`suppress_non_maxima` kills a pixel only through an in-window neighbour scoring at
least as much, and such a neighbour is itself a candidate at every rung the pixel
is. Suppression therefore does not depend on the rung, and the ladder only admits
survivors in descending score. With `optical_flow_detection_num_points_cell = 1`,
which every shipped config sets, the cell's outcome is the best survivor over
that last rung which clears `safe_radius`, the masks and `EDGE_THRESHOLD`.

`fast_cell_select_kernel` is one cube per cell over that cell's own window, with
the same zero rim the host scratch grid gives a neighbour outside the window, and
a shared tree reduction over the packed key
`((255 - score) << 24) | (y << 12) | x`. Integer minimum is associative,
commutative and exact, so the tree is the host's own total order — score
descending, then row, then column, which is what the row-major band walk and a
stable sort produce — and a subgroup fast path would agree with it bit for bit.
Readback is 361 x 4 B per camera against 1,036,800.

The masks stay on the host and are exact there: `cam0OverlapCellsMasksForCam`
pushes `cell` x `cell` rectangles at the cell origins and a cell's candidates lie
strictly inside it, so a cell is wholly masked or wholly clear. The guard is at
the camera level — a rectangle that does not name one cell of *this* camera's
grid, which the mixed-geometry rigs the port supports deliberately produce, sends
the whole camera down the band path. So do `num_points_cell != 1`, a cell no wider
than the FAST ring and a frame 4,096 pixels or more on a side.

MIO10, three interleaved rounds against main + timers (44cbdb0f) on one core:

| Core | frontend_detect ms | track median ms | ATE vs GT cm | tracked/lost | identical |
|---|---:|---:|---:|---:|---|
| base | 1.466 | 5.151 | 1.504 | 412/0 | — |
| this | 0.589 | 4.372 | 1.504 | 412/0 | byte and state |

MGO09, four 640x480 cameras, one round: `frontend_detect` 2.233 ms against
2.241, unchanged; `track_ms` 16.904 against 13.165; ATE vs GT 0.770 against a
0.847 band; 107/0 tracked; byte and state identical. **The stage does not move
on that rig**, and that says where the rest of the time is: the kernels are
microseconds and the readback is now 432 B per camera, so what 2.24 ms buys is
four device round trips. One trip per camera is the floor this shape has.

A hoisted variant that launched every camera and read them all in **one**
download was written, measured and dropped. It is right in isolation — 0.502 ms
against 0.822 for two 960x960 cameras through the scanner alone, and 0.74-0.84
against 1.01-1.04 per frameset through the whole frontend in one process — and
it is wrong under the gate, twice: `frontend_detect` 1.418 and 1.422 ms against
this shape's 0.589, with `track_ms` −8.4% and −8.8% against −15.1%. A counter on
the scanner rules out the obvious explanation: through the real frontend the
batch answered every selection and none fell back. What moves with it is the
estimator — `measure` 2.070 against the base's 2.431 in the same interleaved
run, which no detector change can cause — so the harness is attributing
something the stage timers cannot separate. The finding is recorded rather than
shipped: on a four-camera rig the trip count is still the lever, and a variant
that keeps the single download but leaves the wait where this shape leaves it is
the one to try next.

`tests/gpu_detect.rs` is the exactness gate rather than the A/B run: the whole of
`detectKeypointsWithCells` runs twice over the committed 960x960 MIO10 frames —
through the band walk and through the device selection — and the two
`KeypointsData` are equal corner for corner and response for response, on both
cameras, empty and half-occupied, at three safe radii, under cell-aligned masks
and one that straddles a boundary, at three budgets, over four camera slots
through one reused scanner, and on three frames whose width is not a whole
number of cells. Two ladders that step past their own minimum — 40/6 and 32/5 —
are in it because equality there is what the last-rung fix buys: the same frame
detected at a rung of 6 yields 62 corners against the real ladder's 54, so a
device handed `min_threshold` is separable from one handed the last rung, and
the test asserts that gap before asserting the equality.

The GPU scanner is wrapped in a counting one, because equality alone cannot tell
the two paths apart — a device path that quietly never engaged agrees with the
host walk perfectly — so every case says which path it meant. That is what makes
the three fallbacks assertions rather than assumptions: a straddling mask, a
`num_points_cell` of 2 and a frame `CELL_KEY_LIMIT` pixels wide each have to
report zero selections and a nonzero band count.

The clamps are their own case. `CellGrid::new` floors and centres, so no grid it
derives has a cell that runs past the image and neither strict clamp in the
kernel would ever run; the detector takes its grid from the caller, so the test
supplies two — a last column and row that overhang, and a last column whose
candidate window is empty and must come back as the sentinel. The clamped columns
lie past `width - EDGE_THRESHOLD - 1`, so no corner can come out of them on
either lane: what the equality proves there is that the device stays inside the
image and sees the same zero rim, not that the answer changes.

It is a separate test binary because `the_whole_gpu_path_holds_the_pool_flat`
asserts an exactly flat CubeCL pool and every test in one binary shares one
client.

Two CubeCL traps cost a debugging pass each. A **named** integer constant stays
comptime inside `#[cube]`: seeding a `let mut` from one makes a const variable,
and `stride /= 2` on it panics the expansion on cubecl's own worker thread — the
launch reports success and the buffer comes back as zeros, which is the failure
mode D32 exists for. The sentinel key is spelled as a literal with a
`const _: () = assert!(..)` pinning it to the host's constant, and the reduction
stride is comptime per unrolled step.

## Decision references

The `Dnn` tags in this file and in the README name the project's recorded design decisions. What each one decided, in one line:

- **D09** — FAST detection reuses kornia's grid-cell detector behind a thin wrapper
- **D14** — Accuracy gate: trajectory-level, against basalt C++ on the same decode path, plus ground truth
- **D17** — Threading: Offline (lockstep) mode first; Realtime mode is an enum value reserved for later
- **D31** — Deterministic reductions and the thread budget have an explicit Rust mapping
- **D32** — Panic policy: the core never panics on data; NaN handling mirrors basalt
- **D34** — The shipped VIO path has the landmark and pose damping machinery disabled; the port mirrors that
- **D35** — Numeric gate ladder adopted from the paper dossier
- **D36** — Gate policy per reference segment: tight, standard, no-divergence
- **D41** — Eigen's pivoted LDLT semantics are load-bearing and are ported exactly
- **D44** — Rotation matrices in numerically sensitive paths use Eigen's `toRotationMatrix` operation order
- **D58** — Runtime parity is part of the stopping line; two parallel stages open
- **D59** — The iteration loop: one or two short clips, fail-fast on the ten, speed is a gate clause
- **D60** — The V2 accuracy gate, on the evidence: ground truth inside the C++'s own precision band, the path bound only where the C++ meets it itself, speed on every clip
- **D64** — Reaffirmed for the GPU lanes: no bit-accuracy; the bar is accuracy inside the band and faster than the CPU lane on the same machine
- **D68** — The three unreachable blocks go: squared-form marginalization, nullspace diagnostics, the D34 damping stack
- **D70** — One GPU runtime: the CUDA lane is removed; wgpu is the GPU lane (2026-09-09)
- **D71** — Exponent-bit finite classification and bounded small-angle trig; the MIO14 replay passes its unchanged accuracy limit (2026-09-09)
- **D72** — The GPU detector picks one corner per grid cell on the device; the candidate image never comes back (2026-09-09)

## D74 — Speed profile

Vendored configs stay C++-faithful (D17), and the Rust default LM cap stays 7.
Speed knobs live in `configs/profiles/fast.json`; the first sets
`config.vio_max_iterations` to 4 (at most five LM steps with the inclusive loop).
The benchmark and tracking tools opt in with `--profile fast`. The default
`reference` profile is empty and preserves the vendored text. Unknown overlay
keys raise `KeyError` so a typo cannot silently change the requested run.
- **D73** — The estimator's LM buffers live on the estimator and its hot loops walk columns; no arithmetic changes (2026-09-10)

## D75 — Redetect on demand: the fast profile detects when camera 0 has lost tracks

basalt calls `addPoints` on every frameset and tops up every empty grid cell
(`frame_to_frame_optical_flow.h:637-666`, `frontend/flow.rs`). cuVSLAM instead
detects only once its survivors fall under a fraction of what the last detection
left it, and so pays detection about every fourth frame. On MIO10 that is the
one stage where the two are furthest apart.

**The knob.** `port.redetect_survivor_ratio`, `0.0` by default. At `0` — where
every basalt file and `VioConfig::default` leave it — the frameset always
detects, which is basalt's schedule byte for byte. Above zero the frameset
detects only when camera 0 holds fewer than that fraction of the keypoints the
last **detecting** frameset ended with. `configs/profiles/fast.json` sets `0.85`;
nothing else does.

**Why the key is `port.` and not `config.`.** basalt has no field for it, and the
vendored `configs/*.json` are the documents the C++ reference runs read, key for
key (`tests/test_cpp_reference.py::test_the_vendored_configs_are_the_ones_the_cpp_runs_used`).
So no port-only key is written into them: the profile overlay inserts it,
`slam_rs.reference.PORT_CONFIG_KEYS` allowlists it so a typo is still a
`KeyError`, and `VioConfig` skips serializing it while it is off — a basalt
document still round-trips to exactly the keys it arrived with.

**One decision for the whole rig, taken on camera 0.** `add_points` is a unit:
camera 0's detection, the cross-camera match that carries its new ids into
cameras 1..n, and the non-overlap pass on those cameras. Gating it per camera
would leave a rig half detected, with camera 0's new ids never matched onward.
Camera 0 is also the only camera the keyframe vote reads (D21). The test is a
pure function of the frameset's own state — this frame's camera-0 count, the
count the last detecting frameset ended with, a config field — so a replay
repeats it, and `FrameState` carries the baseline so a refused frameset does not
move it.

**The keyframe coupling, measured.** The vote is
`connected[0] / (connected[0] + unconnected[0]) < 0.7` and the unconnected
observations are all observed ids absent from the landmark database, including
carried tracks that failed triangulation or whose landmarks were removed.
Those tracks can still vote on skipped-detection frames, so gating detection
alone does not imply keyframe starvation; at `0.7`, MIO10's post-warmup cadence goes 7.04 ->
7.33 frames per keyframe and MGO09's stays at 6.71, both inside the 5-9 band the
later scheduling work is priced against.

**Why 0.85, not 0.7 (MIO07, 2026-09-10).** `0.7` passed MIO10 (ATE 1.447 cm) and MGO09 but
failed the 76 s MIO07: 2.624 cm against a 2.29 cm band, while the LM cap alone read 2.093 and
`0.7` without the cap read 2.682 — the gate, not the cap, accumulates drift over a long clip.
`0.85` reads 2.201 cm on MIO07 (3.858 ms median against the cap-only 4.073) and is what the
fast profile ships; `0.7`'s MIO10 numbers above stand as measured. Short clips do not see this
class of regression; MIO07 must be run once per schedule lever.

**Why 0.7 and not 0.5.** Both clear the gate. `0.5` is faster — MIO10 median
2.337 ms against `0.7`'s 2.980, from a 3.549 ms stack — but it detects only
every 6.6 framesets on MIO10 and every 23.5 on MGO09, which halves what the
estimator sees: on MIO10 the mean landmark count goes 46.0 -> 25.7 and tracked
keypoints 119.1 -> 58.4, and MGO09's keyframe cadence goes to 9.40. `0.7` detects every 2.9
framesets on MIO10 and every 5.2 on MGO09, keeps 38.5 landmarks and 84.3 tracked
keypoints, and is the only setting where **both** measured clips score better
than the stack does: MIO10 1.447 cm against 1.525, MGO09 0.757 cm against 0.766.
It gives 0.582 ms of the 0.3 ms this lever was asked for, and leaves the cadence
the next lever is sized against where it was. The `0.5` numbers are recorded so
the trade is re-openable once the wider clip set has been run — the clips with
fast-dying tracks (MIO11, MIO07, MGO13) are where a halved landmark count would
show, and they are not measured here.

- **D74** — Speed profile: vendored configs stay C++-faithful; speed knobs live in `configs/profiles/fast.json`, opted into with `--profile fast` (2026-09-10)
- **D75** — Redetect on demand: the fast profile skips `addPoints` until camera 0 falls under `port.redetect_survivor_ratio` of its last detection (2026-09-10)

## D76 — The fast profile solves the window at keyframes and the newest state alone between them

basalt is a fixed-lag smoother: `measure` re-linearizes and re-solves the whole
sliding window — 7 keyframe pose blocks plus 3 states, ~87 unknowns — on **every**
frameset (`estimator/mod.rs`, `optimize`), although the keyframe cadence on MIO10
is 7.33 framesets. cuVSLAM instead solves only the newest pose against **fixed**
landmarks every frame (`libs/pnp/multicam_pnp.cpp`, 0.064 ms; `soft_inertial_pnp`,
0.450 ms) and runs bundle adjustment at keyframes only; ORB-SLAM makes the same
split. That schedule is most of cuVSLAM's 4x on this clip: after wave 1 the
estimator is 1.519 ms of a 2.974 ms MIO10 call and 6 of every 7 of those
milliseconds buy a joint solve the frame did not need.

**The knob.** `port.frame_update_max_iterations`, `0` by default. At `0` — where
every basalt file and `VioConfig::default` leave it — `measure` runs the joint
solve on every frameset, which is basalt's schedule byte for byte. Above zero, a
frameset that did **not** take a keyframe runs a *frame update* instead, capped at
that many LM steps. `configs/profiles/fast.json` sets `5`; nothing else does. One
knob rather than a `bool` plus a count: the two cannot then be set against each
other, and a zero cap can only mean "off". The key is `port.` for D75's reason —
basalt has no field for it and the vendored `configs/*.json` stay the documents
the C++ reference runs read.

**What the frame update solves.** The 15 unknowns of the newest state (pose,
velocity, both biases) against exactly two factor groups:

* every observation the newest frameset filed on a landmark the window already
  hosts, with the landmark, its host keyframe and every older state **held**.
  The residual is `linearize_point`, the relative pose and its target Jacobian
  are `compute_rel_pose`, and the Huber weight is the landmark block's own
  `compute_error_weight`, now a free function both paths call — there is one
  reprojection model in the crate, not two;
* the IMU factor from the previous state, built by the same `ImuBlock::linearize`
  the window solve uses. Its 30x30 `add_dense_h_b` is formed and the newest
  state's 15x15 corner is taken, which is what holding the previous state means.

The marginalization prior is **absent by construction, not by choice**: it can only
contain blocks frozen at a linearization point (`compute_delta` refuses any other),
and the newest state is appended unfrozen, so the prior's cost does not depend on
the one variable the frame update moves. The code asserts this per frame and falls
back to the joint solve if the prior ever does carry the newest state.

The LM loop is the window loop's shape, constant for constant: `lambda` reset to
`vio_lm_lambda_initial` every frame (D11), `lambda·diag(H)` damping with the same
floor (D10), the iteration budget shared with backtracking (D12), the increment
negated before it is applied (D13), Nielsen's update on an accept, and the same
`1e-6`/`1e-4` convergence pair. The predicted decrease is
`-(inc·b + 0.5·incᵀ H inc)`, which is what the window's `back_substitute`
accumulates when nothing has been eliminated. `damping.lambda_vee` is shared with
the window solve exactly as it is shared between framesets today.

**What does not change.** Observations are filed into the landmark database before
the keyframe vote, so a frame update still feeds the next joint solve everything it
saw. The keyframe decision is unchanged (camera 0's connected ratio below
`vio_new_kf_keypoints_thresh` and at least `vio_min_frames_after_kf` framesets
since the last). `vio_marg_lost_landmarks` still culls from the frameset's own
observations, never from whether the joint solve ran. **Marginalization keeps its
own trigger and runs every frameset** — deferring it to keyframes grows the
ordering by 15 unknowns per skipped frame, `get_dense_h_b` grows quadratically in
that, and the keyframe solve costs more than the six skipped ones saved. Warmup is
the joint solve's: the frame update runs only once `opt_started` is true.

**The FEJ consequence, stated.** `marginalize` freezes `last_state_to_marg` at its
current value and folds the window into the prior. Under this schedule that value
is a frame update's, not a joint solve's, for the framesets between keyframes, so
the prior is anchored at a point that saw its own observations and its own IMU
factor but not the window's second-order coupling. This is the lever's whole risk
and it is why the gate is ATE, measured, and not an argument.

**Expected numbers, before measuring.** ~70 observations and one IMU factor per
frame update against ~440 observations, 55 landmark blocks and an 87x87 dense
build per joint solve: the non-keyframe `measure` should be the 0.097 ms
marginalization plus ~0.05 ms, against 1.519 ms today, for a median gain near
1.3 ms and an amortized `measure` near 0.3 ms. The accuracy band is 1.654 cm on
MIO10 against 1.447 today — 14% of headroom.

**Measured, MIO10, three A/B rounds against the accepted stack.** Median
**2.945 -> 1.311 ms**, `measure` **1.520 -> 0.183**, ATE vs GT **1.447 -> 1.551 cm**
against 1.654 allowed, zero lost framesets, and the three candidate rounds
identical to each other in both the trajectory and the state digest. The
projection was 1.3 ms of median and it is 1.63; the frame update itself is 0.054
of `optimize`'s 0.061 ms, an order below the 0.30 ms the plan priced it at.

**`config.vio_max_iterations` goes back to basalt's 7, and that is part of this
lever.** D74 cut it to 4 because every frameset paid for the window solve. At one
frameset in 7.65 the trade is a different one: eight steps on 13% of framesets
cost about 0.1 ms of mean and nothing on the median, and they are worth
**0.146 cm** of MIO10 ATE — 1.697 cm at the old cap against 1.551 at basalt's.
Nothing else recovered that: five frame-update steps score the same as two
(1.697 against 1.698), so the gap was never the frame update's own convergence.

**What the accuracy costs, and the one structural fix that did not pay.** MGO09,
four cameras, 107 framesets: median **8.654 -> 2.000 ms** but ATE
**0.757 -> 0.960 cm** against 0.8466 allowed — **out of band, and the open item
this lever leaves behind**. The mechanism I could name is that `marginalize`
freezes `last_state_to_marg` — the state one frameset behind the newest — at the
end of the same `measure`, so with the joint solve at keyframes only a state
enters the FEJ prior having had exactly one frame update. Freeing that state too,
a 30-unknown two-state solve over `k−1` and `k` (cuVSLAM's own
`soft_inertial_pnp.cpp` shape; the prior orders neither, so it still contributes
no gradient), was built and measured: **MGO09 0.960 -> 0.927 cm and MIO10 1.551 ->
1.549**, for **0.158 ms** of MIO10 median (1.311 -> 1.469, which is the difference
between meeting the plan's 1.439 ms MIO10 target and missing it). 16% of the MGO09
gap for 12% of the frame: rejected, and recorded here so it is not rebuilt. What
is left of the gap is the landmarks and the seven keyframe poses standing still
between keyframes, which is the lever itself and not a detail of it.

**Re-measured on the stack that shipped `port.redetect_survivor_ratio` at 0.85**
(`redetect-r085`, wave 1's tip; the numbers above are against the 0.7 stack this
branch was cut from). MIO10, three rounds: median **3.313 -> 1.995 ms**,
`measure` **1.554 -> 0.196**, ATE **1.481 -> 1.553 cm** against 1.629 allowed,
zero lost, the three rounds identical in trajectory and state digest, and the
keyframe cadence (7.04), landmark count (44.1) and tracked keypoints (111.4 ->
112.0) all where the joint solve left them — the schedule moves the solve, not
the map. MGO09: **10.335 -> 3.679 ms**, ATE **0.766 -> 0.978** against 0.842.

**Two other things measured and not kept.** `config.vio_max_states` is not
independently tunable: at 5 the window's own invariants break and marginalization
fails with "landmark block host frame ... is not in the absolute ordering" — the
overshoot argument in `estimator/schedule.rs` assumes a state leaves after three
framesets while keyframes are six apart. And the A/B harness builds into a
`CARGO_TARGET_DIR` shared by every worker on the host, so a concurrent build gets
copied out as yours; one MGO09 measurement here was a different branch's binary
before the cores were pre-placed from a private target directory.
- **D76** — The fast profile runs the joint solve at keyframes and a 15-dof fixed-landmark update on the framesets between (2026-09-10)
