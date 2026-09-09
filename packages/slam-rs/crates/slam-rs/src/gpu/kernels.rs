//! Every CubeCL kernel the frontend runs, and the launchers that dispatch them.
//!
//! The bodies are ports of the CPU functions named on each one, not rewrites:
//! [`crate::pyramid::subsample`] is `subsample_kernel`, `patch::build_patch` is
//! `patch_build_kernel`, and `tracker::track_point` with
//! `tracker::track_point_at_level` is `klt_kernel`. Where the CPU port fixed an
//! arithmetic order because `f32` makes it load-bearing, this file fixes the
//! same one: **every reduction over pattern taps runs on unit 0 in ascending
//! tap order** (decision D21's "reductions by hand", decision D31's fixed
//! order), so a repeat run is bit-identical and the only differences from the
//! CPU are the ones shader contraction of `a * b + c` introduces.
//!
//! ## One cube per patch
//!
//! The per-patch kernels launch [`TAP_UNITS`] units per cube and give unit `t`
//! pattern tap `t`. Fifty-two taps sampled in parallel, then a serial
//! reduction on unit 0: the alternative, one unit per patch, leaves 400 units
//! to hide 200 dependent loads each and measured far worse in the design
//! sketch. Every `sync_cube` sits at a cube-uniform point — the validity flag
//! lives in shared memory and is read after a barrier, never inside a
//! per-tap branch — because a barrier some units skip hangs the cube.
//!
//! ## The `meta` array
//!
//! One `u32` buffer carries a pyramid's level geometry and the sampling
//! pattern, so no stage needs a seventh binding (§12.3: at most six buffers per
//! stage, which is also wgpu's floor).
//!
//! `u32` and not `f32`, because a level base is an **index**: it is added to a
//! pixel offset here, and above `2^24` an `f32` holds only every second
//! integer. A 4097x4097 frame puts level 2's base at 16,785,409, which `f32`
//! stores as 16,785,408, and every level-2 sample read one pixel early on both
//! lanes with nothing reporting it (the S25 review). WGSL has no `u16` or `u8`
//! but it does have `u32`, so this is the one exact carrier all three shader
//! compilers store natively. The pattern taps are the only genuinely
//! fractional values in the array, and they ride in it as their bit patterns
//! rather than in a binding of their own: a bitcast is one instruction
//! everywhere, and a seventh buffer is a portability question on every device
//! the portable lane runs on.
//!
//! ```text
//! meta[level * 4 + 0]  base offset of the level inside its buffer, in pixels
//! meta[level * 4 + 1]  width
//! meta[level * 4 + 2]  height
//! meta[level * 4 + 3]  0 = buffer `a`, 1 = buffer `b`  (the level's parity)
//! meta[levels * 4 + tap * 2 + 0]  pattern tap x, as `f32::to_bits`
//! meta[levels * 4 + tap * 2 + 1]  pattern tap y, as `f32::to_bits`
//! ```
//!
//! ## The `store` array
//!
//! One `f32` buffer per patch set, three sections, each with the patch index
//! fast-varying (§12.2). `C` is the capacity, `T` the pattern size, `L` the
//! level count; the offsets are [`data_offset`], [`jacobian_offset`] and
//! [`valid_offset`] on the host side.
//!
//! ```text
//! data     [(level * T + tap) * C + patch]                  from 0
//! h_inv_jt [((level * 3 + row) * T + tap) * C + patch]      from L*T*C
//! valid    [level * C + patch]                              from 4*L*T*C
//! ```
//!
//! ## The transform arrays
//!
//! `[m00, m01, m10, m11, tx, ty]` in six runs of `count`, as
//! [`crate::frontend::tracker::FlowTransforms`] holds them; a tracker result
//! adds a seventh run for the validity flag.
//!
//! ## One CubeCL trap this file works around everywhere
//!
//! An `if` expression used **directly** as the right-hand side of an array or
//! shared-memory store miscompiles. `inverse[r * 3 + c] = if r == c { 1.0 }
//! else { 0.0 }` left the array untouched, which turned every patch Hessian
//! inverse into zeros and dropped every keypoint, with nothing reported
//! anywhere. The same shape *returned* from a `#[cube]` function
//! (`reflect_high`) is fine. So every conditional value below is bound to a
//! local first and the local is stored, which is also why the two kernels that
//! do it allow `unused_assignments`: the initialiser exists to give the local a
//! type, not to be read.

use cubecl::prelude::*;

/// Units per cube on the per-patch kernels.
///
/// One unit per pattern tap, rounded up to two warps: the largest pattern is 52
/// taps ([`crate::frontend::patterns::MAX_PATTERN_SIZE`]), so 64 covers every
/// pattern and the reductions stay inside one cube.
pub const TAP_UNITS: u32 = 64;

/// [`TAP_UNITS`] as the `#[comptime]` length every per-patch shared array is
/// declared with.
///
/// The nine arrays below are indexed by unit, so raising `TAP_UNITS` for a
/// pattern larger than 52 taps while they stayed at a literal 64 would have
/// units 64.. writing past every one of them, under `launch_unchecked` and with
/// no test that would say so. Off one constant, the coupling is the compiler's.
const TAP_SLOTS: usize = TAP_UNITS as usize;

/// Units per cube on the per-element bookkeeping kernels.
const LINEAR_UNITS: u32 = 256;

/// Cube width on the pyramid kernel, the 32x8 tile the kernel comparison test
/// settled on.
pub const TILE_W: u32 = 32;
/// Cube height on the pyramid kernel.
pub const TILE_H: u32 = 8;

// The four numbers below are the CPU lane's own, aliased rather than
// re-declared: a kernel that drifted from its reference by a constant would
// still compile, and `FILTER_LANES` below already shows the shape.
/// `border` on every patch tap, `PATCH_BORDER` (`patch.h:87`).
const PATCH_BORDER: f32 = crate::frontend::patch::PATCH_BORDER;
/// `const int filter_margin = 2` (`frame_to_frame_optical_flow.h:430`).
const FILTER_MARGIN: f32 = crate::frontend::tracker::FILTER_MARGIN;
/// The increment guard at `frame_to_frame_optical_flow.h:425`.
const MAX_INCREMENT_INFINITY_NORM: f32 = crate::frontend::tracker::MAX_INCREMENT_INFINITY_NORM;
/// `Sophus::Constants<float>::epsilon()` (`common.hpp:182-186`).
const SOPHUS_EPSILON: f32 = <f32 as crate::lie::LieScalar>::SOPHUS_EPSILON;

// ── the pyramid ──────────────────────────────────────────────────────────────

/// `border101` for an index past the high end (`image_pyr.h:83`).
///
/// `h - 1 - |h - 1 - x|`, written without an absolute value so it needs no
/// signed intermediate: below `h` it is the identity, at or above it the
/// reflection `2h - 2 - x`. Callers only ever reach `x <= h`.
#[cube]
fn reflect_high(x: usize, h: usize) -> usize {
    if x < h { x } else { h + h - 2usize - x }
}

/// `std::abs(2 * r - k)` (`image_pyr.h:110-111`), which is the *other*
/// reflection: about zero rather than about the far edge. Trap 3 of the
/// architecture dossier is that these two are not one function.
// `usize::abs_diff` is not in CubeCL's kernel language, so the subtraction is
// spelled out.
#[allow(clippy::manual_abs_diff)]
#[cube]
fn reflect_low(twice: usize, k: usize) -> usize {
    if twice >= k { twice - k } else { k - twice }
}

/// One row's five horizontal taps, `[1, 4, 6, 4, 1]`, as an exact `usize` sum.
#[cube]
#[allow(clippy::too_many_arguments)]
fn subsample_band(
    src: &Array<u16>,
    row_base: usize,
    c0: usize,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize,
) -> usize {
    usize::cast_from(src[row_base + c0])
        + 4usize * usize::cast_from(src[row_base + c1])
        + 6usize * usize::cast_from(src[row_base + c2])
        + 4usize * usize::cast_from(src[row_base + c3])
        + usize::cast_from(src[row_base + c4])
}

/// [`crate::pyramid::subsample`] (`image_pyr.h:99-140`) as one fused 5x5 pass.
///
/// The CPU port runs the separable form: a vertical pass into an `i32`
/// accumulator, then a horizontal pass with one rounding at the very end. Both
/// passes are exact integer sums, so the fused 5x5 here is **bit-exact** with
/// it — which is why this kernel's tolerance test asserts equality rather than
/// a bound. Row indices reflect about the source height and column indices
/// about the source width, exactly as the transposed C++ accumulator makes
/// them (see the CPU docstring). Every pixel is non-negative and the
/// accumulator peaks at `65535 * 16 * 16 = 16,776,960`, so `usize` carries it and
/// `>> 8` is the C++ shift rather than a division that would differ on a
/// negative value.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn subsample_kernel(
    src: &Array<u16>,
    dst: &mut Array<u16>,
    src_base: usize,
    src_width: usize,
    src_height: usize,
    dst_base: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let c = usize::cast_from(ABSOLUTE_POS_X);
    let r = usize::cast_from(ABSOLUTE_POS_Y);
    if c >= dst_width || r >= dst_height {
        terminate!();
    }

    let row2 = 2usize * r;
    let col2 = 2usize * c;
    let r0 = reflect_low(row2, 2usize);
    let r1 = reflect_low(row2, 1usize);
    let r3 = reflect_high(row2 + 1usize, src_height);
    let r4 = reflect_high(row2 + 2usize, src_height);
    let c0 = reflect_low(col2, 2usize);
    let c1 = reflect_low(col2, 1usize);
    let c3 = reflect_high(col2 + 1usize, src_width);
    let c4 = reflect_high(col2 + 2usize, src_width);

    let acc = subsample_band(src, src_base + r0 * src_width, c0, c1, col2, c3, c4)
        + 4usize * subsample_band(src, src_base + r1 * src_width, c0, c1, col2, c3, c4)
        + 6usize * subsample_band(src, src_base + row2 * src_width, c0, c1, col2, c3, c4)
        + 4usize * subsample_band(src, src_base + r3 * src_width, c0, c1, col2, c3, c4)
        + subsample_band(src, src_base + r4 * src_width, c0, c1, col2, c3, c4);

    // `T val = ((val_int + (1 << 7)) >> 8)` (`image_pyr.h:135`).
    dst[dst_base + r * dst_width + c] = u16::cast_from((acc + 128usize) >> 8usize);
}

// ── sampling ─────────────────────────────────────────────────────────────────

/// `ImageU16::in_bounds` (`image.h:694-705`).
#[cube]
fn in_bounds(x: f32, y: f32, border: f32, width: usize, height: usize) -> bool {
    border <= x
        && x < (f32::cast_from(width) - border - 1.0f32)
        && border <= y
        && y < (f32::cast_from(height) - border - 1.0f32)
}

/// One pixel of a level, as `f32`.
#[cube]
fn at(image: &Array<u16>, base: usize, stride: usize, x: usize, y: usize) -> f32 {
    f32::cast_from(image[base + y * stride + x])
}

/// `ImageU16::interp` (`image.h:396-415`), with the multiplication grouping and
/// the summation order the CPU port reproduces from Eigen.
#[cube]
fn interp(image: &Array<u16>, base: usize, stride: usize, x: f32, y: f32) -> f32 {
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let dx = x - f32::cast_from(ix);
    let dy = y - f32::cast_from(iy);
    let ddx = 1.0f32 - dx;
    let ddy = 1.0f32 - dy;
    ddx * ddy * at(image, base, stride, ix, iy)
        + ddx * dy * at(image, base, stride, ix, iy + 1usize)
        + dx * ddy * at(image, base, stride, ix + 1usize, iy)
        + dx * dy * at(image, base, stride, ix + 1usize, iy + 1usize)
}

/// `ImageU16::interp_grad` (`image.h:418-469`) writing its three results into
/// three shared arrays at `slot`.
///
/// The value is the bilinear surface; the gradient is the *central difference*
/// of that surface at unit spacing, not its analytic derivative, which is why
/// twelve pixels are read (`ix - 1` through `ix + 2`, `iy - 1` through
/// `iy + 2`) and why the caller must have satisfied `in_bounds(x, y, 1)`.
///
/// Three `&mut SharedMemory` parameters rather than a returned triple: a
/// `#[cube]` function returns one value, and the caller wants these in shared
/// memory anyway so unit 0 can reduce over them.
#[cube]
#[allow(clippy::too_many_arguments)]
fn interp_grad_into(
    image: &Array<u16>,
    base: usize,
    stride: usize,
    x: f32,
    y: f32,
    values: &mut SharedMemory<f32>,
    grad_x: &mut SharedMemory<f32>,
    grad_y: &mut SharedMemory<f32>,
    slot: usize,
) {
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let dx = x - f32::cast_from(ix);
    let dy = y - f32::cast_from(iy);
    let ddx = 1.0f32 - dx;
    let ddy = 1.0f32 - dy;

    let px0y0 = at(image, base, stride, ix, iy);
    let px1y0 = at(image, base, stride, ix + 1usize, iy);
    let px0y1 = at(image, base, stride, ix, iy + 1usize);
    let px1y1 = at(image, base, stride, ix + 1usize, iy + 1usize);
    values[slot] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

    let pxm1y0 = at(image, base, stride, ix - 1usize, iy);
    let pxm1y1 = at(image, base, stride, ix - 1usize, iy + 1usize);
    let res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;
    let px2y0 = at(image, base, stride, ix + 2usize, iy);
    let px2y1 = at(image, base, stride, ix + 2usize, iy + 1usize);
    let res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;
    grad_x[slot] = 0.5f32 * (res_px - res_mx);

    let px0ym1 = at(image, base, stride, ix, iy - 1usize);
    let px1ym1 = at(image, base, stride, ix + 1usize, iy - 1usize);
    let res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;
    let px0y2 = at(image, base, stride, ix, iy + 2usize);
    let px1y2 = at(image, base, stride, ix + 1usize, iy + 2usize);
    let res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;
    grad_y[slot] = 0.5f32 * (res_py - res_my);
}

/// Whether a value is neither infinite nor NaN, without a `classify`.
///
/// `v * 0` is `0` for every finite `v` and `NaN` for an infinity or a NaN, and
/// `NaN == 0` is false. Same predicate as `f32::is_finite`, in two operations
/// every runtime has.
#[cube]
fn is_finite(value: f32) -> bool {
    value * 0.0f32 == 0.0f32
}

// ── Eigen's pivoted LDLT at size three ───────────────────────────────────────

/// `internal::ldlt_inplace<Lower>::unblocked` at size 3 (`LDLT.h:277-380`),
/// in place on a row-major 3x3.
///
/// A transcription of [`crate::frontend::ldlt::ldlt_inverse3`]'s first half,
/// including the part that matters: on a rank-deficient `H` the pivot below
/// `numeric_limits<float>::min()` is *skipped* rather than divided by, so a
/// patch on a one-dimensional texture comes out with a finite zero-valued
/// `H^-1 J^T` and stays valid, as it does in the C++.
///
/// `transpositions` and the early-out for an all-zero diagonal use a flag
/// rather than a `return`, which a `#[cube]` function does not have.
#[cube]
fn ldlt_decompose3(mat: &mut Array<f32>, transpositions: &mut Array<usize>) {
    let mut temp = Array::<f32>::new(3usize);
    let mut done = false;

    for k in 0..3usize {
        if !done {
            // "Find largest diagonal element" (`LDLT.h:305-307`); `maxCoeff`
            // reports the first index of the maximum, so ties take the earliest.
            let mut biggest = k;
            for i in (k + 1usize)..3usize {
                if f32::abs(mat[i * 3usize + i]) > f32::abs(mat[biggest * 3usize + biggest]) {
                    biggest = i;
                }
            }
            transpositions[k] = biggest;

            if k != biggest {
                // "taking care to consider only the lower triangular part"
                // (`LDLT.h:311-324`).
                for j in 0..k {
                    let swap = mat[k * 3usize + j];
                    mat[k * 3usize + j] = mat[biggest * 3usize + j];
                    mat[biggest * 3usize + j] = swap;
                }
                for i in (biggest + 1usize)..3usize {
                    let swap = mat[i * 3usize + k];
                    mat[i * 3usize + k] = mat[i * 3usize + biggest];
                    mat[i * 3usize + biggest] = swap;
                }
                let swap = mat[k * 3usize + k];
                mat[k * 3usize + k] = mat[biggest * 3usize + biggest];
                mat[biggest * 3usize + biggest] = swap;
                for i in (k + 1usize)..biggest {
                    let swap = mat[i * 3usize + k];
                    mat[i * 3usize + k] = mat[biggest * 3usize + i];
                    mat[biggest * 3usize + i] = swap;
                }
            }

            // The delayed column updates through `temp` (`LDLT.h:326-338`).
            if k > 0usize {
                for i in 0..k {
                    temp[i] = mat[i * 3usize + i] * mat[k * 3usize + i];
                }
                let mut correction = 0.0f32;
                for i in 0..k {
                    correction += mat[k * 3usize + i] * temp[i];
                }
                mat[k * 3usize + k] -= correction;
                for row in (k + 1usize)..3usize {
                    let mut update = 0.0f32;
                    for i in 0..k {
                        update += mat[row * 3usize + i] * temp[i];
                    }
                    mat[row * 3usize + k] -= update;
                }
            }

            // LAPACK's cutoff of exactly zero (`LDLT.h:340-361`).
            let pivot = mat[k * 3usize + k];
            let pivot_is_valid = f32::abs(pivot) > 0.0f32;
            if k == 0usize && !pivot_is_valid {
                // "The entire diagonal is zero, there is nothing more to do."
                for j in 0..3usize {
                    transpositions[j] = j;
                }
                done = true;
            }
            if !done && pivot_is_valid {
                for row in (k + 1usize)..3usize {
                    mat[row * 3usize + k] /= pivot;
                }
            }
        }
    }
}

/// `LDLT::_solve_impl_transposed<true>` at size 3 (`LDLT.h:543-577`), in place
/// on a row-major 3x3 right-hand side.
#[cube]
fn ldlt_solve3(mat: &Array<f32>, transpositions: &Array<usize>, rhs: &mut Array<f32>) {
    // `dst = m_transpositions * rhs`: k ascending (`ProductEvaluators.h:1194`).
    for k in 0..3usize {
        let target = transpositions[k];
        if target != k {
            for j in 0..3usize {
                let swap = rhs[k * 3usize + j];
                rhs[k * 3usize + j] = rhs[target * 3usize + j];
                rhs[target * 3usize + j] = swap;
            }
        }
    }
    // `matrixL().solveInPlace(dst)`: unit-lower forward substitution.
    for row in 1..3usize {
        for column in 0..row {
            let factor = mat[row * 3usize + column];
            for j in 0..3usize {
                let value = rhs[column * 3usize + j];
                rhs[row * 3usize + j] -= factor * value;
            }
        }
    }
    // "more precisely, use pseudo-inverse of D" (`LDLT.h:551-568`): the
    // tolerance is `numeric_limits<float>::min()`, the smallest positive normal.
    for i in 0..3usize {
        let d = mat[i * 3usize + i];
        if f32::abs(d) > 1.1754944e-38f32 {
            for j in 0..3usize {
                rhs[i * 3usize + j] /= d;
            }
        } else {
            for j in 0..3usize {
                rhs[i * 3usize + j] = 0.0f32;
            }
        }
    }
    // `matrixL().transpose().solveInPlace(dst)`: unit-upper back substitution.
    for step in 0..3usize {
        let row = 2usize - step;
        for column in (row + 1usize)..3usize {
            let factor = mat[column * 3usize + row];
            for j in 0..3usize {
                let value = rhs[column * 3usize + j];
                rhs[row * 3usize + j] -= factor * value;
            }
        }
    }
    // `dst = m_transpositions.transpose() * dst`: k descending.
    for step in 0..3usize {
        let k = 2usize - step;
        let target = transpositions[k];
        if target != k {
            for j in 0..3usize {
                let swap = rhs[k * 3usize + j];
                rhs[k * 3usize + j] = rhs[target * 3usize + j];
                rhs[target * 3usize + j] = swap;
            }
        }
    }
}

// ── the patch build ──────────────────────────────────────────────────────────

/// `patch::build_patch` (`patch.h:101-166`) over one patch at one level.
///
/// One cube per `(patch, level)`, one unit per pattern tap. Three reductions
/// run on unit 0 in ascending tap order — the tap sum with `grad_sum_se2`, the
/// nine entries of `H = J^T J`, and the finiteness `AND` — because in `f32` the
/// order is the value (decision D31).
///
/// The product-rule correction that comes from differentiating the `1/mean`
/// factor (`patch.h:135`) is applied with the **raw** tap, before the tap is
/// normalised, and the rows of out-of-bounds taps are zeroed: dropping that
/// term gives a Jacobian that looks right and converges to the wrong warp.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn patch_build_kernel(
    pyramid_a: &Array<u16>,
    pyramid_b: &Array<u16>,
    meta: &Array<u32>,
    positions: &Array<f32>,
    store: &mut Array<f32>,
    capacity: usize,
    taps: usize,
    num_levels: usize,
    count: usize,
    pos_x_base: usize,
    pos_y_base: usize,
    selected_base: usize,
) {
    let patch = usize::cast_from(CUBE_POS_X);
    let level = usize::cast_from(CUBE_POS_Y);
    let tap = usize::cast_from(UNIT_POS_X);
    if patch >= count {
        terminate!();
    }

    let data_len = num_levels * taps * capacity;
    let row_stride = taps * capacity;
    let valid_slot = 4usize * data_len + level * capacity + patch;

    // A patch the caller switched off is invalid at every level and never
    // sampled (`SourcePatches::build`'s `selected`).
    if positions[selected_base + patch] == 0.0f32 {
        if tap == 0usize {
            store[valid_slot] = 0.0f32;
        }
        terminate!();
    }

    let mut values = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut grad_x = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut grad_y = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut grad_t = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut okflag = SharedMemory::<usize>::new(TAP_SLOTS);
    let mut red = SharedMemory::<f32>::new(16usize);

    // `const Scalar scale = 1 << level` (`frame_to_frame_optical_flow.h:384`).
    let scale = f32::cast_from(1usize << level);
    let pos_x = positions[pos_x_base + patch] / scale;
    let pos_y = positions[pos_y_base + patch] / scale;

    let base = usize::cast_from(meta[level * 4usize]);
    let width = usize::cast_from(meta[level * 4usize + 1usize]);
    let height = usize::cast_from(meta[level * 4usize + 2usize]);
    let parity = usize::cast_from(meta[level * 4usize + 3usize]);
    let pattern = num_levels * 4usize;

    if tap < taps {
        let tap_x = f32::reinterpret(meta[pattern + tap * 2usize]);
        let tap_y = f32::reinterpret(meta[pattern + tap * 2usize + 1usize]);
        let px = pos_x + tap_x;
        let py = pos_y + tap_y;
        if in_bounds(px, py, PATCH_BORDER, width, height) {
            if parity == 0usize {
                interp_grad_into(
                    pyramid_a,
                    base,
                    width,
                    px,
                    py,
                    &mut values,
                    &mut grad_x,
                    &mut grad_y,
                    tap,
                );
            } else {
                interp_grad_into(
                    pyramid_b,
                    base,
                    width,
                    px,
                    py,
                    &mut values,
                    &mut grad_x,
                    &mut grad_y,
                    tap,
                );
            }
            // `valGrad.tail<2>().transpose() * Jw_se2` with
            // `Jw_se2 = [[1, 0, -tap_y], [0, 1, tap_x]]` (`patch.h:107-115`).
            grad_t[tap] = grad_x[tap] * -tap_y + grad_y[tap] * tap_x;
            okflag[tap] = 1usize;
        } else {
            values[tap] = -1.0f32;
            grad_x[tap] = 0.0f32;
            grad_y[tap] = 0.0f32;
            grad_t[tap] = 0.0f32;
            okflag[tap] = 0usize;
        }
    }
    sync_cube();

    if tap == 0usize {
        let mut sum = 0.0f32;
        let mut valid_points: u32 = 0u32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_t = 0.0f32;
        for i in 0..taps {
            if okflag[i] == 1usize {
                sum += values[i];
                valid_points += 1u32;
                sum_x += grad_x[i];
                sum_y += grad_y[i];
                sum_t += grad_t[i];
            }
        }
        let points = f32::cast_from(valid_points);
        red[0usize] = sum;
        // `mean = sum / n` and `mean_inv = n / sum` (`patch.h:129`, `:131`) are
        // two separate divisions, not reciprocals of each other.
        red[1usize] = sum / points;
        red[2usize] = points / sum;
        red[3usize] = sum_x;
        red[4usize] = sum_y;
        red[5usize] = sum_t;
    }
    sync_cube();

    let sum = red[0usize];
    let mean_inv = red[2usize];
    if tap < taps {
        let raw = values[tap];
        if raw >= 0.0f32 {
            grad_x[tap] = (grad_x[tap] - red[3usize] * raw / sum) * mean_inv;
            grad_y[tap] = (grad_y[tap] - red[4usize] * raw / sum) * mean_inv;
            grad_t[tap] = (grad_t[tap] - red[5usize] * raw / sum) * mean_inv;
            values[tap] = raw * mean_inv;
        } else {
            grad_x[tap] = 0.0f32;
            grad_y[tap] = 0.0f32;
            grad_t[tap] = 0.0f32;
        }
    }
    sync_cube();

    if tap == 0usize {
        // `H_se2 = J^T J` (`patch.h:151`), one rank-1 outer product per tap.
        let mut hessian = Array::<f32>::new(9usize);
        for i in 0..9usize {
            hessian[i] = 0.0f32;
        }
        for i in 0..taps {
            let r0 = grad_x[i];
            let r1 = grad_y[i];
            let r2 = grad_t[i];
            hessian[0usize] += r0 * r0;
            hessian[1usize] += r0 * r1;
            hessian[2usize] += r0 * r2;
            hessian[3usize] += r1 * r0;
            hessian[4usize] += r1 * r1;
            hessian[5usize] += r1 * r2;
            hessian[6usize] += r2 * r0;
            hessian[7usize] += r2 * r1;
            hessian[8usize] += r2 * r2;
        }
        let mut transpositions = Array::<usize>::new(3usize);
        ldlt_decompose3(&mut hessian, &mut transpositions);
        let mut inverse = Array::<f32>::new(9usize);
        for i in 0..9usize {
            inverse[i] = 0.0f32;
        }
        inverse[0usize] = 1.0f32;
        inverse[4usize] = 1.0f32;
        inverse[8usize] = 1.0f32;
        ldlt_solve3(&hessian, &transpositions, &mut inverse);
        for i in 0..9usize {
            red[7usize + i] = inverse[i];
        }
    }
    sync_cube();

    if tap < taps {
        // `H_se2_inv_J_se2_T.col(i) = H^-1 * J^T.col(i)` (`patch.h:156`).
        let c0 = grad_x[tap];
        let c1 = grad_y[tap];
        let c2 = grad_t[tap];
        let p0 = red[7usize] * c0 + red[8usize] * c1 + red[9usize] * c2;
        let p1 = red[10usize] * c0 + red[11usize] * c1 + red[12usize] * c2;
        let p2 = red[13usize] * c0 + red[14usize] * c1 + red[15usize] * c2;
        let slot = data_len + (level * 3usize * taps + tap) * capacity + patch;
        store[slot] = p0;
        store[slot + row_stride] = p1;
        store[slot + 2usize * row_stride] = p2;
        let value = values[tap];
        store[(level * taps + tap) * capacity + patch] = value;
        let finite = is_finite(p0) && is_finite(p1) && is_finite(p2) && is_finite(value);
        let mut flag: usize = 0usize;
        if finite {
            flag = 1usize;
        }
        okflag[tap] = flag;
    } else {
        okflag[tap] = 1usize;
    }
    sync_cube();

    if tap == 0usize {
        // `valid = mean > eps && H^-1 J^T finite && data finite` (`patch.h:164`).
        let mut finite = true;
        for i in 0..taps {
            if okflag[i] == 0usize {
                finite = false;
            }
        }
        let mut valid: f32 = 0.0f32;
        if red[1usize] > f32::new(f32::EPSILON) && finite {
            valid = 1.0f32;
        }
        store[valid_slot] = valid;
    }
}

// ── the KLT tracker ──────────────────────────────────────────────────────────

/// `Sophus::SE2<float>::exp` composed onto the running warp, on unit 0.
///
/// Writes the composed `[m00, m01, m10, m11, tx, ty]` back into `state`. Three
/// details of `tracker::se2_exp` are load-bearing and reproduced literally: the
/// `SO2(cos, sin)` constructor **normalises** by `hypot`, the `V` factor divides
/// by the *normalised* components, and the small-angle branch is Sophus's
/// Taylor series below `Constants<float>::epsilon()`. `hypot` is spelled
/// `sqrt(c*c + s*s)` here, which is the one place this kernel can differ from
/// the CPU by an ulp for reasons other than fused multiply-add.
#[cube]
fn compose_se2_exp(state: &mut SharedMemory<f32>, t0: f32, t1: f32, theta: f32) {
    let cos_theta = f32::cos(theta);
    let sin_theta = f32::sin(theta);
    let length = f32::sqrt(cos_theta * cos_theta + sin_theta * sin_theta);
    let real = cos_theta / length;
    let imaginary = sin_theta / length;

    let mut sin_by_theta = imaginary / theta;
    let mut one_minus_cos_by_theta = (1.0f32 - real) / theta;
    if f32::abs(theta) < SOPHUS_EPSILON {
        let theta_sq = theta * theta;
        sin_by_theta = 1.0f32 - (1.0f32 / 6.0f32) * theta_sq;
        one_minus_cos_by_theta = 0.5f32 * theta - (1.0f32 / 24.0f32) * theta * theta_sq;
    }
    let exp_x = sin_by_theta * t0 - one_minus_cos_by_theta * t1;
    let exp_y = one_minus_cos_by_theta * t0 + sin_by_theta * t1;

    // `transform *= exp`: `linear * exp.linear`, `linear * exp.t + t`.
    let m00 = state[0usize];
    let m01 = state[1usize];
    let m10 = state[2usize];
    let m11 = state[3usize];
    state[0usize] = m00 * real + m01 * imaginary;
    state[1usize] = m00 * -imaginary + m01 * real;
    state[2usize] = m10 * real + m11 * imaginary;
    state[3usize] = m10 * -imaginary + m11 * real;
    state[4usize] = m00 * exp_x + m01 * exp_y + state[4usize];
    state[5usize] = m10 * exp_x + m11 * exp_y + state[5usize];
}

/// `tracker::track_point` and `track_point_at_level`
/// (`frame_to_frame_optical_flow.h:377-438`), one cube per patch.
///
/// The coarse-to-fine sweep and the Gauss-Newton loop both run to their fixed
/// bounds with a shared `alive` flag standing in for the C++ loop conditions —
/// the CPU port already made that substitution and proved it changes no value
/// (§12.2). Every `sync_cube` here is cube-uniform: the flag lives in shared
/// memory and is read into a local *after* a barrier, so a unit never skips a
/// barrier its neighbours reach.
///
/// `check_guess_bounds` is the `valid = t2 in [0, w) x [0, h)` test at `:346`,
/// which the forward pass makes and the backward pass does not. When it fails
/// the C++ returns the plain identity rather than the composed warp, and so
/// does this: `alive[2usize]` records whether the sweep ran at all.
#[cube(launch, launch_unchecked)]
// `!(x < y)` is deliberate: it is how the C++ spells the increment guard, so a
// NaN takes the branch it takes there. The initialisers of the locals that
// stand in for an `if` expression are never read, which is the point.
#[allow(
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    unused_assignments
)]
fn klt_kernel(
    pyramid_a: &Array<u16>,
    pyramid_b: &Array<u16>,
    meta: &Array<u32>,
    store: &Array<f32>,
    // One binding, read and written, not two views of the same buffer. Every
    // caller passes the same handle for both roles — the C++ composes the warp
    // in place (`:399`) — and wgpu refuses a buffer bound `STORAGE_READ_ONLY`
    // and `STORAGE_READ_WRITE` in one dispatch, which is a validation error on
    // any adapter whose pool does not merge the two slices into one binding.
    // The cap's Mali G610 is that adapter.
    transforms: &mut Array<f32>,
    capacity: usize,
    taps: usize,
    num_levels: usize,
    count: usize,
    max_iterations: usize,
    check_guess_bounds: usize,
) {
    let patch = usize::cast_from(CUBE_POS_X);
    let tap = usize::cast_from(UNIT_POS_X);
    if patch >= count {
        terminate!();
    }

    let data_len = num_levels * taps * capacity;
    let row_stride = taps * capacity;
    let pattern = num_levels * 4usize;

    let mut residual = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut product_0 = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut product_1 = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut product_2 = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut okflag = SharedMemory::<usize>::new(TAP_SLOTS);
    // 0..6 the running warp, 6 the tap sum, 7 the in-bounds tap count,
    // 8 the level scale.
    let mut state = SharedMemory::<f32>::new(16usize);
    // 0 the running validity, 1 whether this level was entered alive,
    // 2 whether the guess passed the bounds test at all.
    let mut alive = SharedMemory::<usize>::new(4usize);

    if tap == 0usize {
        let guess_x = transforms[4usize * count + patch];
        let guess_y = transforms[5usize * count + patch];
        let level0_width = f32::cast_from(meta[1usize]);
        let level0_height = f32::cast_from(meta[2usize]);
        let mut inside = true;
        if check_guess_bounds == 1usize
            && (guess_x < 0.0f32
                || guess_y < 0.0f32
                || guess_x >= level0_width
                || guess_y >= level0_height)
        {
            inside = false;
        }
        let mut entered: usize = 0usize;
        if inside {
            entered = 1usize;
        }
        alive[0usize] = entered;
        alive[2usize] = entered;
        // `transform.linear = identity`, `transform.translation = guess` (`:381`).
        state[0usize] = 1.0f32;
        state[1usize] = 0.0f32;
        state[2usize] = 0.0f32;
        state[3usize] = 1.0f32;
        state[4usize] = guess_x;
        state[5usize] = guess_y;
    }
    sync_cube();

    for step in 0..num_levels {
        let level = num_levels - 1usize - step;
        if tap == 0usize {
            alive[1usize] = alive[0usize];
            if alive[0usize] == 1usize {
                // `transform.translation /= scale` (`:386`), exact for a power of two.
                let scale = f32::cast_from(1usize << level);
                state[8usize] = scale;
                state[4usize] /= scale;
                state[5usize] /= scale;
                // `patch_valid &= patches.valid(level, index)` (`:389`).
                if store[4usize * data_len + level * capacity + patch] == 0.0f32 {
                    alive[0usize] = 0usize;
                }
            }
        }
        sync_cube();

        let base = usize::cast_from(meta[level * 4usize]);
        let width = usize::cast_from(meta[level * 4usize + 1usize]);
        let height = usize::cast_from(meta[level * 4usize + 2usize]);
        let parity = usize::cast_from(meta[level * 4usize + 3usize]);

        for _iteration in 0..max_iterations {
            let sampling = alive[0usize] == 1usize;
            // `residual(img, transform * pattern2, res)` (`patch.h:168-202`),
            // with the warp applied per tap.
            if sampling && tap < taps {
                let tap_x = f32::reinterpret(meta[pattern + tap * 2usize]);
                let tap_y = f32::reinterpret(meta[pattern + tap * 2usize + 1usize]);
                let warped_x = state[0usize] * tap_x + state[1usize] * tap_y + state[4usize];
                let warped_y = state[2usize] * tap_x + state[3usize] * tap_y + state[5usize];
                if in_bounds(warped_x, warped_y, PATCH_BORDER, width, height) {
                    let mut sampled: f32 = 0.0f32;
                    if parity == 0usize {
                        sampled = interp(pyramid_a, base, width, warped_x, warped_y);
                    } else {
                        sampled = interp(pyramid_b, base, width, warped_x, warped_y);
                    }
                    residual[tap] = sampled;
                    okflag[tap] = 1usize;
                } else {
                    residual[tap] = -1.0f32;
                    okflag[tap] = 0usize;
                }
            }
            sync_cube();

            if sampling && tap == 0usize {
                let mut sum = 0.0f32;
                let mut valid_points: u32 = 0u32;
                for i in 0..taps {
                    if okflag[i] == 1usize {
                        sum += residual[i];
                        valid_points += 1u32;
                    }
                }
                // An all-black target cannot be normalised (`patch.h:183-186`).
                if sum < f32::new(f32::EPSILON) {
                    alive[0usize] = 0usize;
                }
                state[6usize] = sum;
                state[7usize] = f32::cast_from(valid_points);
            }
            sync_cube();

            let solving = alive[0usize] == 1usize;
            if solving && tap < taps {
                // `res[i] = num_valid_points * val / sum - data[i]` (`patch.h:193`).
                let sum = state[6usize];
                let points = state[7usize];
                let stored = store[(level * taps + tap) * capacity + patch];
                let sampled = residual[tap];
                if sampled >= 0.0f32 && stored >= 0.0f32 {
                    residual[tap] = points * sampled / sum - stored;
                    okflag[tap] = 1usize;
                } else {
                    residual[tap] = 0.0f32;
                    okflag[tap] = 0usize;
                }
                let slot = data_len + (level * 3usize * taps + tap) * capacity + patch;
                product_0[tap] = store[slot] * residual[tap];
                product_1[tap] = store[slot + row_stride] * residual[tap];
                product_2[tap] = store[slot + 2usize * row_stride] * residual[tap];
            }
            sync_cube();

            if solving && tap == 0usize {
                let mut residuals: u32 = 0u32;
                for i in 0..taps {
                    if okflag[i] == 1usize {
                        residuals += 1u32;
                    }
                }
                // `return num_residuals > PATTERN_SIZE / 2` (`patch.h:201`).
                if residuals * 2u32 <= u32::cast_from(taps) {
                    alive[0usize] = 0usize;
                } else {
                    // `inc = -H_se2_inv_J_se2_T * res` (`:419`), taps ascending.
                    let mut sum_0 = 0.0f32;
                    let mut sum_1 = 0.0f32;
                    let mut sum_2 = 0.0f32;
                    for i in 0..taps {
                        sum_0 += product_0[i];
                        sum_1 += product_1[i];
                        sum_2 += product_2[i];
                    }
                    let inc_0 = -sum_0;
                    let inc_1 = -sum_1;
                    let inc_2 = -sum_2;
                    let mut ok = is_finite(inc_0) && is_finite(inc_1) && is_finite(inc_2);
                    // `inc.lpNorm<Infinity>()` is `(a < b) ? b : a` from element 0,
                    // spelled out so a NaN takes the branch it takes in C++.
                    let mut infinity_norm = f32::abs(inc_0);
                    if infinity_norm < f32::abs(inc_1) {
                        infinity_norm = f32::abs(inc_1);
                    }
                    if infinity_norm < f32::abs(inc_2) {
                        infinity_norm = f32::abs(inc_2);
                    }
                    if !(infinity_norm < MAX_INCREMENT_INFINITY_NORM) {
                        ok = false;
                    }
                    if ok {
                        compose_se2_exp(&mut state, inc_0, inc_1, inc_2);
                        // The new centre must stay two pixels inside (`:430-432`).
                        if !in_bounds(state[4usize], state[5usize], FILTER_MARGIN, width, height) {
                            ok = false;
                        }
                    }
                    if !ok {
                        alive[0usize] = 0usize;
                    }
                }
            }
            sync_cube();
        }

        if tap == 0usize && alive[1usize] == 1usize {
            // `transform.translation *= scale` (`:396`), which the C++ runs even
            // when the level just failed.
            let scale = state[8usize];
            state[4usize] *= scale;
            state[5usize] *= scale;
        }
        sync_cube();
    }

    if tap == 0usize {
        if alive[2usize] == 1usize {
            // `transform.linear = old_linear * transform.linear` (`:399`).
            // Read into locals before the first write: the four coefficients
            // are at the indices the composed warp overwrites.
            let o00 = transforms[patch];
            let o01 = transforms[count + patch];
            let o10 = transforms[2usize * count + patch];
            let o11 = transforms[3usize * count + patch];
            transforms[patch] = o00 * state[0usize] + o01 * state[2usize];
            transforms[count + patch] = o00 * state[1usize] + o01 * state[3usize];
            transforms[2usize * count + patch] = o10 * state[0usize] + o11 * state[2usize];
            transforms[3usize * count + patch] = o10 * state[1usize] + o11 * state[3usize];
            transforms[4usize * count + patch] = state[4usize];
            transforms[5usize * count + patch] = state[5usize];
            transforms[6usize * count + patch] = f32::cast_from(alive[0usize]);
        } else {
            transforms[patch] = 1.0f32;
            transforms[count + patch] = 0.0f32;
            transforms[2usize * count + patch] = 0.0f32;
            transforms[3usize * count + patch] = 1.0f32;
            transforms[4usize * count + patch] = 0.0f32;
            transforms[5usize * count + patch] = 0.0f32;
            transforms[6usize * count + patch] = 0.0f32;
        }
    }
}

/// The backward pass's inputs, from the forward result (`:355-359`).
///
/// `off = t2 - t2_guess` with `t2 == t1` at that point, so
/// `off == source position - guess`, and `t1_recovered = forward + off`; the
/// linear part carries over from the forward result. Both terms of `off` are on
/// the host when a frame starts, so the tracker uploads it with the backward
/// patch set's positions and this kernel only adds it.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn prepare_backward_kernel(
    forward: &Array<f32>,
    offsets: &Array<f32>,
    out: &mut Array<f32>,
    count: usize,
    offset_x_base: usize,
    offset_y_base: usize,
) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= count {
        terminate!();
    }
    for k in 0..4usize {
        out[k * count + index] = forward[k * count + index];
    }
    out[4usize * count + index] = forward[4usize * count + index] + offsets[offset_x_base + index];
    out[5usize * count + index] = forward[5usize * count + index] + offsets[offset_y_base + index];
    out[6usize * count + index] = forward[6usize * count + index];
}

/// The recovered-distance test that decides a track (`:362-364`).
///
/// The warp published is always the *forward* one; only the validity flag comes
/// from the backward pass, exactly as `trackPoints` keeps `transform_1` and
/// tests `(t1 - t1_recovered).squaredNorm()`.
#[cube(launch, launch_unchecked)]
// As `klt_kernel`: `!(dist2 < max)` is the C++'s own spelling of the test.
#[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
fn finish_kernel(
    forward: &Array<f32>,
    backward: &Array<f32>,
    positions: &Array<f32>,
    out: &mut Array<f32>,
    count: usize,
    pos_x_base: usize,
    pos_y_base: usize,
    max_recovered_dist2: f32,
) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= count {
        terminate!();
    }
    for k in 0..6usize {
        out[k * count + index] = forward[k * count + index];
    }
    let mut ok =
        forward[6usize * count + index] != 0.0f32 && backward[6usize * count + index] != 0.0f32;
    if ok {
        let dx = positions[pos_x_base + index] - backward[4usize * count + index];
        let dy = positions[pos_y_base + index] - backward[5usize * count + index];
        if !(dx * dx + dy * dy < max_recovered_dist2) {
            ok = false;
        }
    }
    let mut flag: f32 = 0.0f32;
    if ok {
        flag = 1.0f32;
    }
    out[6usize * count + index] = flag;
}

// ── launchers ────────────────────────────────────────────────────────────────
//
// Every launcher is generic over `R: Runtime` and every per-frame one uses
// `launch_unchecked`: the default checked mode adds a `select` per array access
// and the shape checks the stage traits already make are what put every index
// in range (`Robocap.md`, "Kernel-design rules learned"). The per-kernel
// tolerance tests, which assert the pyramid **bit-exact** against the CPU, are
// what validates that claim on every `cargo test --features gpu-wgpu`. The one
// exception is [`launch_probe`], which is off that path and says why.

/// A device buffer and the element count the kernel will see in it.
///
/// Spelled once rather than at each of the launcher parameters below, all of
/// which take exactly this: the count is a promise `ArrayArg::from_raw_parts`
/// cannot check, so keeping it beside the handle is what makes the promise
/// visible at the call site.
pub(super) type Buffer<'a> = (&'a cubecl::server::Handle, usize);

/// The dispatch every image-shaped kernel uses: one unit per pixel, over
/// [`TILE_W`] x [`TILE_H`] tiles.
///
/// Two-dimensional cube dims with `ABSOLUTE_POS_X`/`_Y` rather than a linear
/// index and a `div`/`mod`, which is the layout the kernel comparison
/// measurement settled on.
fn tile_2d(width: usize, height: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(
            (width as u32).div_ceil(TILE_W),
            (height as u32).div_ceil(TILE_H),
            1,
        ),
        CubeDim {
            x: TILE_W,
            y: TILE_H,
            z: 1,
        },
    )
}

/// Cubes per dispatch dimension a WebGPU implementation must allow, and the
/// number every one of them stops at.
///
/// wgpu reports its adapter's own `max_compute_workgroups_per_dimension` and on
/// every adapter measured that is exactly this floor, so the portable lane
/// treats it as the limit rather than as a minimum.
const MAX_CUBES_PER_DIM: u32 = 65_535;

/// The dispatch every per-element bookkeeping kernel uses: one unit per element,
/// found through [`ABSOLUTE_POS`].
///
/// Two dimensions rather than one, because a single row of cubes reaches only
/// `65,535 * 256 = 2^24` elements and the frame copy is one unit per **pixel**:
/// a 4097x4097 frame is 16,785,409 of them, and the portable lane failed that
/// dispatch at validation — a wgpu error on its own thread, so the caller saw a
/// panicking read rather than a refusal. Both compilers flatten `ABSOLUTE_POS`
/// as `z * cubes_x * units * cubes_y + y * cubes_x * units + x`, so the second
/// row is a continuation of the first and the mapping is the same dense
/// enumeration either way. Anything under the ceiling — every shipped frame and
/// every keypoint buffer — still dispatches exactly one row, unchanged.
fn linear_1d(count: usize) -> (CubeCount, CubeDim) {
    let cubes: u32 = (count as u32).div_ceil(LINEAR_UNITS);
    (
        CubeCount::Static(
            cubes.min(MAX_CUBES_PER_DIM),
            cubes.div_ceil(MAX_CUBES_PER_DIM),
            1,
        ),
        CubeDim {
            x: LINEAR_UNITS,
            y: 1,
            z: 1,
        },
    )
}

/// One level of one pyramid: `source` down into `target` at half its size.
pub(super) fn launch_subsample<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    source: super::pyramid::Level,
    target: super::pyramid::Level,
) {
    let (cubes, units) = tile_2d(target.width, target.height);
    unsafe {
        subsample_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(src.0.clone(), src.1),
            ArrayArg::from_raw_parts(dst.0.clone(), dst.1),
            source.base,
            source.width,
            source.height,
            target.base,
            target.width,
            target.height,
        );
    }
}

/// The shape every per-patch launcher needs: capacity, taps, levels, count.
#[derive(Debug, Clone, Copy)]
pub(super) struct PatchShape {
    /// Patch slots the store was allocated for.
    pub capacity: usize,
    /// Pattern taps.
    pub taps: usize,
    /// Pyramid levels, which is `optical_flow_levels + 1`.
    pub num_levels: usize,
    /// Patches actually filled.
    pub count: usize,
}

/// Where a launcher reads positions and the selection flag inside one buffer.
#[derive(Debug, Clone, Copy)]
pub(super) struct PositionBases {
    /// Index of patch 0's `x`.
    pub x: usize,
    /// Index of patch 0's `y`.
    pub y: usize,
    /// Index of patch 0's selection flag, non-zero meaning selected.
    pub selected: usize,
}

/// Sample every patch of `positions` at every level of the pyramid.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_patch_build<R: Runtime>(
    client: &ComputeClient<R>,
    pyramid: (
        &cubecl::server::Handle,
        usize,
        &cubecl::server::Handle,
        usize,
    ),
    meta: Buffer<'_>,
    positions: Buffer<'_>,
    store: Buffer<'_>,
    shape: PatchShape,
    bases: PositionBases,
) {
    unsafe {
        patch_build_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(shape.count as u32, shape.num_levels as u32, 1),
            CubeDim {
                x: TAP_UNITS,
                y: 1,
                z: 1,
            },
            ArrayArg::from_raw_parts(pyramid.0.clone(), pyramid.1),
            ArrayArg::from_raw_parts(pyramid.2.clone(), pyramid.3),
            ArrayArg::from_raw_parts(meta.0.clone(), meta.1),
            ArrayArg::from_raw_parts(positions.0.clone(), positions.1),
            ArrayArg::from_raw_parts(store.0.clone(), store.1),
            shape.capacity,
            shape.taps,
            shape.num_levels,
            shape.count,
            bases.x,
            bases.y,
            bases.selected,
        );
    }
}

/// Track every patch coarse to fine into the pyramid handed in.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_klt<R: Runtime>(
    client: &ComputeClient<R>,
    pyramid: (
        &cubecl::server::Handle,
        usize,
        &cubecl::server::Handle,
        usize,
    ),
    meta: Buffer<'_>,
    store: Buffer<'_>,
    transforms: Buffer<'_>,
    shape: PatchShape,
    max_iterations: usize,
    check_guess_bounds: bool,
) {
    unsafe {
        klt_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(shape.count as u32, 1, 1),
            CubeDim {
                x: TAP_UNITS,
                y: 1,
                z: 1,
            },
            ArrayArg::from_raw_parts(pyramid.0.clone(), pyramid.1),
            ArrayArg::from_raw_parts(pyramid.2.clone(), pyramid.3),
            ArrayArg::from_raw_parts(meta.0.clone(), meta.1),
            ArrayArg::from_raw_parts(store.0.clone(), store.1),
            ArrayArg::from_raw_parts(transforms.0.clone(), transforms.1),
            shape.capacity,
            shape.taps,
            shape.num_levels,
            shape.count,
            max_iterations,
            usize::from(check_guess_bounds),
        );
    }
}

/// A device copy of `count` elements of one width.
///
/// Two callers, and the second is why the kernel is generic rather than fixed
/// to the width the probe needs. [`super::probe_storage`] copies a known
/// pattern of four widths to refuse a runtime that cannot store one of them;
/// [`launch_copy_level0`] copies the frame at `u16`, because the upload and the
/// pyramid want different shapes. The upload wants to be exactly as long as the
/// frame — `create_from_slice` is CubeCL 0.10's only host-to-device write and it
/// copies the payload on the host before the bus sees it, so every byte over
/// the frame is paid for twice — while the pyramid wants level 0 at offset zero
/// of the buffer that also holds levels 2 and 4, because the per-patch kernels
/// reach a level through two bindings split by parity. One device copy of the
/// frame, a few microseconds at this card's bandwidth, buys both, and leaves
/// the even allocation to be made once at `allocate` rather than replaced every
/// frame.
#[cube(launch, launch_unchecked)]
fn probe_kernel<N: Numeric>(src: &Array<N>, dst: &mut Array<N>, count: usize) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= count {
        terminate!();
    }
    dst[index] = src[index];
}

/// Copy `count` elements of type `N` from `src` to `dst`.
///
/// The one launcher here that keeps CubeCL's bounds checks — `launch`, not
/// `launch_unchecked`. It runs once per process, off the per-frame path, and it
/// is the kernel whose whole job is to prove the runtime is sound: bounds checks
/// off is the wrong shape for that, and 256 elements cost nothing either way.
/// The `unsafe` that remains is `ArrayArg::from_raw_parts`, which every launcher
/// needs to name a handle's element count, not the launch.
pub(super) fn launch_probe<N: Numeric, R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    count: usize,
) {
    let (cubes, units) = linear_1d(count);
    unsafe {
        probe_kernel::launch::<N, R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(src.0.clone(), src.1),
            ArrayArg::from_raw_parts(dst.0.clone(), dst.1),
            count,
        );
    }
}

/// Copy `count` pixels from the upload buffer to the front of `dst`.
///
/// [`probe_kernel`] instantiated at `u16`, not a second kernel: the body was
/// the same three lines, so a copy kernel of its own meant two
/// SPIR-V modules compiled for one copy. `launch_unchecked` here where
/// [`launch_probe`] takes the checked one — this is the per-frame path.
pub(super) fn launch_copy_level0<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    count: usize,
) {
    let (cubes, units) = linear_1d(count);
    unsafe {
        probe_kernel::launch_unchecked::<u16, R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(src.0.clone(), src.1),
            ArrayArg::from_raw_parts(dst.0.clone(), dst.1),
            count,
        );
    }
}

/// Build the backward pass's transform inputs from the forward result.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_prepare_backward<R: Runtime>(
    client: &ComputeClient<R>,
    forward: Buffer<'_>,
    offsets: Buffer<'_>,
    out: Buffer<'_>,
    count: usize,
    bases: PositionBases,
) {
    let (cubes, units) = linear_1d(count);
    unsafe {
        prepare_backward_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(forward.0.clone(), forward.1),
            ArrayArg::from_raw_parts(offsets.0.clone(), offsets.1),
            ArrayArg::from_raw_parts(out.0.clone(), out.1),
            count,
            bases.x,
            bases.y,
        );
    }
}

/// Combine the two passes into the result the CPU downloads.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_finish<R: Runtime>(
    client: &ComputeClient<R>,
    forward: Buffer<'_>,
    backward: Buffer<'_>,
    positions: Buffer<'_>,
    out: Buffer<'_>,
    count: usize,
    bases: PositionBases,
    max_recovered_dist2: f32,
) {
    let (cubes, units) = linear_1d(count);
    unsafe {
        finish_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(forward.0.clone(), forward.1),
            ArrayArg::from_raw_parts(backward.0.clone(), backward.1),
            ArrayArg::from_raw_parts(positions.0.clone(), positions.1),
            ArrayArg::from_raw_parts(out.0.clone(), out.1),
            count,
            bases.x,
            bases.y,
            max_recovered_dist2,
        );
    }
}

// ── FAST detection ───────────────────────────────────────────────────────────

/// The Bresenham ring's row and column offsets, biased by 3 so they are
/// unsigned: `ring[k]` is the row and `ring[16 + k]` the column of ring point
/// `k`, each plus 3. Every caller is at least 3 pixels inside the image, so the
/// bias cancels without a signed intermediate.
pub(super) const RING_BIAS: usize = 3;

/// kornia's `corner_score_9_scalar` (`fast.rs:838`) over every pixel.
///
/// The score is FAST-9's own: the maximum over the sixteen arc starts of the
/// minimum over nine consecutive saturating differences, on the bright and the
/// dark side. It is written for **every** pixel, corner or not, because the
/// in-block local-maximum filter below compares raw scores.
///
/// This is what makes a GPU detector exact rather than approximate. kornia's
/// candidate test at threshold `t` is `p > center + t` or `p < center - t` for
/// nine consecutive ring points, which is the same statement as
/// `corner_score_9 > t` — including the saturation, since a saturated bound
/// forces the corresponding difference under `t` too. So one dense score image
/// answers every rung of the detector's threshold ladder, and
/// `tests/fast_model.rs` checks that claim against kornia itself at five rungs
/// and two widths before any of this runs.
#[cube(launch, launch_unchecked)]
fn fast_score_kernel(
    frame: &Array<u16>,
    ring: &Array<u32>,
    score: &mut Array<u8>,
    width: usize,
    height: usize,
    margin: usize,
) {
    let x = usize::cast_from(ABSOLUTE_POS_X);
    let y = usize::cast_from(ABSOLUTE_POS_Y);
    if x >= width || y >= height {
        terminate!();
    }
    let slot = y * width + x;
    if x < margin || y < margin || x + margin >= width || y + margin >= height {
        score[slot] = 0u8;
        terminate!();
    }

    // `sub_ptr[x] = (sub_img_raw(x, y) >> 8)` (`keypoints.cpp:156`), on the
    // device: uploading the `u16` frame and shifting here costs 0.9 MB more over
    // the bus and saves a whole-frame pass on the host, which measured the
    // larger of the two.
    let center = u32::cast_from(frame[slot]) >> 8u32;
    // `#[unroll]` on all three loops below, and not for the sake of this card.
    // `dark` and `bright` are indexed dynamically (`dark[(k + i) % 16]`) from
    // loops with comptime bounds, so a rolled kernel cannot keep them in
    // registers: 16 x 4 B x 2 arrays x 256 units is 32 kB of local memory per
    // cube in the frame's hottest kernel, plus seventeen uncoalesced global
    // loads per pixel. Unrolled, every index is a literal and both arrays fit
    // in registers. On a 5090 at 7-11 % utilisation this changes nothing
    // measurable and nothing should be concluded from measuring it here; it is
    // for the shared-LPDDR targets the portable lane exists for, where
    // `Robocap.md`'s Pi 5 lesson is that a memory-bound kernel loses to the CPU
    // outright. The corner-scan equality tests on both lanes are what say it
    // changed no value.
    let mut dark = Array::<u32>::new(16usize);
    let mut bright = Array::<u32>::new(16usize);
    #[unroll]
    for k in 0..16usize {
        let row = y + usize::cast_from(ring[k]) - RING_BIAS;
        let column = x + usize::cast_from(ring[16usize + k]) - RING_BIAS;
        let p = u32::cast_from(frame[row * width + column]) >> 8u32;
        let mut below: u32 = 0u32;
        if center > p {
            below = center - p;
        }
        let mut above: u32 = 0u32;
        if p > center {
            above = p - center;
        }
        dark[k] = below;
        bright[k] = above;
    }

    // Step by 2: each iteration shares the seven-element core `[k+1 .. k+8]`
    // between the arc starting at `k` and the one starting at `k+1`, which is
    // the canonical FAST-9 structure and the order kornia sums it in.
    let mut dark_score: u32 = 0u32;
    let mut bright_score: u32 = 0u32;
    #[unroll]
    for step in 0..8usize {
        let k = 2usize * step;
        let mut core_dark = dark[(k + 1usize) % 16usize];
        let mut core_bright = bright[(k + 1usize) % 16usize];
        #[unroll]
        for i in 2..9usize {
            core_dark = min(core_dark, dark[(k + i) % 16usize]);
            core_bright = min(core_bright, bright[(k + i) % 16usize]);
        }
        dark_score = max(dark_score, min(core_dark, dark[k]));
        dark_score = max(dark_score, min(core_dark, dark[(k + 9usize) % 16usize]));
        bright_score = max(bright_score, min(core_bright, bright[k]));
        bright_score = max(
            bright_score,
            min(core_bright, bright[(k + 9usize) % 16usize]),
        );
    }
    score[slot] = u8::cast_from(max(dark_score, bright_score));
}

/// Lanes per block of kornia's local-maximum filter, off the CPU detector's own
/// constant so the kernel and the host cannot disagree about the block
/// alignment that decides the corner set.
const FILTER_LANES: usize = crate::frontend::detect::FAST_FILTER_LANES;
/// The last lane of a block, which has no right-hand neighbour to beat.
const FILTER_LAST: usize = FILTER_LANES - 1;

/// kornia's in-block local-maximum filter (`fast.rs:539-553`).
///
/// Turned on at `width >= 800`, where dense-corner images emit so many
/// candidates that `Vec::push` dominates the CPU kernel; kornia keeps a lane
/// only when its score beats its two neighbours **inside its own sixteen-lane
/// block**, with zero outside the block. The blocks are aligned to the image's
/// own left margin and the tail past the last whole block takes the scalar path,
/// which has no filter — so the alignment and the tail are part of the corner
/// set, not an implementation detail, and both are reproduced here.
///
/// The output is the score where a candidate survives and zero elsewhere, which
/// makes `kept > t` the exact candidate set for every threshold `t >= 0`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fast_localmax_kernel(
    score: &Array<u8>,
    kept: &mut Array<u8>,
    width: usize,
    height: usize,
    margin: usize,
    filtered_end: usize,
    use_filter: usize,
) {
    let x = usize::cast_from(ABSOLUTE_POS_X);
    let y = usize::cast_from(ABSOLUTE_POS_Y);
    if x >= width || y >= height {
        terminate!();
    }
    let slot = y * width + x;
    let value = u32::cast_from(score[slot]);
    let mut survivor = value;
    if use_filter == 1usize && x >= margin && x < filtered_end {
        let lane = (x - margin) % FILTER_LANES;
        let mut left: u32 = 0u32;
        if lane != 0usize {
            left = u32::cast_from(score[slot - 1usize]);
        }
        let mut right: u32 = 0u32;
        if lane != FILTER_LAST {
            right = u32::cast_from(score[slot + 1usize]);
        }
        if value <= left || value <= right {
            survivor = 0u32;
        }
    }
    kept[slot] = u8::cast_from(survivor);
}

/// The dense FAST-9 score image of one frame.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_fast_score<R: Runtime>(
    client: &ComputeClient<R>,
    frame: Buffer<'_>,
    ring: Buffer<'_>,
    score: Buffer<'_>,
    width: usize,
    height: usize,
    margin: usize,
) {
    let (cubes, units) = tile_2d(width, height);
    unsafe {
        fast_score_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(frame.0.clone(), frame.1),
            ArrayArg::from_raw_parts(ring.0.clone(), ring.1),
            ArrayArg::from_raw_parts(score.0.clone(), score.1),
            width,
            height,
            margin,
        );
    }
}

/// The candidate image: the score where the local-maximum filter keeps it.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_fast_localmax<R: Runtime>(
    client: &ComputeClient<R>,
    score: Buffer<'_>,
    kept: Buffer<'_>,
    width: usize,
    height: usize,
    margin: usize,
    filtered_end: usize,
    use_filter: bool,
) {
    let (cubes, units) = tile_2d(width, height);
    unsafe {
        fast_localmax_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(score.0.clone(), score.1),
            ArrayArg::from_raw_parts(kept.0.clone(), kept.1),
            width,
            height,
            margin,
            filtered_end,
            usize::from(use_filter),
        );
    }
}

/// Words of [`MASK_BITS`] columns each in one row of the candidate bitmask.
pub(super) const MASK_BITS: usize = 32;

/// One bit per column of the candidate image, packed thirty-two to a word.
///
/// The detector asks for up to eighty row bands per camera per frame and each
/// one is the whole image width, so walking the candidate image byte by byte
/// costs more than the FAST sweep it replaced: a byte-by-byte host walk measured
/// 1.53 ms against kornia's 1.81 ms, so the whole GPU scan was barely ahead of
/// the CPU one. With a bitmask the host reads one word per thirty-two columns
/// and only touches the score where a bit is set.
#[cube(launch, launch_unchecked)]
fn fast_mask_kernel(
    kept: &Array<u8>,
    mask: &mut Array<u32>,
    width: usize,
    height: usize,
    words: usize,
) {
    let word = usize::cast_from(ABSOLUTE_POS_X);
    let y = usize::cast_from(ABSOLUTE_POS_Y);
    if word >= words || y >= height {
        terminate!();
    }
    let first = word * MASK_BITS;
    let mut bits: u32 = 0u32;
    for bit in 0..MASK_BITS {
        let x = first + bit;
        if x < width && kept[y * width + x] != 0u8 {
            bits |= 1u32 << u32::cast_from(bit);
        }
    }
    mask[y * words + word] = bits;
}

/// Pack the candidate image into one bit per column.
pub(super) fn launch_fast_mask<R: Runtime>(
    client: &ComputeClient<R>,
    kept: Buffer<'_>,
    mask: Buffer<'_>,
    width: usize,
    height: usize,
    words: usize,
) {
    let (cubes, units) = tile_2d(words, height);
    unsafe {
        fast_mask_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(kept.0.clone(), kept.1),
            ArrayArg::from_raw_parts(mask.0.clone(), mask.1),
            width,
            height,
            words,
        );
    }
}
