//! The global BA's PCG kernels and their launchers.
//!
//! Every sum has one fixed order, so two runs on one device are bit-equal and
//! no kernel uses a float atomic:
//! - [`block_gemv_kernel`]: one unit per block row, `t = J x` over the block's
//!   columns in order;
//! - [`block_gemv_t_kernel`]: one unit per block column, `J^T t` over eight
//!   lanes then a pairwise sum, the CPU lane's `dot_lanes` order;
//! - [`pose_reduce_kernel`]: one unit per unknown, `lambda x` then every block
//!   that holds the pose in block order (the pose-major CSR reduction);
//! - the PCG kernels run on one cube of [`PCG_UNITS`] units: unit `u` owns
//!   poses `u, u + 256, ...` (six entries each, in order). Every PCG scalar
//!   (`r . z`, `r . r`, `p . q`, `x . b`, `x . r`) is a compensated f32 sum:
//!   a Neumaier accumulator per unit, then a fixed tree of
//!   TwoSums that carries the rounding errors beside the partials. The scalar
//!   logic of `kornia_staging_slam::sqrt_ba::pcg` runs on unit 0; once it
//!   stops, a flag turns every later kernel into a no-op, so the host reads
//!   the state only every `GBA_CG_CHECK_EVERY` iterations.
use cubecl::prelude::*;

use super::finite::is_finite;
use super::layout::{linear_1d, Buffer};

/// Units of the single-cube PCG kernels; the fixed reduction width.
pub(crate) const PCG_UNITS: usize = 256;
/// `ustate[1]` while the PCG runs.
pub(crate) const RUNNING: u32 = 0;
/// The `Q` criterion held, or the residual is exactly zero.
pub(crate) const CONVERGED: u32 = 1;
/// `p' A p <= 0` or infinite.
pub(crate) const INDEFINITE: u32 = 2;
/// `rho`, `beta` or `alpha` zero or not finite.
pub(crate) const NUMERICAL_FAILURE: u32 = 3;
/// The iteration cap.
pub(crate) const MAX_ITERATIONS: u32 = 4;
/// `fstate` slots.
pub(crate) const RHO: usize = 0;
pub(crate) const Q0: usize = 1;
/// `ustate` slots: stopped flag, termination, iterations.
pub(crate) const STOPPED: usize = 0;
pub(crate) const TERMINATION: usize = 1;
pub(crate) const ITERATIONS: usize = 2;

#[cube]
fn is_infinite(value: f32) -> bool {
    (u32::reinterpret(value) & 0x7fffffffu32) == 0x7f800000u32
}

/// `value`, hidden from the shader compiler: `hide` is a runtime zero, so
/// the compiler cannot prove an identity between `value` and the operands it
/// came from. GPU compilers (SPIR-V drivers, Metal by default) reassociate
/// float algebra and would fold the error terms below to zero.
#[cube]
fn opaque(value: f32, hide: u32) -> f32 {
    f32::reinterpret(u32::reinterpret(value) ^ hide)
}

/// Add `value` to unit `unit`'s compensated sum: Neumaier's variant of
/// Kahan summation, `hi` the running sum and `lo` the rounding it dropped.
#[cube]
fn neumaier_add(
    hi: &mut Shared<[f32]>,
    lo: &mut Shared<[f32]>,
    unit: usize,
    value: f32,
    hide: u32,
) {
    let sum = hi[unit];
    let t = opaque(sum + value, hide);
    if f32::abs(sum) >= f32::abs(value) {
        lo[unit] += opaque(sum - t, hide) + value;
    } else {
        lo[unit] += opaque(value - t, hide) + sum;
    }
    hi[unit] = t;
}

/// Unit `unit`'s share of `a . b`, compensated: the six entries of groups
/// `unit, unit + 256, ...` in order.
#[cube]
fn unit_dot(
    a: &[f32],
    b: &[f32],
    hi: &mut Shared<[f32]>,
    lo: &mut Shared<[f32]>,
    unit: usize,
    poses: usize,
    hide: u32,
) {
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            let n = 6usize * k + i;
            neumaier_add(hi, lo, unit, a[n] * b[n], hide);
        }
    }
}

/// Add the units' compensated sums in a fixed tree of error-free TwoSums
/// (Knuth), the errors carried in `lo`, with a cube-uniform barrier per step.
/// Every unit gets the total `hi[0] + lo[0]`.
#[cube]
fn compensated_tree(hi: &mut Shared<[f32]>, lo: &mut Shared<[f32]>, unit: usize, hide: u32) -> f32 {
    sync_cube();
    #[unroll]
    for step in 0..comptime!(PCG_UNITS.ilog2() as usize) {
        let stride = PCG_UNITS >> (step + 1usize);
        if unit < stride {
            let a = hi[unit];
            let b = hi[unit + stride];
            let s = opaque(a + b, hide);
            let bb = opaque(s - a, hide);
            hi[unit] = s;
            lo[unit] += lo[unit + stride] + ((a - opaque(s - bb, hide)) + (b - bb));
        }
        sync_cube();
    }
    hi[0] + lo[0]
}

/// Unit 0 records why the PCG stopped.
#[cube]
fn stop(ustate: &mut [u32], termination: u32, iteration: u32) {
    ustate[STOPPED] = 1u32;
    ustate[TERMINATION] = termination;
    ustate[ITERATIONS] = iteration;
}

/// `t = J x` for one block row.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn block_gemv_kernel(
    data: &[f32],
    row_meta: &[u32],
    block_meta: &[u32],
    slot_kf: &[u32],
    x: &[f32],
    t: &mut [f32],
    ustate: &[u32],
    rows: usize,
) {
    let g = usize::cast_from(ABSOLUTE_POS);
    if g >= rows || ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let blk = usize::cast_from(row_meta[2usize * g]) * 5usize;
    let r = usize::cast_from(row_meta[2usize * g + 1usize]);
    let offset = usize::cast_from(block_meta[blk]);
    let m = usize::cast_from(block_meta[blk + 1usize]);
    let t_offset = usize::cast_from(block_meta[blk + 2usize]);
    let slot0 = usize::cast_from(block_meta[blk + 3usize]);
    let k = usize::cast_from(block_meta[blk + 4usize]);
    let mut acc = 0.0f32;
    for s in 0..k {
        let base = 6usize * usize::cast_from(slot_kf[slot0 + s]);
        let column = offset + 6usize * s * m + r;
        #[unroll]
        for j in 0..6usize {
            acc += data[column + j * m] * x[base + j];
        }
    }
    t[t_offset + r] = acc;
}

/// `contrib = J^T t` for one block column: eight lanes, then a pairwise sum
/// and the tail.
#[cube(launch_unchecked)]
fn block_gemv_t_kernel(
    data: &[f32],
    col_meta: &[u32],
    block_meta: &[u32],
    t: &[f32],
    contrib: &mut [f32],
    ustate: &[u32],
    cols: usize,
) {
    let g = usize::cast_from(ABSOLUTE_POS);
    if g >= cols || ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let blk = usize::cast_from(col_meta[2usize * g]) * 5usize;
    let c = usize::cast_from(col_meta[2usize * g + 1usize]);
    let m = usize::cast_from(block_meta[blk + 1usize]);
    let column = usize::cast_from(block_meta[blk]) + c * m;
    let t_offset = usize::cast_from(block_meta[blk + 2usize]);
    let slot0 = usize::cast_from(block_meta[blk + 3usize]);
    let full = m - m % 8usize;
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut a4 = 0.0f32;
    let mut a5 = 0.0f32;
    let mut a6 = 0.0f32;
    let mut a7 = 0.0f32;
    for i in range_stepped(0usize, full, 8usize) {
        let d = column + i;
        let e = t_offset + i;
        a0 += data[d] * t[e];
        a1 += data[d + 1usize] * t[e + 1usize];
        a2 += data[d + 2usize] * t[e + 2usize];
        a3 += data[d + 3usize] * t[e + 3usize];
        a4 += data[d + 4usize] * t[e + 4usize];
        a5 += data[d + 5usize] * t[e + 5usize];
        a6 += data[d + 6usize] * t[e + 6usize];
        a7 += data[d + 7usize] * t[e + 7usize];
    }
    let mut sum = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));
    for i in full..m {
        sum += data[column + i] * t[t_offset + i];
    }
    contrib[6usize * slot0 + c] = sum;
}

/// `y = lambda x + sum of the block contributions`, pose-major.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pose_reduce_kernel(
    contrib: &[f32],
    pose_ptr: &[u32],
    pose_slots: &[u32],
    x: &[f32],
    y: &mut [f32],
    ustate: &[u32],
    lambda: f32,
    unknowns: usize,
) {
    let i = usize::cast_from(ABSOLUTE_POS);
    if i >= unknowns || ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let k = i / 6usize;
    let j = i % 6usize;
    let mut acc = lambda * x[i];
    let end = usize::cast_from(pose_ptr[k + 1usize]);
    for e in usize::cast_from(pose_ptr[k])..end {
        acc += contrib[6usize * usize::cast_from(pose_slots[e]) + j];
    }
    y[i] = acc;
}

/// PCG steps 1-3: `z = M^-1 r`, `rho = r . z`, the `rho`/`beta` checks, and
/// `p = z + beta p` (`p = z` on the first iteration).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pcg_precondition_kernel(
    minv: &[f32],
    r: &[f32],
    z: &mut [f32],
    p: &mut [f32],
    fstate: &mut [f32],
    ustate: &mut [u32],
    poses: usize,
    iteration: u32,
    hide: u32,
) {
    if ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let unit = usize::cast_from(UNIT_POS);
    let mut rz_hi = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut rz_lo = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut rr_hi = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut rr_lo = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut beta_shared = Shared::<[f32]>::new_slice(1usize);
    let mut stopped = Shared::<[u32]>::new_slice(1usize);
    rz_hi[unit] = 0.0f32;
    rz_lo[unit] = 0.0f32;
    rr_hi[unit] = 0.0f32;
    rr_lo[unit] = 0.0f32;
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            let mut zi = 0.0f32;
            #[unroll]
            for j in 0..6usize {
                zi += minv[36usize * k + 6usize * i + j] * r[6usize * k + j];
            }
            z[6usize * k + i] = zi;
            let ri = r[6usize * k + i];
            neumaier_add(&mut rz_hi, &mut rz_lo, unit, ri * zi, hide);
            neumaier_add(&mut rr_hi, &mut rr_lo, unit, ri * ri, hide);
        }
    }
    let rho = compensated_tree(&mut rz_hi, &mut rz_lo, unit, hide);
    let rr = compensated_tree(&mut rr_hi, &mut rr_lo, unit, hide);
    if unit == 0usize {
        let last_rho = fstate[RHO];
        fstate[RHO] = rho;
        let mut termination = RUNNING.runtime();
        let mut beta = 0.0f32;
        if rho == 0.0f32 && rr == 0.0f32 {
            termination = CONVERGED;
        } else if rho == 0.0f32 || !is_finite(rho) {
            termination = NUMERICAL_FAILURE;
        } else if iteration > 1u32 {
            beta = rho / last_rho;
            if beta == 0.0f32 || !is_finite(beta) {
                termination = NUMERICAL_FAILURE;
            }
        }
        if termination != RUNNING {
            stop(ustate, termination, iteration);
        }
        stopped[0] = termination;
        beta_shared[0] = beta;
    }
    sync_cube();
    if stopped[0] != RUNNING {
        terminate!();
    }
    let beta = beta_shared[0];
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            let n = 6usize * k + i;
            if iteration == 1u32 {
                p[n] = z[n];
            } else {
                p[n] = z[n] + beta * p[n];
            }
        }
    }
}

/// [`pcg_update_kernel`] after `q = A p`: `x += alpha p`, `r -= alpha q`, the `Q` test.
pub(crate) const STEP: u32 = 0;
/// After `q = A p` on an iteration that replaces the residual: `x += alpha p` only.
pub(crate) const ADVANCE: u32 = 1;
/// After `q = A x`: `r = b - q`, the `Q` test.
pub(crate) const REPLACE: u32 = 2;

/// PCG steps 4-6 in one of three modes ([`STEP`], [`ADVANCE`], [`REPLACE`]).
/// `alpha = rho / p . q` and its checks precede the `x` update; the `Q` test
/// `i (Q_i - Q_{i-1}) / Q_i < eta` with `Q_i = -(x . b + x . r)` follows `r`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pcg_update_kernel(
    p: &[f32],
    q: &[f32],
    x: &mut [f32],
    r: &mut [f32],
    b: &[f32],
    fstate: &mut [f32],
    ustate: &mut [u32],
    poses: usize,
    iteration: u32,
    mode: u32,
    max_iterations: u32,
    eta: f32,
    hide: u32,
) {
    if ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let unit = usize::cast_from(UNIT_POS);
    let mut xb_hi = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut xb_lo = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut xr_hi = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut xr_lo = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut alpha_shared = Shared::<[f32]>::new_slice(1usize);
    let mut stopped = Shared::<[u32]>::new_slice(1usize);
    let mut alpha = 0.0f32;
    if mode != REPLACE {
        xb_hi[unit] = 0.0f32;
        xb_lo[unit] = 0.0f32;
        unit_dot(p, q, &mut xb_hi, &mut xb_lo, unit, poses, hide);
        let pq = compensated_tree(&mut xb_hi, &mut xb_lo, unit, hide);
        if unit == 0usize {
            let mut termination = RUNNING.runtime();
            let mut value = 0.0f32;
            if pq <= 0.0f32 || is_infinite(pq) {
                termination = INDEFINITE;
            } else {
                value = fstate[RHO] / pq;
                if !is_finite(value) {
                    termination = NUMERICAL_FAILURE;
                }
            }
            if termination != RUNNING {
                stop(ustate, termination, iteration);
            }
            stopped[0] = termination;
            alpha_shared[0] = value;
        }
        sync_cube();
        if stopped[0] != RUNNING {
            terminate!();
        }
        alpha = alpha_shared[0];
    }
    xb_hi[unit] = 0.0f32;
    xb_lo[unit] = 0.0f32;
    xr_hi[unit] = 0.0f32;
    xr_lo[unit] = 0.0f32;
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            let n = 6usize * k + i;
            let mut xn = x[n];
            let mut rn = r[n];
            if mode == REPLACE {
                rn = b[n] - q[n];
            } else {
                xn += alpha * p[n];
                x[n] = xn;
                rn -= alpha * q[n];
            }
            if mode != ADVANCE {
                r[n] = rn;
                neumaier_add(&mut xb_hi, &mut xb_lo, unit, xn * b[n], hide);
                neumaier_add(&mut xr_hi, &mut xr_lo, unit, xn * rn, hide);
            }
        }
    }
    if mode != ADVANCE {
        let xb = compensated_tree(&mut xb_hi, &mut xb_lo, unit, hide);
        let xr = compensated_tree(&mut xr_hi, &mut xr_lo, unit, hide);
        if unit == 0usize {
            let q1 = -(xb + xr);
            let zeta = f32::cast_from(iteration) * (q1 - fstate[Q0]) / q1;
            if zeta < eta {
                stop(ustate, CONVERGED, iteration);
            } else {
                fstate[Q0] = q1;
                if iteration >= max_iterations {
                    stop(ustate, MAX_ITERATIONS, iteration);
                }
            }
        }
    }
}

/// `out[0] = a . b` over `poses` six-entry groups, in the PCG kernels' order.
#[cube(launch_unchecked)]
fn dot_kernel(a: &[f32], b: &[f32], out: &mut [f32], poses: usize, hide: u32) {
    let unit = usize::cast_from(UNIT_POS);
    let mut hi = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut lo = Shared::<[f32]>::new_slice(PCG_UNITS);
    hi[unit] = 0.0f32;
    lo[unit] = 0.0f32;
    unit_dot(a, b, &mut hi, &mut lo, unit, poses, hide);
    let total = compensated_tree(&mut hi, &mut lo, unit, hide);
    if unit == 0usize {
        out[0] = total;
    }
}

fn arg<R: Runtime>(buffer: Buffer<'_>) -> BufferArg<R> {
    // SAFETY: each launcher's caller sized the handle for `buffer.1` elements.
    unsafe { BufferArg::from_raw_parts(buffer.0.clone(), buffer.1) }
}

fn single_cube() -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(PCG_UNITS as u32),
    )
}

/// One page of blocks: `t` rows, then `J^T t` columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_block_gemv<R: Runtime>(
    client: &ComputeClient<R>,
    data: Buffer<'_>,
    row_meta: Buffer<'_>,
    col_meta: Buffer<'_>,
    block_meta: Buffer<'_>,
    slot_kf: Buffer<'_>,
    x: Buffer<'_>,
    t: Buffer<'_>,
    contrib: Buffer<'_>,
    ustate: Buffer<'_>,
) {
    let rows = row_meta.1 / 2;
    let cols = col_meta.1 / 2;
    let (cubes, units) = linear_1d(rows);
    // SAFETY: the metadata indexes only inside `data`, `slot_kf`, `x` and `t`
    // (built and checked together by `linalg::Layout`).
    unsafe {
        block_gemv_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(data),
            arg(row_meta),
            arg(block_meta),
            arg(slot_kf),
            arg(x),
            arg(t),
            arg(ustate),
            rows,
        );
    }
    let (cubes, units) = linear_1d(cols);
    // SAFETY: as above, for `contrib`.
    unsafe {
        block_gemv_t_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(data),
            arg(col_meta),
            arg(block_meta),
            arg(t),
            arg(contrib),
            arg(ustate),
            cols,
        );
    }
}

/// `y = lambda x + contributions`, one unit per unknown.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_pose_reduce<R: Runtime>(
    client: &ComputeClient<R>,
    contrib: Buffer<'_>,
    pose_ptr: Buffer<'_>,
    pose_slots: Buffer<'_>,
    x: Buffer<'_>,
    y: Buffer<'_>,
    ustate: Buffer<'_>,
    lambda: f32,
) {
    let unknowns = x.1;
    let (cubes, units) = linear_1d(unknowns);
    // SAFETY: `pose_ptr` has `unknowns / 6 + 1` ascending entries into `pose_slots`,
    // whose slots index `contrib` in groups of six.
    unsafe {
        pose_reduce_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(contrib),
            arg(pose_ptr),
            arg(pose_slots),
            arg(x),
            arg(y),
            arg(ustate),
            lambda,
            unknowns,
        );
    }
}

/// The device vectors of one PCG solve, all `6 * poses` long except the
/// `36 * poses` inverses and the two state buffers.
pub(crate) struct PcgBuffers<'a> {
    pub minv: Buffer<'a>,
    pub b: Buffer<'a>,
    pub x: Buffer<'a>,
    pub r: Buffer<'a>,
    pub z: Buffer<'a>,
    pub p: Buffer<'a>,
    pub q: Buffer<'a>,
    pub fstate: Buffer<'a>,
    pub ustate: Buffer<'a>,
}

pub(crate) fn launch_pcg_precondition<R: Runtime>(
    client: &ComputeClient<R>,
    v: &PcgBuffers<'_>,
    iteration: u32,
) {
    let (cubes, units) = single_cube();
    // SAFETY: every vector holds `6 * poses` entries and `minv` `36 * poses`.
    unsafe {
        pcg_precondition_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(v.minv),
            arg(v.r),
            arg(v.z),
            arg(v.p),
            arg(v.fstate),
            arg(v.ustate),
            v.x.1 / 6,
            iteration,
            0u32,
        );
    }
}

pub(crate) fn launch_pcg_update<R: Runtime>(
    client: &ComputeClient<R>,
    v: &PcgBuffers<'_>,
    iteration: u32,
    mode: u32,
    max_iterations: u32,
    eta: f32,
) {
    let (cubes, units) = single_cube();
    // SAFETY: as `launch_pcg_precondition`.
    unsafe {
        pcg_update_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(v.p),
            arg(v.q),
            arg(v.x),
            arg(v.r),
            arg(v.b),
            arg(v.fstate),
            arg(v.ustate),
            v.x.1 / 6,
            iteration,
            mode,
            max_iterations,
            eta,
            0u32,
        );
    }
}

/// `out[0] = a . b` in the PCG kernels' order.
pub(crate) fn launch_dot<R: Runtime>(
    client: &ComputeClient<R>,
    a: Buffer<'_>,
    b: Buffer<'_>,
    out: Buffer<'_>,
) {
    let (cubes, units) = single_cube();
    // SAFETY: `a` and `b` hold the same multiple of six entries; `out` one.
    unsafe {
        dot_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(a),
            arg(b),
            arg(out),
            a.1 / 6,
            0u32,
        );
    }
}
