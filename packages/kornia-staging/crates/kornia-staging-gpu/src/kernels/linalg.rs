//! The global BA's PCG kernels (DESIGN §7.3, PR e1) and their launchers.
//!
//! Every sum has one fixed order, so two runs on one device are bit-equal and
//! no kernel uses a float atomic:
//! - [`block_gemv_kernel`]: one unit per block row, `t = J x` over the block's
//!   columns in order;
//! - [`block_gemv_t_kernel`]: one unit per block column, `J^T t` over eight
//!   lanes then a pairwise sum, the CPU lane's `dot_lanes` order;
//! - [`pose_reduce_kernel`]: one unit per unknown, `lambda x` then every block
//!   that holds the pose in block order (the pose-major CSR reduction);
//! - the three PCG kernels run on one cube of [`PCG_UNITS`] units: unit `u`
//!   owns poses `u, u + 256, ...` (six entries each, in order), and the cube
//!   sums its partials in a fixed tree. The scalar logic of
//!   `kornia_staging_slam::sqrt_ba::pcg` runs on unit 0; once it stops, a flag
//!   turns every later kernel into a no-op, so the host reads the state only
//!   every `GBA_CG_CHECK_EVERY` iterations.
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

/// Sum a power-of-two shared array into slot 0 in a fixed tree, with a
/// cube-uniform barrier per step.
#[cube]
fn sum_tree(shared: &mut Shared<[f32]>, unit: usize, #[comptime] slots: usize) {
    sync_cube();
    #[unroll]
    for step in 0..comptime!(slots.ilog2() as usize) {
        let stride = slots >> (step + 1usize);
        if unit < stride {
            shared[unit] += shared[unit + stride];
        }
        sync_cube();
    }
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
) {
    if ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let unit = usize::cast_from(UNIT_POS);
    let mut rz = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut rr = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut beta_shared = Shared::<[f32]>::new_slice(1usize);
    let mut stopped = Shared::<[u32]>::new_slice(1usize);
    let mut acc_rz = 0.0f32;
    let mut acc_rr = 0.0f32;
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
            acc_rz += ri * zi;
            acc_rr += ri * ri;
        }
    }
    rz[unit] = acc_rz;
    rr[unit] = acc_rr;
    sum_tree(&mut rz, unit, PCG_UNITS);
    sum_tree(&mut rr, unit, PCG_UNITS);
    if unit == 0usize {
        let rho = rz[0];
        let last_rho = fstate[RHO];
        fstate[RHO] = rho;
        let mut termination = RUNNING.runtime();
        let mut beta = 0.0f32;
        if rho == 0.0f32 && rr[0] == 0.0f32 {
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

/// The `Q` test on unit 0 after `x . b` and `x . r` are in slot 0.
#[cube]
#[allow(clippy::too_many_arguments)]
fn model_test(
    xb: &Shared<[f32]>,
    xr: &Shared<[f32]>,
    fstate: &mut [f32],
    ustate: &mut [u32],
    unit: usize,
    iteration: u32,
    max_iterations: u32,
    eta: f32,
) {
    if unit == 0usize {
        let q1 = -(xb[0] + xr[0]);
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

/// PCG steps 4-6 after `q = A p`: `alpha = rho / p . q`, `x += alpha p`, and
/// unless this iteration replaces the residual, `r -= alpha q` and the `Q` test.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pcg_step_kernel(
    p: &[f32],
    q: &[f32],
    x: &mut [f32],
    r: &mut [f32],
    b: &[f32],
    fstate: &mut [f32],
    ustate: &mut [u32],
    poses: usize,
    iteration: u32,
    replace: u32,
    max_iterations: u32,
    eta: f32,
) {
    if ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let unit = usize::cast_from(UNIT_POS);
    let mut pq = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut xb = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut xr = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut alpha_shared = Shared::<[f32]>::new_slice(1usize);
    let mut stopped = Shared::<[u32]>::new_slice(1usize);
    let mut acc = 0.0f32;
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            acc += p[6usize * k + i] * q[6usize * k + i];
        }
    }
    pq[unit] = acc;
    sum_tree(&mut pq, unit, PCG_UNITS);
    if unit == 0usize {
        let value = pq[0];
        let mut termination = RUNNING.runtime();
        let mut alpha = 0.0f32;
        if value <= 0.0f32 || is_infinite(value) {
            termination = INDEFINITE;
        } else {
            alpha = fstate[RHO] / value;
            if !is_finite(alpha) {
                termination = NUMERICAL_FAILURE;
            }
        }
        if termination != RUNNING {
            stop(ustate, termination, iteration);
        }
        stopped[0] = termination;
        alpha_shared[0] = alpha;
    }
    sync_cube();
    if stopped[0] != RUNNING {
        terminate!();
    }
    let alpha = alpha_shared[0];
    let mut acc_xb = 0.0f32;
    let mut acc_xr = 0.0f32;
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            let n = 6usize * k + i;
            let xn = x[n] + alpha * p[n];
            x[n] = xn;
            if replace == 0u32 {
                let rn = r[n] - alpha * q[n];
                r[n] = rn;
                acc_xb += xn * b[n];
                acc_xr += xn * rn;
            }
        }
    }
    if replace == 0u32 {
        xb[unit] = acc_xb;
        xr[unit] = acc_xr;
        sum_tree(&mut xb, unit, PCG_UNITS);
        sum_tree(&mut xr, unit, PCG_UNITS);
        model_test(
            &xb,
            &xr,
            fstate,
            ustate,
            unit,
            iteration,
            max_iterations,
            eta,
        );
    }
}

/// Residual replacement after `q = A x`: `r = b - q`, then the `Q` test.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pcg_replace_kernel(
    q: &[f32],
    x: &[f32],
    r: &mut [f32],
    b: &[f32],
    fstate: &mut [f32],
    ustate: &mut [u32],
    poses: usize,
    iteration: u32,
    max_iterations: u32,
    eta: f32,
) {
    if ustate[STOPPED] != 0u32 {
        terminate!();
    }
    let unit = usize::cast_from(UNIT_POS);
    let mut xb = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut xr = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut acc_xb = 0.0f32;
    let mut acc_xr = 0.0f32;
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            let n = 6usize * k + i;
            let rn = b[n] - q[n];
            r[n] = rn;
            acc_xb += x[n] * b[n];
            acc_xr += x[n] * rn;
        }
    }
    xb[unit] = acc_xb;
    xr[unit] = acc_xr;
    sum_tree(&mut xb, unit, PCG_UNITS);
    sum_tree(&mut xr, unit, PCG_UNITS);
    model_test(
        &xb,
        &xr,
        fstate,
        ustate,
        unit,
        iteration,
        max_iterations,
        eta,
    );
}

/// `out[0] = a . b` over `poses` six-entry groups, in the PCG kernels' order.
#[cube(launch_unchecked)]
fn dot_kernel(a: &[f32], b: &[f32], out: &mut [f32], poses: usize) {
    let unit = usize::cast_from(UNIT_POS);
    let mut partial = Shared::<[f32]>::new_slice(PCG_UNITS);
    let mut acc = 0.0f32;
    for k in range_stepped(unit, poses, PCG_UNITS) {
        #[unroll]
        for i in 0..6usize {
            acc += a[6usize * k + i] * b[6usize * k + i];
        }
    }
    partial[unit] = acc;
    sum_tree(&mut partial, unit, PCG_UNITS);
    if unit == 0usize {
        out[0] = partial[0];
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
        );
    }
}

pub(crate) fn launch_pcg_step<R: Runtime>(
    client: &ComputeClient<R>,
    v: &PcgBuffers<'_>,
    iteration: u32,
    replace: bool,
    max_iterations: u32,
    eta: f32,
) {
    let (cubes, units) = single_cube();
    // SAFETY: as `launch_pcg_precondition`.
    unsafe {
        pcg_step_kernel::launch_unchecked::<R>(
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
            u32::from(replace),
            max_iterations,
            eta,
        );
    }
}

pub(crate) fn launch_pcg_replace<R: Runtime>(
    client: &ComputeClient<R>,
    v: &PcgBuffers<'_>,
    iteration: u32,
    max_iterations: u32,
    eta: f32,
) {
    let (cubes, units) = single_cube();
    // SAFETY: as `launch_pcg_precondition`.
    unsafe {
        pcg_replace_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            arg(v.q),
            arg(v.x),
            arg(v.r),
            arg(v.b),
            arg(v.fstate),
            arg(v.ustate),
            v.x.1 / 6,
            iteration,
            max_iterations,
            eta,
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
        dot_kernel::launch_unchecked::<R>(client, cubes, units, arg(a), arg(b), arg(out), a.1 / 6);
    }
}
