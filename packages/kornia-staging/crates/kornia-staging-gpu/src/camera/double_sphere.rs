//! Double sphere projection, Jacobian and closed-form inverse on the portable device.
//!
//! Packed parameters: `fx fy cx cy xi alpha w2`, with `w2` the validated CPU model's
//! domain bound (`DoubleSphere::domain_bound`). Usenko et al., "The Double Sphere
//! Camera Model", 3DV 2018, eqs. 40-47.
#![allow(missing_docs)]
use super::brown8::{BELOW_DEPTH, NON_FINITE, OUTSIDE_DOMAIN, VALID};
use crate::kernels::finite::is_finite;
use cubecl::prelude::*;

/// Packed f32 count.
pub const PARAMETERS: usize = 7;

#[cube]
fn project_status(
    params: &[f32],
    offset: usize,
    x: f32,
    y: f32,
    z: f32,
    out: &mut Array<f32>,
    #[comptime] jacobian: bool,
) -> u32 {
    let mut status = u32::cast_from(VALID);
    let r2 = x * x + y * y;
    let d1 = f32::sqrt(r2 + z * z);
    if !is_finite(x) || !is_finite(y) || !is_finite(z) || !is_finite(d1) {
        status = NON_FINITE;
    } else if d1 == 0.0f32 {
        status = BELOW_DEPTH;
    } else if z <= -params[offset + 6usize] * d1 {
        status = OUTSIDE_DOMAIN;
    } else {
        let fx = params[offset];
        let fy = params[offset + 1usize];
        let xi = params[offset + 4usize];
        let alpha = params[offset + 5usize];
        let k = xi * d1 + z;
        let d2 = f32::sqrt(r2 + k * k);
        let n = alpha * d2 + (1.0f32 - alpha) * k;
        out[0usize] = fx * (x / n) + params[offset + 2usize];
        out[1usize] = fy * (y / n) + params[offset + 3usize];
        if jacobian {
            let n2 = n * n;
            let dn = (xi * (1.0f32 - alpha) / d1 + alpha * (xi * k / d1 + 1.0f32) / d2) / n2;
            let tt2 = xi * z / d1 + 1.0f32;
            let dz = ((1.0f32 - alpha) * tt2 + alpha * k * tt2 / d2) / n2;
            out[2usize] = fx * (1.0f32 / n - x * x * dn);
            out[3usize] = -fx * x * y * dn;
            out[4usize] = -fx * x * dz;
            out[5usize] = -fy * x * y * dn;
            out[6usize] = fy * (1.0f32 / n - y * y * dn);
            out[7usize] = -fy * y * dz;
        }
        if !is_finite(out[0usize]) || !is_finite(out[1usize]) {
            status = NON_FINITE;
        }
    }
    if status != VALID {
        out[0usize] = 0.0f32;
        out[1usize] = 0.0f32;
    }
    status
}

/// Project one point with the CPU `project_with_status` rules.
///
/// Returns 0 valid, 1 non-finite, 2 at the origin, or 3 outside `z > -w2 * |p|`.
/// `out` holds at least two entries, zero on rejection.
#[cube]
pub fn project(params: &[f32], offset: usize, x: f32, y: f32, z: f32, out: &mut Array<f32>) -> u32 {
    project_status(params, offset, x, y, z, out, false)
}

/// Project and write the 2x3 point Jacobian row-major into `out[2..8]`.
#[cube]
pub fn project_with_point_jacobian(
    params: &[f32],
    offset: usize,
    x: f32,
    y: f32,
    z: f32,
    out: &mut Array<f32>,
) -> u32 {
    project_status(params, offset, x, y, z, out, true)
}

/// Invert one pixel to a unit bearing in closed form.
///
/// Returns 0 valid, 1 non-finite, or 3 outside the image disc or the forward domain.
#[cube]
pub fn unproject(params: &[f32], offset: usize, px: f32, py: f32, out: &mut Array<f32>) -> u32 {
    out[0usize] = 0.0f32;
    out[1usize] = 0.0f32;
    out[2usize] = 0.0f32;
    let mut status = u32::cast_from(NON_FINITE);
    let mx = (px - params[offset + 2usize]) / params[offset];
    let my = (py - params[offset + 3usize]) / params[offset + 1usize];
    if is_finite(px) && is_finite(py) && is_finite(mx) && is_finite(my) {
        let xi = params[offset + 4usize];
        let alpha = params[offset + 5usize];
        let r2 = mx * mx + my * my;
        let two_alpha = alpha + alpha - 1.0f32;
        if alpha > 0.5f32 && r2 * two_alpha >= 1.0f32 {
            status = OUTSIDE_DOMAIN;
        } else {
            let mz = (1.0f32 - alpha * alpha * r2)
                / (alpha * f32::sqrt(1.0f32 - two_alpha * r2) + 1.0f32 - alpha);
            let k = (mz * xi + f32::sqrt(mz * mz + (1.0f32 - xi * xi) * r2)) / (mz * mz + r2);
            let rx = k * mx;
            let ry = k * my;
            let rz = k * mz - xi;
            if !is_finite(rx) || !is_finite(ry) || !is_finite(rz) {
                status = NON_FINITE;
            } else if rz <= -params[offset + 6usize] {
                status = OUTSIDE_DOMAIN;
            } else {
                status = VALID;
                out[0usize] = rx;
                out[1usize] = ry;
                out[2usize] = rz;
            }
        }
    }
    status
}
