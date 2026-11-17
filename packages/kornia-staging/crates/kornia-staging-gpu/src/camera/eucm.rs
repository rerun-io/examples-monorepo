//! Enhanced unified camera (EUCM) projection, Jacobian and closed-form inverse on the portable device.
//!
//! Packed parameters: `fx fy cx cy alpha beta w`, with `w` the validated CPU model's
//! domain bound (`Eucm::domain_bound`). Khomutenko et al., "An Enhanced Unified
//! Camera Model", RA-L 2016.
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
    let alpha = params[offset + 4usize];
    let beta = params[offset + 5usize];
    let rho = f32::sqrt(beta * (x * x + y * y) + z * z);
    if !is_finite(x) || !is_finite(y) || !is_finite(z) || !is_finite(rho) {
        status = NON_FINITE;
    } else if rho == 0.0f32 {
        status = BELOW_DEPTH;
    } else if z <= -params[offset + 6usize] * rho {
        status = OUTSIDE_DOMAIN;
    } else {
        let fx = params[offset];
        let fy = params[offset + 1usize];
        let n = alpha * rho + (1.0f32 - alpha) * z;
        out[0usize] = fx * (x / n) + params[offset + 2usize];
        out[1usize] = fy * (y / n) + params[offset + 3usize];
        if jacobian {
            let denom = n * n * rho;
            let ab = alpha * beta;
            let add = n * rho;
            let dz = alpha * z + (1.0f32 - alpha) * rho;
            out[2usize] = fx * (add - ab * x * x) / denom;
            out[3usize] = -fx * ab * x * y / denom;
            out[4usize] = -fx * x * dz / denom;
            out[5usize] = -fy * ab * x * y / denom;
            out[6usize] = fy * (add - ab * y * y) / denom;
            out[7usize] = -fy * y * dz / denom;
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
/// Returns 0 valid, 1 non-finite, 2 at the origin, or 3 outside `z > -w * rho`.
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
        let alpha = params[offset + 4usize];
        let beta = params[offset + 5usize];
        let r2 = mx * mx + my * my;
        let bound = (alpha + alpha - 1.0f32) * beta;
        if alpha > 0.5f32 && r2 * bound >= 1.0f32 {
            status = OUTSIDE_DOMAIN;
        } else {
            let mz = (1.0f32 - alpha * alpha * beta * r2)
                / (alpha * f32::sqrt(1.0f32 - bound * r2) + 1.0f32 - alpha);
            let norm = f32::sqrt(r2 + mz * mz);
            let rx = mx / norm;
            let ry = my / norm;
            let rz = mz / norm;
            let rho = f32::sqrt(beta * (rx * rx + ry * ry) + rz * rz);
            if !is_finite(rx) || !is_finite(ry) || !is_finite(rz) {
                status = NON_FINITE;
            } else if rz <= -params[offset + 6usize] * rho {
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
