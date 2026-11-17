//! Equirectangular (longitude-latitude) projection, Jacobian and inverse on the portable device.
//!
//! Packed parameters: `fx fy cx cy lon_max`. Camera frame RDF; `u = fx atan2(x, z) + cx`,
//! `v = fy atan2(y, hypot(x, z)) + cy`. The pole test uses the CPU f32 model's epsilon
//! (`Scalar::sophus_epsilon_sqrt`), folded at compile time.
#![allow(missing_docs)]
use super::brown8::{BELOW_DEPTH, NON_FINITE, OUTSIDE_DOMAIN, VALID};
use crate::kernels::{finite::is_finite, trig};
use cubecl::prelude::*;
use kornia_staging_algebra::Scalar;

/// Packed f32 count.
pub const PARAMETERS: usize = 5;

/// Overflow-safe `hypot` without a native one.
#[cube]
fn hypot(a: f32, b: f32) -> f32 {
    let scale = f32::max(f32::abs(a), f32::abs(b));
    let mut result = 0.0f32;
    if scale > 0.0f32 {
        let sa = a / scale;
        let sb = b / scale;
        result = scale * f32::sqrt(sa * sa + sb * sb);
    }
    result
}

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
    if !is_finite(x) || !is_finite(y) || !is_finite(z) {
        status = NON_FINITE;
    } else {
        let rho = hypot(x, z);
        let lon = f32::atan2(x, z);
        if rho == 0.0f32 && y == 0.0f32 {
            status = BELOW_DEPTH;
        } else if rho <= comptime!(<f32 as Scalar>::sophus_epsilon_sqrt()) * hypot(rho, y)
            || f32::abs(lon) > params[offset + 4usize]
        {
            // Poles have no longitude; the window keeps the image from wrapping.
            status = OUTSIDE_DOMAIN;
        } else {
            let fx = params[offset];
            let fy = params[offset + 1usize];
            out[0usize] = fx * lon + params[offset + 2usize];
            out[1usize] = fy * f32::atan2(y, rho) + params[offset + 3usize];
            if jacobian {
                let rho2 = rho * rho;
                let n2 = rho2 + y * y;
                out[2usize] = fx * z / rho2;
                out[3usize] = 0.0f32;
                out[4usize] = -fx * x / rho2;
                out[5usize] = -fy * x * y / (n2 * rho);
                out[6usize] = fy * rho / n2;
                out[7usize] = -fy * z * y / (n2 * rho);
            }
            if !is_finite(out[0usize]) || !is_finite(out[1usize]) {
                status = NON_FINITE;
            }
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
/// Returns 0 valid, 1 non-finite, 2 at the origin, or 3 at a pole or beyond `lon_max`.
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

/// Invert one pixel to a unit bearing.
///
/// Returns 0 valid, 1 non-finite, or 3 beyond `lon_max` or within the pole epsilon.
#[cube]
pub fn unproject(params: &[f32], offset: usize, px: f32, py: f32, out: &mut Array<f32>) -> u32 {
    out[0usize] = 0.0f32;
    out[1usize] = 0.0f32;
    out[2usize] = 0.0f32;
    let mut status = u32::cast_from(NON_FINITE);
    let lon = (px - params[offset + 2usize]) / params[offset];
    let lat = (py - params[offset + 3usize]) / params[offset + 1usize];
    if is_finite(px) && is_finite(py) && is_finite(lon) && is_finite(lat) {
        let cos_lat = f32::cos(lat);
        if f32::abs(lon) > params[offset + 4usize]
            || f32::abs(lat) >= comptime!(std::f32::consts::FRAC_PI_2)
            || cos_lat <= comptime!(<f32 as Scalar>::sophus_epsilon_sqrt())
        {
            status = OUTSIDE_DOMAIN;
        } else {
            status = VALID;
            out[0usize] = cos_lat * trig::sin(lon);
            out[1usize] = trig::sin(lat);
            out[2usize] = cos_lat * f32::cos(lon);
        }
    }
    status
}
