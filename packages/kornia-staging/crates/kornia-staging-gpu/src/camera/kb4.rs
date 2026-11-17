//! Kannala-Brandt KB4 projection, Jacobian and bracketed inverse on the portable device.
//!
//! Packed parameters: `fx fy cx cy k1 k2 k3 k4 theta_max r_max`, the last two taken
//! from the validated CPU model (`KannalaBrandt4::max_angle`, `max_radius`).
//! Native f32 `atan2` stays within 2.7e-7 rad of f64 on the Vulkan lane
//! (`kernels::tests::native_inverse_trig_error_against_f64`), so the CPU's
//! f64-rounded angle is matched without a polynomial.
#![allow(missing_docs)]
use super::brown8::{BELOW_DEPTH, NON_FINITE, NO_CONVERGENCE, OUTSIDE_DOMAIN, VALID};
use crate::kernels::{finite::is_finite, trig};
use cubecl::prelude::*;

/// Packed f32 count.
pub const PARAMETERS: usize = 10;

/// `theta * (1 + k1 t^2 + ...)` in the CPU's Horner order.
#[cube]
fn distorted_radius(params: &[f32], offset: usize, theta: f32) -> f32 {
    let t2 = theta * theta;
    ((((params[offset + 7usize] * t2 + params[offset + 6usize]) * t2 + params[offset + 5usize])
        * t2
        + params[offset + 4usize])
        * t2
        + 1.0f32)
        * theta
}

/// Project one point with the CPU `project_with_status` rules.
///
/// # Arguments
/// * `params`, `offset` - Ten packed KB4 values at `offset`.
/// * `x`, `y`, `z` - Camera-frame coordinates; rays beyond 90 degrees are valid up to `theta_max`.
/// * `out` - At least two entries, zero on rejection.
///
/// Returns 0 valid, 1 non-finite, 2 depth (on-axis, `z <= 0`), or 3 beyond `theta_max`.
///
/// With `jacobian`, also writes the 2x3 point Jacobian row-major into `out[2..8]` (at least eight entries,
/// meaningful only for status 0).
#[cube]
pub fn project(
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
    if !is_finite(r2) || !is_finite(z) {
        status = NON_FINITE;
    } else {
        let r = f32::sqrt(r2);
        let theta = f32::atan2(r, z);
        if r == 0.0f32 && z <= 0.0f32 {
            status = BELOW_DEPTH;
        } else if theta > params[offset + 8usize] {
            status = OUTSIDE_DOMAIN;
        } else {
            let fx = params[offset];
            let fy = params[offset + 1usize];
            if r == 0.0f32 {
                // On the axis KB4 is the pinhole camera.
                out[0usize] = fx * x / z + params[offset + 2usize];
                out[1usize] = fy * y / z + params[offset + 3usize];
                if jacobian {
                    out[2usize] = fx / z;
                    out[3usize] = 0.0f32;
                    out[4usize] = -fx * x / (z * z);
                    out[5usize] = 0.0f32;
                    out[6usize] = fy / z;
                    out[7usize] = -fy * y / (z * z);
                }
            } else {
                let rt = distorted_radius(params, offset, theta);
                out[0usize] = fx * (x * rt / r) + params[offset + 2usize];
                out[1usize] = fy * (y * rt / r) + params[offset + 3usize];
                if jacobian {
                    let t2 = theta * theta;
                    let slope = (((9.0f32 * params[offset + 7usize] * t2
                        + 7.0f32 * params[offset + 6usize])
                        * t2
                        + 5.0f32 * params[offset + 5usize])
                        * t2
                        + 3.0f32 * params[offset + 4usize])
                        * t2
                        + 1.0f32;
                    let tmp = z * z + r2;
                    let tx = x / r * z / tmp;
                    let ty = y / r * z / tmp;
                    let tz = -r / tmp;
                    out[2usize] = fx * (rt * r + x * r * slope * tx - x * x * rt / r) / r2;
                    out[3usize] = fx * x * (slope * ty * r - y * rt / r) / r2;
                    out[4usize] = fx * x * slope * tz / r;
                    out[5usize] = fy * y * (slope * tx * r - x * rt / r) / r2;
                    out[6usize] = fy * (rt * r + y * r * slope * ty - y * y * rt / r) / r2;
                    out[7usize] = fy * y * slope * tz / r;
                }
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

/// Invert one pixel to a unit bearing with the CPU's bracketed Newton on theta.
///
/// Returns 0 valid, 1 non-finite, 3 beyond `r_max` or at pi, or 5 no convergence.
#[cube]
pub fn unproject(params: &[f32], offset: usize, px: f32, py: f32, out: &mut Array<f32>) -> u32 {
    out[0usize] = 0.0f32;
    out[1usize] = 0.0f32;
    out[2usize] = 0.0f32;
    let mut status = u32::cast_from(NON_FINITE);
    let x = (px - params[offset + 2usize]) / params[offset];
    let y = (py - params[offset + 3usize]) / params[offset + 1usize];
    if is_finite(px) && is_finite(py) && is_finite(x) && is_finite(y) {
        let r = f32::sqrt(x * x + y * y);
        let theta_max = params[offset + 8usize];
        let limit = params[offset + 9usize];
        let epsilon = comptime!(kornia_staging_3d::camera::inverse_epsilon::<f32>());
        let endpoint = epsilon * (1.0f32 + f32::abs(limit));
        let mut theta = 0.0f32;
        if !is_finite(r) {
            status = NON_FINITE;
        } else if r == 0.0f32 {
            status = VALID;
        } else if r > limit + endpoint {
            status = OUTSIDE_DOMAIN;
        } else if f32::abs(r - limit) <= endpoint {
            theta = theta_max;
            status = VALID;
        } else {
            status = NO_CONVERGENCE;
            let mut low = 0.0f32;
            let mut high = theta_max;
            theta = f32::min(r, theta_max);
            for _iteration in 0..80usize {
                let t2 = theta * theta;
                // The CPU `angular::radial` loop, unrolled from k4 down to k1.
                let mut value = params[offset + 7usize] * t2;
                let mut slope = 9.0f32 * params[offset + 7usize] * t2;
                value = (value + params[offset + 6usize]) * t2;
                slope = (slope + 7.0f32 * params[offset + 6usize]) * t2;
                value = (value + params[offset + 5usize]) * t2;
                slope = (slope + 5.0f32 * params[offset + 5usize]) * t2;
                value = (value + params[offset + 4usize]) * t2;
                slope = (slope + 3.0f32 * params[offset + 4usize]) * t2;
                let residual = theta * (1.0f32 + value) - r;
                slope += 1.0f32;
                if f32::abs(residual) <= epsilon * (1.0f32 + r) {
                    status = VALID;
                    break;
                }
                if residual > 0.0f32 {
                    high = theta;
                } else {
                    low = theta;
                }
                let newton = theta - residual / slope;
                if slope > 0.0f32 && newton > low && newton < high {
                    theta = newton;
                } else {
                    theta = 0.5f32 * (low + high);
                }
            }
        }
        if status == VALID {
            if r == 0.0f32 {
                out[2usize] = 1.0f32;
            } else if theta >= comptime!(std::f32::consts::PI) {
                // At pi the azimuth is undefined.
                status = OUTSIDE_DOMAIN;
            } else {
                let sin = trig::sin(theta);
                out[0usize] = sin * x / r;
                out[1usize] = sin * y / r;
                out[2usize] = f32::cos(theta);
            }
        }
    }
    status
}
