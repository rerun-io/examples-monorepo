//! Brown8 projection and damped two-dimensional inverse on the portable device.
#![allow(missing_docs)]
use crate::kernels::finite::is_finite;
// CubeCL generates undocumented expansion modules for inline device functions.
use cubecl::prelude::*;
use kornia_staging_algebra::Scalar;

/// Valid result.
pub const VALID: u32 = 0;
/// Non-finite input or arithmetic.
pub const NON_FINITE: u32 = 1;
/// Point lies below the minimum depth.
pub const BELOW_DEPTH: u32 = 2;
/// Point lies outside the calibrated domain.
pub const OUTSIDE_DOMAIN: u32 = 3;
/// Inverse Jacobian is singular.
pub const SINGULAR: u32 = 4;
/// Inverse failed to converge.
pub const NO_CONVERGENCE: u32 = 5;

#[cube]
fn distort(
    params: &[f32],
    offset: usize,
    x: f32,
    y: f32,
    out: &mut Array<f32>,
    #[comptime] jacobian: bool,
) -> bool {
    let k1 = params[offset + 4usize];
    let k2 = params[offset + 5usize];
    let p1 = params[offset + 6usize];
    let p2 = params[offset + 7usize];
    let k3 = params[offset + 8usize];
    let k4 = params[offset + 9usize];
    let k5 = params[offset + 10usize];
    let k6 = params[offset + 11usize];
    let r = x * x + y * y;
    let r4 = r * r;
    let num = 1.0f32 + r * (k1 + r * (k2 + r * k3));
    let den = 1.0f32 + r * (k4 + r * (k5 + r * k6));
    let radial = num / den;
    let dx = 2.0f32 * p1 * x * y + p2 * (r + 2.0f32 * x * x);
    let dy = 2.0f32 * p2 * x * y + p1 * (r + 2.0f32 * y * y);
    out[0usize] = x * radial + dx;
    out[1usize] = y * radial + dy;
    if jacobian {
        let dr = ((k1 + 2.0f32 * k2 * r + 3.0f32 * k3 * r4) * den
            - num * (k4 + 2.0f32 * k5 * r + 3.0f32 * k6 * r4))
            / (den * den);
        out[2usize] = radial + 2.0f32 * x * x * dr + 2.0f32 * p1 * y + 6.0f32 * p2 * x;
        out[3usize] = 2.0f32 * x * y * dr + 2.0f32 * p1 * x + 2.0f32 * p2 * y;
        out[4usize] = radial + 2.0f32 * y * y * dr + 6.0f32 * p1 * y + 2.0f32 * p2 * x;
    }
    // The CPU Brown family evaluates zero prism terms even for Brown8. An
    // overflowing r4 makes those terms non-finite; preserve its rejection.
    is_finite(r4)
}

/// Project one point using validated Brown8 coefficients and the CPU depth/radius rules.
///
/// # Arguments
/// * `params`, `offset` - A buffer containing 13 validated coefficients at `offset`.
/// * `x`, `y`, `z` - Camera-frame coordinates.
/// * `out` - At least two entries, set to zero on rejection.
///
/// Returns 0 for success, 1 for non-finite, 2 for depth, or 3 for radius rejection.
/// The caller must test the status before using the output in downstream maths.
#[cube]
pub fn project(params: &[f32], offset: usize, x: f32, y: f32, z: f32, out: &mut Array<f32>) -> u32 {
    let mut status = u32::cast_from(VALID);
    out[0usize] = 0.0f32;
    out[1usize] = 0.0f32;
    if !is_finite(x) || !is_finite(y) || !is_finite(z) {
        status = NON_FINITE;
    } else if z < comptime!(<f32 as Scalar>::sophus_epsilon_sqrt()) {
        status = BELOW_DEPTH;
    } else {
        let nx = x / z;
        let ny = y / z;
        let radius = params[offset + 12usize];
        if radius > 0.0f32 && nx * nx + ny * ny > radius * radius {
            status = OUTSIDE_DOMAIN;
        } else {
            let mut d = Array::<f32>::new(5usize);
            let finite_radius = distort(params, offset, nx, ny, &mut d, false);
            if finite_radius {
                let px = params[offset] * d[0usize] + params[offset + 2usize];
                let py = params[offset + 1usize] * d[1usize] + params[offset + 3usize];
                if !is_finite(px) || !is_finite(py) {
                    status = NON_FINITE;
                } else {
                    out[0usize] = px;
                    out[1usize] = py;
                }
            } else {
                status = NON_FINITE;
            }
        }
    }
    status
}

/// Invert one pixel with the CPU's 80-step, 20-backtrack damped Newton solve.
///
/// # Arguments
/// * `params`, `offset` - A buffer containing 13 validated coefficients at `offset`.
/// * `px`, `py` - Pixel coordinates.
/// * `out` - At least three entries, set to zero on rejection.
///
/// Returns 0 for success, 1 for non-finite, 3 for radius, 4 for singularity, or
/// 5 for non-convergence. Only success supplies a unit bearing; callers must
/// reject other statuses before using the output in downstream maths.
#[cube]
pub fn unproject(params: &[f32], offset: usize, px: f32, py: f32, out: &mut Array<f32>) -> u32 {
    out[0usize] = 0.0f32;
    out[1usize] = 0.0f32;
    out[2usize] = 0.0f32;
    let mut status = u32::cast_from(NON_FINITE);
    if is_finite(px) && is_finite(py) {
        let tx = (px - params[offset + 2usize]) / params[offset];
        let ty = (py - params[offset + 3usize]) / params[offset + 1usize];
        if is_finite(tx) && is_finite(ty) {
            let mut x = tx;
            let mut y = ty;
            let mut d = Array::<f32>::new(5usize);
            let tolerance = comptime!(kornia_staging_3d::camera::inverse_epsilon::<f32>())
                * (1.0f32 + f32::max(f32::abs(tx), f32::abs(ty)));
            status = NO_CONVERGENCE;
            let mut finite_radius = distort(params, offset, x, y, &mut d, true);
            for _iteration in 0..80usize {
                if !finite_radius || !is_finite(d[0usize]) || !is_finite(d[1usize]) {
                    status = NON_FINITE;
                    break;
                }
                let rx = d[0usize] - tx;
                let ry = d[1usize] - ty;
                let norm = f32::max(f32::abs(rx), f32::abs(ry));
                if norm <= tolerance {
                    status = VALID;
                    break;
                }
                let det = d[2usize] * d[4usize] - d[3usize] * d[3usize];
                if !is_finite(det) || det == 0.0f32 {
                    status = SINGULAR;
                    break;
                }
                let sx = (d[4usize] * rx - d[3usize] * ry) / det;
                let sy = (-d[3usize] * rx + d[2usize] * ry) / det;
                let mut scale = 1.0f32;
                let mut accepted = false;
                for _backtrack in 0..20usize {
                    let cx = x - scale * sx;
                    let cy = y - scale * sy;
                    // Smaller steps cannot move a point that already rounds to itself.
                    if cx == x && cy == y {
                        break;
                    }
                    finite_radius = distort(params, offset, cx, cy, &mut d, true);
                    if finite_radius && is_finite(d[0usize]) && is_finite(d[1usize]) {
                        let next = f32::max(f32::abs(d[0usize] - tx), f32::abs(d[1usize] - ty));
                        if next < norm {
                            x = cx;
                            y = cy;
                            accepted = true;
                            break;
                        }
                    }
                    scale *= 0.5f32;
                }
                if !accepted {
                    break;
                }
            }
            if status == VALID {
                let radius = params[offset + 12usize];
                if radius > 0.0f32 && x * x + y * y > radius * radius {
                    status = OUTSIDE_DOMAIN;
                } else {
                    // Scaling matches hypot's finite range without relying on a native hypot.
                    let scale = f32::max(1.0f32, f32::max(f32::abs(x), f32::abs(y)));
                    let nx = x / scale;
                    let ny = y / scale;
                    let nz = 1.0f32 / scale;
                    let norm = f32::sqrt(nx * nx + ny * ny + nz * nz);
                    out[0usize] = nx / norm;
                    out[1usize] = ny / norm;
                    out[2usize] = nz / norm;
                }
            }
        }
    }
    status
}
