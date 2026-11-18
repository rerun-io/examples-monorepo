//! Mei (unified sphere) projection, Jacobian and damped-Newton inverse on the portable device.
//!
//! Packed parameters: `fx fy cx cy xi cos_half_field d0..d12`, with `cos_half_field` the
//! validated CPU model's field bound (`Mei::cos_half_field`) and `d0..d12` the V6
//! coefficients. One layout covers both Insta360 versions: a V3 camera packs
//! `k1 k2 k3` into `d0..d2` and its tangential `p2 p1` into `d5 d6` (the V6 terms
//! `a = d5 + d7 r2`, `b = d6 + d8 r2` reduce to them) and zeroes the rest, so the
//! V6 formula equals the V3 one; only the rounding of the tangential sum differs.
//! C. Mei and P. Rives, ICRA 2007.
#![allow(missing_docs)]
use super::brown8::{BELOW_DEPTH, NON_FINITE, NO_CONVERGENCE, OUTSIDE_DOMAIN, SINGULAR, VALID};
use crate::kernels::finite::is_finite;
use cubecl::prelude::*;

/// Packed f32 count.
pub const PARAMETERS: usize = 19;

/// Distort the sphere point `m` into `out[0..2]`; with `jacobian`, `d(x_d, y_d) / d(m)` row-major into
/// `out[2..6]`. Operation order is the CPU `MeiDistortion::V6` one.
#[cube]
fn distort(
    params: &[f32],
    offset: usize,
    mx: f32,
    my: f32,
    out: &mut Array<f32>,
    #[comptime] jacobian: bool,
) {
    let d = offset + 6usize;
    let r2 = mx * mx + my * my;
    let mut power = 1.0f32;
    let mut gain = 1.0f32;
    // d gain / d r2.
    let mut slope = 0.0f32;
    let mut order = 1.0f32;
    for term in 0..5usize {
        let k = params[d + term];
        slope += order * k * power;
        power *= r2;
        gain += k * power;
        order += 1.0f32;
    }
    let a = params[d + 5usize] + params[d + 7usize] * r2;
    let b = params[d + 6usize] + params[d + 8usize] * r2;
    let sx = r2 + 2.0f32 * mx * mx;
    let sy = r2 + 2.0f32 * my * my;
    let xy = 2.0f32 * mx * my;
    out[0usize] =
        mx * gain + sx * a + xy * b + params[d + 9usize] * r2 + params[d + 11usize] * r2 * r2;
    out[1usize] =
        my * gain + sy * b + xy * a + params[d + 10usize] * r2 + params[d + 12usize] * r2 * r2;
    if jacobian {
        // Every r2-dependent term differentiates through d r2 = 2 m.
        let x_r2 = mx * slope
            + a
            + sx * params[d + 7usize]
            + xy * params[d + 8usize]
            + params[d + 9usize]
            + 2.0f32 * params[d + 11usize] * r2;
        let y_r2 = my * slope
            + b
            + sy * params[d + 8usize]
            + xy * params[d + 7usize]
            + params[d + 10usize]
            + 2.0f32 * params[d + 12usize] * r2;
        out[2usize] = gain + 4.0f32 * mx * a + 2.0f32 * my * b + 2.0f32 * mx * x_r2;
        out[3usize] = 2.0f32 * mx * b + 2.0f32 * my * x_r2;
        out[4usize] = 2.0f32 * my * a + 2.0f32 * mx * y_r2;
        out[5usize] = gain + 4.0f32 * my * b + 2.0f32 * mx * a + 2.0f32 * my * y_r2;
    }
}

/// Project one point with the CPU `project_with_status` rules.
///
/// Returns 0 valid, 1 non-finite, 2 at the origin, or 3 outside the field, `z < |P| cos_half_field`.
/// `out` holds at least two entries, zero on rejection.
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
    let norm = f32::sqrt(x * x + y * y + z * z);
    if !is_finite(x) || !is_finite(y) || !is_finite(z) || !is_finite(norm) {
        status = NON_FINITE;
    } else if norm == 0.0f32 {
        status = BELOW_DEPTH;
    } else if z < norm * params[offset + 5usize] {
        status = OUTSIDE_DOMAIN;
    } else {
        let fx = params[offset];
        let fy = params[offset + 1usize];
        let xi = params[offset + 4usize];
        let den = z + xi * norm;
        let mx = x / den;
        let my = y / den;
        let mut dd = Array::<f32>::new(6usize);
        distort(params, offset, mx, my, &mut dd, jacobian);
        out[0usize] = fx * dd[0usize] + params[offset + 2usize];
        out[1usize] = fy * dd[1usize] + params[offset + 3usize];
        if jacobian {
            // dm/dP = ([I 0] - m (xi P / |P| + e_z)^T) / den.
            let ex = xi * x / norm;
            let ey = xi * y / norm;
            let ez = xi * z / norm + 1.0f32;
            let m00 = (1.0f32 - mx * ex) / den;
            let m01 = -mx * ey / den;
            let m02 = -mx * ez / den;
            let m10 = -my * ex / den;
            let m11 = (1.0f32 - my * ey) / den;
            let m12 = -my * ez / den;
            out[2usize] = fx * (dd[2usize] * m00 + dd[3usize] * m10);
            out[3usize] = fx * (dd[2usize] * m01 + dd[3usize] * m11);
            out[4usize] = fx * (dd[2usize] * m02 + dd[3usize] * m12);
            out[5usize] = fy * (dd[4usize] * m00 + dd[5usize] * m10);
            out[6usize] = fy * (dd[4usize] * m01 + dd[5usize] * m11);
            out[7usize] = fy * (dd[4usize] * m02 + dd[5usize] * m12);
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

/// Invert one pixel with the CPU's 80-step, 20-backtrack damped Newton on the distortion, then lift
/// through the sphere in closed form.
///
/// Returns 0 valid, 1 non-finite, 3 outside the sphere's one-to-one region or the field, 4 singular, or
/// 5 no convergence. `out` holds at least three entries, zero on rejection.
#[cube]
pub fn unproject(params: &[f32], offset: usize, px: f32, py: f32, out: &mut Array<f32>) -> u32 {
    out[0usize] = 0.0f32;
    out[1usize] = 0.0f32;
    out[2usize] = 0.0f32;
    let mut status = u32::cast_from(NON_FINITE);
    let tx = (px - params[offset + 2usize]) / params[offset];
    let ty = (py - params[offset + 3usize]) / params[offset + 1usize];
    if is_finite(px) && is_finite(py) && is_finite(tx) && is_finite(ty) {
        let mut mx = tx;
        let mut my = ty;
        let mut d = Array::<f32>::new(6usize);
        let tolerance = comptime!(kornia_staging_3d::camera::inverse_epsilon::<f32>())
            * (1.0f32 + f32::max(f32::abs(tx), f32::abs(ty)));
        status = NO_CONVERGENCE;
        // The accepted line-search value and Jacobian carry into the next iteration.
        distort(params, offset, mx, my, &mut d, true);
        for _iteration in 0..80usize {
            if !is_finite(d[0usize]) || !is_finite(d[1usize]) {
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
            let det = d[2usize] * d[5usize] - d[3usize] * d[4usize];
            if !is_finite(det) || det == 0.0f32 {
                status = SINGULAR;
                break;
            }
            let sx = (d[5usize] * rx - d[3usize] * ry) / det;
            let sy = (-d[4usize] * rx + d[2usize] * ry) / det;
            let mut scale = 1.0f32;
            let mut accepted = false;
            for _backtrack in 0..20usize {
                let cx = mx - scale * sx;
                let cy = my - scale * sy;
                // Smaller steps cannot move a point that already rounds to itself.
                if cx == mx && cy == my {
                    break;
                }
                distort(params, offset, cx, cy, &mut d, true);
                if is_finite(d[0usize]) && is_finite(d[1usize]) {
                    let next = f32::max(f32::abs(d[0usize] - tx), f32::abs(d[1usize] - ty));
                    if next < norm {
                        mx = cx;
                        my = cy;
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
            let xi = params[offset + 4usize];
            let r2 = mx * mx + my * my;
            // The root is real only inside the sphere's one-to-one region.
            let disc = 1.0f32 + (1.0f32 - xi * xi) * r2;
            if disc < 0.0f32 {
                status = OUTSIDE_DOMAIN;
            } else {
                let lambda = (xi + f32::sqrt(disc)) / (1.0f32 + r2);
                let rx = lambda * mx;
                let ry = lambda * my;
                let rz = lambda - xi;
                if !is_finite(rx) || !is_finite(ry) || !is_finite(rz) {
                    status = NON_FINITE;
                } else if rz < params[offset + 5usize] {
                    status = OUTSIDE_DOMAIN;
                } else {
                    out[0usize] = rx;
                    out[1usize] = ry;
                    out[2usize] = rz;
                }
            }
        }
    }
    status
}
