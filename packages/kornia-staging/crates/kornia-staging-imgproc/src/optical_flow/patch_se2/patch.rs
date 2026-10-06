//! Mean-normalized patches and cached inverse-compositional SE(2) factors.
//!
//! Sampling computes raw taps and gradients, then applies the derivative of the
//! normalization factor. Build `H = JᵀJ` in tap order using rank-one updates,
//! invert the 3x3 factor with guarded pivoted LDLT, then apply it to each stored
//! Jacobian column. The product-rule correction is required for the right warp.
//!
//! Builds write into caller-owned structure-of-arrays storage.
//! The largest temporary is a 3x3 matrix plus a 3-vector, avoiding a full patch
//! Jacobian in per-thread GPU storage. Residual sampling warps one tap at a time,
//! avoiding a transient transformed-pattern matrix.

use nalgebra::Matrix3;

use super::ldlt::ldlt_inverse3;
use super::patterns::Pattern;
use super::se2::AffineCompact2;
use kornia_image::Image;

use super::simd::F32x4;

/// # Arguments
/// `source` is a dense image; `positions` holds four centres; `data` and `jacobian` have at least `4 * P::SIZE` and `12 * P::SIZE` elements.
///
/// # Preconditions
/// Call [`super::patterns::validate_pattern`] once for `P` before using this low-level kernel.
/// Strides and buffers must cover every addressed element.
///
/// Build four independent patches in `[row][tap][lane]` storage.
/// Each lane follows the scalar oracle's operation order. The pivoted
/// factorization remains scalar because each point can choose different pivots.
///
/// # Panics
/// If data has fewer than `4 * P::SIZE` elements or the factor fewer than `12 * P::SIZE`.
pub fn build_patch_group<P: Pattern>(
    source: &Image<u16, 1>,
    positions: [[f32; 2]; 4],
    data: &mut [f32],
    jacobian: &mut [f32],
) -> ([f32; 4], [bool; 4]) {
    let mut sum = F32x4::ZERO;
    let mut count = F32x4::ZERO;
    let mut grad_sum = [F32x4::ZERO; 3];
    let row_stride = 4 * P::SIZE;
    let pos_x = F32x4(positions.map(|p| p[0]));
    let pos_y = F32x4(positions.map(|p| p[1]));
    for (tap, &[x, y]) in P::OFFSETS.iter().take(P::SIZE).enumerate() {
        let (raw, [gx, gy], valid) =
            sample_group::<true>(source, pos_x + F32x4::splat(x), pos_y + F32x4::splat(y));
        let rows = [gx, gy, gx * F32x4::splat(-y) + gy * F32x4::splat(x)];
        raw.store(&mut data[4 * tap..]);
        sum = sum + F32x4::select(valid, raw, F32x4::ZERO);
        count = count + F32x4(valid.map(|ok| if ok { 1.0 } else { 0.0 }));
        for row in 0..3 {
            let values = F32x4::select(valid, rows[row], F32x4::ZERO);
            values.store(&mut jacobian[row * row_stride + 4 * tap..]);
            // A masked tap must leave even a negative-zero accumulator unchanged.
            grad_sum[row] = F32x4::select(valid, grad_sum[row] + values, grad_sum[row]);
        }
    }
    let positive = std::array::from_fn(|lane| sum.0[lane] > 0.0 && count.0[lane] > 0.0);
    let safe_count = F32x4::select(positive, count, F32x4::splat(1.0));
    let safe_sum = F32x4::select(positive, sum, F32x4::splat(1.0));
    let mean = F32x4::select(positive, sum / safe_count, F32x4::ZERO);
    let mean_inv = F32x4::select(positive, count / safe_sum, F32x4::ZERO);
    let mut h = [[F32x4::ZERO; 3]; 3];
    for tap in 0..P::SIZE {
        let raw = F32x4::load(&data[4 * tap..]);
        let valid = raw.0.map(|value| value >= 0.0);
        let mut rows = [F32x4::ZERO; 3];
        for row in 0..3 {
            let slot = &mut jacobian[row * row_stride + 4 * tap..];
            rows[row] = F32x4::select(
                valid,
                (F32x4::load(slot) - grad_sum[row] * raw / safe_sum) * mean_inv,
                F32x4::ZERO,
            );
            rows[row].store(slot);
        }
        F32x4::select(valid, raw * mean_inv, raw).store(&mut data[4 * tap..]);
        for col in 0..3 {
            for row in 0..3 {
                h[row][col] = h[row][col] + rows[col] * rows[row];
            }
        }
    }
    let inverses: [Matrix3<f32>; 4] = std::array::from_fn(|lane| {
        ldlt_inverse3(&Matrix3::from_fn(|row, col| h[row][col].0[lane]))
    });
    let inverse: [[F32x4; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|col| F32x4(inverses.map(|matrix| matrix[(row, col)])))
    });
    let mut finite = [true; 4];
    for tap in 0..P::SIZE {
        let column: [F32x4; 3] =
            std::array::from_fn(|row| F32x4::load(&jacobian[row * row_stride + 4 * tap..]));
        for row in 0..3 {
            let product = inverse[row][0] * column[0]
                + inverse[row][1] * column[1]
                + inverse[row][2] * column[2];
            product.store(&mut jacobian[row * row_stride + 4 * tap..]);
            for (lane, ok) in finite.iter_mut().enumerate() {
                *ok &= product.0[lane].is_finite();
            }
        }
        for lane in 0..4 {
            finite[lane] &= data[4 * tap + lane].is_finite();
        }
    }
    (
        mean.0,
        std::array::from_fn(|lane| mean.0[lane] > f32::EPSILON && finite[lane]),
    )
}

/// Two-pixel patch border, keeping taps and their gradient stencils inside the image.
pub const PATCH_BORDER: f32 = 2.0;

/// Accumulate the three SE(2) rows in independent vector lanes. Each row still
/// visits every tap in scalar order; the fourth lane is unused.
///
/// # Panics
/// If factor or residual storage is too short for the pattern and strides.
pub fn patch_increment_rows<P: Pattern>(
    factor: &[f32],
    element_stride: usize,
    row_stride: usize,
    residual: &[f32],
) -> [f32; 3] {
    let mut sum = F32x4::ZERO;
    for (tap, &value) in residual[..P::SIZE].iter().enumerate() {
        let offset = tap * element_stride;
        let rows = F32x4([
            factor[offset],
            factor[row_stride + offset],
            factor[2 * row_stride + offset],
            0.0,
        ]);
        sum = sum + rows * F32x4::splat(value);
    }
    [sum.0[0], sum.0[1], sum.0[2]]
}

/// # Arguments
/// `data` contains source taps at `stride`; `source` and `transform` specify target sampling; `residual` holds `P::SIZE` outputs.
///
/// # Preconditions
/// Call [`super::patterns::validate_pattern`] once for `P` before using this low-level kernel.
/// Strides and buffers must cover every addressed element.
///
/// Sample four taps of one point at a time. The sum still visits individual
/// taps in ascending order, so this can serve scalar tracking call sites
/// without changing their arithmetic or needing adjacent points' guesses.
///
/// # Panics
/// If data or residual storage is too short for the pattern and stride.
pub fn patch_residual_taps<P: Pattern>(
    data: &[f32],
    stride: usize,
    source: &Image<u16, 1>,
    transform: &AffineCompact2<f32>,
    residual: &mut [f32],
) -> bool {
    let warp = transform.coefficients().map(F32x4::splat);
    let mut sum = 0.0;
    let mut count = 0;
    for base in (0..P::SIZE).step_by(4) {
        let lanes = (P::SIZE - base).min(4);
        let taps: [[f32; 2]; 4] = std::array::from_fn(|lane| {
            if lane < lanes {
                P::OFFSETS[base + lane]
            } else {
                [0.0; 2]
            }
        });
        let x = F32x4(taps.map(|tap| tap[0]));
        let y = F32x4(taps.map(|tap| tap[1]));
        let px = warp[0] * x + warp[1] * y + warp[4];
        let py = warp[2] * x + warp[3] * y + warp[5];
        let (values, _, valid) = sample_group::<false>(source, px, py);
        for lane in 0..lanes {
            residual[base + lane] = values.0[lane];
            if valid[lane] {
                sum += values.0[lane];
                count += 1;
            }
        }
    }
    if !sum.is_finite() || sum < f32::EPSILON {
        residual[..P::SIZE].fill(0.0);
        return false;
    }
    let mut num_residuals = 0;
    for base in (0..P::SIZE).step_by(4) {
        let lanes = (P::SIZE - base).min(4);
        let values = F32x4(std::array::from_fn(|lane| {
            if lane < lanes {
                residual[base + lane]
            } else {
                -1.0
            }
        }));
        let stored = F32x4(std::array::from_fn(|lane| {
            if lane < lanes {
                data[(base + lane) * stride]
            } else {
                -1.0
            }
        }));
        let normalized = F32x4::splat(count as f32) * values / F32x4::splat(sum) - stored;
        for lane in 0..lanes {
            if values.0[lane] >= 0.0 && stored.0[lane] >= 0.0 {
                residual[base + lane] = normalized.0[lane];
                num_residuals += 1;
            } else {
                residual[base + lane] = 0.0;
            }
        }
    }
    num_residuals > P::SIZE / 2
}

/// Four KLT samples with a two-pixel border. Gather pixels once their whole
/// stencil is in bounds, then keep the scalar bilinear operation order in
/// each SIMD lane. Invalid lanes return -1; callers mask their gradients.
#[inline]
#[allow(unsafe_code)] // Bounds cover the whole bilinear stencil before gathering.
pub(crate) fn sample_group<const GRAD: bool>(
    image: &Image<u16, 1>,
    x: F32x4,
    y: F32x4,
) -> (F32x4, [F32x4; 2], [bool; 4]) {
    let ix = x.0.map(|v| v as usize);
    let iy = y.0.map(|v| v as usize);
    let valid = std::array::from_fn(|lane| {
        crate::interpolation::in_bounds_u16(image, x.0[lane], y.0[lane], 2.0)
            // Integer checks also cover dimensions beyond f32's exact range.
            && ix[lane] >= 1 && ix[lane] < image.width().saturating_sub(2)
            && iy[lane] >= 1 && iy[lane] < image.height().saturating_sub(2)
    });
    let dx = x - F32x4(ix.map(|v| v as f32));
    let dy = y - F32x4(iy.map(|v| v as f32));
    let ddx = F32x4::splat(1.0) - dx;
    let ddy = F32x4::splat(1.0) - dy;
    let weights = [ddx * ddy, ddx * dy, dx * ddy, dx * dy];
    // Borrow dense pixels once for the whole stencil.
    let pixels = image.as_slice();
    let pixel = |ox: usize, oy: usize| {
        F32x4(std::array::from_fn(|lane| {
            if valid[lane] {
                let offset = (iy[lane] - 1 + oy) * image.width() + ix[lane] - 1 + ox;
                // SAFETY: valid covers the entire [-1, +2] stencil; image
                // construction guarantees width * height contiguous storage.
                f32::from(unsafe { *pixels.get_unchecked(offset) })
            } else {
                0.0
            }
        }))
    };
    let interpolate =
        |a, b, c, d| weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d;
    let p00 = pixel(1, 1);
    let p01 = pixel(1, 2);
    let p10 = pixel(2, 1);
    let p11 = pixel(2, 2);
    let value = F32x4::select(valid, interpolate(p00, p01, p10, p11), F32x4::splat(-1.0));
    let gradients = if GRAD {
        let mx = interpolate(pixel(0, 1), pixel(0, 2), p00, p01);
        let px = interpolate(p10, p11, pixel(3, 1), pixel(3, 2));
        let my = interpolate(pixel(1, 0), p00, pixel(2, 0), p10);
        let py = interpolate(p01, pixel(1, 3), p11, pixel(2, 3));
        [F32x4::splat(0.5) * (px - mx), F32x4::splat(0.5) * (py - my)]
    } else {
        [F32x4::ZERO; 2]
    };
    (value, gradients, valid)
}
