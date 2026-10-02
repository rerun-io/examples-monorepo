//! The KB4 virtual-camera map kernel: a scalar path and a NEON path (aarch64) with the same operation order, so both give the
//! same float32 results (no fused multiply-adds). (Reciprocal estimates + Newton steps instead of `fdiv`/`fsqrt` measured slower
//! on both the A55 and the A76.)

/// The per-call constants of one KB4 map.
pub(super) struct Kb4 {
    /// virtual_from_source's first row (the ray's `a` coefficient).
    pub row0: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub k: [f32; 4],
    pub min_z: f32,
}

const TAN_3PI_8: f32 = 2.414_213_6;
const TAN_PI_8: f32 = 0.414_213_57;
// Cephes atanf's polynomial on [0, tan(pi/8)] (error ~2e-7 relative).
const P0: f32 = 8.053_744_5e-2;
const P1: f32 = 1.387_768_6e-1;
const P2: f32 = 1.997_771_1e-1;
const P3: f32 = 3.333_295e-1;

/// One pixel: the source pixel of the ray `a · row0 + base`, NaN when its z is at or below `min_z`.
#[inline(always)]
fn kb4_pixel(lens: &Kb4, base: [f32; 3], a: f32) -> (f32, f32) {
    let (x, y, z) = (a * lens.row0[0] + base[0], a * lens.row0[1] + base[1], a * lens.row0[2] + base[2]);
    let radius = (x * x + y * y).sqrt();
    let zc = z.max(lens.min_z);
    // theta = atan2(radius, z) with one division: t > tan(3pi/8) -> pi/2 + atan(-z / r); t > tan(pi/8) -> pi/4 + atan((r - z) / (r + z)).
    let far = radius > TAN_3PI_8 * zc;
    let mid = radius > TAN_PI_8 * zc;
    let numerator = if far { -zc } else if mid { radius - zc } else { radius };
    let denominator = if far { radius } else if mid { radius + zc } else { zc };
    let t = numerator / denominator;
    let offset = if far { std::f32::consts::FRAC_PI_2 } else if mid { std::f32::consts::FRAC_PI_4 } else { 0.0 };
    let q = t * t;
    let theta = offset + (((((P0 * q - P1) * q + P2) * q - P3) * q) * t + t);
    let t2 = theta * theta;
    let theta_d = theta * (1.0 + t2 * (lens.k[0] + t2 * (lens.k[1] + t2 * (lens.k[2] + t2 * lens.k[3]))));
    // On the axis theta_d -> 0 with radius, and x = y = 0: the pixel is the principal point either way.
    let scale = theta_d / radius.max(1e-12);
    if z > lens.min_z { (lens.fx * scale * x + lens.cx, lens.fy * scale * y + lens.cy) } else { (f32::NAN, f32::NAN) }
}

/// One block of a row: `columns[i]` is pixel i's `a`; writes `xs[i]`, `ys[i]`.
pub(super) fn kb4_row(lens: &Kb4, base: [f32; 3], columns: &[f32], xs: &mut [f32], ys: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    let done = neon::kb4_row(lens, base, columns, xs, ys);
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    for ((x_out, y_out), &a) in xs.iter_mut().zip(ys.iter_mut()).zip(columns).skip(done) {
        (*x_out, *y_out) = kb4_pixel(lens, base, a);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{Kb4, P0, P1, P2, P3, TAN_3PI_8, TAN_PI_8};

    /// The NEON path over whole groups of 4 pixels; returns how many pixels it wrote.
    pub(super) fn kb4_row(lens: &Kb4, base: [f32; 3], columns: &[f32], xs: &mut [f32], ys: &mut [f32]) -> usize {
        let n = columns.len().min(xs.len()).min(ys.len()) / 4 * 4;
        // SAFETY: NEON is part of the aarch64 baseline; every load and store below reads or writes exactly 4 f32 inside a
        // `chunks_exact(4)` chunk of a slice.
        unsafe {
            let splat = |v: f32| vdupq_n_f32(v);
            let (r0, r1, r2) = (splat(lens.row0[0]), splat(lens.row0[1]), splat(lens.row0[2]));
            let (b0, b1, b2) = (splat(base[0]), splat(base[1]), splat(base[2]));
            let min_z = splat(lens.min_z);
            for ((a4, x4), y4) in columns[..n].chunks_exact(4).zip(xs[..n].chunks_exact_mut(4)).zip(ys[..n].chunks_exact_mut(4)) {
                let a = vld1q_f32(a4.as_ptr());
                let x = vaddq_f32(vmulq_f32(a, r0), b0);
                let y = vaddq_f32(vmulq_f32(a, r1), b1);
                let z = vaddq_f32(vmulq_f32(a, r2), b2);
                let radius = vsqrtq_f32(vaddq_f32(vmulq_f32(x, x), vmulq_f32(y, y)));
                let zc = vmaxq_f32(z, min_z);
                let far = vcgtq_f32(radius, vmulq_f32(splat(TAN_3PI_8), zc));
                let mid = vcgtq_f32(radius, vmulq_f32(splat(TAN_PI_8), zc));
                let numerator = vbslq_f32(far, vnegq_f32(zc), vbslq_f32(mid, vsubq_f32(radius, zc), radius));
                let denominator = vbslq_f32(far, radius, vbslq_f32(mid, vaddq_f32(radius, zc), zc));
                let t = vdivq_f32(numerator, denominator);
                let offset = vbslq_f32(far, splat(std::f32::consts::FRAC_PI_2), vbslq_f32(mid, splat(std::f32::consts::FRAC_PI_4), splat(0.0)));
                let q = vmulq_f32(t, t);
                let mut p = vsubq_f32(vmulq_f32(splat(P0), q), splat(P1));
                p = vaddq_f32(vmulq_f32(p, q), splat(P2));
                p = vsubq_f32(vmulq_f32(p, q), splat(P3));
                p = vmulq_f32(p, q);
                let theta = vaddq_f32(offset, vaddq_f32(vmulq_f32(p, t), t));
                let t2 = vmulq_f32(theta, theta);
                let mut poly = vaddq_f32(splat(lens.k[2]), vmulq_f32(t2, splat(lens.k[3])));
                poly = vaddq_f32(splat(lens.k[1]), vmulq_f32(t2, poly));
                poly = vaddq_f32(splat(lens.k[0]), vmulq_f32(t2, poly));
                poly = vaddq_f32(splat(1.0), vmulq_f32(t2, poly));
                let theta_d = vmulq_f32(theta, poly);
                let scale = vdivq_f32(theta_d, vmaxq_f32(radius, splat(1e-12)));
                let ok = vcgtq_f32(z, min_z);
                let u = vaddq_f32(vmulq_f32(vmulq_f32(splat(lens.fx), scale), x), splat(lens.cx));
                let v = vaddq_f32(vmulq_f32(vmulq_f32(splat(lens.fy), scale), y), splat(lens.cy));
                vst1q_f32(x4.as_mut_ptr(), vbslq_f32(ok, u, splat(f32::NAN)));
                vst1q_f32(y4.as_mut_ptr(), vbslq_f32(ok, v, splat(f32::NAN)));
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_matches_the_scalar_pixel_formula() {
        let lens = Kb4 { row0: [0.9, 0.1, -0.4], fx: 630.0, fy: 628.0, cx: 946.0, cy: 539.0, k: [0.077, -0.063, 0.080, -0.029], min_z: 1e-6 };
        let base = [0.05, -0.2, 0.6];
        let columns: Vec<f32> = (0..64).map(|i| -0.4 + 0.0125 * i as f32).collect();
        let (mut xs, mut ys) = (vec![0f32; 64], vec![0f32; 64]);
        kb4_row(&lens, base, &columns, &mut xs, &mut ys);
        for (i, &a) in columns.iter().enumerate() {
            let (x, y) = kb4_pixel(&lens, base, a);
            assert!(x.to_bits() == xs[i].to_bits() && y.to_bits() == ys[i].to_bits(), "{i}: ({x}, {y}) vs ({}, {})", xs[i], ys[i]);
        }
    }
}
