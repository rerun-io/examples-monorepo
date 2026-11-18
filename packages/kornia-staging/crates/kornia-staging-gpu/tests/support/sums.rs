//! Host mirrors of the device PCG's scalar sums (`kernels/linalg.rs`): unit
//! `u` of 256 adds the six entries of groups `u, u + 256, ...` in order, then
//! a fixed pairwise tree adds the units.

/// The device's units per sum.
const UNITS: usize = 256;

/// `a . b` as the device sums it: a Neumaier (compensated) sum per unit, then
/// a tree of TwoSums that carries the units' errors.
pub fn fixed_order_dot(a: &[f32], b: &[f32]) -> f32 {
    let (mut hi, mut lo) = ([0.0f32; UNITS], [0.0f32; UNITS]);
    for (k, (ga, gb)) in a.chunks(6).zip(b.chunks(6)).enumerate() {
        let u = k % UNITS;
        for (x, y) in ga.iter().zip(gb) {
            let (sum, value) = (hi[u], x * y);
            let t = sum + value;
            lo[u] += if sum.abs() >= value.abs() {
                (sum - t) + value
            } else {
                (value - t) + sum
            };
            hi[u] = t;
        }
    }
    let mut stride = UNITS / 2;
    while stride > 0 {
        for u in 0..stride {
            let (a, b) = (hi[u], hi[u + stride]);
            let s = a + b;
            let bb = s - a;
            hi[u] = s;
            lo[u] += lo[u + stride] + ((a - (s - bb)) + (b - bb));
        }
        stride /= 2;
    }
    hi[0] + lo[0]
}

/// `a . b` in the same unit order without compensation: the device sum
/// before the e1 review.
#[allow(dead_code, reason = "used by gba_lanes; linalg compiles a subset")]
pub fn plain_fixed_order_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut partial = [0.0f32; UNITS];
    for (k, (ga, gb)) in a.chunks(6).zip(b.chunks(6)).enumerate() {
        for (x, y) in ga.iter().zip(gb) {
            partial[k % UNITS] += x * y;
        }
    }
    let mut stride = UNITS / 2;
    while stride > 0 {
        for u in 0..stride {
            partial[u] += partial[u + stride];
        }
        stride /= 2;
    }
    partial[0]
}
