//! Eigen's two-sided Jacobi SVD of a 4x4, and the Jacobi rotation the QR
//! shares with it.
//!
//! `crate::ba_base::triangulate` is the only user, and basalt's
//! `0 < inv_dist < 3` acceptance gate (`sqrt_keypoint_vio.cpp:534`) turns a
//! borderline SVD into a landmark that either exists or does not — which is why
//! this is ported rather than delegated (see the module header of
//! [`crate::eigen`]).

use nalgebra::{Matrix4, Vector4};

use crate::lie::LieScalar;
use crate::lie::c;

/// A two-sided Jacobi SVD of a 4x4, ported from Eigen's `JacobiSVD`.
///
/// Ported step for step from
/// `thirdparty/basalt-headers/thirdparty/eigen/Eigen/src/SVD/JacobiSVD.h:700-820`,
/// `Eigen/src/misc/RealSvd2x2.h:22-46` and `Eigen/src/Jacobi/Jacobi.h:92-116`
/// (`makeJacobi`) and `:286-300` (`apply_rotation_in_the_plane`), because
/// [`triangulate`] is the only user and basalt's `0 < inv_dist < 3` acceptance
/// gate (`sqrt_keypoint_vio.cpp:534`) turns a borderline SVD into a landmark
/// that either exists or does not.
///
/// A 4x4 takes Eigen's **square** path: `rows() == cols()` skips both QR
/// preconditioners entirely (`JacobiSVD.h:729-740`), so the whole algorithm is
/// the scaling, the sweep of 2x2 real Jacobi rotations, and the final sort. Only
/// `V` is accumulated, which is what `ComputeFullV` asks for.
///
/// Returns `(singular_values, v)` with the singular values in decreasing order
/// and `v`'s columns permuted to match, or `None` when the input holds a
/// non-finite coefficient — Eigen's `InvalidInput` branch (`:721-727`), which
/// leaves `m_matrixV` **uninitialized** and which basalt then reads. The port
/// refuses instead; see [`triangulate`].
pub(crate) fn jacobi_svd_4x4_full_v<S: LieScalar>(
    a: &Matrix4<S>,
) -> Option<(Vector4<S>, Matrix4<S>)> {
    // `precision = 2 * NumTraits<Scalar>::epsilon()` (`JacobiSVD.h:712`).
    let precision: S = c::<S>(2.0) * S::default_epsilon();
    // `considerAsZero = numeric_limits<RealScalar>::min()` (`:715`).
    let consider_as_zero: S = S::min_positive();

    // `scale = matrix.cwiseAbs().maxCoeff<PropagateNaN>()` (`:718`).
    let mut scale: S = S::zero();
    for value in a.iter() {
        let abs: S = value.abs();
        if !abs.to_f64().is_finite() {
            return None;
        }
        if abs > scale {
            scale = abs;
        }
    }
    if scale == S::zero() {
        scale = S::one();
    }

    let mut work: Matrix4<S> = a / scale;
    let mut v: Matrix4<S> = Matrix4::identity();

    // `maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff()` (`:742`).
    let mut max_diag_entry: S = work.diagonal().abs().max();

    let mut finished: bool = false;
    while !finished {
        finished = true;
        for p in 1..4 {
            for q in 0..p {
                // NaN never exceeds the threshold, which is what stops the loop
                // spinning forever on a degenerate input (`:759-761`).
                let threshold: S = if consider_as_zero > precision * max_diag_entry {
                    consider_as_zero
                } else {
                    precision * max_diag_entry
                };
                if work[(p, q)].abs() > threshold || work[(q, p)].abs() > threshold {
                    finished = false;
                    // For a real scalar `svd_precondition_2x2_block_to_be_real`
                    // is the no-op specialization that returns true
                    // (`JacobiSVD.h:363-367`).
                    let (j_left, j_right) = real_2x2_jacobi_svd(&work, p, q);
                    apply_on_the_left(&mut work, p, q, j_left);
                    apply_on_the_right(&mut work, p, q, j_right);
                    apply_on_the_right(&mut v, p, q, j_right);
                    let biggest: S = if work[(p, p)].abs() > work[(q, q)].abs() {
                        work[(p, p)].abs()
                    } else {
                        work[(q, q)].abs()
                    };
                    if biggest > max_diag_entry {
                        max_diag_entry = biggest;
                    }
                }
            }
        }
    }

    // Step 3 (`:785-801`). `computeU()` is false here, so the sign of the
    // diagonal never has to be pushed into a `U` column.
    let mut singular_values: Vector4<S> = Vector4::zeros();
    for i in 0..4 {
        singular_values[i] = work[(i, i)].abs();
    }
    singular_values *= scale;

    // Step 4 (`:805-818`): selection sort, decreasing, stopping at the first
    // exactly-zero maximum. Eigen's `maxCoeff(&pos)` keeps the **first**
    // maximum (a strict `>` in its visitor).
    for i in 0..4 {
        let mut pos: usize = i;
        for j in (i + 1)..4 {
            if singular_values[j] > singular_values[pos] {
                pos = j;
            }
        }
        if singular_values[pos] == S::zero() {
            break;
        }
        if pos != i {
            singular_values.swap_rows(i, pos);
            v.swap_columns(i, pos);
        }
    }

    Some((singular_values, v))
}

/// A Jacobi rotation `(c, s)`, `Eigen::JacobiRotation`
/// (`Eigen/src/Jacobi/Jacobi.h:30-84`), real scalars only.
///
/// Shared with the linearization stage, whose Givens QR path rotates with them
/// (`landmark_block_abs_dynamic.hpp:429-439`); `makeGivens` lives in
/// [`crate::linearize`] next to its only caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JacobiRotation<S: LieScalar> {
    pub(crate) c: S,
    pub(crate) s: S,
}

impl<S: LieScalar> JacobiRotation<S> {
    /// `transpose()` (`Jacobi.h:60-63`).
    fn transpose(self) -> Self {
        Self {
            c: self.c,
            s: -self.s,
        }
    }

    /// `operator*` (`Jacobi.h:53-58`), with every `conj` an identity.
    fn mul(self, other: Self) -> Self {
        Self {
            c: self.c * other.c - self.s * other.s,
            s: self.c * other.s + self.s * other.c,
        }
    }

    /// `makeJacobi(x, y, z)` (`Jacobi.h:92-116`): the rotation that diagonalises
    /// the symmetric `[[x, y], [y, z]]`.
    fn make_jacobi(x: S, y: S, z: S) -> Self {
        let deno: S = c::<S>(2.0) * y.abs();
        if deno < S::min_positive() {
            return Self {
                c: S::one(),
                s: S::zero(),
            };
        }
        let tau: S = (x - z) / deno;
        let w: S = (tau * tau + S::one()).sqrt();
        let t: S = if tau > S::zero() {
            S::one() / (tau + w)
        } else {
            S::one() / (tau - w)
        };
        let sign_t: S = if t > S::zero() { S::one() } else { -S::one() };
        let n: S = S::one() / (t * t + S::one()).sqrt();
        Self {
            s: -sign_t * (y / y.abs()) * t.abs() * n,
            c: n,
        }
    }
}

/// `real_2x2_jacobi_svd` (`Eigen/src/misc/RealSvd2x2.h:22-46`).
fn real_2x2_jacobi_svd<S: LieScalar>(
    matrix: &Matrix4<S>,
    p: usize,
    q: usize,
) -> (JacobiRotation<S>, JacobiRotation<S>) {
    let mut m: [[S; 2]; 2] = [
        [matrix[(p, p)], matrix[(p, q)]],
        [matrix[(q, p)], matrix[(q, q)]],
    ];
    let t: S = m[0][0] + m[1][1];
    let d: S = m[1][0] - m[0][1];

    let rot1: JacobiRotation<S> = if d.abs() < S::min_positive() {
        JacobiRotation {
            c: S::one(),
            s: S::zero(),
        }
    } else {
        let u: S = t / d;
        let tmp: S = (S::one() + u * u).sqrt();
        JacobiRotation {
            s: S::one() / tmp,
            c: u / tmp,
        }
    };

    // `m.applyOnTheLeft(0, 1, rot1)` on the local 2x2.
    let [row0, row1] = &mut m;
    for (x, y) in row0.iter_mut().zip(row1.iter_mut()) {
        let xi: S = *x;
        let yi: S = *y;
        *x = rot1.c * xi + rot1.s * yi;
        *y = -rot1.s * xi + rot1.c * yi;
    }

    let j_right: JacobiRotation<S> = JacobiRotation::make_jacobi(m[0][0], m[0][1], m[1][1]);
    let j_left: JacobiRotation<S> = rot1.mul(j_right.transpose());
    (j_left, j_right)
}

/// `MatrixBase::applyOnTheLeft(p, q, j)` (`Jacobi.h:261-266`): rows `p` and `q`.
fn apply_on_the_left<S: LieScalar>(m: &mut Matrix4<S>, p: usize, q: usize, j: JacobiRotation<S>) {
    for col in 0..4 {
        let xi: S = m[(p, col)];
        let yi: S = m[(q, col)];
        m[(p, col)] = j.c * xi + j.s * yi;
        m[(q, col)] = -j.s * xi + j.c * yi;
    }
}

/// `MatrixBase::applyOnTheRight(p, q, j)` (`Jacobi.h:276-281`): columns `p` and
/// `q`, with `j.transpose()` — hence the flipped sign of `s`.
fn apply_on_the_right<S: LieScalar>(m: &mut Matrix4<S>, p: usize, q: usize, j: JacobiRotation<S>) {
    let jt: JacobiRotation<S> = j.transpose();
    for row in 0..4 {
        let xi: S = m[(row, p)];
        let yi: S = m[(row, q)];
        m[(row, p)] = jt.c * xi + jt.s * yi;
        m[(row, q)] = -jt.s * xi + jt.c * yi;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use approx::assert_abs_diff_eq;

    use super::*;

    /// The same LCG `ba_base`'s tests draw from, so the case is the one the
    /// triangulation was checked on before this moved here.
    fn pseudo_random(seed: u64, n: usize) -> Vec<f64> {
        let mut state: u64 = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit: f64 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                unit * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn the_svd_reproduces_the_input() {
        // A * V should have orthogonal columns whose norms are the singular
        // values, in decreasing order.
        let values: Vec<f64> = pseudo_random(77, 16);
        let a: Matrix4<f64> = Matrix4::from_iterator(values);
        let (sigma, v) = jacobi_svd_4x4_full_v(&a).unwrap();
        for i in 0..3 {
            assert!(sigma[i] >= sigma[i + 1]);
        }
        assert_abs_diff_eq!(v.transpose() * v, Matrix4::identity(), epsilon = 1e-13);
        let av: Matrix4<f64> = a * v;
        for i in 0..4 {
            assert_abs_diff_eq!(av.column(i).norm(), sigma[i], epsilon = 1e-13);
        }
    }

    /// Eigen's `InvalidInput` branch (`JacobiSVD.h:721-727`) leaves `m_matrixV`
    /// uninitialized and basalt then reads it; the port refuses.
    #[test]
    fn a_non_finite_input_is_refused() {
        let mut a: Matrix4<f64> = Matrix4::identity();
        a[(0, 0)] = f64::NAN;
        assert!(jacobi_svd_4x4_full_v(&a).is_none());
    }
}
