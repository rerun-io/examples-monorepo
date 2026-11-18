//! Givens landmark damping with an exact undo.
//!
//! The three zero rows at the bottom of a [`super::LandmarkQr`] block receive
//! `sqrt(lambda)` on the (Jacobi-scaled) landmark diagonal, and six Givens
//! rotations fold them into the triangular `Q1^T J_l` rows (rootba
//! `set_landmark_damping`). Rootba undoes this by applying the transposed
//! rotations, which leaves rounding residue in every touched entry. The
//! rotations touch only the first three and the last three rows, so this
//! keeps a copy of those six rows and restores it: the undo is bit-exact.
use kornia_staging_algebra::Scalar;

/// The six saved rows of one landmark block and whether it is damped now.
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkDamping<S: Scalar> {
    saved: Vec<S>,
    damped: bool,
}

impl<S: Scalar> Default for LandmarkDamping<S> {
    fn default() -> Self {
        Self {
            saved: Vec::new(),
            damped: false,
        }
    }
}

impl<S: Scalar> LandmarkDamping<S> {
    /// Whether [`Self::set`] left the block damped.
    pub fn is_damped(&self) -> bool {
        self.damped
    }

    /// Undo earlier damping, then damp with `lambda` (zero leaves it undamped).
    ///
    /// `packed` is column-major `rows x cols`; landmark columns start at
    /// `lm_col`; the last three rows must be zero when undamped.
    ///
    /// # Panics
    /// Panics if `rows < 6` or the buffer is shorter than `rows * cols`.
    pub fn set(&mut self, packed: &mut [S], rows: usize, cols: usize, lm_col: usize, lambda: S) {
        self.undo(packed, rows, cols);
        if lambda <= S::zero() {
            return;
        }
        assert!(
            rows >= 6 && packed.len() >= rows * cols,
            "landmark block too small to damp"
        );
        self.saved.clear();
        for c in 0..cols {
            let col = &packed[c * rows..(c + 1) * rows];
            self.saved.extend_from_slice(&col[..3]);
            self.saved.extend_from_slice(&col[rows - 3..]);
        }
        self.damped = true;
        let sqrt_lambda = lambda.sqrt();
        for k in 0..3 {
            packed[(lm_col + k) * rows + rows - 3 + k] = sqrt_lambda;
        }
        // Same order as rootba: for each landmark axis n, cancel the damping
        // rows that hold a nonzero in column n against row n.
        for n in 0..3 {
            for m in 0..=n {
                let d = rows - 3 + n - m;
                let a = packed[(lm_col + n) * rows + n];
                let b = packed[(lm_col + n) * rows + d];
                let r = a.hypot(b);
                if r == S::zero() || b == S::zero() {
                    continue;
                }
                let (c, s) = (a / r, b / r);
                for col in 0..cols {
                    let x = packed[col * rows + n];
                    let y = packed[col * rows + d];
                    packed[col * rows + n] = c * x + s * y;
                    packed[col * rows + d] = c * y - s * x;
                }
                packed[(lm_col + n) * rows + d] = S::zero();
            }
        }
    }

    /// Forget the damping without restoring: the caller rewrote the block.
    /// Keeps the snapshot's capacity for the next [`Self::set`].
    pub fn discard(&mut self) {
        self.damped = false;
    }

    /// Restore the saved rows if the block is damped. Bit-exact.
    pub fn undo(&mut self, packed: &mut [S], rows: usize, cols: usize) {
        if !self.damped {
            return;
        }
        for c in 0..cols {
            let saved = &self.saved[c * 6..c * 6 + 6];
            let col = &mut packed[c * rows..(c + 1) * rows];
            col[..3].copy_from_slice(&saved[..3]);
            col[rows - 3..].copy_from_slice(&saved[3..]);
        }
        self.damped = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqrt_ba::LandmarkQr;
    use nalgebra::DMatrix;

    fn block(obs: usize) -> (LandmarkQr<f64>, DMatrix<f64>) {
        let mut qr = LandmarkQr::<f64>::new(obs, 12, (0..12).collect()).unwrap();
        let mut m = DMatrix::zeros(qr.rows(), qr.columns());
        for r in 0..2 * obs {
            for c in 0..qr.columns() {
                if (12..qr.landmark_column()).contains(&c) {
                    continue;
                }
                m[(r, c)] = ((r * 31 + c * 17 + 3) as f64).sin();
            }
        }
        qr.eliminate_householder_unchecked(m.as_mut_slice());
        (qr, m)
    }

    #[test]
    fn undo_restores_every_bit() {
        let (qr, m) = block(5);
        let mut work = m.clone();
        let mut damping = LandmarkDamping::default();
        for lambda in [1e-3, 0.7, 1e-12, 0.0, 4.0] {
            damping.set(
                work.as_mut_slice(),
                qr.rows(),
                qr.columns(),
                qr.landmark_column(),
                lambda,
            );
            assert_eq!(damping.is_damped(), lambda > 0.0);
            damping.undo(work.as_mut_slice(), qr.rows(), qr.columns());
            assert!(work
                .iter()
                .zip(m.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn damping_adds_lambda_to_the_landmark_normal_equations() {
        // Q^T [J; sqrt(lambda) E] keeps the normal matrix: R'R = Jl'Jl + lambda I,
        // and the damping rows stay out of the landmark columns.
        let (qr, m) = block(4);
        let lambda = 0.25;
        let mut work = m.clone();
        LandmarkDamping::default().set(
            work.as_mut_slice(),
            qr.rows(),
            qr.columns(),
            qr.landmark_column(),
            lambda,
        );
        let before = m.view((0, 0), (qr.rows(), qr.columns())).into_owned();
        let mut normal_before = before.transpose() * &before;
        for k in 0..3 {
            normal_before[(qr.landmark_column() + k, qr.landmark_column() + k)] += lambda;
        }
        let normal_after = work.transpose() * &work;
        assert!((normal_after - normal_before).amax() < 1e-12);
        for k in 0..3 {
            for r in 3..qr.rows() {
                assert!(
                    work[(r, qr.landmark_column() + k)].abs() < 1e-12,
                    "row {r} lm {k}"
                );
            }
        }
    }
}
