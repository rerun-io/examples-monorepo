//! Numerical elimination over caller-owned padded landmark rows.
use kornia_staging_algebra::{
    linalg::qr::{apply_householder_unchecked, make_householder_unchecked, Givens},
    Scalar,
};
use nalgebra::{DMatrixView, Matrix3, Vector3};

use super::SqrtBaError;
/// Outcome of the original triangular solve. A non-finite result must never be
/// applied to a landmark, but QR itself preserves full-width NaN propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackSubstitution<S: Scalar> {
    /// The absolute product of the triangular diagonal is exactly zero.
    Singular,
    /// The solve or model-cost contraction produced a non-finite value.
    NonFinite,
    /// A finite landmark increment and the model-cost term to subtract.
    Solved {
        /// Increment of [direction x, direction y, inverse distance].
        increment: [S; 3],
        /// Absolute, left-associated product of the triangular diagonal.
        determinant: S,
        /// Caller subtracts this term from its model cost change, in block order.
        cost_term: S,
    },
}
/// Validated padded layout, active pose columns and reusable reflection scratch.
/// Storage stays with the caller. Assemble rows before elimination: inactive
/// pose columns and the final three damping rows must initially be zero.
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkQr<S: Scalar> {
    active_cols: Vec<usize>,
    qr_full_width: bool,
    padding_idx: usize,
    lm_idx: usize,
    res_idx: usize,
    num_rows: usize,
    work_essential: Vec<S>,
}
impl<S: Scalar> LandmarkQr<S> {
    /// Validate dimensions and active columns once and allocate reusable scratch.
    /// Rows are `2*observations+3`; columns are pose, padding to four, landmark(3), residual.
    /// # Errors
    /// Rejects overflowing/unaddressable dimensions and unsorted, duplicate or out-of-range active columns.
    /// ```
    /// use kornia_staging_slam::sqrt_ba::LandmarkQr;
    /// let mut qr = LandmarkQr::<f64>::new(2, 6, (0..6).collect())?;
    /// let mut storage = vec![0.0; qr.rows()*qr.columns()];
    /// qr.eliminate_householder_unchecked(&mut storage);
    /// # Ok::<(), kornia_staging_slam::sqrt_ba::SqrtBaError>(())
    /// ```
    pub fn new(
        observations: usize,
        pose_columns: usize,
        active_cols: Vec<usize>,
    ) -> Result<Self, SqrtBaError> {
        let padding_idx = pose_columns;
        let num_rows = observations
            .checked_mul(2)
            .and_then(|n| n.checked_add(3))
            .ok_or(SqrtBaError::LayoutOverflow)?;
        let padding_size = (4 - padding_idx % 4) % 4;
        let lm_idx = padding_idx
            .checked_add(padding_size)
            .ok_or(SqrtBaError::LayoutOverflow)?;
        let res_idx = lm_idx.checked_add(3).ok_or(SqrtBaError::LayoutOverflow)?;
        let num_cols = res_idx.checked_add(1).ok_or(SqrtBaError::LayoutOverflow)?;
        if !num_rows
            .checked_mul(num_cols)
            .and_then(|n| n.checked_mul(size_of::<S>()))
            .is_some_and(|n| n <= isize::MAX as usize)
        {
            return Err(SqrtBaError::BlockTooLarge {
                rows: num_rows,
                cols: num_cols,
            });
        }
        if active_cols.last().is_some_and(|&c| c >= padding_idx)
            || active_cols.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(SqrtBaError::ActiveColumns);
        }
        Ok(Self {
            active_cols,
            qr_full_width: false,
            padding_idx,
            lm_idx,
            res_idx,
            num_rows,
            work_essential: vec![S::zero(); num_rows],
        })
    }
    /// Total rows, including the three zero damping rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.num_rows
    }
    /// Padded column count.
    #[inline]
    pub fn columns(&self) -> usize {
        // The constructor checked this addition; the residual is the final column.
        self.res_idx + 1
    }
    /// Pose width before padding.
    #[inline]
    pub fn pose_columns(&self) -> usize {
        self.padding_idx
    }
    /// Offset of the three landmark columns.
    #[inline]
    pub fn landmark_column(&self) -> usize {
        self.lm_idx
    }
    /// Offset of the residual column.
    #[inline]
    pub fn residual_column(&self) -> usize {
        self.res_idx
    }
    /// Ascending unique pose columns written by row assembly.
    #[inline]
    pub fn active_columns(&self) -> &[usize] {
        &self.active_cols
    }
    /// Whether the last elimination transformed every column, including inactive columns.
    #[inline]
    pub fn used_full_width(&self) -> bool {
        self.qr_full_width
    }
    /// Three reflections over active columns, at full width for non-finite axes.
    /// Storage must match the validated layout. Initial inactive columns must be zero.
    pub fn eliminate_householder_unchecked(&mut self, packed: &mut [S]) {
        self.qr_full_width = false;
        for k in 0..3 {
            // Exclude damping rows from reflection. Skip an empty reflection when fewer
            // than two observations leave insufficient rows; malformed sparse blocks must
            // not cause an out-of-range access.
            let remaining_rows: usize = self.num_rows.saturating_sub(k + 3);
            if remaining_rows == 0 {
                continue;
            }
            let (tau, _beta) = make_householder_unchecked(
                &packed[(self.lm_idx + k) * self.num_rows + k
                    ..(self.lm_idx + k) * self.num_rows + k + remaining_rows],
                &mut self.work_essential,
            );
            let axis = &self.work_essential[..remaining_rows];
            if axis.iter().all(|value| value.is_finite()) {
                // A finite reflection leaves an untouched zero column at zero.
                // Each live column still uses nalgebra's original dot/update.
                for column in self
                    .active_cols
                    .iter()
                    .copied()
                    .chain(self.lm_idx..self.columns())
                {
                    apply_householder_unchecked(
                        &mut packed[column * self.num_rows + k..],
                        remaining_rows,
                        1,
                        self.num_rows,
                        axis,
                        tau,
                    );
                }
            } else {
                // Non-finite axes must propagate through the unobserved columns.
                self.qr_full_width = true;
                apply_householder_unchecked(
                    &mut packed[k..],
                    remaining_rows,
                    self.columns(),
                    self.num_rows,
                    axis,
                    tau,
                );
            }
        }
    }

    /// Givens QR (Golub & Van Loan Algorithm 5.2.4), retained to check Householder elimination.
    /// Storage must match the validated layout.
    pub fn eliminate_givens_unchecked(&mut self, packed: &mut [S]) {
        self.qr_full_width = true;
        // Guard empty observation sets before subtracting the row offset.
        if self.num_rows < 4 {
            return;
        }
        for n in 0..3 {
            let mut m: usize = self.num_rows - 4;
            while m > n {
                let rot = Givens::cancel_y(
                    packed[(self.lm_idx + n) * self.num_rows + m - 1],
                    packed[(self.lm_idx + n) * self.num_rows + m],
                );
                rot.apply_unchecked(&mut packed[m - 1..], self.columns(), self.num_rows);
                m -= 1;
            }
        }
    }

    /// Recover a step after the caller established both packed lengths.
    /// `pose_inc_is_finite` is shared across blocks by the absolute-system caller;
    /// false retains all zero-times-NaN multiplications of the original kernel.
    pub fn back_substitute_unchecked(
        &self,
        packed: &[S],
        pose_inc: &[S],
        pose_inc_is_finite: bool,
    ) -> BackSubstitution<S> {
        let storage = DMatrixView::from_slice(packed, self.num_rows, self.columns());
        // `Q1Jl`, the upper triangle of the 3x3 at the top of the
        // landmark columns.
        let mut q1jl: Matrix3<S> = Matrix3::zeros();
        for r in 0..3 {
            for col in r..3 {
                q1jl[(r, col)] = storage[(r, self.lm_idx + col)];
            }
        }

        // The product of the triangular diagonal detects a singular landmark.
        let det: S = (q1jl[(0, 0)] * q1jl[(1, 1)] * q1jl[(2, 2)]).abs();
        if det == S::zero() {
            return BackSubstitution::Singular;
        }
        // `Q1Jr + Q1Jp * pose_inc`.
        let mut rhs: Vector3<S> = Vector3::zeros();
        for r in 0..3 {
            let mut acc: S = S::zero();
            for k in self.back_substitution_columns(pose_inc_is_finite) {
                acc += storage[(r, k)] * pose_inc[k];
            }
            rhs[r] = storage[(r, self.res_idx)] + acc;
        }

        // Back-substitute from the last row, then negate the solution.
        let mut inc: Vector3<S> = Vector3::zeros();
        for r in (0..3usize).rev() {
            let mut acc: S = S::zero();
            for k in (r + 1)..3 {
                acc += q1jl[(r, k)] * inc[k];
            }
            inc[r] = (rhs[r] - acc) / q1jl[(r, r)];
        }
        inc = -inc;

        // No landmark damping is applied before the model cost change. The
        // three damping rows are provably still zero — `storage` starts zeroed,
        // the observations fill rows `0..2*obs`, and both QR paths stop at
        // `num_rows - 3` — so there is nothing to undo.

        // `QJinc = storage.topLeftCorner(num_rows - 3, padding_idx) * pose_inc`
        // then `QJinc.head<3>() += Q1Jl * inc` with `Q1Jl`
        // re-read from the now-undamped storage.
        let q2_rows: usize = self.num_rows - 3;
        let mut qjinc: nalgebra::DVector<S> = nalgebra::DVector::zeros(q2_rows);
        for r in 0..q2_rows {
            let mut acc: S = S::zero();
            for k in self.back_substitution_columns(pose_inc_is_finite) {
                acc += storage[(r, k)] * pose_inc[k];
            }
            qjinc[r] = acc;
        }
        for r in 0..3.min(q2_rows) {
            let mut acc: S = S::zero();
            for k in r..3 {
                acc += storage[(r, self.lm_idx + k)] * inc[k];
            }
            qjinc[r] += acc;
        }

        // `diff = QJinc^T * (0.5 * QJinc + Qr)`.
        let mut diff: S = S::zero();
        for r in 0..q2_rows {
            diff += qjinc[r] * (S::from_literal(0.5) * qjinc[r] + storage[(r, self.res_idx)]);
        }
        if !inc.iter().all(|v| v.is_finite()) || !diff.is_finite() {
            return BackSubstitution::NonFinite;
        }
        BackSubstitution::Solved {
            increment: inc.into(),
            determinant: det,
            cost_term: diff,
        }
    }
    /// Finite-axis Householder QR leaves inactive columns at zero. Full-width
    /// QR or a non-finite increment needs the original multiply order.
    fn back_substitution_columns(
        &self,
        pose_inc_is_finite: bool,
    ) -> impl Iterator<Item = usize> + '_ {
        let (active, full) = if pose_inc_is_finite && !self.qr_full_width {
            (self.active_cols.as_slice(), 0..0)
        } else {
            (&[][..], 0..self.padding_idx)
        };
        active.iter().copied().chain(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    fn schur_equivalence<S: Scalar>(tolerance: f64) {
        for givens in [false, true] {
            let mut qr = LandmarkQr::<S>::new(4, 6, vec![0, 1, 2, 3, 4, 5]).unwrap();
            let mut matrix = DMatrix::zeros(qr.rows(), qr.columns());
            for r in 0..8 {
                for c in 0..qr.columns() {
                    if (6..qr.landmark_column()).contains(&c) {
                        continue;
                    }
                    matrix[(r, c)] = S::from_literal(((r * 13 + c * 7 + 1) as f64).sin());
                }
            }
            // A well-conditioned landmark system independent of the pose columns.
            for k in 0..3 {
                matrix[(k, qr.landmark_column() + k)] += S::from_literal(5.0);
            }
            let j = matrix.columns(0, 6).into_owned();
            let l = matrix.columns(qr.landmark_column(), 3).into_owned();
            let expected = j.transpose() * &j
                - j.transpose()
                    * &l
                    * (l.transpose() * &l).try_inverse().unwrap()
                    * l.transpose()
                    * &j;
            if givens {
                qr.eliminate_givens_unchecked(matrix.as_mut_slice());
            } else {
                qr.eliminate_householder_unchecked(matrix.as_mut_slice());
            }
            let reduced = matrix.view((3, 0), (qr.rows() - 3, 6));
            let actual = reduced.transpose() * reduced;
            assert!((actual - expected).amax().to_f64() < tolerance);
        }
    }
    #[test]
    fn eliminated_rows_match_schur_complement_in_both_precisions() {
        schur_equivalence::<f64>(1e-11);
        schur_equivalence::<f32>(2e-4);
    }
    #[test]
    fn triangular_solve_and_cost_have_known_values() {
        let qr = LandmarkQr::<f64>::new(2, 0, vec![]).unwrap();
        let mut matrix = DMatrix::zeros(qr.rows(), qr.columns());
        for (i, (diagonal, residual)) in [(2.0, 2.0), (3.0, 6.0), (4.0, 12.0)]
            .into_iter()
            .enumerate()
        {
            matrix[(i, qr.landmark_column() + i)] = diagonal;
            matrix[(i, qr.residual_column())] = residual;
        }
        assert_eq!(
            qr.back_substitute_unchecked(matrix.as_slice(), &[], true),
            BackSubstitution::Solved {
                increment: [-1.0, -2.0, -3.0],
                determinant: 24.0,
                cost_term: -92.0
            }
        );
        matrix[(2, qr.landmark_column() + 2)] = 0.0;
        assert_eq!(
            qr.back_substitute_unchecked(matrix.as_slice(), &[], true),
            BackSubstitution::Singular
        );
    }
    #[test]
    fn nonfinite_reflector_reaches_inactive_columns_and_resets_next_time() {
        let mut qr = LandmarkQr::<f64>::new(2, 6, vec![0]).unwrap();
        let mut data = vec![0.0; qr.rows() * qr.columns()];
        data[qr.landmark_column() * qr.rows()] = f64::NAN;
        qr.eliminate_householder_unchecked(&mut data);
        assert!(qr.used_full_width());
        assert!(data[qr.rows()].is_nan());
        data.fill(0.0);
        qr.eliminate_householder_unchecked(&mut data);
        assert!(!qr.used_full_width());
        qr.eliminate_givens_unchecked(&mut data);
        assert!(qr.used_full_width());
    }
    #[test]
    fn invalid_landmark_layout_is_rejected() {
        assert!(LandmarkQr::<f64>::new(2, 6, vec![4, 3]).is_err());
        assert!(LandmarkQr::<f64>::new(usize::MAX, 6, vec![]).is_err());
        let qr = LandmarkQr::<f64>::new(2, 6, (0..6).collect()).unwrap();
        assert_eq!(
            (
                qr.rows(),
                qr.columns(),
                qr.landmark_column(),
                qr.residual_column()
            ),
            (7, 12, 8, 11)
        );
    }
}
