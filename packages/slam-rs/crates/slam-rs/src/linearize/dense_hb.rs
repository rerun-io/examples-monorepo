//! Reusable storage for deterministic dense landmark reduction.

use nalgebra::{DMatrix, DVector};

use super::LinearizeError;
use super::landmark_block::{DenseHbScratch, LandmarkBlock};
use super::reduce::{Reducible, deterministic_reduce};
use crate::lie::LieScalar;

/// One subtree's partial `(H, b)` of the dense reduction, and the columns it holds.
///
/// C++ gives every TBB task a full `total_size` x `total_size` partial and adds
/// the whole square at each join (`:513-542`), but a subtree only ever writes
/// the pose columns its landmarks observe — 22 of 85 on the median MIO10 frame
/// for one landmark, and the union of a subtree's landmarks above that. The
/// rest is `+0.0` on both sides of a join and `+0.0` after a reset, so keeping
/// the square but touching only `columns` is the same arithmetic; see
/// [`LandmarkBlock::active_cols`] for why `+= +0.0` here is the identity.
///
/// This partial is the one destination that argument holds for: it is created
/// zeroed and [`Self::reset`] puts back `+0.0`, never `-0.0`, so no coefficient
/// a skipped write would have changed exists. A block whose own columns are not
/// the identity to skip — [`LandmarkBlock::active_writeback_is_exact`] — is
/// added at full width instead.
#[derive(Debug, Clone)]
struct DensePartial<S: LieScalar> {
    /// The partial `H`, full size, zero outside `columns` x `columns`.
    h: DMatrix<S>,
    /// The partial `b`, full size, zero outside `columns`.
    b: DVector<S>,
    /// Which columns have been written, indexed by column.
    written: Vec<bool>,
    /// The same set ascending, which is the order `h`'s column-major storage wants.
    columns: Vec<usize>,
    /// The same set again as `(start, length)` runs of consecutive columns.
    ///
    /// [`Reducible::reset`] and [`Reducible::join`] between them are the whole
    /// cost of the reduction's interior — 54 joins and 54 resets over 55
    /// landmark blocks — and both touch `columns` x `columns` of a column-major
    /// square. Walking `columns` reaches every coefficient through a `usize`
    /// out of a `Vec`, which LLVM has to treat as a gather, plus a bounds check
    /// per coefficient. The set is almost never scattered: a landmark's own
    /// columns are two runs of six, and a subtree's union saturates towards the
    /// contiguous `0..opt_size`. Holding the runs turns both operations into
    /// slice work on `h`'s own storage. It is the same set, so the same
    /// coefficients are touched, and both operations are elementwise, so no sum
    /// is reassociated.
    runs: Vec<(usize, usize)>,
}

impl<S: LieScalar> DensePartial<S> {
    /// An identity accumulator for an `n`-column ordering.
    fn zeros(n: usize) -> Self {
        Self {
            h: DMatrix::zeros(n, n),
            b: DVector::zeros(n),
            written: vec![false; n],
            columns: Vec::with_capacity(n),
            runs: Vec::new(),
        }
    }

    /// Back to the identity for an `n`-column ordering, reusing the buffers.
    ///
    /// [`Reducible::reset`] is the cheaper reset and is what the reduction's own
    /// subtree buffers take; this one zeroes the whole square, because the
    /// accumulator the reduction hands back is written by three more parties —
    /// the IMU blocks, the marginalization prior and the caller that pins a
    /// fixed keyframe's rows — none of which record the columns they touched.
    /// The window's landmarks cover nearly every column of the ordering
    /// anyway, so on the frame that matters the two resets zero the same
    /// square; what this saves is the allocation, not the memset.
    fn reset_sized(&mut self, n: usize) {
        if self.b.nrows() == n {
            self.h.fill(S::zero());
            self.b.fill(S::zero());
            self.columns.clear();
            self.runs.clear();
            self.written.fill(false);
        } else {
            *self = Self::zeros(n);
        }
    }

    /// Record that `columns` have been written, keeping the list ascending.
    ///
    /// Every column is in range, so this indexes rather than absorbing an
    /// out-of-range one (decision D32): [`Self::accumulate`] marks only a block
    /// the writeback has accepted, whose check is `padding_idx <= h.ncols()`,
    /// and both a block's `active_cols` and its `pose_columns` are inside its
    /// own `padding_idx` ([`LandmarkBlock::allocate`] refuses a pose block that
    /// is not); [`Self::join`] marks a partial of the same ordering.
    fn mark(&mut self, columns: impl IntoIterator<Item = usize>) {
        let mut added: bool = false;
        for column in columns {
            added |= !self.written[column];
            self.written[column] = true;
        }
        if added {
            self.columns.clear();
            self.runs.clear();
            for column in 0..self.written.len() {
                if !self.written[column] {
                    continue;
                }
                self.columns.push(column);
                match self.runs.last_mut() {
                    // Consecutive with the run being built, so extend it.
                    Some((start, length)) if *start + *length == column => *length += 1,
                    _ => self.runs.push((column, 1)),
                }
            }
        }
    }

    /// Add one landmark block's `(H, b)` and record the columns it wrote.
    ///
    /// The two halves belong together: [`Self::mark`] records what a join and a
    /// reset will touch and the writeback is what writes it, so a drift between
    /// them would leave coefficients no join adds and no reset clears — a
    /// silently wrong reduction that no test would catch. Marking **after** the
    /// add is what makes every column in range: the add is the check on the
    /// block's layout.
    ///
    /// Which of the two writebacks runs is the block's own answer: the observed
    /// columns when skipping the rest is the identity, the full width when it is
    /// not, and then every column is marked because the full width writes every
    /// column — a NaN spread into a column the block never observed is part of
    /// the sum basalt takes (decision D32).
    fn accumulate(
        &mut self,
        block: &LandmarkBlock<S>,
        scratch: &mut DenseHbScratch<S>,
    ) -> Result<(), LinearizeError> {
        if block.active_writeback_is_exact() {
            block.add_dense_h_b_active(&mut self.h, &mut self.b, scratch)?;
            self.mark(block.active_cols().iter().copied());
        } else {
            block.add_dense_h_b(&mut self.h, &mut self.b, scratch)?;
            self.mark(block.pose_columns());
        }
        Ok(())
    }
}

impl<S: LieScalar> Reducible for DensePartial<S> {
    /// A second `opt_size`-wide accumulator, which is what C++'s split
    /// constructor allocates per task (`:513-542`).
    fn identity_like(&self) -> Self {
        Self::zeros(self.b.nrows())
    }

    /// Back to the identity, zeroing only what was written.
    fn reset(&mut self) {
        let stride: usize = self.h.nrows();
        {
            let h: &mut [S] = self.h.as_mut_slice();
            for &(first_col, cols) in &self.runs {
                for column in first_col..(first_col + cols) {
                    let base: usize = column * stride;
                    for &(first_row, rows) in &self.runs {
                        h[(base + first_row)..(base + first_row + rows)].fill(S::zero());
                    }
                }
            }
        }
        {
            let b: &mut [S] = self.b.as_mut_slice();
            for &(first, length) in &self.runs {
                b[first..(first + length)].fill(S::zero());
            }
        }
        self.columns.clear();
        self.runs.clear();
        self.written.fill(false);
    }

    /// `H_ += b.H_; b_ += b.b_` (`:532-535`), over the right side's columns.
    fn join(&mut self, right: &Self) {
        debug_assert_eq!(self.h.nrows(), right.h.nrows());
        let stride: usize = self.h.nrows();
        {
            let destination: &mut [S] = self.h.as_mut_slice();
            let source: &[S] = right.h.as_slice();
            for &(first_col, cols) in &right.runs {
                for column in first_col..(first_col + cols) {
                    let base: usize = column * stride;
                    for &(first_row, rows) in &right.runs {
                        let lo: usize = base + first_row;
                        let target: &mut [S] = &mut destination[lo..(lo + rows)];
                        for (slot, value) in target.iter_mut().zip(source[lo..(lo + rows)].iter()) {
                            *slot += *value;
                        }
                    }
                }
            }
        }
        {
            let destination: &mut [S] = self.b.as_mut_slice();
            let source: &[S] = right.b.as_slice();
            for &(first, length) in &right.runs {
                let target: &mut [S] = &mut destination[first..(first + length)];
                for (slot, value) in target
                    .iter_mut()
                    .zip(source[first..(first + length)].iter())
                {
                    *slot += *value;
                }
            }
        }
        self.mark(right.columns.iter().copied());
    }
}

/// The buffers [`super::LinearizationAbsQR::get_dense_h_b_into`] reduces in, held
/// across calls and across frames.
///
/// One call over `n` landmark blocks needs the accumulator it returns plus
/// `ceil(log2 n)` subtree partials, each a full `opt_size` square, and the
/// linearizer's own leaf scratch: seven `87x87` `f32` matrices on the median
/// MIO10 frame. The Levenberg-Marquardt loop calls it **once per inner step** —
/// seven times on that frame — and every one of those buffers is either zeroed
/// on entry (`DensePartial::reset_sized`) or reset by the reduction before a
/// leaf writes it (`Reducible::reset`, which restores exactly `+0.0` over the
/// columns that were written), so a buffer that persists is the same
/// arithmetic on the same values.
///
/// The window changes size, so the buffers are keyed on `opt_size` and dropped
/// when it moves; that happens when a keyframe enters or leaves, not per frame.
#[derive(Debug, Clone)]
pub struct DenseHbWorkspace<S: LieScalar> {
    /// What the reduction accumulates into and the caller reads.
    accumulator: DensePartial<S>,
    /// One subtree partial per recursion depth, as `deterministic_reduce` wants
    /// them.
    depths: Vec<Option<DensePartial<S>>>,
    /// The per-block transpose buffer of [`LandmarkBlock::add_dense_h_b`].
    leaf: DenseHbScratch<S>,
}

impl<S: LieScalar> Default for DenseHbWorkspace<S> {
    fn default() -> Self {
        Self {
            accumulator: DensePartial::zeros(0),
            depths: Vec::new(),
            leaf: DenseHbScratch::default(),
        }
    }
}

impl<S: LieScalar> DenseHbWorkspace<S> {
    /// Every buffer at the identity for an `n`-column ordering.
    fn prepare(&mut self, n: usize) {
        if self.accumulator.b.nrows() != n {
            // A subtree partial is only ever built by `identity_like` off the
            // accumulator, so dropping them here is what keeps the two agreeing
            // on the ordering's width.
            self.depths.clear();
        }
        self.accumulator.reset_sized(n);
    }
}

impl<S: LieScalar> DenseHbWorkspace<S> {
    /// Reduce landmark blocks in their existing deterministic join order.
    pub(super) fn reduce(
        &mut self,
        opt_size: usize,
        blocks: &[LandmarkBlock<S>],
    ) -> Result<(&mut DMatrix<S>, &mut DVector<S>), LinearizeError> {
        self.prepare(opt_size);
        let DenseHbWorkspace {
            accumulator,
            depths,
            leaf,
        } = self;
        deterministic_reduce::<DensePartial<S>, LinearizeError>(
            blocks.len(),
            accumulator,
            depths,
            &mut |i: usize, acc: &mut DensePartial<S>| {
                let block: &LandmarkBlock<S> =
                    blocks.get(i).ok_or(LinearizeError::LayoutOverflow)?;
                acc.accumulate(block, leaf)
            },
        )?;
        let DensePartial { h, b, .. } = accumulator;

        Ok((h, b))
    }

    /// Move the assembled system out without copying its buffers.
    pub(super) fn into_result(self) -> (DMatrix<S>, DVector<S>) {
        let DensePartial { h, b, .. } = self.accumulator;
        (h, b)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::calib::{CameraModel, PinholeParams};
    use crate::camera::CameraEnum;
    use crate::landmark::Landmark;
    use crate::lie::Se3;
    use crate::lie::So3;
    use crate::linearize::{LandmarkBlockOptions, RelPoseLin};
    use crate::types::POSE_SIZE;
    use crate::types::{AbsOrderMap, LandmarkId, TimeCamId};
    use nalgebra::{Matrix4, Matrix6};
    use nalgebra::{Vector2, Vector3};

    /// A two-frame ordering with one landmark hosted in the first frame and seen
    /// in both cameras, the second observation non-finite.
    ///
    /// `Landmark::add_observation` accepts that keypoint, the residual and its
    /// Huber weight carry the NaN past the Jacobian checks of
    /// `linearize_landmark`, and the QR spreads it across whole rows.
    fn a_block_carrying_a_nan() -> LandmarkBlock<f64> {
        let mut aom: AbsOrderMap = AbsOrderMap::new();
        for frame in 0..2i64 {
            aom.push(frame, POSE_SIZE).unwrap();
        }
        let host: TimeCamId = TimeCamId::new(0, 0);
        let mut lm: Landmark<f64> =
            Landmark::new(LandmarkId(7), host, Vector2::new(0.01, -0.02), 0.25);
        lm.obs.insert(host, Vector2::new(505.0, 510.0));
        lm.obs
            .insert(TimeCamId::new(0, 1), Vector2::new(f64::NAN, 512.0));

        let rel: Vec<RelPoseLin<f64>> = vec![
            RelPoseLin {
                t_t_h: Matrix4::identity(),
                d_rel_d_h: Matrix6::zeros(),
                d_rel_d_t: Matrix6::zeros(),
            },
            RelPoseLin {
                t_t_h: Se3::<f64>::new(So3::identity(), Vector3::new(0.1, 0.0, 0.0)).matrix(),
                d_rel_d_h: Matrix6::identity(),
                d_rel_d_t: -Matrix6::identity(),
            },
        ];
        let model: CameraModel<f64> = CameraModel::Pinhole(PinholeParams {
            fx: 379.0,
            fy: 379.0,
            cx: 505.0,
            cy: 510.0,
        });
        let cameras: Vec<CameraEnum<f64>> = vec![
            CameraEnum::from_model(&model).unwrap(),
            CameraEnum::from_model(&model).unwrap(),
        ];
        let options: LandmarkBlockOptions<f64> = LandmarkBlockOptions {
            huber_parameter: 0.5,
            obs_std_dev: 2.0,
            ..Default::default()
        };

        let mut block: LandmarkBlock<f64> = LandmarkBlock::allocate(
            lm.id,
            &lm,
            &|_, target: TimeCamId| Some(usize::from(target.cam_id == 1)),
            &aom,
            false,
        )
        .unwrap();
        block
            .linearize_landmark(&lm, &rel, &cameras, &options)
            .unwrap();
        block.perform_qr(&options).unwrap();
        block
    }

    /// The reduction adds a non-finite block at full width, bit for bit with the
    /// system the C++ path writes, and marks every column it wrote.
    ///
    /// Skipping the block's unobserved columns would drop those NaNs — the
    /// reduced camera system would come out finite where basalt's is not
    /// (decision D32) — and leave coefficients no join adds and no reset clears.
    #[test]
    fn a_non_finite_block_is_reduced_at_full_width() {
        let block: LandmarkBlock<f64> = a_block_carrying_a_nan();
        let n: usize = block.pose_columns().len();
        let mut scratch: DenseHbScratch<f64> = DenseHbScratch::default();

        let mut h: DMatrix<f64> = DMatrix::zeros(n, n);
        let mut b: DVector<f64> = DVector::zeros(n);
        block.add_dense_h_b(&mut h, &mut b, &mut scratch).unwrap();
        assert!(
            h.iter().any(|value| value.is_nan()),
            "the fixture was supposed to carry a NaN into H"
        );
        assert!(
            (POSE_SIZE..n).any(|column| h[(column, column)].is_nan()),
            "and past the columns the block observes"
        );

        let mut partial: DensePartial<f64> = DensePartial::zeros(n);
        partial.accumulate(&block, &mut scratch).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    partial.h[(i, j)].to_bits(),
                    h[(i, j)].to_bits(),
                    "H({i}, {j})"
                );
            }
            assert_eq!(partial.b[i].to_bits(), b[i].to_bits(), "b({i})");
        }

        // The join and the reset go over `columns`, so the fallback has to have
        // marked the whole width it wrote.
        assert_eq!(partial.columns, (0..n).collect::<Vec<usize>>());
        partial.reset();
        assert!(
            partial
                .h
                .iter()
                .chain(partial.b.iter())
                .all(|value| value.to_bits() == 0.0f64.to_bits()),
            "the reset left a coefficient behind"
        );
    }
}
