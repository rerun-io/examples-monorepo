//! Reusable storage for deterministic dense landmark reduction.

use nalgebra::{DMatrix, DVector};

use super::LinearizeError;
use super::landmark_block::{DenseHbScratch, LandmarkBlock};
use crate::lie::LieScalar;

/// Dense accumulator, reset to positive zero before each assembly.
#[derive(Debug, Clone)]
struct DensePartial<S: LieScalar> {
    h: DMatrix<S>,
    b: DVector<S>,
}

impl<S: LieScalar> DensePartial<S> {
    fn zeros(n: usize) -> Self {
        Self {
            h: DMatrix::zeros(n, n),
            b: DVector::zeros(n),
        }
    }

    /// Clear the full system: IMU, prior and fixed-keyframe writes also use it.
    fn reset_sized(&mut self, n: usize) {
        if self.b.nrows() == n {
            self.h.fill(S::zero());
            self.b.fill(S::zero());
        } else {
            *self = Self::zeros(n);
        }
    }

    /// Add one landmark, preserving full-width writes when required.
    fn accumulate(
        &mut self,
        block: &LandmarkBlock<S>,
        scratch: &mut DenseHbScratch<S>,
    ) -> Result<(), LinearizeError> {
        if block.active_writeback_is_exact() {
            block.add_dense_h_b_active(&mut self.h, &mut self.b, scratch)?;
        } else {
            block.add_dense_h_b(&mut self.h, &mut self.b, scratch)?;
        }
        Ok(())
    }
}

/// Reusable dense accumulator and per-landmark transpose scratch.
/// Blocks accumulate sequentially in their existing order. Buffers resize when
/// the window ordering changes and are cleared before each assembly.
#[derive(Debug, Clone)]
pub struct DenseHbWorkspace<S: LieScalar> {
    /// What the reduction accumulates into and the caller reads.
    accumulator: DensePartial<S>,
    /// The per-block transpose buffer of [`LandmarkBlock::add_dense_h_b`].
    leaf: DenseHbScratch<S>,
}

impl<S: LieScalar> Default for DenseHbWorkspace<S> {
    fn default() -> Self {
        Self {
            accumulator: DensePartial::zeros(0),
            leaf: DenseHbScratch::default(),
        }
    }
}

impl<S: LieScalar> DenseHbWorkspace<S> {
    /// Accumulate landmark blocks in their existing order.
    pub(super) fn reduce(
        &mut self,
        opt_size: usize,
        blocks: &[LandmarkBlock<S>],
    ) -> Result<(&mut DMatrix<S>, &mut DVector<S>), LinearizeError> {
        self.accumulator.reset_sized(opt_size);
        let DenseHbWorkspace { accumulator, leaf } = self;
        for block in blocks {
            accumulator.accumulate(block, leaf)?;
        }
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

    /// Non-finite blocks must write every column so unobserved columns retain NaNs.
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

        partial.reset_sized(partial.b.nrows());
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
