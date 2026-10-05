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

    /// Compute one triangle while preserving row and landmark addition order.
    fn accumulate_symmetric(
        &mut self,
        block: &LandmarkBlock<S>,
        scratch: &mut DenseHbScratch<S>,
        rows: &mut Vec<S>,
    ) -> Result<(), LinearizeError> {
        const LANES: usize = 8;
        block.check_dense_h_b_size(&self.h, &self.b)?;
        if !block.active_writeback_is_exact() {
            return block.add_dense_h_b(&mut self.h, &mut self.b, scratch);
        }
        let columns = block.active_cols();
        let live = columns.len();
        let count = block.num_q2rows();
        let stride = (live + 1).div_ceil(LANES) * LANES;
        rows.clear();
        rows.resize(count * stride, S::zero());
        let storage = block.storage();
        for (slot, &column) in columns.iter().enumerate() {
            for row in 0..count {
                rows[row * stride + slot] = storage[(row + 3, column)];
            }
        }
        let residual = block.layout().4;
        for row in 0..count {
            rows[row * stride + live] = storage[(row + 3, residual)];
        }
        for (slot, &i) in columns.iter().enumerate() {
            for lo in ((slot / LANES * LANES)..stride).step_by(LANES) {
                let mut partial = [S::zero(); LANES];
                for row in rows.chunks_exact(stride) {
                    let factor = row[slot];
                    for (sum, &value) in partial.iter_mut().zip(&row[lo..lo + LANES]) {
                        *sum += factor * value;
                    }
                }
                for (offset, value) in partial.into_iter().enumerate() {
                    let j = lo + offset;
                    if j < slot {
                        continue;
                    }
                    if j < live {
                        let column = columns[j];
                        self.h[(i, column)] += value;
                        if j != slot {
                            self.h[(column, i)] += value;
                        }
                    } else if j == live {
                        self.b[i] += value;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reusable dense accumulator and row scratch for serial symmetric assembly.
/// Blocks scatter in their existing order. Buffers resize with the window
/// ordering and are cleared before each assembly.
#[derive(Debug, Clone)]
pub struct DenseHbWorkspace<S: LieScalar> {
    /// What the reduction accumulates into and the caller reads.
    accumulator: DensePartial<S>,
    /// One transpose buffer reused across blocks on the serial path.
    leaf: DenseHbScratch<S>,
    /// Row-major active columns plus residual, reused across landmarks.
    rows: Vec<S>,
}

impl<S: LieScalar> Default for DenseHbWorkspace<S> {
    fn default() -> Self {
        Self {
            accumulator: DensePartial::zeros(0),
            leaf: DenseHbScratch::default(),
            rows: Vec::new(),
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
        for block in blocks {
            self.accumulator
                .accumulate_symmetric(block, &mut self.leaf, &mut self.rows)?;
        }
        let accumulator = &mut self.accumulator;
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

    /// Scalar reference: every coefficient uses the original row and landmark order.
    fn reference_reduce<S: LieScalar>(
        n: usize,
        blocks: &[LandmarkBlock<S>],
    ) -> (DMatrix<S>, DVector<S>) {
        let mut h = DMatrix::zeros(n, n);
        let mut b = DVector::zeros(n);
        for block in blocks {
            let columns: Vec<_> = if block.active_writeback_is_exact() {
                block.active_cols().to_vec()
            } else {
                block.pose_columns().collect()
            };
            let storage = block.storage();
            for &i in &columns {
                for &j in &columns {
                    let mut value = S::zero();
                    for row in 3..3 + block.num_q2rows() {
                        value += storage[(row, i)] * storage[(row, j)];
                    }
                    h[(i, j)] += value;
                }
                let mut value = S::zero();
                for row in 3..3 + block.num_q2rows() {
                    value += storage[(row, i)] * storage[(row, block.layout().4)];
                }
                b[i] += value;
            }
        }
        (h, b)
    }

    #[test]
    fn symmetric_dense_assembly_matches_reference_in_estimator_windows() {
        compare_symmetric_dense::<f32>();
        compare_symmetric_dense::<f64>();
    }

    fn compare_symmetric_dense<S: LieScalar>() {
        use crate::camera::CameraEnum;
        use crate::estimator::{FlowObservations, FrameOutcome, SqrtKeypointVio};
        use crate::types::KeypointId;
        use std::sync::Arc;
        let calibration = crate::calib::Calibration::<f64>::from_json_str(include_str!(
            "../../tests/fixtures/msdmi_calib.json"
        ))
        .unwrap();
        let mut config = crate::config::VioConfig::from_json_str(include_str!(
            "../../../../configs/msdmi_config.json"
        ))
        .unwrap();
        config.vio_min_frames_after_kf = 5;
        config.vio_new_kf_keypoints_thresh = 2.0;
        let mut candidate =
            SqrtKeypointVio::<S>::with_default_gravity(calibration.cast(), config).unwrap();
        let mut observations = FlowObservations::new(0, calibration.t_i_c.len());
        for id in 0..36 {
            let point = calibration.t_i_c[0]
                * Vector3::new(
                    (id % 6) as f64 * 0.15 - 0.4,
                    (id / 6) as f64 * 0.15 - 0.4,
                    3.0 + (id % 3) as f64 * 0.2,
                );
            for (cam, pixels) in observations.cameras.iter_mut().enumerate() {
                let p = calibration.t_i_c[cam].inverse() * point;
                let model = CameraEnum::from_model(&calibration.intrinsics[cam]).unwrap();
                let mut pixel = Vector2::zeros();
                let mut jac = nalgebra::Matrix2x4::zeros();
                assert!(model.project_with_jacobian(
                    &nalgebra::Vector4::new(p.x, p.y, p.z, 1.0),
                    &mut pixel,
                    &mut jac
                ));
                pixels.insert(KeypointId(id), pixel.cast());
            }
        }
        for n in 0..=65 {
            let imu = crate::imu::ImuSample {
                t_ns: n * 5_000_000,
                gyro: Vector3::zeros(),
                accel: Vector3::new(0.0, 0.0, 9.81),
            };
            candidate.push_imu(imu);
        }
        let mut solved = false;
        for frame in 0..16 {
            observations.t_ns = frame * 20_000_000;
            let frame = Arc::new(observations.clone());
            let actual = candidate.process_frame(frame, None).unwrap();
            if let FrameOutcome::Measured(stats) = actual {
                solved |= !stats.lm.is_empty() && stats.num_landmarks > 0;
            }
            let mut order = AbsOrderMap::new();
            for &t in candidate.ba.frame_poses.keys() {
                order.push(t, POSE_SIZE).unwrap();
            }
            for &t in candidate.ba.frame_states.keys() {
                order.push(t, crate::types::POSE_VEL_BIAS_SIZE).unwrap();
            }
            let inputs = crate::linearize::LinearizationInputs::default();
            let mut linearizer = crate::linearize::LinearizationAbsQR::new(
                &candidate.ba,
                &order,
                crate::linearize::LinearizationOptions::default(),
                &inputs,
            )
            .unwrap();
            linearizer
                .linearize_problem(&candidate.ba, &inputs, None)
                .unwrap();
            linearizer.perform_qr(None).unwrap();
            let (h, b) = reference_reduce(order.total_size(), linearizer.landmark_blocks());
            let mut workspace = DenseHbWorkspace::default();
            let (actual_h, actual_b) = workspace
                .reduce(order.total_size(), linearizer.landmark_blocks())
                .unwrap();
            for (expected, actual) in h
                .iter()
                .chain(b.iter())
                .zip(actual_h.iter().chain(actual_b.iter()))
            {
                assert_eq!(actual.to_f64().to_bits(), expected.to_f64().to_bits());
            }
        }
        assert!(
            solved,
            "the synthetic stereo window must exercise a joint solve"
        );
    }

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

    /// Symmetric assembly retains the typed size refusal.
    #[test]
    fn an_undersized_system_returns_a_size_error() {
        let block = a_block_carrying_a_nan();
        let expected = block.pose_columns().len();
        let found = expected - 1;
        let mut workspace = DenseHbWorkspace::default();
        assert!(matches!(
            workspace.reduce(found, &[block]),
            Err(LinearizeError::StackedSystemSize { expected: e, found: f })
                if e == expected && f == found
        ));
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
        partial
            .accumulate_symmetric(&block, &mut scratch, &mut Vec::new())
            .unwrap();
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

        let mut workspace = DenseHbWorkspace::default();
        let (parallel_h, parallel_b) = workspace.reduce(n, &[block]).unwrap();
        for (expected, actual) in h
            .iter()
            .chain(b.iter())
            .zip(parallel_h.iter().chain(parallel_b.iter()))
        {
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }
}
