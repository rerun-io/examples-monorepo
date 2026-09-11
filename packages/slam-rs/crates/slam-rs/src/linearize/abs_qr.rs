//! Absolute-pose square-root linearization.
//! Landmark blocks, IMU factors and the prior form the reduced camera system.
//! Inputs are borrowed per call, allowing the estimator to retain the linearizer
//! while updating the window without self-referential storage.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Matrix4, Matrix6};

use crate::ba_base::BundleAdjustmentBase;
use crate::imu::{ImuBlock, ImuLinData, IntegratedImuMeasurement};
use crate::landmark::Landmark;
use crate::lie::{LieScalar, Se3};
use crate::linearize::landmark_block::{LandmarkBlock, LandmarkBlockOptions};
use crate::linearize::{DenseHbWorkspace, LinearizeError, RelPoseLin, linearize_relative_pose};
use crate::types::{AbsOrderMap, FrameId, LandmarkId, MargLinData, POSE_VEL_BIAS_SIZE, TimeCamId};

/// `LinearizationBase<Scalar, POSE_SIZE>::Options`,
/// without the `linearization_type` field: only `ABS_QR` is ported (decision D13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearizationOptions<S: LieScalar> {
    /// The landmark blocks' options.
    pub lb_options: LandmarkBlockOptions<S>,
}

impl<S: LieScalar> Default for LinearizationOptions<S> {
    /// Explicit defaults avoid requiring `S: Default` beyond `LieScalar`.
    fn default() -> Self {
        Self {
            lb_options: LandmarkBlockOptions::default(),
        }
    }
}

/// IMU measurements indexed by start timestamp; end time is start plus duration.
#[derive(Debug, Clone)]
pub struct ImuInput<'a, S: LieScalar> {
    /// Gravity and the two bias random-walk square-root weights.
    pub lin_data: ImuLinData<S>,
    /// The preintegrated intervals, in start-timestamp order.
    pub measurements: Vec<(i64, &'a IntegratedImuMeasurement<S>)>,
}

/// Borrowed inputs needed for a linearization call.
#[derive(Debug)]
pub struct LinearizationInputs<'a, S: LieScalar> {
    /// The square-root marginalization prior, if there is one.
    pub marg: Option<&'a MargLinData<S>>,
    /// The preintegrated IMU intervals, if the estimator is inertial.
    pub imu: Option<&'a ImuInput<'a, S>>,
    /// Restrict the landmarks to those hosted by these frames.
    pub used_frames: Option<&'a BTreeSet<FrameId>>,
    /// or to these landmarks.
    pub lost_landmarks: Option<&'a BTreeSet<LandmarkId>>,
    /// Frames whose pose Jacobians are zeroed and whose landmarks
    /// are not optimised.
    pub fixed_frames: Option<&'a BTreeSet<FrameId>>,
}

impl<S: LieScalar> Default for LinearizationInputs<'_, S> {
    /// An unrestricted visual-only problem without an IMU factor or prior.
    fn default() -> Self {
        Self {
            marg: None,
            imu: None,
            used_frames: None,
            lost_landmarks: None,
            fixed_frames: None,
        }
    }
}

/// Where one IMU block's two states sit in the ordering, resolved once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImuMeta {
    start_t: i64,
    end_t: i64,
    start_idx: usize,
    end_idx: usize,
}

/// The linearizer: one landmark block per landmark, plus IMU and prior.
#[derive(Debug, Clone)]
pub struct LinearizationAbsQR<S: LieScalar> {
    options: LinearizationOptions<S>,
    /// `landmark_ids`, sorted.
    landmark_ids: Vec<LandmarkId>,
    /// `landmark_blocks`, parallel to `landmark_ids`.
    landmark_blocks: Vec<LandmarkBlock<S>>,
    /// `landmark_block_idx` : the prefix sum of `numQ2rows()`.
    landmark_block_idx: Vec<usize>,
    /// `num_rows_Q2r`.
    num_rows_q2r: usize,
    /// `relative_pose_lin` as a dense table, so the blocks hold an
    /// index rather than a pointer. The `(host, target)` -> slot map that
    /// builds it is a local in [`Self::new`]: nothing needs it once the blocks
    /// have their indices.
    rel_pose_pairs: Vec<(TimeCamId, TimeCamId)>,
    rel_pose_lin: Vec<RelPoseLin<S>>,
    /// One per preintegrated interval.
    imu_meta: Vec<ImuMeta>,
    /// Filled by [`Self::linearize_problem`]; empty before it runs.
    imu_blocks: Vec<ImuBlock<S>>,
    /// `aom`.
    aom: AbsOrderMap,
}

impl<S: LieScalar> LinearizationAbsQR<S> {
    /// Allocate relative poses and landmark blocks in sorted landmark order,
    /// then build the prefix sum of reduced row counts. Sorting fixes row order.
    /// Copy Huber threshold and observation deviation from the estimator so options
    /// cannot disagree.
    pub fn new(
        estimator: &BundleAdjustmentBase<S>,
        aom: &AbsOrderMap,
        mut options: LinearizationOptions<S>,
        inputs: &LinearizationInputs<'_, S>,
    ) -> Result<Self, LinearizeError> {
        options.lb_options.huber_parameter = estimator.huber_thresh;
        options.lb_options.obs_std_dev = estimator.obs_std_dev;

        // one `RelPoseLin` per (host, target) pair.
        let mut rel_pose_pairs: Vec<(TimeCamId, TimeCamId)> = Vec::new();
        let mut rel_pose_index: BTreeMap<(TimeCamId, TimeCamId), usize> = BTreeMap::new();
        for (tcid_h, target_map) in estimator.lmdb.observations() {
            for tcid_t in target_map.keys() {
                let key: (TimeCamId, TimeCamId) = (*tcid_h, *tcid_t);
                if rel_pose_index.insert(key, rel_pose_pairs.len()).is_none() {
                    rel_pose_pairs.push(key);
                }
            }
        }
        let rel_pose_lin: Vec<RelPoseLin<S>> = vec![RelPoseLin::default(); rel_pose_pairs.len()];

        // Landmarks already iterate in sorted id order.
        let mut landmark_ids: Vec<LandmarkId> = Vec::new();
        for lm in estimator.lmdb.landmarks() {
            let keep: bool = match (inputs.used_frames, inputs.lost_landmarks) {
                (None, None) => true,
                (used, lost) => {
                    used.is_some_and(|f| f.contains(&lm.host_kf_id.frame_id))
                        || lost.is_some_and(|l| l.contains(&lm.id))
                }
            };
            if keep {
                landmark_ids.push(lm.id);
            }
        }

        let mut landmark_blocks: Vec<LandmarkBlock<S>> = Vec::with_capacity(landmark_ids.len());
        for &lm_id in &landmark_ids {
            let lm: &Landmark<S> = estimator
                .lmdb
                .get_landmark(lm_id)
                .ok_or(LinearizeError::UnknownLandmark(lm_id))?;
            let is_fixed: bool = inputs
                .fixed_frames
                .is_some_and(|f| f.contains(&lm.host_kf_id.frame_id));
            landmark_blocks.push(LandmarkBlock::allocate(
                lm_id,
                lm,
                &|host, target| rel_pose_index.get(&(host, target)).copied(),
                aom,
                is_fixed,
            )?);
        }

        let mut landmark_block_idx: Vec<usize> = Vec::with_capacity(landmark_blocks.len());
        let mut num_rows_q2r: usize = 0;
        for block in &landmark_blocks {
            landmark_block_idx.push(num_rows_q2r);
            num_rows_q2r = num_rows_q2r
                .checked_add(block.num_q2rows())
                .ok_or(LinearizeError::LayoutOverflow)?;
        }

        // Resolve IMU endpoints and full-state block sizes once at construction.
        // Check timestamp addition and offsets before the repeated scatter path (D32).
        let mut imu_meta: Vec<ImuMeta> = Vec::new();
        if let Some(imu) = inputs.imu {
            for (start_t, meas) in &imu.measurements {
                let end_t: i64 = start_t.checked_add(meas.get_dt_ns()).ok_or(
                    LinearizeError::ImuIntervalOverflow {
                        start: *start_t,
                        dt_ns: meas.get_dt_ns(),
                    },
                )?;
                let (start_idx, start_size) =
                    aom.get(*start_t).ok_or(LinearizeError::UnknownImuFrames {
                        start: *start_t,
                        end: end_t,
                    })?;
                let (end_idx, end_size) =
                    aom.get(end_t).ok_or(LinearizeError::UnknownImuFrames {
                        start: *start_t,
                        end: end_t,
                    })?;
                // Each IMU endpoint needs 15 columns; a six-column pose slot would overwrite its neighbor.
                for (frame, size) in [(*start_t, start_size), (end_t, end_size)] {
                    if size != POSE_VEL_BIAS_SIZE {
                        return Err(LinearizeError::ImuStateNotFullSize { frame, size });
                    }
                }
                imu_meta.push(ImuMeta {
                    start_t: *start_t,
                    end_t,
                    start_idx,
                    end_idx,
                });
            }
        }

        Ok(Self {
            options,
            landmark_ids,
            landmark_blocks,
            landmark_block_idx,
            num_rows_q2r,
            rel_pose_pairs,
            rel_pose_lin,
            imu_meta,
            imu_blocks: Vec::new(),
            aom: aom.clone(),
        })
    }

    /// Linearize in fixed order: relative poses, landmark blocks, IMU blocks, prior.
    /// Relative-pose Jacobians use frozen linearization points while values are
    /// reevaluated at current states (trap 7). Landmark folds are independent of
    /// thread count. Return error and numerical validity.
    pub fn linearize_problem(
        &mut self,
        estimator: &BundleAdjustmentBase<S>,
        inputs: &LinearizationInputs<'_, S>,
    ) -> Result<(S, bool), LinearizeError> {
        // 1. the relative poses.
        for (i, (tcid_h, tcid_t)) in self.rel_pose_pairs.iter().enumerate() {
            let Some(rpl) = self.rel_pose_lin.get_mut(i) else {
                // Unreachable: the two vectors are built together.
                return Err(LinearizeError::LayoutOverflow);
            };
            if tcid_h == tcid_t {
                rpl.t_t_h = Matrix4::identity();
                rpl.d_rel_d_h = Matrix6::zeros();
                rpl.d_rel_d_t = Matrix6::zeros();
                continue;
            }
            let state_h = estimator.get_pose_state_with_lin(tcid_h.frame_id)?;
            let state_t = estimator.get_pose_state_with_lin(tcid_t.frame_id)?;
            let t_i_c_h: &Se3<S> =
                estimator
                    .calib
                    .t_i_c
                    .get(tcid_h.cam_id)
                    .ok_or(LinearizeError::UnknownCamera {
                        cam_id: tcid_h.cam_id,
                        camera_count: estimator.calib.t_i_c.len(),
                    })?;
            let t_i_c_t: &Se3<S> =
                estimator
                    .calib
                    .t_i_c
                    .get(tcid_t.cam_id)
                    .ok_or(LinearizeError::UnknownCamera {
                        cam_id: tcid_t.cam_id,
                        camera_count: estimator.calib.t_i_c.len(),
                    })?;

            // Jacobians at the linearization point.
            let mut d_rel_d_h: Matrix6<S> = Matrix6::zeros();
            let mut d_rel_d_t: Matrix6<S> = Matrix6::zeros();
            let t_t_h: Se3<S> = linearize_relative_pose(
                &state_h,
                &state_t,
                t_i_c_h,
                t_i_c_t,
                Some(&mut d_rel_d_h),
                Some(&mut d_rel_d_t),
            );

            if let Some(fixed) = inputs.fixed_frames {
                if fixed.contains(&tcid_h.frame_id) {
                    d_rel_d_h = Matrix6::zeros();
                }
                if fixed.contains(&tcid_t.frame_id) {
                    d_rel_d_t = Matrix6::zeros();
                }
            }

            rpl.t_t_h = t_t_h.matrix();
            rpl.d_rel_d_h = d_rel_d_h;
            rpl.d_rel_d_t = d_rel_d_t;
        }

        // 2. Fold errors in landmark order and AND the validity flags.
        // Accumulate sequentially in landmark order.
        let cameras = estimator.cameras();
        let lb_options: LandmarkBlockOptions<S> = self.options.lb_options;
        let blocks: &mut [LandmarkBlock<S>] = &mut self.landmark_blocks;
        let ids: &[LandmarkId] = &self.landmark_ids;
        let rel_pose_lin: &[RelPoseLin<S>] = &self.rel_pose_lin;
        let mut numerically_valid: bool = true;
        let mut error: S = (0..blocks.len()).try_fold(S::zero(), |acc, i| {
            let lm_id: LandmarkId = *ids.get(i).ok_or(LinearizeError::LayoutOverflow)?;
            let lm: &Landmark<S> = estimator
                .lmdb
                .get_landmark(lm_id)
                .ok_or(LinearizeError::UnknownLandmark(lm_id))?;
            let block: &mut LandmarkBlock<S> =
                blocks.get_mut(i).ok_or(LinearizeError::LayoutOverflow)?;
            let contribution: S =
                block.linearize_landmark(lm, rel_pose_lin, cameras, &lb_options)?;
            numerically_valid = numerically_valid && !block.is_numerical_failure();
            Ok::<S, LinearizeError>(acc + contribution)
        })?;

        // 3a. the IMU blocks.
        self.imu_blocks.clear();
        if let Some(imu) = inputs.imu {
            for (meta, (_, meas)) in self.imu_meta.iter().zip(imu.measurements.iter()) {
                let start_state = estimator.frame_states.get(&meta.start_t).ok_or(
                    LinearizeError::UnknownImuFrames {
                        start: meta.start_t,
                        end: meta.end_t,
                    },
                )?;
                let end_state = estimator.frame_states.get(&meta.end_t).ok_or(
                    LinearizeError::UnknownImuFrames {
                        start: meta.start_t,
                        end: meta.end_t,
                    },
                )?;
                let block: ImuBlock<S> =
                    ImuBlock::linearize(meas, &imu.lin_data, start_state, end_state);
                error += block.error;
                self.imu_blocks.push(block);
            }
        }

        // 3b. the marginalization prior.
        if let Some(marg) = inputs.marg {
            error += estimator.compute_marg_prior_error(marg)?;
        }

        Ok((error, numerically_valid))
    }

    /// `performQR()` : eliminate every block's landmark columns.
    pub fn perform_qr(&mut self) -> Result<(), LinearizeError> {
        let options: LandmarkBlockOptions<S> = self.options.lb_options;
        for block in &mut self.landmark_blocks {
            block.perform_qr(&options)?;
        }
        Ok(())
    }

    /// Build the dense reduced camera system with a reusable accumulator.
    /// Fold landmarks in deterministic order, then add IMU factors and the prior.
    /// There are no pose-damping rows (D68).
    pub fn get_dense_h_b(
        &self,
        estimator: &BundleAdjustmentBase<S>,
        inputs: &LinearizationInputs<'_, S>,
    ) -> Result<(DMatrix<S>, DVector<S>), LinearizeError> {
        let mut workspace: DenseHbWorkspace<S> = DenseHbWorkspace::default();
        self.get_dense_h_b_into(estimator, inputs, &mut workspace)?;
        // The workspace is this call's own, so the assembled system moves out of
        // it rather than being copied: the borrow above ends with the statement.
        Ok(workspace.into_result())
    }

    /// [`Self::get_dense_h_b`] into buffers the caller keeps.
    ///
    /// The reduced system is the workspace's own accumulator, handed back by
    /// reference: the Levenberg-Marquardt loop builds one per inner step and
    /// throws it away, so nothing wants an owned copy. The caller may write
    /// into both — pins a fixed keyframe's
    /// rows in place — because the next call zeroes the whole square rather
    /// than only the columns the reduction recorded.
    pub fn get_dense_h_b_into<'w>(
        &self,
        estimator: &BundleAdjustmentBase<S>,
        inputs: &LinearizationInputs<'_, S>,
        workspace: &'w mut DenseHbWorkspace<S>,
    ) -> Result<(&'w mut DMatrix<S>, &'w mut DVector<S>), LinearizeError> {
        let opt_size: usize = self.aom.total_size();
        let (h, b) = workspace.reduce(opt_size, &self.landmark_blocks)?;

        // `add_dense_H_b_imu`.
        for (block, meta) in self.imu_blocks.iter().zip(self.imu_meta.iter()) {
            block.add_dense_h_b(meta.start_idx, meta.end_idx, h, b);
        }

        // `add_dense_H_b_marg_prior`. 's pose-damping
        // diagonal is not here: nothing sets it (D68).
        if let Some(marg) = inputs.marg {
            estimator.linearize_marg_prior(marg, &self.aom, h, b)?;
        }

        Ok((h, b))
    }

    /// Export the stacked square-root system for marginalization.
    /// Rows are landmark contributions, 15 per IMU interval, then the prior.
    /// No unused pose-damping rows are reserved (D68).
    pub fn get_dense_q2jp_q2r(
        &self,
        estimator: &BundleAdjustmentBase<S>,
        inputs: &LinearizationInputs<'_, S>,
    ) -> Result<(DMatrix<S>, DVector<S>), LinearizeError> {
        let poses_size: usize = self.aom.total_size();
        let mut total_size: usize = self.num_rows_q2r;

        let imu_start_idx: usize = total_size;
        total_size += self.imu_meta.len() * POSE_VEL_BIAS_SIZE;

        let marg_start_idx: usize = total_size;
        if let Some(marg) = inputs.marg {
            total_size += marg.h.nrows();
        }

        let mut q2jp: DMatrix<S> = DMatrix::zeros(total_size, poses_size);
        let mut q2r: DVector<S> = DVector::zeros(total_size);

        for (block, &start) in self
            .landmark_blocks
            .iter()
            .zip(self.landmark_block_idx.iter())
        {
            block.get_dense_q2jp_q2r(&mut q2jp, &mut q2r, start)?;
        }

        let mut start_idx: usize = imu_start_idx;
        for (block, meta) in self.imu_blocks.iter().zip(self.imu_meta.iter()) {
            block.add_dense_q2jp_q2r(meta.start_idx, meta.end_idx, start_idx, &mut q2jp, &mut q2r);
            start_idx += POSE_VEL_BIAS_SIZE;
        }

        // `get_dense_Q2Jp_Q2r_marg_prior`, trap 8: the prior's
        // residual is re-anchored at the current state as `H * delta + b`.
        if let Some(marg) = inputs.marg {
            // Prior columns occupy the window prefix. Check ordering in both dense and
            // square-root exports to avoid attaching one frame's columns to another.
            estimator.check_marg_prior_order(marg, &self.aom)?;
            let delta: DVector<S> = estimator.compute_delta(&marg.order)?;
            let (marg_rows, marg_cols) = (marg.h.nrows(), marg.h.ncols());
            for i in 0..marg_rows {
                for j in 0..marg_cols {
                    q2jp[(marg_start_idx + i, j)] = marg.h[(i, j)];
                }
                let mut acc: S = S::zero();
                for j in 0..marg_cols {
                    acc += marg.h[(i, j)] * delta[j];
                }
                q2r[marg_start_idx + i] = acc + marg.b[i];
            }
        }

        Ok((q2jp, q2r))
    }

    /// Apply pose increments to landmarks and return the predicted cost change.
    /// Sum blocks in order, then IMU factors, then the prior. The input increment
    /// must already be negated to compensate for the residual sign.
    pub fn back_substitute(
        &mut self,
        estimator: &mut BundleAdjustmentBase<S>,
        inputs: &LinearizationInputs<'_, S>,
        pose_inc: &DVector<S>,
    ) -> Result<S, LinearizeError> {
        if pose_inc.nrows() != self.aom.total_size() {
            return Err(LinearizeError::PoseIncrementSize {
                expected: self.aom.total_size(),
                found: pose_inc.nrows(),
            });
        }

        // Fold in landmark order, subtracting each block's change from the
        // accumulator. The fold is deterministic and independent of thread count.
        let blocks: &mut [LandmarkBlock<S>] = &mut self.landmark_blocks;
        let ids: &[LandmarkId] = &self.landmark_ids;
        let lmdb: &mut crate::landmark::LandmarkDatabase<S> = &mut estimator.lmdb;
        let mut l_diff: S = (0..blocks.len()).try_fold(S::zero(), |acc, i| {
            let lm_id: LandmarkId = *ids.get(i).ok_or(LinearizeError::LayoutOverflow)?;
            let lm: &mut Landmark<S> = lmdb
                .get_landmark_mut(lm_id)
                .ok_or(LinearizeError::UnknownLandmark(lm_id))?;
            let block: &mut LandmarkBlock<S> =
                blocks.get_mut(i).ok_or(LinearizeError::LayoutOverflow)?;
            let mut value: S = acc;
            block.back_substitute(lm, pose_inc, &mut value)?;
            Ok::<S, LinearizeError>(value)
        })?;

        for (block, meta) in self.imu_blocks.iter().zip(self.imu_meta.iter()) {
            block.back_substitute(meta.start_idx, meta.end_idx, pose_inc, &mut l_diff);
        }

        if let Some(marg) = inputs.marg {
            let marg_size: usize = marg.h.ncols();
            let marg_pose_inc: DVector<S> = pose_inc.rows(0, marg_size).into_owned();
            l_diff += estimator.compute_marg_prior_model_cost_change(marg, &marg_pose_inc)?;
        }

        Ok(l_diff)
    }

    /// The options the blocks were built with, `options_`
    /// including the Huber threshold and the
    /// pixel sigma copied off the estimator.
    pub fn options(&self) -> &LinearizationOptions<S> {
        &self.options
    }

    /// The landmark blocks, in the order the stacked system uses them.
    pub fn landmark_blocks(&self) -> &[LandmarkBlock<S>] {
        &self.landmark_blocks
    }

    /// The landmark ids, parallel to [`Self::landmark_blocks`].
    pub fn landmark_ids(&self) -> &[LandmarkId] {
        &self.landmark_ids
    }

    /// The relative poses, indexed as the blocks index them.
    pub fn relative_poses(&self) -> &[RelPoseLin<S>] {
        &self.rel_pose_lin
    }
}
