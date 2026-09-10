//! The absolute-pose square-root linearizer.
//!
//! `LinearizationAbsQR<Scalar, POSE_SIZE>`
//! (`include/basalt/linearization/linearization_abs_qr.hpp`,
//! `src/linearization/linearization_abs_qr.cpp`): owns one landmark block per
//! landmark, one IMU block per preintegrated interval, and the marginalization
//! prior, and turns them into the dense reduced camera system the
//! Levenberg-Marquardt loop solves.
//!
//! **Borrowing.** C++ keeps raw pointers to the estimator, the marginalization
//! data and the IMU data for the linearizer's whole life (`:56-64`). The port
//! passes them in at each call instead — `estimator` and a
//! [`LinearizationInputs`] — so nothing here is self-referential and stage S8
//! can hold the linearizer across an iteration while it mutates the window.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Matrix4, Matrix6};

use crate::ba_base::{BundleAdjustmentBase, compute_rel_pose};
use crate::imu::{ImuBlock, ImuLinData, IntegratedImuMeasurement};
use crate::landmark::Landmark;
use crate::lie::{LieScalar, Se3};
use crate::linearize::landmark_block::{LandmarkBlock, LandmarkBlockOptions};
use crate::linearize::reduce::deterministic_reduce_scalar;
use crate::linearize::{DenseHbWorkspace, LinearizeError, RelPoseLin};
use crate::types::{AbsOrderMap, FrameId, LandmarkId, MargLinData, POSE_VEL_BIAS_SIZE, TimeCamId};

/// `LinearizationBase<Scalar, POSE_SIZE>::Options` (`linearization_base.hpp:23-26`),
/// without the `linearization_type` field: only `ABS_QR` is ported (decision D13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearizationOptions<S: LieScalar> {
    /// The landmark blocks' options (`:24`).
    pub lb_options: LandmarkBlockOptions<S>,
}

impl<S: LieScalar> Default for LinearizationOptions<S> {
    /// basalt's defaults. Written out rather than derived: `#[derive(Default)]`
    /// would demand `S: Default`, which `LieScalar` does not.
    fn default() -> Self {
        Self {
            lb_options: LandmarkBlockOptions::default(),
        }
    }
}

/// `ImuLinData<Scalar>` plus the measurements it indexes
/// (`imu_types.h:306-315`), which C++ carries as a `std::map` of pointers.
///
/// The key is the start timestamp of the interval; the end timestamp is
/// `start + measurement.get_dt_ns()` (`imu_block.hpp:29-30`).
#[derive(Debug, Clone)]
pub struct ImuInput<'a, S: LieScalar> {
    /// Gravity and the two bias random-walk square-root weights.
    pub lin_data: ImuLinData<S>,
    /// The preintegrated intervals, in start-timestamp order.
    pub measurements: Vec<(i64, &'a IntegratedImuMeasurement<S>)>,
}

/// Everything `LinearizationAbsQR`'s constructor takes as a pointer in C++
/// (`linearization_abs_qr.hpp:41-46`).
///
/// `last_state_to_marg` is not here: C++ takes it and immediately `UNUSED`s it
/// (`linearization_abs_qr.cpp:67`).
#[derive(Debug)]
pub struct LinearizationInputs<'a, S: LieScalar> {
    /// The square-root marginalization prior, if there is one.
    pub marg: Option<&'a MargLinData<S>>,
    /// The preintegrated IMU intervals, if the estimator is inertial.
    pub imu: Option<&'a ImuInput<'a, S>>,
    /// Restrict the landmarks to those hosted by these frames (`:117-122`).
    pub used_frames: Option<&'a BTreeSet<FrameId>>,
    /// ...or to these landmarks (`:120-121`).
    pub lost_landmarks: Option<&'a BTreeSet<LandmarkId>>,
    /// Frames whose pose Jacobians are zeroed (`:223-224`) and whose landmarks
    /// are not optimised (`:140`).
    pub fixed_frames: Option<&'a BTreeSet<FrameId>>,
}

impl<S: LieScalar> Default for LinearizationInputs<'_, S> {
    /// Every C++ pointer null: a visual-only problem with no prior and no
    /// restriction (`linearization_abs_qr.hpp:41-46` default arguments).
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
    /// `landmark_ids` (`:104`), sorted (`:127`).
    landmark_ids: Vec<LandmarkId>,
    /// `landmark_blocks` (`:105`), parallel to `landmark_ids`.
    landmark_blocks: Vec<LandmarkBlock<S>>,
    /// `landmark_block_idx` (`:111`): the prefix sum of `numQ2rows()`.
    landmark_block_idx: Vec<usize>,
    /// `num_rows_Q2r` (`:135`).
    num_rows_q2r: usize,
    /// `relative_pose_lin` (`:126`) as a dense table, so the blocks hold an
    /// index rather than a pointer. The `(host, target)` -> slot map that
    /// builds it is a local in [`Self::new`]: nothing needs it once the blocks
    /// have their indices.
    rel_pose_pairs: Vec<(TimeCamId, TimeCamId)>,
    rel_pose_lin: Vec<RelPoseLin<S>>,
    /// One per preintegrated interval (`:106`, `:163-167`).
    imu_meta: Vec<ImuMeta>,
    /// Filled by [`Self::linearize_problem`]; empty before it runs.
    imu_blocks: Vec<ImuBlock<S>>,
    /// `aom` (`:119`).
    aom: AbsOrderMap,
}

impl<S: LieScalar> LinearizationAbsQR<S> {
    /// The constructor (`linearization_abs_qr.cpp:50-172`).
    ///
    /// Allocates one relative pose per (host, target) pair of the landmark
    /// database, selects and **sorts** the landmark ids (`:115-127` — the
    /// comment says "for better visualization", but it is also what makes the
    /// row order of the stacked system reproducible), allocates every landmark
    /// block, and builds the prefix sum of their `Q₂` row counts.
    ///
    /// C++ asserts that the options' Huber threshold and observation sigma are
    /// the estimator's (`:69-73`); the port copies them from the estimator
    /// instead, so they cannot disagree.
    pub fn new(
        estimator: &BundleAdjustmentBase<S>,
        aom: &AbsOrderMap,
        mut options: LinearizationOptions<S>,
        inputs: &LinearizationInputs<'_, S>,
    ) -> Result<Self, LinearizeError> {
        options.lb_options.huber_parameter = estimator.huber_thresh;
        options.lb_options.obs_std_dev = estimator.obs_std_dev;

        // `:76-111`: one `RelPoseLin` per (host, target) pair.
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

        // `:115-127`. The database already keeps its landmarks id-sorted, so the
        // `std::sort` of `:127` is the iteration order here.
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

        // `:132-150`.
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

        // `:152-161`.
        let mut landmark_block_idx: Vec<usize> = Vec::with_capacity(landmark_blocks.len());
        let mut num_rows_q2r: usize = 0;
        for block in &landmark_blocks {
            landmark_block_idx.push(num_rows_q2r);
            num_rows_q2r = num_rows_q2r
                .checked_add(block.num_q2rows())
                .ok_or(LinearizeError::LayoutOverflow)?;
        }

        // `:163-167`. C++ computes `start_t + dt` unchecked and then indexes
        // the ordering with `at()` inside every method that uses the block
        // (`imu_block.hpp:89-93` and its four siblings), which throws on a
        // missing frame and silently misplaces a 15x15 write on a frame that is
        // in the ordering with a *pose-sized* slot. Both are resolved here,
        // once, so the per-iteration path cannot fail (decision D32).
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
                // An IMU factor is 15 columns wide at each end
                // (`imu_block.hpp:21`). A pose-only slot next to one is not a
                // narrower factor, it is a different problem: C++ would write
                // fifteen columns over a six-column slot and into its neighbour.
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

    /// `linearizeProblem(&numerically_valid)` (`:200-277`): the residual, its
    /// Jacobians and the total error at the current state.
    ///
    /// Three stages, in this order and with this summation order:
    ///
    /// 1. every relative pose and its two Jacobians (`:207-241`), taken at the
    ///    **linearization point** (`getPoseLin`) and then re-evaluated for its
    ///    value alone at the current state when either end is frozen
    ///    (`:229-232`) — first-estimate Jacobians, trap 7;
    /// 2. the landmark blocks, a `parallel_deterministic_reduce` in C++
    ///    (**site 1 of 4**, `:262`), reproduced here through
    ///    `crate::linearize::reduce`, which is TBB's balanced join tree and
    ///    not a fold;
    /// 3. the IMU blocks (`:266-268`) and then the marginalization prior
    ///    (`:270-274`), both serial in C++ too.
    ///
    /// Returns `(error, numerically_valid)`.
    pub fn linearize_problem(
        &mut self,
        estimator: &BundleAdjustmentBase<S>,
        inputs: &LinearizationInputs<'_, S>,
    ) -> Result<(S, bool), LinearizeError> {
        // `:201-204`.

        // 1. the relative poses (`:207-241`).
        for (i, (tcid_h, tcid_t)) in self.rel_pose_pairs.iter().enumerate() {
            let Some(rpl) = self.rel_pose_lin.get_mut(i) else {
                // Unreachable: the two vectors are built together.
                return Err(LinearizeError::LayoutOverflow);
            };
            if tcid_h == tcid_t {
                // `:235-239`.
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

            // `:219-221`: Jacobians at the linearization point.
            let mut d_rel_d_h: Matrix6<S> = Matrix6::zeros();
            let mut d_rel_d_t: Matrix6<S> = Matrix6::zeros();
            let mut t_t_h: Se3<S> = compute_rel_pose(
                state_h.pose_lin(),
                t_i_c_h,
                state_t.pose_lin(),
                t_i_c_t,
                Some(&mut d_rel_d_h),
                Some(&mut d_rel_d_t),
            );

            // `:223-224`.
            if let Some(fixed) = inputs.fixed_frames {
                if fixed.contains(&tcid_h.frame_id) {
                    d_rel_d_h = Matrix6::zeros();
                }
                if fixed.contains(&tcid_t.frame_id) {
                    d_rel_d_t = Matrix6::zeros();
                }
            }

            // `:229-232`: the value, and only the value, at the current state.
            if state_h.is_linearized() || state_t.is_linearized() {
                t_t_h =
                    compute_rel_pose(state_h.pose(), t_i_c_h, state_t.pose(), t_i_c_t, None, None);
            }

            rpl.t_t_h = t_t_h.matrix();
            rpl.d_rel_d_h = d_rel_d_h;
            rpl.d_rel_d_t = d_rel_d_t;
        }

        // 2. the landmark blocks. **Reduction site 1 of 4** (`:246-262`):
        // `tbb::parallel_deterministic_reduce` over `[0, num_landmarks)`,
        // summing the per-block error and ANDing the validity. The sum follows
        // TBB's balanced join tree, not a left fold — see
        // [`crate::linearize::reduce`]. The `&&` needs no order.
        let cameras = estimator.cameras();
        let lb_options: LandmarkBlockOptions<S> = self.options.lb_options;
        let blocks: &mut [LandmarkBlock<S>] = &mut self.landmark_blocks;
        let ids: &[LandmarkId] = &self.landmark_ids;
        let rel_pose_lin: &[RelPoseLin<S>] = &self.rel_pose_lin;
        let mut numerically_valid: bool = true;
        let mut error: S =
            deterministic_reduce_scalar::<S, LinearizeError>(blocks.len(), &mut |i, acc| {
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
                Ok(acc + contribution)
            })?;

        // 3a. the IMU blocks (`:266-268`).
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

        // 3b. the marginalization prior (`:270-274`).
        if let Some(marg) = inputs.marg {
            error += estimator.compute_marg_prior_error(marg)?;
        }

        Ok((error, numerically_valid))
    }

    /// `performQR()` (`:280-287`): eliminate every block's landmark columns.
    pub fn perform_qr(&mut self) -> Result<(), LinearizeError> {
        let options: LandmarkBlockOptions<S> = self.options.lb_options;
        for block in &mut self.landmark_blocks {
            block.perform_qr(&options)?;
        }
        Ok(())
    }

    /// `get_dense_H_b(H, b)` (`:511-563`): the reduced camera system.
    ///
    /// **Reduction site 4 of 4** (`:550`). C++ gives every TBB task its own full
    /// `total_size` x `total_size` partial and adds them at the joins
    /// (`:513-542`); the port walks the same join tree through
    /// `crate::linearize::reduce`, which reuses one accumulator per recursion
    /// **depth** rather than one per task — the same sum, `ceil(log2 n)`
    /// matrices instead of `n`.
    ///
    /// The order of the additions after the landmark blocks is basalt's: IMU
    /// (`:553`), then the marginalization prior (`:559`). `:556`'s pose damping
    /// is not one of them (D68).
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
    /// into both — `sqrt_keypoint_vio.cpp:1395-1406` pins a fixed keyframe's
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

        // `add_dense_H_b_imu` (`:640-653`).
        for (block, meta) in self.imu_blocks.iter().zip(self.imu_meta.iter()) {
            block.add_dense_h_b(meta.start_idx, meta.end_idx, h, b);
        }

        // `add_dense_H_b_marg_prior` (`:600-631`). `:595-598`'s pose-damping
        // diagonal is not here: nothing sets it (D68).
        if let Some(marg) = inputs.marg {
            estimator.linearize_marg_prior(marg, &self.aom, h, b)?;
        }

        Ok((h, b))
    }

    /// `get_dense_Q2Jp_Q2r(Q2Jp, Q2r)` (`:462-509`): the stacked square-root
    /// system the marginalization of stage S7 consumes.
    ///
    /// The row budget, in this order (`:464-482`): the landmark blocks'
    /// `num_rows_Q2r`, then 15 rows per IMU interval, then the marginalization
    /// prior's rows. C++ reserves `aom.total_size` rows of pose damping between
    /// the last two; nothing sets it (D68), so the port has no such rows.
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

        // `:484-493`.
        for (block, &start) in self
            .landmark_blocks
            .iter()
            .zip(self.landmark_block_idx.iter())
        {
            block.get_dense_q2jp_q2r(&mut q2jp, &mut q2r, start)?;
        }

        // `:496-502`.
        let mut start_idx: usize = imu_start_idx;
        for (block, meta) in self.imu_blocks.iter().zip(self.imu_meta.iter()) {
            block.add_dense_q2jp_q2r(meta.start_idx, meta.end_idx, start_idx, &mut q2jp, &mut q2r);
            start_idx += POSE_VEL_BIAS_SIZE;
        }

        // `get_dense_Q2Jp_Q2r_marg_prior` (`:573-593`), trap 8: the prior's
        // residual is re-anchored at the current state as `H * delta + b`.
        if let Some(marg) = inputs.marg {
            // The prior's columns are written into the *first* `marg_cols`
            // columns of the stacked system (`:587-589`), which is only correct
            // if its ordering is the window's prefix. `linearizeMargPrior`
            // asserts exactly that before it does the same thing in Hessian
            // form (`ba_base.cpp:383-388`); the square-root export in C++ does
            // not, and would silently attach a frame's columns to another
            // frame. The port checks both paths.
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

    /// `backSubstitute(pose_inc)` (`:297-321`): apply the pose increment to the
    /// landmarks and return the model cost change the whole problem predicts.
    ///
    /// **Reduction site 2 of 4** (`:307`): the per-block `l_diff` is summed in
    /// block order. The IMU blocks (`:309-311`) and the prior (`:313-318`) add
    /// theirs afterwards, in that order.
    ///
    /// The increment this takes is basalt's **negated** one
    /// (`sqrt_keypoint_vio.cpp:1450`), the other half of the flipped residual
    /// sign of `ba_utils.h:117`.
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

        // **Reduction site 2 of 4** (`:301-307`): TBB's join tree again, with
        // the subtraction inside the leaf, exactly as `backSubstitute` mutates
        // the accumulator it is handed (`:302`).
        let blocks: &mut [LandmarkBlock<S>] = &mut self.landmark_blocks;
        let ids: &[LandmarkId] = &self.landmark_ids;
        let lmdb: &mut crate::landmark::LandmarkDatabase<S> = &mut estimator.lmdb;
        let mut l_diff: S =
            deterministic_reduce_scalar::<S, LinearizeError>(blocks.len(), &mut |i, acc| {
                let lm_id: LandmarkId = *ids.get(i).ok_or(LinearizeError::LayoutOverflow)?;
                let lm: &mut Landmark<S> = lmdb
                    .get_landmark_mut(lm_id)
                    .ok_or(LinearizeError::UnknownLandmark(lm_id))?;
                let block: &mut LandmarkBlock<S> =
                    blocks.get_mut(i).ok_or(LinearizeError::LayoutOverflow)?;
                let mut value: S = acc;
                block.back_substitute(lm, pose_inc, &mut value)?;
                Ok(value)
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
    /// (`linearization_abs_qr.hpp:102`), including the Huber threshold and the
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
