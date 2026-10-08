//! Landmark rows `[J_p | pad | J_l(3) | r]` and their QR elimination.
//! Three reflections triangularize landmark columns. The first three rows
//! support back-substitution; remaining rows form the reduced camera system.

use kornia_staging_algebra::Scalar;
use kornia_staging_slam::sqrt_ba::DenseBlock;
use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Vector2};

use crate::ba_base::linearize_point;
use crate::camera::SlamCamera;
use crate::landmark::Landmark;
use crate::lie::{c};
use crate::linearize::{LinearizeError, RelPoseLin};
use crate::types::{AbsOrderMap, LandmarkId, POSE_SIZE, TimeCamId};
use kornia_staging_slam::factors::LinearizePointOut;
use kornia_staging_slam::sqrt_ba::{BackSubstitution, LandmarkQr};

/// `LandmarkBlock<Scalar>::Options`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkBlockOptions<S: Scalar> {
    /// Select Householder instead of Givens elimination; defaults to true.
    pub use_householder: bool,
    /// Zero the residual and Jacobian of a projection the camera rejected
    /// rather than keeping whatever the model wrote.
    pub use_valid_projections_only: bool,
    /// Huber threshold in **raw pixels**, or zero for a plain squared norm
    pub huber_parameter: S,
    /// Standard deviation of the reprojection error, in pixels.
    pub obs_std_dev: S,
}

impl<S: Scalar> Default for LandmarkBlockOptions<S> {
    /// Default landmark-block options.
    fn default() -> Self {
        Self {
            use_householder: true,
            use_valid_projections_only: true,
            huber_parameter: S::zero(),
            obs_std_dev: S::one(),
        }
    }
}

/// `LandmarkBlock<Scalar>::State`.
///
/// `Uninitialized` is unreachable in the port — [`LandmarkBlock::allocate`] is a
/// constructor, so there is no half-built block to be in that state — and is
/// kept because the numbering is part of the ported interface and
/// `printStorage` prints it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkBlockState {
    /// Nothing allocated yet.
    Uninitialized,
    /// Storage sized, nothing written.
    Allocated,
    /// The linearization at this state is unusable.
    NumericalFailure,
    /// `linearizeLandmark` has run.
    Linearized,
    /// `performQR` has run: the landmark columns are eliminated.
    Marginalized,
}

/// Observation offsets and relative-pose index resolved at allocation.
/// This avoids repeated map lookups and fallible indexing during linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockObservation {
    /// Where the landmark was seen.
    tcid_t: TimeCamId,
    /// Relative-pose index, or `None` when marginalization drops this observation.
    rel_pose: Option<usize>,
    /// Column offset of the host frame's pose block.
    abs_h_idx: usize,
    /// Column offset of the target frame's pose block.
    abs_t_idx: usize,
}

/// One landmark's block of the linear system.
///
/// Layout (architecture §1.3):
///
/// ```text
/// num_rows    = 2 * observations + 3          // 3 landmark-damping rows ()
/// padding_idx = aom.total_size                // ()
/// padding_size= (4 - padding_idx % 4) % 4     // 16-byte alignment ()
/// lm_idx      = padding_idx + padding_size    // ()
/// res_idx     = lm_idx + 3                    // ()
/// num_cols    = res_idx + 1, asserted % 4 == 0 ()
/// ```
///
/// Storage is column major. Each reflection updates views of the existing
/// matrix using a preallocated unit-axis buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkBlock<S: Scalar> {
    /// `storage` : `[ J_p | pad | J_l | r ]`, `num_rows` x `num_cols`.
    storage: DMatrix<S>,
    /// One entry per observation, in `lm.obs` order.
    observations: Vec<BlockObservation>,
    /// Numerical layout, active columns and elimination scratch.
    qr: LandmarkQr<S>,
    /// The landmark this block belongs to.
    lm_id: LandmarkId,
    /// `lm_ptr->host_kf_id`.
    host_kf_id: TimeCamId,
    /// `is_fixed_` : the landmark is not optimised.
    is_fixed: bool,
    /// `state`.
    state: LandmarkBlockState,
}

/// `compute_error_weight`.
///
/// Returns `(weighted_error, weight)`. Note the Huber test is on the
/// **squared** residual against the squared threshold, and that both are in
/// raw pixels: the `1 / obs_std_dev` scaling happens afterwards,
/// which is the "effective 2 sigma" deviation of papers-part2 §13.
///
/// A free function rather than a method because the estimator's non-keyframe
/// frame update weights its residuals with the same rule and the same
/// association (D76), and one reprojection model in the crate means one Huber
/// in the crate.
pub fn compute_error_weight<S: Scalar>(
    res_squared: S,
    options: &LandmarkBlockOptions<S>,
) -> (S, S) {
    if options.huber_parameter > S::zero() {
        let huber_weight: S = if res_squared <= options.huber_parameter * options.huber_parameter {
            S::one()
        } else {
            options.huber_parameter / res_squared.sqrt()
        };
        let error: S = c::<S>(0.5) * (c::<S>(2.0) - huber_weight) * huber_weight * res_squared;
        (error, huber_weight)
    } else {
        (c::<S>(0.5) * res_squared, S::one())
    }
}

impl<S: Scalar> LandmarkBlock<S> {
    /// Allocate a landmark block from the ordering and relative-pose index table.
    /// Targets outside the ordering retain zero-contribution observations with no
    /// relative pose. A missing host or required pose returns a typed error (D32).
    pub fn allocate(
        lm_id: LandmarkId,
        lm: &Landmark<S>,
        rel_pose_index: &dyn Fn(TimeCamId, TimeCamId) -> Option<usize>,
        aom: &AbsOrderMap,
        is_fixed: bool,
    ) -> Result<Self, LinearizeError> {
        let host: TimeCamId = lm.host_kf_id;
        // A landmark block requires its host frame in the ordering.
        let (abs_h_idx, _) = aom
            .get(host.frame_id)
            .ok_or(LinearizeError::HostNotInOrdering {
                frame_id: host.frame_id,
            })?;

        let mut observations: Vec<BlockObservation> = Vec::with_capacity(lm.obs.len());
        for &tcid_t in lm.obs.keys() {
            let rel_pose: Option<usize> = match aom.get(tcid_t.frame_id) {
                //  — in the ordering, so the pair must have a relative pose.
                Some(_) => Some(rel_pose_index(host, tcid_t).ok_or(
                    LinearizeError::MissingRelativePose {
                        host: host.frame_id,
                        host_cam: host.cam_id,
                        target: tcid_t.frame_id,
                        target_cam: tcid_t.cam_id,
                    },
                )?),
                //  — dropped for marginalization.
                None => None,
            };
            let abs_t_idx: usize = aom.get(tcid_t.frame_id).map_or(0, |(idx, _)| idx);
            observations.push(BlockObservation {
                tcid_t,
                rel_pose,
                abs_h_idx,
                abs_t_idx,
            });
        }

        let padding_idx = aom.total_size();
        // Check that every six-column pose block fits before writing.
        for obs in &observations {
            if obs.rel_pose.is_some() {
                let end: usize = obs
                    .abs_t_idx
                    .max(obs.abs_h_idx)
                    .checked_add(POSE_SIZE)
                    .ok_or(kornia_staging_slam::sqrt_ba::SqrtBaError::LayoutOverflow)?;
                if end > padding_idx {
                    return Err(LinearizeError::PoseBlockOutOfRange {
                        offset: obs.abs_t_idx.max(obs.abs_h_idx),
                        total_size: padding_idx,
                    });
                }
            }
        }

        let mut active_cols: Vec<usize> = Vec::with_capacity(2 * POSE_SIZE * observations.len());
        for obs in &observations {
            // A dropped observation is skipped at and writes nothing;
            // its `abs_t_idx` is the `0` sentinel, not a column it owns.
            if obs.rel_pose.is_none() {
                continue;
            }
            // Both offsets are inside `padding_idx`: the check above refuses
            // the block otherwise, and it covers exactly these observations.
            for offset in [obs.abs_h_idx, obs.abs_t_idx] {
                active_cols.extend(offset..offset + POSE_SIZE);
            }
        }
        active_cols.sort_unstable();
        active_cols.dedup();

        let qr = LandmarkQr::new(observations.len(), padding_idx, active_cols)?;
        Ok(Self {
            storage: DMatrix::zeros(qr.rows(), qr.columns()),
            observations,
            qr,
            lm_id,
            host_kf_id: host,
            is_fixed,
            state: LandmarkBlockState::Allocated,
        })
    }

    /// `linearizeLandmark` : fill the block at the current
    /// linearization point and return this landmark's share of the error.
    ///
    /// Three behaviours are deliberate and are kept (trap 11, decision D32):
    /// a projection the camera rejected contributes **nothing** when
    /// `use_valid_projections_only` is set; a non-finite Jacobian block
    /// is **zeroed with a warning**, not an error (which the comment
    /// at says used to set `NumericalFailure`); and the two pose
    /// blocks are accumulated with `+=`, which is what makes a
    /// landmark observed in its own host frame — where the host and target
    /// columns coincide — come out right.
    pub fn linearize_landmark(
        &mut self,
        lm: &Landmark<S>,
        rel_pose_lin: &[RelPoseLin<S>],
        cameras: &[SlamCamera<S>],
        options: &LandmarkBlockOptions<S>,
    ) -> Result<S, LinearizeError> {
        // `storage.setZero()`.
        self.storage.fill(S::zero());

        let mut error_sum: S = S::zero();

        for (i, obs) in self.observations.iter().enumerate() {
            let Some(rel_idx) = obs.rel_pose else {
                // `if (pose_lin_vec[i])` : a dropped measurement.
                continue;
            };
            let rel: &RelPoseLin<S> =
                rel_pose_lin
                    .get(rel_idx)
                    .ok_or(LinearizeError::MissingRelativePose {
                        host: self.host_kf_id.frame_id,
                        host_cam: self.host_kf_id.cam_id,
                        target: obs.tcid_t.frame_id,
                        target_cam: obs.tcid_t.cam_id,
                    })?;
            let cam: &SlamCamera<S> =
                cameras
                    .get(obs.tcid_t.cam_id)
                    .ok_or(LinearizeError::UnknownCamera {
                        cam_id: obs.tcid_t.cam_id,
                        camera_count: cameras.len(),
                    })?;
            let kpt_obs: &Vector2<S> =
                lm.obs
                    .get(&obs.tcid_t)
                    .ok_or(LinearizeError::MissingObservation {
                        lm_id: self.lm_id,
                        target: obs.tcid_t.frame_id,
                        target_cam: obs.tcid_t.cam_id,
                    })?;

            let obs_idx: usize = i * 2;
            let mut res: Vector2<S> = Vector2::zeros();
            let mut d_res_d_xi: Matrix2x6<S> = Matrix2x6::zeros();
            let mut d_res_d_p: Matrix2x3<S> = Matrix2x3::zeros();

            let valid: bool = linearize_point(
                kpt_obs,
                lm,
                &rel.t_t_h,
                cam,
                &mut res,
                &mut LinearizePointOut {
                    d_res_d_xi: Some(&mut d_res_d_xi),
                    d_res_d_p: Some(&mut d_res_d_p),
                    proj: None,
                },
            );

            // `if (is_fixed_) d_res_d_p.setZero()`.
            if self.is_fixed {
                d_res_d_p.fill(S::zero());
            }

            if options.use_valid_projections_only && !valid {
                continue;
            }

            // zeroed, never fatal.
            if !d_res_d_xi.iter().all(|v| v.to_f64().is_finite()) {
                log::warn!(
                    "d_res_d_xi is not valid, lm = Landmark(id={:?}, host_kf_id={:?})",
                    self.lm_id,
                    self.host_kf_id
                );
                d_res_d_xi.fill(S::zero());
            }
            if !d_res_d_p.iter().all(|v| v.to_f64().is_finite()) {
                log::warn!(
                    "d_res_d_p is not valid, lm = Landmark(id={:?}, host_kf_id={:?})",
                    self.lm_id,
                    self.host_kf_id
                );
                d_res_d_p.fill(S::zero());
            }

            // `res.squaredNorm()` is a contiguous two-coefficient
            // reduction, so there is only one order to take.
            let res_squared: S = res[0] * res[0] + res[1] * res[1];
            let (weighted_error, weight) = compute_error_weight(res_squared, options);
            let sqrt_weight: S = weight.sqrt() / options.obs_std_dev;
            error_sum += weighted_error / (options.obs_std_dev * options.obs_std_dev);

            for r in 0..2 {
                for col in 0..3 {
                    self.storage[(obs_idx + r, self.qr.landmark_column() + col)] =
                        sqrt_weight * d_res_d_p[(r, col)];
                }
                self.storage[(obs_idx + r, self.qr.residual_column())] = sqrt_weight * res[r];
            }

            // The scaling happens once, in place, and then both pose
            // blocks are accumulated.
            d_res_d_xi *= sqrt_weight;
            let host_block: Matrix2x6<S> = d_res_d_xi * rel.d_rel_d_h;
            let target_block: Matrix2x6<S> = d_res_d_xi * rel.d_rel_d_t;
            for r in 0..2 {
                for col in 0..POSE_SIZE {
                    self.storage[(obs_idx + r, obs.abs_h_idx + col)] += host_block[(r, col)];
                }
            }
            for r in 0..2 {
                for col in 0..POSE_SIZE {
                    self.storage[(obs_idx + r, obs.abs_t_idx + col)] += target_block[(r, col)];
                }
            }
        }

        self.state = LandmarkBlockState::Linearized;
        Ok(error_sum)
    }

    /// `performQR` : eliminate the three landmark columns.
    pub fn perform_qr(&mut self, options: &LandmarkBlockOptions<S>) -> Result<(), LinearizeError> {
        if self.state != LandmarkBlockState::Linearized {
            return Err(LinearizeError::WrongState {
                expected: LandmarkBlockState::Linearized,
                found: self.state,
            });
        }
        if options.use_householder {
            self.qr
                .eliminate_householder_unchecked(self.storage.as_mut_slice());
        } else {
            self.qr
                .eliminate_givens_unchecked(self.storage.as_mut_slice());
        }
        self.state = LandmarkBlockState::Marginalized;
        Ok(())
    }

    /// Recover and apply the landmark increment, adding its predicted cost decrease.
    /// A singular `Q1Jl` warns and skips the landmark; a small determinant only warns.
    /// Project inverse distance with `max(0, inv_dist + inc[2])` (traps 11–12).
    /// There is no landmark damping or Jacobian scaling (D34, D68).
    /// The pose increment must span the full ordering.
    #[cfg(test)]
    fn back_substitute(
        &mut self,
        lm: &mut Landmark<S>,
        pose_inc: &DVector<S>,
        l_diff: &mut S,
    ) -> Result<(), LinearizeError> {
        self.back_substitute_with_finite_pose(
            lm,
            pose_inc,
            pose_inc.iter().all(|value| value.is_finite()),
            l_diff,
        )
    }

    /// The window checks the shared pose increment once for all its blocks.
    pub(super) fn back_substitute_with_finite_pose(
        &mut self,
        lm: &mut Landmark<S>,
        pose_inc: &DVector<S>,
        pose_inc_is_finite: bool,
        l_diff: &mut S,
    ) -> Result<(), LinearizeError> {
        if self.state != LandmarkBlockState::Marginalized {
            return Err(LinearizeError::WrongState {
                expected: LandmarkBlockState::Marginalized,
                found: self.state,
            });
        }
        // `if (is_fixed_) return`.
        if self.is_fixed {
            return Ok(());
        }
        if pose_inc.nrows() != self.qr.pose_columns() {
            return Err(LinearizeError::PoseIncrementSize {
                expected: self.qr.pose_columns(),
                found: pose_inc.nrows(),
            });
        }

        let (inc, det, cost_term) = match self.qr.back_substitute_unchecked(
            self.storage.as_slice(),
            pose_inc.as_slice(),
            pose_inc_is_finite,
        ) {
            BackSubstitution::Singular => {
                log::warn!(
                    "det(Q1Jl) == 0, skipping backsubstitution for lm: Landmark(id={:?}, host_kf_id={:?})",
                    self.lm_id,
                    self.host_kf_id
                );
                return Ok(());
            }
            BackSubstitution::NonFinite => {
                return Err(LinearizeError::NonFiniteLandmarkStep { lm_id: self.lm_id });
            }
            BackSubstitution::Solved {
                increment,
                determinant,
                cost_term,
            } => (increment, determinant, cost_term),
        };
        if det < c::<S>(0.01) {
            log::warn!(
                "Unusually small det(Q1Jl)={}, lm: Landmark(id={:?}, host_kf_id={:?})",
                det.to_f64(),
                self.lm_id,
                self.host_kf_id
            );
        }
        *l_diff -= cost_term;

        // No column-scale multiplication is needed because Jacobian scaling is absent (D68).
        lm.direction[0] += inc[0];
        lm.direction[1] += inc[1];
        let updated: S = lm.inv_dist + inc[2];
        lm.inv_dist = if updated > S::zero() {
            updated
        } else {
            S::zero()
        };
        Ok(())
    }

    /// `get_dense_Q2Jp_Q2r(Q2Jp, Q2r, start_idx)` : the null-space
    /// rows, the ones the marginalization QR of stage S7 consumes.
    ///
    /// Rows `3..num_rows` of the block become rows
    /// `start_idx..start_idx + num_rows - 3` of the stacked system; the columns
    /// are the `padding_idx` pose columns, i.e. the padding, the landmark
    /// columns and the residual are all left behind.
    pub fn get_dense_q2jp_q2r(
        &self,
        q2jp: &mut DMatrix<S>,
        q2r: &mut DVector<S>,
        start_idx: usize,
    ) -> Result<(), LinearizeError> {
        let rows: usize = self.num_q2rows();
        // Validate destination rows and columns before writing.
        if q2jp.ncols() != self.qr.pose_columns() {
            return Err(LinearizeError::StackedSystemSize {
                expected: self.qr.pose_columns(),
                found: q2jp.ncols(),
            });
        }
        let end: usize = start_idx
            .checked_add(rows)
            .ok_or(kornia_staging_slam::sqrt_ba::SqrtBaError::LayoutOverflow)?;
        if end > q2jp.nrows() || end > q2r.nrows() {
            return Err(LinearizeError::StackedSystemSize {
                expected: end,
                found: q2jp.nrows().min(q2r.nrows()),
            });
        }
        self.dense_block()
            .write_stacked_unchecked(q2jp, q2r, start_idx);
        Ok(())
    }

    /// The observed pose columns.
    pub fn active_cols(&self) -> &[usize] {
        self.qr.active_columns()
    }

    /// Every pose column of the dense system, `0..padding_idx`: what
    /// a full dense contraction writes.
    pub fn pose_columns(&self) -> std::ops::Range<usize> {
        0..self.qr.pose_columns()
    }

    /// Borrow numerical storage without copying the landmark or its state.
    pub(super) fn dense_block(&self) -> DenseBlock<'_, S> {
        DenseBlock::new(&self.qr, self.storage.as_slice())
    }

    /// Stored row count minus the three reserved damping rows; includes the `Q₁` rows.
    pub fn num_q2rows(&self) -> usize {
        self.qr.rows() - 3
    }

    /// The landmark this block belongs to.
    pub fn landmark_id(&self) -> LandmarkId {
        self.lm_id
    }

    /// `getState()`.
    pub fn state(&self) -> LandmarkBlockState {
        self.state
    }

    /// Numerical-failure status remains false: invalid Jacobians are zeroed instead.
    pub fn is_numerical_failure(&self) -> bool {
        self.state == LandmarkBlockState::NumericalFailure
    }

    /// The layout: `(num_rows, num_cols, padding_idx, lm_idx, res_idx)`.
    ///
    /// Layout arithmetic is the part most likely to go wrong in a
    /// port, so it is observable and the fixture checks all five numbers.
    pub fn layout(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.qr.rows(),
            self.qr.columns(),
            self.qr.pose_columns(),
            self.qr.landmark_column(),
            self.qr.residual_column(),
        )
    }

    /// The block buffer, `storage`, row `r`, column `c`.
    pub fn storage(&self) -> &DMatrix<S> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::calib::{BasaltCamera, Kb4Params};
    use crate::lie::Se3;
    use crate::types::LandmarkId;
    use nalgebra::{Matrix4, Matrix6, Vector3, Vector4};

    /// A one-frame ordering and a landmark hosted in it, seen twice.
    fn fixture(order_frames: usize) -> (AbsOrderMap, Landmark<f64>, Vec<RelPoseLin<f64>>) {
        let mut aom: AbsOrderMap = AbsOrderMap::new();
        for i in 0..order_frames {
            aom.push(i as i64, POSE_SIZE).unwrap();
        }
        let host: TimeCamId = TimeCamId::new(0, 0);
        let mut lm: Landmark<f64> =
            Landmark::new(LandmarkId(7), host, Vector2::new(0.01, -0.02), 0.25);
        lm.obs.insert(host, Vector2::new(505.0, 510.0));
        lm.obs
            .insert(TimeCamId::new(0, 1), Vector2::new(500.0, 512.0));
        let rel: Vec<RelPoseLin<f64>> = vec![
            RelPoseLin {
                t_t_h: Matrix4::identity(),
                d_rel_d_h: Matrix6::zeros(),
                d_rel_d_t: Matrix6::zeros(),
            },
            RelPoseLin {
                t_t_h: Se3::<f64>::new(crate::lie::So3::identity(), Vector3::new(0.1, 0.0, 0.0))
                    .matrix(),
                d_rel_d_h: Matrix6::identity(),
                d_rel_d_t: -Matrix6::identity(),
            },
        ];
        (aom, lm, rel)
    }

    fn index(host: TimeCamId, target: TimeCamId) -> Option<usize> {
        let _ = host;
        Some(usize::from(target.cam_id == 1))
    }

    #[test]
    fn stacked_matrix_and_residual_can_have_different_capacities() {
        let (aom, lm, rel) = fixture(2);
        let mut block = LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        block.perform_qr(&options()).unwrap();
        let rows = block.num_q2rows();
        for extra in [0, 2] {
            let mut j = DMatrix::zeros(rows + extra, aom.total_size());
            let mut r = DVector::zeros(rows + 2 - extra);
            block.get_dense_q2jp_q2r(&mut j, &mut r, 0).unwrap();
            for row in 0..rows {
                for col in 0..aom.total_size() {
                    assert_eq!(
                        j[(row, col)].to_bits(),
                        block.storage[(row + 3, col)].to_bits()
                    );
                }
                assert_eq!(
                    r[row].to_bits(),
                    block.storage[(row + 3, block.qr.residual_column())].to_bits()
                );
            }
        }
    }

    fn cameras() -> Vec<SlamCamera<f64>> {
        let model: BasaltCamera<f64> = BasaltCamera::Kb4(Kb4Params {
            fx: 379.045,
            fy: 379.008,
            cx: 505.512,
            cy: 509.969,
            k1: 0.00693023,
            k2: -0.0013828,
            k3: -0.000272596,
            k4: -0.000452646,
        });
        vec![
            SlamCamera::from_model(&model).unwrap(),
            SlamCamera::from_model(&model).unwrap(),
        ]
    }

    fn options() -> LandmarkBlockOptions<f64> {
        LandmarkBlockOptions {
            huber_parameter: 0.5,
            obs_std_dev: 2.0,
            ..Default::default()
        }
    }

    /// `add_dense_h_b` reads only the observed pose columns, and the rest of
    /// `0..padding_idx` really is zero — through the linearization and the QR,
    /// which is the whole life of a block.
    ///
    /// The optimization that skips them is only the identity because of this:
    /// a column of zeros makes every product `+/-0.0`, so the accumulator stays
    /// `+0.0`, and `H += +0.0` cannot move an `H` that is never `-0.0` either.
    #[test]
    fn dense_h_b_touches_only_observed_columns() {
        let (aom, lm, rel) = fixture(4);
        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        // Both observations are in frame 0, so only its six columns are live.
        assert_eq!(
            block.qr.active_columns(),
            (0..POSE_SIZE).collect::<Vec<usize>>()
        );

        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        let zero_after = |block: &LandmarkBlock<f64>, stage: &str| {
            for column in POSE_SIZE..block.qr.pose_columns() {
                for row in 0..block.qr.rows() {
                    assert_eq!(
                        block.storage[(row, column)],
                        0.0,
                        "column {column} moved off zero at {stage}"
                    );
                }
            }
        };
        zero_after(&block, "linearizeLandmark");
        block.perform_qr(&options()).unwrap();
        zero_after(&block, "performQR");

        // And the dense system it produces is the one the whole `padding_idx`
        // loop would have produced.
        assert_dense_h_b_is_the_full_loop(&block);
    }

    /// The system `add_dense_h_b_over` writes over `active_cols` equals the
    /// one a loop over the whole `padding_idx` square produces, coefficient by
    /// coefficient and **bit for bit**. Both tests of the skip rest on this.
    fn assert_dense_h_b_is_the_full_loop(block: &LandmarkBlock<f64>) {
        let mut h: DMatrix<f64> = DMatrix::zeros(block.qr.pose_columns(), block.qr.pose_columns());
        let mut b: DVector<f64> = DVector::zeros(block.qr.pose_columns());
        block.dense_block().coefficients_unchecked(
            block.active_cols(),
            &mut kornia_staging_slam::sqrt_ba::DenseHbWorkspace::default(),
            |i, j, value| {
                if let Some(j) = j {
                    h[(i, j)] += value;
                } else {
                    b[i] += value;
                }
            },
        );
        let rows: usize = block.num_q2rows();
        for i in 0..block.qr.pose_columns() {
            for j in 0..block.qr.pose_columns() {
                let mut acc: f64 = 0.0;
                for r in 0..rows {
                    acc += block.storage[(3 + r, i)] * block.storage[(3 + r, j)];
                }
                assert_eq!(h[(i, j)].to_bits(), acc.to_bits(), "H({i}, {j})");
            }
            let mut acc: f64 = 0.0;
            for r in 0..rows {
                acc +=
                    block.storage[(3 + r, i)] * block.storage[(3 + r, block.qr.residual_column())];
            }
            assert_eq!(b[i].to_bits(), acc.to_bits(), "b({i})");
        }
    }

    /// The public writeback writes every column, whatever the destination holds.
    ///
    /// `-0.0 + (+0.0)` is `+0.0`, so a destination coefficient at `-0.0` tells
    /// the two paths apart where no value comparison can: the full width writes
    /// it and the sign goes, the skip leaves it. This is the contract a caller
    /// who owns `h` and `b` gets, and the reason the skip is not it.
    #[test]
    fn the_public_writeback_covers_a_negative_zero_destination() {
        let (aom, lm, rel) = fixture(4);
        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        block.perform_qr(&options()).unwrap();
        // Both observations are in frame 0: columns `6..24` are the skipped ones.
        assert_eq!(
            block.qr.active_columns(),
            (0..POSE_SIZE).collect::<Vec<usize>>()
        );

        let n: usize = block.qr.pose_columns();
        let mut h: DMatrix<f64> = DMatrix::from_element(n, n, -0.0);
        let mut b: DVector<f64> = DVector::from_element(n, -0.0);
        block.dense_block().add_full_unchecked(
            &mut h,
            &mut b,
            &mut kornia_staging_slam::sqrt_ba::DenseHbWorkspace::default(),
        );

        for i in 0..n {
            for j in 0..n {
                if i < POSE_SIZE && j < POSE_SIZE {
                    continue;
                }
                // The block contributes `+/-0.0` here, and `-0.0 + 0.0 = +0.0`.
                assert_eq!(h[(i, j)].to_bits(), 0.0f64.to_bits(), "H({i}, {j})");
            }
            if i >= POSE_SIZE {
                assert_eq!(b[i].to_bits(), 0.0f64.to_bits(), "b({i})");
            }
        }
    }

    /// A block carrying a non-finite observation is not one the skip may take.
    ///
    /// `Landmark::add_observation` accepts the keypoint, the Huber weight
    /// carries the NaN past the Jacobian checks, and the Householder spreads it
    /// across whole rows — including the columns this block never observed. The
    /// reduction's answer to that is the full-width path
    /// (`a_non_finite_block_is_reduced_at_full_width` in `abs_qr`); this pins
    /// the two facts underneath it.
    #[test]
    fn a_non_finite_observation_spreads_past_the_observed_columns() {
        let (aom, mut lm, rel) = fixture(4);
        lm.obs
            .insert(TimeCamId::new(0, 1), Vector2::new(f64::NAN, 512.0));
        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        block.perform_qr(&options()).unwrap();

        assert!(
            !block.dense_block().active_writeback_is_exact(),
            "a NaN block must not take the skip"
        );
        let spread: bool = (POSE_SIZE..block.qr.pose_columns()).any(|column| {
            (0..block.num_q2rows()).any(|r| !block.storage[(3 + r, column)].is_finite())
        });
        assert!(
            spread,
            "the QR was supposed to spread the NaN off the block's own columns"
        );
    }

    /// A measurement dropped during marginalization writes no pose columns.
    ///
    /// `linearizeLandmark` skips an observation with no relative pose,
    /// so nothing ever writes at its `abs_t_idx` — which is the `0` sentinel of
    /// another frame's first column. The block here is hosted in frame 1,
    /// so that sentinel is not the host's own offset and the two are told apart:
    /// columns `0..6` stay zero through the linearization and the QR, and the
    /// dense system is the full loop's either way.
    #[test]
    fn a_dropped_observation_writes_no_columns() {
        let (aom, _, rel) = fixture(4);
        let host: TimeCamId = TimeCamId::new(1, 0);
        let mut lm: Landmark<f64> =
            Landmark::new(LandmarkId(7), host, Vector2::new(0.01, -0.02), 0.25);
        lm.obs.insert(host, Vector2::new(505.0, 510.0));
        lm.obs
            .insert(TimeCamId::new(1, 1), Vector2::new(500.0, 512.0));
        // Frame 9 is not in the ordering, so this one is dropped.
        lm.obs
            .insert(TimeCamId::new(9, 0), Vector2::new(498.0, 507.0));

        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        let live: std::ops::Range<usize> = POSE_SIZE..2 * POSE_SIZE;
        assert_eq!(
            block.qr.active_columns(),
            live.clone().collect::<Vec<usize>>(),
            "the dropped observation's sentinel offset is not a column it writes"
        );
        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        block.perform_qr(&options()).unwrap();

        for column in (0..block.qr.pose_columns()).filter(|column| !live.contains(column)) {
            for row in 0..block.qr.rows() {
                assert_eq!(
                    block.storage[(row, column)],
                    0.0,
                    "column {column} moved off zero"
                );
            }
        }
        assert_dense_h_b_is_the_full_loop(&block);
    }

    /// The layout arithmetic of on every remainder of the padding rule.
    #[test]
    fn the_layout_pads_to_a_multiple_of_four() {
        for frames in 1..=6usize {
            let (aom, lm, _) = fixture(frames);
            let block: LandmarkBlock<f64> =
                LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
            let total: usize = frames * POSE_SIZE;
            let (num_rows, num_cols, padding_idx, lm_idx, res_idx) = block.layout();
            assert_eq!(padding_idx, total);
            assert_eq!(lm_idx, total + (4 - total % 4) % 4);
            assert_eq!(res_idx, lm_idx + 3);
            assert_eq!(num_cols, res_idx + 1);
            assert_eq!(num_cols % 4, 0, "the `% 4` assertion of `:96`");
            assert_eq!(num_rows, 2 * lm.obs.len() + 3, "`:85`");
            assert_eq!(block.storage().nrows(), num_rows);
            assert_eq!(block.storage().ncols(), num_cols);
        }
    }

    /// A host frame outside the ordering is 's assertion, typed.
    #[test]
    fn a_host_outside_the_ordering_is_an_error() {
        let (_, lm, _) = fixture(1);
        let empty: AbsOrderMap = AbsOrderMap::new();
        assert_eq!(
            LandmarkBlock::allocate(lm.id, &lm, &index, &empty, false).unwrap_err(),
            LinearizeError::HostNotInOrdering { frame_id: 0 }
        );
    }

    /// A missing relative pose is 's assertion, typed.
    #[test]
    fn a_missing_relative_pose_is_an_error() {
        let (aom, lm, _) = fixture(1);
        let err = LandmarkBlock::allocate(lm.id, &lm, &|_, _| None, &aom, false).unwrap_err();
        assert!(matches!(err, LinearizeError::MissingRelativePose { .. }));
    }

    /// Out-of-order block operations return typed state errors.
    #[test]
    fn methods_refuse_to_run_out_of_order() {
        let (aom, lm, rel) = fixture(1);
        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        assert_eq!(block.state(), LandmarkBlockState::Allocated);
        assert!(!block.is_numerical_failure());

        // `performQR` asserts `Linearized`.
        assert_eq!(
            block.perform_qr(&options()).unwrap_err(),
            LinearizeError::WrongState {
                expected: LandmarkBlockState::Linearized,
                found: LandmarkBlockState::Allocated,
            }
        );

        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        assert_eq!(block.state(), LandmarkBlockState::Linearized);

        block.perform_qr(&options()).unwrap();
        assert_eq!(block.state(), LandmarkBlockState::Marginalized);

        // the increment is the whole ordering.
        let mut l_diff: f64 = 0.0;
        let mut moved: Landmark<f64> = lm.clone();
        assert_eq!(
            block
                .back_substitute(&mut moved, &DVector::zeros(3), &mut l_diff)
                .unwrap_err(),
            LinearizeError::PoseIncrementSize {
                expected: POSE_SIZE,
                found: 3,
            }
        );
    }

    #[test]
    fn nonfinite_back_substitution_leaves_landmark_and_cost_unchanged() {
        let (aom, mut landmark, _) = fixture(1);
        let before = landmark.clone();
        let mut block =
            LandmarkBlock::allocate(landmark.id, &landmark, &index, &aom, false).unwrap();
        block.state = LandmarkBlockState::Marginalized;
        for i in 0..3 {
            block.storage[(i, block.qr.landmark_column() + i)] = 1.0;
        }
        block.storage[(0, block.qr.residual_column())] = f64::NAN;
        let mut cost = 5.0;
        assert!(matches!(
            block.back_substitute(&mut landmark, &DVector::zeros(POSE_SIZE), &mut cost),
            Err(LinearizeError::NonFiniteLandmarkStep { .. })
        ));
        assert_eq!(landmark.direction, before.direction);
        assert_eq!(landmark.inv_dist, before.inv_dist);
        assert_eq!(cost, 5.0);
    }

    /// Singular and fixed landmarks are skipped without error (trap 11).
    /// One observation gives rank at most two, so the third triangular diagonal is
    /// zero and back-substitution leaves the landmark untouched. Fixed landmarks
    /// return even earlier.
    #[test]
    fn a_singular_landmark_block_is_skipped_not_an_error() {
        let (aom, mut lm, rel) = fixture(1);
        // One observation: the host's own image.
        lm.obs.remove(&TimeCamId::new(0, 1));

        let mut singular: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        singular
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        singular.perform_qr(&options()).unwrap();
        let mut l_diff: f64 = 0.0;
        let mut moved: Landmark<f64> = lm.clone();
        singular
            .back_substitute(&mut moved, &DVector::zeros(POSE_SIZE), &mut l_diff)
            .unwrap();
        assert_eq!(
            moved.direction, lm.direction,
            "a singular block moved a landmark"
        );
        assert_eq!(moved.inv_dist, lm.inv_dist);
        assert_eq!(l_diff, 0.0);

        // And a fixed block returns before it even looks.
        let (aom, lm, rel) = fixture(1);
        let mut fixed: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, true).unwrap();
        fixed
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        // `is_fixed_` zeroes `d_res_d_p`, so the landmark columns are
        // empty before the QR ever runs.
        let (_, _, _, lm_idx, _) = fixed.layout();
        for row in 0..4 {
            for col in 0..3 {
                assert_eq!(fixed.storage()[(row, lm_idx + col)], 0.0);
            }
        }
        fixed.perform_qr(&options()).unwrap();
        let mut l_diff: f64 = 0.0;
        let mut moved: Landmark<f64> = lm.clone();
        fixed
            .back_substitute(&mut moved, &DVector::zeros(POSE_SIZE), &mut l_diff)
            .unwrap();
        assert_eq!(moved.inv_dist, lm.inv_dist);
        assert_eq!(l_diff, 0.0);
    }

    /// An observation outside the ordering retains two zero rows and contributes
    /// nothing after marginalization drops its relative pose.
    #[test]
    fn an_observation_outside_the_ordering_leaves_its_rows_zero() {
        let (aom, mut lm, rel) = fixture(1);
        // A second image, in a frame the ordering does not have.
        lm.obs
            .insert(TimeCamId::new(9, 0), Vector2::new(500.0, 512.0));
        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();

        let (num_rows, num_cols, _, _, _) = block.layout();
        assert_eq!(
            num_rows,
            2 * 3 + 3,
            "the dropped observation still gets rows"
        );
        // `lm.obs` is ordered by `TimeCamId`, so frame 9 is last: rows 4 and 5.
        for row in 4..6 {
            for col in 0..num_cols {
                assert_eq!(
                    block.storage()[(row, col)],
                    0.0,
                    "a dropped observation wrote to ({row}, {col})"
                );
            }
        }
        // and the two live observations did write.
        assert!(block.storage().rows(0, 4).iter().any(|v| *v != 0.0));
    }

    /// The Huber branch of `compute_error_weight` : below the
    /// threshold the weight is one, above it the error grows linearly rather
    /// than quadratically.
    #[test]
    fn the_huber_weight_crosses_at_the_threshold() {
        let opt: LandmarkBlockOptions<f64> = options();
        let delta: f64 = opt.huber_parameter;

        let (error_in, weight_in) = compute_error_weight(0.25 * delta * delta, &opt);
        assert_eq!(weight_in, 1.0);
        assert_eq!(error_in, 0.5 * 0.25 * delta * delta);

        // Exactly at the threshold the comparison is `<=`, so the weight is one.
        let (_, weight_at) = compute_error_weight(delta * delta, &opt);
        assert_eq!(weight_at, 1.0);

        let res_squared: f64 = 4.0 * delta * delta;
        let (error_out, weight_out) = compute_error_weight(res_squared, &opt);
        assert!((weight_out - 0.5).abs() < 1e-15, "{weight_out}");
        assert!((error_out - 0.5 * 1.5 * 0.5 * res_squared).abs() < 1e-15);

        // With the threshold off it is a plain squared norm.
        let plain: LandmarkBlockOptions<f64> = LandmarkBlockOptions {
            huber_parameter: 0.0,
            ..opt
        };
        let (error, weight) = compute_error_weight(res_squared, &plain);
        assert_eq!(weight, 1.0);
        assert_eq!(error, 0.5 * res_squared);
    }

    /// Refuse matrix-size overflow before nalgebra allocation can abort (D32).
    #[test]
    fn a_block_too_large_to_allocate_is_an_error() {
        let (_, lm, _) = fixture(1);
        let mut aom: AbsOrderMap = AbsOrderMap::new();
        // The `% 4` rule needs a multiple of four, and the product of the two
        // dimensions has to overflow: 7 rows times this is far past `isize::MAX`
        // bytes long before it wraps.
        aom.push(0, usize::MAX / 2 - (usize::MAX / 2) % 4).unwrap();
        let err = LandmarkBlock::<f64>::allocate(lm.id, &lm, &index, &aom, false).unwrap_err();
        assert!(
            matches!(
                err,
                LinearizeError::SqrtBa(
                    kornia_staging_slam::sqrt_ba::SqrtBaError::BlockTooLarge { .. }
                )
            ),
            "{err:?}"
        );
    }

    /// `Vector4` is only here to keep the import list honest about what the
    /// fixture builds.
    #[test]
    fn the_fixture_landmark_is_in_front_of_the_camera() {
        let (_, lm, _) = fixture(1);
        let bearing: Vector4<f64> = crate::landmark::StereographicParam::unproject(&lm.direction);
        assert!(bearing[2] > 0.0);
    }
}
