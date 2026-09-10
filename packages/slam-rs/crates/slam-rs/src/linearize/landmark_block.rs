//! One landmark's rows of the linear system, and the QR that eliminates it.
//!
//! `LandmarkBlockAbsDynamic<Scalar, POSE_SIZE>`
//! (`include/basalt/linearization/landmark_block_abs_dynamic.hpp`), the block of
//! CVPR 2021 Fig. 2: one dense buffer laid out `[ J_p | pad | J_l(3) | r ]`,
//! three Householder reflections that rotate the landmark columns to upper
//! triangular, and the two halves that fall out of it — rows `0..3` (`Q₁`) for
//! back-substitution, rows `3..` (`Q₂`) for the reduced camera system.

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Matrix3, Vector2, Vector3};

use crate::ba_base::{LinearizePointOut, linearize_point};
use crate::camera::CameraEnum;
use crate::eigen::qr::{
    ColumnRedux, apply_householder_on_the_left, apply_rotation_on_the_left, make_givens,
    make_householder,
};
use crate::eigen::svd::JacobiRotation;
use crate::landmark::Landmark;
use crate::lie::{LieScalar, c};
use crate::linearize::{LinearizeError, RelPoseLin};
use crate::types::{AbsOrderMap, LandmarkId, POSE_SIZE, TimeCamId};

/// `LandmarkBlock<Scalar>::Options` (`landmark_block.hpp:31-48`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkBlockOptions<S: LieScalar> {
    /// Householder rather than Givens for the elimination (`:33`). basalt ships
    /// `true` and the shipped VIO never changes it.
    pub use_householder: bool,
    /// Zero the residual and Jacobian of a projection the camera rejected
    /// (`:37`), rather than keeping whatever the model wrote.
    pub use_valid_projections_only: bool,
    /// Huber threshold in **raw pixels**, or zero for a plain squared norm
    /// (`:40`).
    pub huber_parameter: S,
    /// Standard deviation of the reprojection error, in pixels (`:43`).
    pub obs_std_dev: S,
}

impl<S: LieScalar> Default for LandmarkBlockOptions<S> {
    /// basalt's defaults (`landmark_block.hpp:33-47`).
    fn default() -> Self {
        Self {
            use_householder: true,
            use_valid_projections_only: true,
            huber_parameter: S::zero(),
            obs_std_dev: S::one(),
        }
    }
}

/// How many coefficients of one row of `H` are accumulated at a time.
///
/// The accumulator is a fixed-size array rather than a slice of the scratch
/// buffer for two reasons, and both decide whether the loop vectorises: its
/// length is a constant, so the body unrolls with no runtime trip count and no
/// scalar tail, and it is a local, so the compiler can see that writing it
/// cannot change the row it is reading. Written against `Q2Jp` — two pose
/// blocks of six columns plus the residual on the median frame — eight is one
/// SSE pair and holds a whole pose block plus its neighbour.
const LANES: usize = 8;

/// The buffer [`LandmarkBlock::add_dense_h_b`] works in, reused across blocks.
///
/// `H += Q2Jp^T Q2Jp` is a dot product per coefficient down two columns of
/// `storage`, and an `f32` sum cannot be reassociated, so the reduction over
/// rows has to stay sequential. Holding one accumulator per **column** instead
/// of one per coefficient turns the inner loop into `acc[j] += t * row[j]`,
/// which each accumulator still walks in row order — the same additions in the
/// same order — and which vectorises, where the dot product does not.
#[derive(Debug, Clone)]
pub struct DenseHbScratch<S: LieScalar> {
    /// The `Q2` rows of `storage` over the written columns then the residual,
    /// row-major so one row is contiguous, and padded to a whole number of
    /// [`LANES`] with `+0.0`.
    rows: Vec<S>,
}

impl<S: LieScalar> Default for DenseHbScratch<S> {
    fn default() -> Self {
        Self { rows: Vec::new() }
    }
}

/// `LandmarkBlock<Scalar>::State` (`landmark_block.hpp:50`).
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

/// One observation's place in the block, resolved once at allocation.
///
/// C++ keeps two parallel vectors — `pose_lin_vec` (a `RelPoseLin*`, null when
/// the target frame is not in the ordering, `:70-75`) and `pose_tcid_vec` — and
/// looks the two absolute offsets up with `at()` on every linearization
/// (`:139-140`). The port resolves both at allocation and stores them, so the
/// per-observation path has no map lookup and no way to throw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockObservation {
    /// Where the landmark was seen.
    tcid_t: TimeCamId,
    /// Index into the driver's relative-pose table, or `None` for an
    /// observation dropped for marginalization (C++'s null pointer, `:74`).
    rel_pose: Option<usize>,
    /// Column offset of the host frame's pose block.
    abs_h_idx: usize,
    /// Column offset of the target frame's pose block.
    abs_t_idx: usize,
}

/// One landmark's block of the linear system.
///
/// Layout (`landmark_block_abs_dynamic.hpp:83-96`, architecture §1.3):
///
/// ```text
/// num_rows    = 2 * observations + 3          // 3 landmark-damping rows (:85)
/// padding_idx = aom.total_size                // (:83)
/// padding_size= (4 - padding_idx % 4) % 4     // 16-byte alignment (:87-88)
/// lm_idx      = padding_idx + padding_size    // (:90)
/// res_idx     = lm_idx + 3                    // (:91)
/// num_cols    = res_idx + 1, asserted % 4 == 0 (:92-96)
/// ```
///
/// **Storage order.** C++'s buffer is `Eigen::RowMajor` (`:530`); nalgebra's
/// `DMatrix` is column major. Nothing here depends on the layout — every loop
/// is written out — but it is why `makeHouseholder`'s reduction is a sequential
/// fold rather than a vectorised one (see `crate::eigen::qr`).
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkBlock<S: LieScalar> {
    /// `storage` (`:530`): `[ J_p | pad | J_l | r ]`, `num_rows` x `num_cols`.
    storage: DMatrix<S>,
    /// One entry per observation, in `lm.obs` order.
    observations: Vec<BlockObservation>,
    /// The pose columns the observations write into, ascending and deduplicated.
    ///
    /// Every other column of `0..padding_idx` stays exactly zero for the block's
    /// whole life: `linearizeLandmark` only ever writes `block<2, 6>` at an
    /// observation's host and target offsets (`:178-179`), and the Householder
    /// reflections and Givens rotations that follow act on rows, which cannot
    /// move a zero column off zero. An observation dropped for marginalization
    /// writes nothing at all — `:137` skips it for want of a relative pose, and
    /// its `abs_t_idx` is the `0` sentinel of `:74`, which is a column of
    /// whichever frame the ordering puts first — so it is left out.
    /// [`Self::add_dense_h_b_active`] is the only reader, and
    /// `dense_h_b_touches_only_observed_columns` and
    /// `a_dropped_observation_writes_no_columns` pin the invariant.
    ///
    /// The "cannot move a zero column off zero" step is finite arithmetic: a
    /// reflection whose coefficients are non-finite writes NaN everywhere,
    /// which is why [`Self::active_writeback_is_exact`] checks rather than
    /// assumes.
    active_cols: Vec<usize>,
    /// The landmark this block belongs to.
    lm_id: LandmarkId,
    /// `lm_ptr->host_kf_id` (`:525`).
    host_kf_id: TimeCamId,
    /// `is_fixed_` (`:552`): the landmark is not optimised.
    is_fixed: bool,
    padding_idx: usize,
    lm_idx: usize,
    res_idx: usize,
    num_rows: usize,
    num_cols: usize,
    /// `state` (`:547`).
    state: LandmarkBlockState,
    /// The `tempVector1` of `performQRHouseholder` (`:442`), preallocated: the
    /// per-observation and per-reflection paths must not allocate.
    work_row: Vec<S>,
    /// The `tempVector2` of `:443`, the essential part of the reflector.
    work_essential: Vec<S>,
}

impl<S: LieScalar> LandmarkBlock<S> {
    /// `allocateLandmark` (`:35-104`).
    ///
    /// `rel_pose_index` maps a `(host, target)` pair to its slot in the driver's
    /// relative-pose table; C++ stores the pointer itself (`:67-75`). An
    /// observation whose target frame is not in `aom` is kept with no relative
    /// pose, which is how a measurement dropped during marginalization survives
    /// allocation and contributes nothing (`:70-75`, and the comment at
    /// `:125-135` that says this should not happen in the first place).
    ///
    /// C++ asserts the host frame is in the ordering (`:62`) and that the
    /// relative pose exists (`:68`); both are typed errors here (decision D32).
    pub fn allocate(
        lm_id: LandmarkId,
        lm: &Landmark<S>,
        rel_pose_index: &dyn Fn(TimeCamId, TimeCamId) -> Option<usize>,
        aom: &AbsOrderMap,
        is_fixed: bool,
    ) -> Result<Self, LinearizeError> {
        let host: TimeCamId = lm.host_kf_id;
        // `BASALT_ASSERT(aom.abs_order_map.count(lm.host_kf_id.frame_id) > 0)`
        // (`:62`) — a landmark block without its host frame cannot be built.
        let (abs_h_idx, _) = aom
            .get(host.frame_id)
            .ok_or(LinearizeError::HostNotInOrdering {
                frame_id: host.frame_id,
            })?;

        let mut observations: Vec<BlockObservation> = Vec::with_capacity(lm.obs.len());
        for &tcid_t in lm.obs.keys() {
            let rel_pose: Option<usize> = match aom.get(tcid_t.frame_id) {
                // `:71` — in the ordering, so the pair must have a relative pose.
                Some(_) => Some(rel_pose_index(host, tcid_t).ok_or(
                    LinearizeError::MissingRelativePose {
                        host: host.frame_id,
                        host_cam: host.cam_id,
                        target: tcid_t.frame_id,
                        target_cam: tcid_t.cam_id,
                    },
                )?),
                // `:74` — dropped for marginalization.
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

        // The layout arithmetic of `:83-96`, every step checked: the sizes come
        // from a caller-supplied ordering, and an overflow here would silently
        // wrap into a buffer that aliases its own blocks (decision D32).
        let padding_idx: usize = aom.total_size();
        let num_rows: usize = observations
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(3))
            .ok_or(LinearizeError::LayoutOverflow)?;
        let pad: usize = padding_idx % 4;
        let padding_size: usize = if pad != 0 { 4 - pad } else { 0 };
        let lm_idx: usize = padding_idx
            .checked_add(padding_size)
            .ok_or(LinearizeError::LayoutOverflow)?;
        let res_idx: usize = lm_idx
            .checked_add(3)
            .ok_or(LinearizeError::LayoutOverflow)?;
        let num_cols: usize = res_idx
            .checked_add(1)
            .ok_or(LinearizeError::LayoutOverflow)?;
        // `BASALT_ASSERT(num_cols % 4 == 0)` (`:96`).
        if num_cols % 4 != 0 {
            return Err(LinearizeError::UnalignedBlock { num_cols });
        }
        // `storage.resize(num_rows, num_cols)` (`:98`). Each dimension being
        // representable is not enough: nalgebra multiplies them, and the
        // allocator wants the byte count, which must fit in an `isize`
        // (decision D32). An ordering carrying one absurd block size reaches
        // here with both dimensions individually fine.
        let elements: usize =
            num_rows
                .checked_mul(num_cols)
                .ok_or(LinearizeError::BlockTooLarge {
                    rows: num_rows,
                    cols: num_cols,
                })?;
        let fits: bool = elements
            .checked_mul(size_of::<S>())
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
        if !fits {
            return Err(LinearizeError::BlockTooLarge {
                rows: num_rows,
                cols: num_cols,
            });
        }
        // Every pose block must fit: C++ writes `block<2, 6>(obs_idx, abs_idx)`
        // with no bound check (`:178-179`).
        for obs in &observations {
            if obs.rel_pose.is_some() {
                let end: usize = obs
                    .abs_t_idx
                    .max(obs.abs_h_idx)
                    .checked_add(POSE_SIZE)
                    .ok_or(LinearizeError::LayoutOverflow)?;
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
            // A dropped observation is skipped at `:137` and writes nothing;
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

        Ok(Self {
            storage: DMatrix::zeros(num_rows, num_cols),
            observations,
            active_cols,
            lm_id,
            host_kf_id: host,
            is_fixed,
            padding_idx,
            lm_idx,
            res_idx,
            num_rows,
            num_cols,
            state: LandmarkBlockState::Allocated,
            work_row: vec![S::zero(); num_cols],
            work_essential: vec![S::zero(); num_rows],
        })
    }

    /// `compute_error_weight` (`:456-470`).
    ///
    /// Returns `(weighted_error, weight)`. Note the Huber test is on the
    /// **squared** residual against the squared threshold, and that both are in
    /// raw pixels: the `1 / obs_std_dev` scaling happens afterwards (`:170-172`),
    /// which is the "effective 2 sigma" deviation of papers-part2 §13.
    fn compute_error_weight(&self, res_squared: S, options: &LandmarkBlockOptions<S>) -> (S, S) {
        if options.huber_parameter > S::zero() {
            let huber_weight: S =
                if res_squared <= options.huber_parameter * options.huber_parameter {
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

    /// `linearizeLandmark` (`:110-191`): fill the block at the current
    /// linearization point and return this landmark's share of the error.
    ///
    /// Three behaviours are deliberate and are kept (trap 11, decision D32):
    /// a projection the camera rejected contributes **nothing** when
    /// `use_valid_projections_only` is set (`:152`); a non-finite Jacobian block
    /// is **zeroed with a warning**, not an error (`:153-163`, which the comment
    /// at `:165-166` says used to set `NumericalFailure`); and the two pose
    /// blocks are accumulated with `+=` (`:178-179`), which is what makes a
    /// landmark observed in its own host frame — where the host and target
    /// columns coincide — come out right.
    pub fn linearize_landmark(
        &mut self,
        lm: &Landmark<S>,
        rel_pose_lin: &[RelPoseLin<S>],
        cameras: &[CameraEnum<S>],
        options: &LandmarkBlockOptions<S>,
    ) -> Result<S, LinearizeError> {
        // `storage.setZero()` (`:115-117`).
        self.storage.fill(S::zero());

        let mut error_sum: S = S::zero();

        for (i, obs) in self.observations.iter().enumerate() {
            let Some(rel_idx) = obs.rel_pose else {
                // `if (pose_lin_vec[i])` (`:137`): a dropped measurement.
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
            let cam: &CameraEnum<S> =
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

            // `if (is_fixed_) d_res_d_p.setZero()` (`:150`).
            if self.is_fixed {
                d_res_d_p.fill(S::zero());
            }

            // `:152`.
            if options.use_valid_projections_only && !valid {
                continue;
            }

            // `:153-163`: zeroed, never fatal.
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

            // `:168-172`. `res.squaredNorm()` is a contiguous two-coefficient
            // reduction, so there is only one order to take.
            let res_squared: S = res[0] * res[0] + res[1] * res[1];
            let (weighted_error, weight) = self.compute_error_weight(res_squared, options);
            let sqrt_weight: S = weight.sqrt() / options.obs_std_dev;
            error_sum += weighted_error / (options.obs_std_dev * options.obs_std_dev);

            // `:174-175`.
            for r in 0..2 {
                for col in 0..3 {
                    self.storage[(obs_idx + r, self.lm_idx + col)] =
                        sqrt_weight * d_res_d_p[(r, col)];
                }
                self.storage[(obs_idx + r, self.res_idx)] = sqrt_weight * res[r];
            }

            // `:177-179`. The scaling happens once, in place, and then both pose
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

    /// `performQR` (`:193-206`): eliminate the three landmark columns.
    pub fn perform_qr(&mut self, options: &LandmarkBlockOptions<S>) -> Result<(), LinearizeError> {
        if self.state != LandmarkBlockState::Linearized {
            return Err(LinearizeError::WrongState {
                expected: LandmarkBlockState::Linearized,
                found: self.state,
            });
        }
        if options.use_householder {
            self.perform_qr_householder();
        } else {
            self.perform_qr_givens();
        }
        self.state = LandmarkBlockState::Marginalized;
        Ok(())
    }

    /// `performQRHouseholder` (`:441-454`): three reflections, each applied to
    /// the whole width of the block.
    fn perform_qr_householder(&mut self) {
        for k in 0..3 {
            // `remainingRows = num_rows - k - 3` (`:446`): the damping rows are
            // excluded, so the reflection never touches them.
            //
            // **Deviation.** With fewer than two observations the count runs
            // out and C++ calls `makeHouseholder` on a segment of length zero,
            // whose `tail` is then a block of length -1 — an assertion in a
            // debug build and undefined behaviour in a release one. basalt
            // never builds such a block (`min_num_obs = 2`,
            // `landmark_database.cpp:207`), but the port must not be the thing
            // that crashes if one ever appears, so the reflection is skipped.
            let remaining_rows: usize = self.num_rows.saturating_sub(k + 3);
            if remaining_rows == 0 {
                continue;
            }
            let (tau, _beta) = make_householder(
                &self.storage,
                self.lm_idx + k,
                k,
                remaining_rows,
                // `storage` stands for an `Eigen::RowMajor` matrix (`:530`), so
                // its columns are strided and reduce sequentially.
                ColumnRedux::Strided,
                &mut self.work_essential,
            );
            apply_householder_on_the_left(
                &mut self.storage,
                k,
                remaining_rows,
                &self.work_essential[..remaining_rows.saturating_sub(1)],
                tau,
                &mut self.work_row,
            );
        }
    }

    /// `performQRGivens` (`:429-439`): Golub & Van Loan Algorithm 5.2.4.
    ///
    /// Not reached with basalt's shipped options (`use_householder = true`), and
    /// kept because it is the reference the Householder path is checked against.
    fn perform_qr_givens(&mut self) {
        // C++'s `num_rows - 4` underflows on a block with no observations; see
        // the note in [`Self::perform_qr_householder`].
        if self.num_rows < 4 {
            return;
        }
        for n in 0..3 {
            let mut m: usize = self.num_rows - 4;
            while m > n {
                let rot: JacobiRotation<S> = make_givens(
                    self.storage[(m - 1, self.lm_idx + n)],
                    self.storage[(m, self.lm_idx + n)],
                );
                apply_rotation_on_the_left(&mut self.storage, m, m - 1, rot);
                m -= 1;
            }
        }
    }

    /// `backSubstitute(pose_inc, l_diff)` (`:253-327`): recover this landmark's
    /// increment from the pose increment, add its share of the model cost
    /// change, and apply it.
    ///
    /// Two behaviours worth naming:
    ///
    /// * a singular `Q1Jl` is **warned about and skipped**, not an error
    ///   (`:263-269`, trap 11), and an unusually small determinant only warns;
    /// * the inverse distance is **projected**, not clamped:
    ///   `max(0, inv_dist + inc[2])` (`:326`, trap 12).
    ///
    /// C++ also undoes the damping before the model cost change (`:310`) and
    /// scales the increment by `Jl_col_scale` after it (`:322-323`); the port
    /// carries neither, because nothing damps or scales (D34, D68).
    ///
    /// `pose_inc` must be `aom.total_size` long (`:259`).
    pub fn back_substitute(
        &mut self,
        lm: &mut Landmark<S>,
        pose_inc: &DVector<S>,
        l_diff: &mut S,
    ) -> Result<(), LinearizeError> {
        if self.state != LandmarkBlockState::Marginalized {
            return Err(LinearizeError::WrongState {
                expected: LandmarkBlockState::Marginalized,
                found: self.state,
            });
        }
        // `if (is_fixed_) return` (`:256`).
        if self.is_fixed {
            return Ok(());
        }
        if pose_inc.nrows() != self.padding_idx {
            return Err(LinearizeError::PoseIncrementSize {
                expected: self.padding_idx,
                found: pose_inc.nrows(),
            });
        }

        // `Q1Jl` (`:261`), the upper triangle of the 3x3 at the top of the
        // landmark columns.
        let mut q1jl: Matrix3<S> = Matrix3::zeros();
        for r in 0..3 {
            for col in r..3 {
                q1jl[(r, col)] = self.storage[(r, self.lm_idx + col)];
            }
        }

        // `abs(Q1Jl.determinant())` (`:263`). `TriangularView::determinant()` is
        // `m_matrix.diagonal().prod()`, and that diagonal is strided, so Eigen
        // takes the scalar unroller's `Length / 2` split — `d0 * (d1 * d2)` — in
        // both precisions (`Redux.h:98-108`).
        let det: S = (q1jl[(0, 0)] * (q1jl[(1, 1)] * q1jl[(2, 2)])).abs();
        if det == S::zero() {
            // `:264-266`, trap 11: skip this landmark, keep the rest.
            log::warn!(
                "det(Q1Jl) == 0, skipping backsubstitution for lm: Landmark(id={:?}, host_kf_id={:?})",
                self.lm_id,
                self.host_kf_id
            );
            return Ok(());
        } else if det < c::<S>(0.01) {
            // `:267-269`.
            log::warn!(
                "Unusually small det(Q1Jl)={}, lm: Landmark(id={:?}, host_kf_id={:?})",
                det.to_f64(),
                self.lm_id,
                self.host_kf_id
            );
        }

        // `Q1Jr + Q1Jp * pose_inc` (`:271-274`).
        let mut rhs: Vector3<S> = Vector3::zeros();
        for r in 0..3 {
            let mut acc: S = S::zero();
            for k in 0..self.padding_idx {
                acc += self.storage[(r, k)] * pose_inc[k];
            }
            rhs[r] = self.storage[(r, self.res_idx)] + acc;
        }

        // `-Q1Jl.solve(...)`: the upper-triangular back substitution, in Eigen's
        // order — the dot product of the row's tail against the already solved
        // tail of the solution, then one division.
        let mut inc: Vector3<S> = Vector3::zeros();
        for r in (0..3usize).rev() {
            let mut acc: S = S::zero();
            for k in (r + 1)..3 {
                acc += q1jl[(r, k)] * inc[k];
            }
            inc[r] = (rhs[r] - acc) / q1jl[(r, r)];
        }
        inc = -inc;

        // `:310` calls `setLandmarkDamping(0)` here, to undo the damping before
        // the model cost change. The port has no damping (D34, D68) and the
        // three damping rows are provably still zero — `storage` starts zeroed,
        // the observations fill rows `0..2*obs`, and both QR paths stop at
        // `num_rows - 3` — so there is nothing to undo.

        // `QJinc = storage.topLeftCorner(num_rows - 3, padding_idx) * pose_inc`
        // (`:313`), then `QJinc.head<3>() += Q1Jl * inc` (`:316`) with `Q1Jl`
        // re-read from the now-undamped storage.
        let q2_rows: usize = self.num_rows - 3;
        let mut qjinc: DVector<S> = DVector::zeros(q2_rows);
        for r in 0..q2_rows {
            let mut acc: S = S::zero();
            for k in 0..self.padding_idx {
                acc += self.storage[(r, k)] * pose_inc[k];
            }
            qjinc[r] = acc;
        }
        for r in 0..3.min(q2_rows) {
            let mut acc: S = S::zero();
            for k in r..3 {
                acc += self.storage[(r, self.lm_idx + k)] * inc[k];
            }
            qjinc[r] += acc;
        }

        // `diff = QJinc^T * (0.5 * QJinc + Qr)` (`:318-320`).
        let mut diff: S = S::zero();
        for r in 0..q2_rows {
            diff += qjinc[r] * (c::<S>(0.5) * qjinc[r] + self.storage[(r, self.res_idx)]);
        }
        *l_diff -= diff;

        // `:322-326`, without `:322-323`'s `Jl_col_scale` multiply: the scale
        // is all ones with nothing to scale the columns (D68).
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

    /// `get_dense_Q2Jp_Q2r(Q2Jp, Q2r, start_idx)` (`:472-478`): the null-space
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
        // C++ asserts the column count (`:475`) and indexes the rows unchecked.
        if q2jp.ncols() != self.padding_idx {
            return Err(LinearizeError::StackedSystemSize {
                expected: self.padding_idx,
                found: q2jp.ncols(),
            });
        }
        let end: usize = start_idx
            .checked_add(rows)
            .ok_or(LinearizeError::LayoutOverflow)?;
        if end > q2jp.nrows() || end > q2r.nrows() {
            return Err(LinearizeError::StackedSystemSize {
                expected: end,
                found: q2jp.nrows().min(q2r.nrows()),
            });
        }
        for r in 0..rows {
            q2r[start_idx + r] = self.storage[(3 + r, self.res_idx)];
            for k in 0..self.padding_idx {
                q2jp[(start_idx + r, k)] = self.storage[(3 + r, k)];
            }
        }
        Ok(())
    }

    /// The pose columns `Self::add_dense_h_b_active` writes; see the field.
    pub fn active_cols(&self) -> &[usize] {
        &self.active_cols
    }

    /// Every pose column of the dense system, `0..padding_idx`: what
    /// [`Self::add_dense_h_b`] writes.
    pub fn pose_columns(&self) -> std::ops::Range<usize> {
        0..self.padding_idx
    }

    /// `add_dense_H_b(H, b)` (`:494-500`): `H += JᵀJ`, `b += Jᵀr` over the same
    /// `Q₂` rows.
    ///
    /// **Full width**: every column of `0..padding_idx` is written, as the C++
    /// writes it, because `h` and `b` are the caller's and this method knows
    /// nothing about what is in them. `Self::add_dense_h_b_active` is the same
    /// sum over the observed columns only, for the one caller that owns its
    /// destination and can prove the rest is the identity.
    ///
    /// Both sums run over the block's rows in increasing order, one output
    /// coefficient at a time. Eigen calls its general matrix product here, whose
    /// blocking may associate a long sum differently; over `num_rows - 3` rows —
    /// at most a few tens — the difference is at the last bits and the fixture
    /// measures it.
    pub fn add_dense_h_b(
        &self,
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
        scratch: &mut DenseHbScratch<S>,
    ) -> Result<(), LinearizeError> {
        self.check_dense_h_b_size(h, b)?;
        // The cold path — the tests and the reduction's non-finite fallback —
        // so the column list is built here rather than kept in `scratch`.
        let columns: Vec<usize> = self.pose_columns().collect();
        self.add_dense_h_b_over(&columns, h, b, scratch);
        Ok(())
    }

    /// [`Self::add_dense_h_b`] over [`Self::active_cols`] only.
    ///
    /// Bit-identical to the full width when the destination holds `+0.0` at
    /// every column this block does not observe and
    /// [`Self::active_writeback_is_exact`] holds; the dense reduction is the one
    /// caller that can say both, on its own accumulators.
    pub(crate) fn add_dense_h_b_active(
        &self,
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
        scratch: &mut DenseHbScratch<S>,
    ) -> Result<(), LinearizeError> {
        self.check_dense_h_b_size(h, b)?;
        self.add_dense_h_b_over(&self.active_cols, h, b, scratch);
        Ok(())
    }

    /// Whether the skip [`Self::add_dense_h_b_active`] makes is the identity on
    /// a destination that is `+0.0` outside [`Self::active_cols`].
    ///
    /// The skipped writes are `x += Σ (a * 0)` over the columns the block does
    /// not observe. That is `x += ±0.0`, which leaves a `+0.0` `x` alone — but
    /// only while every factor is finite and those columns really are zero.
    /// Neither is free: [`Landmark::add_observation`] accepts a non-finite
    /// keypoint, the Huber weight carries the NaN past the Jacobian checks of
    /// [`Self::linearize_landmark`], and the Householder reflections of
    /// `crate::eigen::qr`'s reflections act on whole rows, which spreads it into columns the
    /// block never observed. One pass over the `Q₂` rows decides both, and a
    /// block that fails takes the full-width path so those NaNs are written
    /// (decision D32: NaN handling mirrors basalt).
    pub(crate) fn active_writeback_is_exact(&self) -> bool {
        let rows: usize = self.num_q2rows();
        let mut active = self.active_cols.iter().copied().peekable();
        for column in self.pose_columns() {
            if active.next_if_eq(&column).is_some() {
                for r in 0..rows {
                    if !self.storage[(3 + r, column)].to_f64().is_finite() {
                        return false;
                    }
                }
            } else {
                // `-0.0 == 0.0`, which is what this wants: either zero makes
                // every product `±0.0`.
                for r in 0..rows {
                    if self.storage[(3 + r, column)] != S::zero() {
                        return false;
                    }
                }
            }
        }
        // The residual column, the other factor of `b`.
        (0..rows).all(|r| self.storage[(3 + r, self.res_idx)].to_f64().is_finite())
    }

    /// C++ asserts the destination is big enough (`:496`); a typed error here.
    fn check_dense_h_b_size(&self, h: &DMatrix<S>, b: &DVector<S>) -> Result<(), LinearizeError> {
        if h.nrows() < self.padding_idx
            || h.ncols() < self.padding_idx
            || b.nrows() < self.padding_idx
        {
            return Err(LinearizeError::StackedSystemSize {
                expected: self.padding_idx,
                found: h.nrows().min(b.nrows()),
            });
        }
        Ok(())
    }

    /// The sum itself, over `columns` — ascending, inside `0..padding_idx`.
    fn add_dense_h_b_over(
        &self,
        columns: &[usize],
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
        scratch: &mut DenseHbScratch<S>,
    ) {
        let rows: usize = self.num_q2rows();
        let live: usize = columns.len();
        // `[ the columns | the residual ]`, one contiguous row per `Q2` row;
        // see [`DenseHbScratch`] for why the transpose is worth its copy. The
        // row is padded to a whole number of [`LANES`] so every chunk below is
        // a full one; the padding is `+0.0`, contributes `factor * 0.0` to a
        // coefficient nothing reads, and is never written back.
        let width: usize = live + 1;
        let stride: usize = width.div_ceil(LANES) * LANES;
        scratch.rows.clear();
        scratch.rows.resize(rows * stride, S::zero());
        for (slot, &column) in columns.iter().enumerate() {
            for r in 0..rows {
                scratch.rows[r * stride + slot] = self.storage[(3 + r, column)];
            }
        }
        for r in 0..rows {
            scratch.rows[r * stride + live] = self.storage[(3 + r, self.res_idx)];
        }

        let transposed: &[S] = &scratch.rows;
        for (slot, &i) in columns.iter().enumerate() {
            for lo in (0..stride).step_by(LANES) {
                // One row of `H` over `LANES` of its columns. Every coefficient
                // still sums over the `Q2` rows in ascending row order — `r` is
                // the loop below and each lane is touched once per row — so
                // this is the same sum as one accumulator per column was, in
                // the same order, over `LANES` columns at a time.
                let mut partial: [S; LANES] = [S::zero(); LANES];
                for r in 0..rows {
                    let row: &[S] = &transposed[(r * stride)..((r + 1) * stride)];
                    let factor: S = row[slot];
                    let source: &[S] = &row[lo..(lo + LANES)];
                    for (accumulator, &value) in partial.iter_mut().zip(source.iter()) {
                        *accumulator += factor * value;
                    }
                }
                let end: usize = (lo + LANES).min(width);
                for (offset, &value) in partial.iter().enumerate().take(end - lo) {
                    let j: usize = lo + offset;
                    if j < live {
                        h[(i, columns[j])] += value;
                    } else {
                        // `j == live`: the residual column.
                        b[i] += value;
                    }
                }
            }
        }
    }

    /// `numQ2rows()` (`:426`): `num_rows - 3`, the rows the reduced camera
    /// system takes. Note this counts the `Q₁` rows too — the name is basalt's.
    pub fn num_q2rows(&self) -> usize {
        self.num_rows - 3
    }

    /// The landmark this block belongs to.
    pub fn landmark_id(&self) -> LandmarkId {
        self.lm_id
    }

    /// `getState()` (`:424`).
    pub fn state(&self) -> LandmarkBlockState {
        self.state
    }

    /// `isNumericalFailure()` (`:22`).
    ///
    /// Always false in the port for the same reason it is almost always false in
    /// C++: the comment at `:165-166` records that the `NumericalFailure` branch
    /// was removed in favour of zeroing the offending Jacobian.
    pub fn is_numerical_failure(&self) -> bool {
        self.state == LandmarkBlockState::NumericalFailure
    }

    /// The layout: `(num_rows, num_cols, padding_idx, lm_idx, res_idx)`.
    ///
    /// The arithmetic of `:83-96` is the thing most likely to go wrong in a
    /// port, so it is observable and the fixture checks all five numbers.
    pub fn layout(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.num_rows,
            self.num_cols,
            self.padding_idx,
            self.lm_idx,
            self.res_idx,
        )
    }

    /// The block buffer, `storage` (`:530`), row `r`, column `c`.
    pub fn storage(&self) -> &DMatrix<S> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::calib::{CameraModel, Kb4Params};
    use crate::lie::Se3;
    use crate::types::LandmarkId;
    use nalgebra::{Matrix4, Matrix6, Vector4};

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

    fn cameras() -> Vec<CameraEnum<f64>> {
        let model: CameraModel<f64> = CameraModel::Kb4(Kb4Params {
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
            CameraEnum::from_model(&model).unwrap(),
            CameraEnum::from_model(&model).unwrap(),
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
        assert_eq!(block.active_cols, (0..POSE_SIZE).collect::<Vec<usize>>());

        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        let zero_after = |block: &LandmarkBlock<f64>, stage: &str| {
            for column in POSE_SIZE..block.padding_idx {
                for row in 0..block.num_rows {
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

    /// The system `add_dense_h_b_active` writes over `active_cols` equals the
    /// one a loop over the whole `padding_idx` square produces, coefficient by
    /// coefficient and **bit for bit**. Both tests of the skip rest on this.
    fn assert_dense_h_b_is_the_full_loop(block: &LandmarkBlock<f64>) {
        let mut h: DMatrix<f64> = DMatrix::zeros(block.padding_idx, block.padding_idx);
        let mut b: DVector<f64> = DVector::zeros(block.padding_idx);
        block
            .add_dense_h_b_active(&mut h, &mut b, &mut DenseHbScratch::default())
            .unwrap();
        let rows: usize = block.num_q2rows();
        for i in 0..block.padding_idx {
            for j in 0..block.padding_idx {
                let mut acc: f64 = 0.0;
                for r in 0..rows {
                    acc += block.storage[(3 + r, i)] * block.storage[(3 + r, j)];
                }
                assert_eq!(h[(i, j)].to_bits(), acc.to_bits(), "H({i}, {j})");
            }
            let mut acc: f64 = 0.0;
            for r in 0..rows {
                acc += block.storage[(3 + r, i)] * block.storage[(3 + r, block.res_idx)];
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
        assert_eq!(block.active_cols, (0..POSE_SIZE).collect::<Vec<usize>>());

        let n: usize = block.padding_idx;
        let mut h: DMatrix<f64> = DMatrix::from_element(n, n, -0.0);
        let mut b: DVector<f64> = DVector::from_element(n, -0.0);
        block
            .add_dense_h_b(&mut h, &mut b, &mut DenseHbScratch::default())
            .unwrap();

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
            !block.active_writeback_is_exact(),
            "a NaN block must not take the skip"
        );
        let spread: bool = (POSE_SIZE..block.padding_idx).any(|column| {
            (0..block.num_q2rows()).any(|r| !block.storage[(3 + r, column)].is_finite())
        });
        assert!(
            spread,
            "the QR was supposed to spread the NaN off the block's own columns"
        );
    }

    /// A measurement dropped during marginalization writes no pose columns.
    ///
    /// `linearizeLandmark` skips an observation with no relative pose (`:137`),
    /// so nothing ever writes at its `abs_t_idx` — which is the `0` sentinel of
    /// `:74`, another frame's first column. The block here is hosted in frame 1,
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
            block.active_cols,
            live.clone().collect::<Vec<usize>>(),
            "the dropped observation's sentinel offset is not a column it writes"
        );
        block
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        block.perform_qr(&options()).unwrap();

        for column in (0..block.padding_idx).filter(|column| !live.contains(column)) {
            for row in 0..block.num_rows {
                assert_eq!(
                    block.storage[(row, column)],
                    0.0,
                    "column {column} moved off zero"
                );
            }
        }
        assert_dense_h_b_is_the_full_loop(&block);
    }

    /// The layout arithmetic of `:83-96` on every remainder of the padding rule.
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

    /// A host frame outside the ordering is `:62`'s assertion, typed.
    #[test]
    fn a_host_outside_the_ordering_is_an_error() {
        let (_, lm, _) = fixture(1);
        let empty: AbsOrderMap = AbsOrderMap::new();
        assert_eq!(
            LandmarkBlock::allocate(lm.id, &lm, &index, &empty, false).unwrap_err(),
            LinearizeError::HostNotInOrdering { frame_id: 0 }
        );
    }

    /// A missing relative pose is `:68`'s assertion, typed.
    #[test]
    fn a_missing_relative_pose_is_an_error() {
        let (aom, lm, _) = fixture(1);
        let err = LandmarkBlock::allocate(lm.id, &lm, &|_, _| None, &aom, false).unwrap_err();
        assert!(matches!(err, LinearizeError::MissingRelativePose { .. }));
    }

    /// The state machine of `landmark_block.hpp:50`: every method that C++
    /// asserts a state for returns a typed error instead.
    #[test]
    fn methods_refuse_to_run_out_of_order() {
        let (aom, lm, rel) = fixture(1);
        let mut block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        assert_eq!(block.state(), LandmarkBlockState::Allocated);
        assert!(!block.is_numerical_failure());

        // `performQR` asserts `Linearized` (`:194`).
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

        // `:259`: the increment is the whole ordering.
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

    /// Two of basalt's deliberate non-failures (trap 11).
    ///
    /// A landmark seen **once** gives a `2 x 3` `J_l`, so after the QR the third
    /// diagonal of `Q1Jl` is zero, `det(Q1Jl) == 0`, and `backSubstitute` warns
    /// and returns without touching the landmark (`:263-266`). A **fixed**
    /// landmark returns even earlier (`:256`). Neither is an error, and turning
    /// either into one would change the estimator.
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

        // And a fixed block returns before it even looks (`:256`).
        let (aom, lm, rel) = fixture(1);
        let mut fixed: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, true).unwrap();
        fixed
            .linearize_landmark(&lm, &rel, &cameras(), &options())
            .unwrap();
        // `is_fixed_` zeroes `d_res_d_p` (`:150`), so the landmark columns are
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

    /// An observation whose target frame is not in the ordering is dropped:
    /// C++ stores a null `RelPoseLin*` for it (`:70-75`) and skips it at `:137`,
    /// leaving its two rows zero. That is how a measurement dropped during
    /// marginalization survives allocation and contributes nothing.
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
        // ...and the two live observations did write.
        assert!(block.storage().rows(0, 4).iter().any(|v| *v != 0.0));
    }

    /// The Huber branch of `compute_error_weight` (`:456-470`): below the
    /// threshold the weight is one, above it the error grows linearly rather
    /// than quadratically.
    #[test]
    fn the_huber_weight_crosses_at_the_threshold() {
        let (aom, lm, _) = fixture(1);
        let block: LandmarkBlock<f64> =
            LandmarkBlock::allocate(lm.id, &lm, &index, &aom, false).unwrap();
        let opt: LandmarkBlockOptions<f64> = options();
        let delta: f64 = opt.huber_parameter;

        let (error_in, weight_in) = block.compute_error_weight(0.25 * delta * delta, &opt);
        assert_eq!(weight_in, 1.0);
        assert_eq!(error_in, 0.5 * 0.25 * delta * delta);

        // Exactly at the threshold the comparison is `<=`, so the weight is one.
        let (_, weight_at) = block.compute_error_weight(delta * delta, &opt);
        assert_eq!(weight_at, 1.0);

        let res_squared: f64 = 4.0 * delta * delta;
        let (error_out, weight_out) = block.compute_error_weight(res_squared, &opt);
        assert!((weight_out - 0.5).abs() < 1e-15, "{weight_out}");
        assert!((error_out - 0.5 * 1.5 * 0.5 * res_squared).abs() < 1e-15);

        // With the threshold off it is a plain squared norm (`:466-468`).
        let plain: LandmarkBlockOptions<f64> = LandmarkBlockOptions {
            huber_parameter: 0.0,
            ..opt
        };
        let (error, weight) = block.compute_error_weight(res_squared, &plain);
        assert_eq!(weight, 1.0);
        assert_eq!(error, 0.5 * res_squared);
    }

    /// An ordering with one absurd block size passes every per-dimension check
    /// and then multiplies out to a matrix nalgebra cannot allocate. C++ calls
    /// `storage.resize(num_rows, num_cols)` (`:98`), which would abort in the
    /// allocator; the port refuses first (decision D32).
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
            matches!(err, LinearizeError::BlockTooLarge { .. }),
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
