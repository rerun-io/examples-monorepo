//! The bundle-adjustment base: sliding-window state, the reprojection residual
//! and DLT triangulation.
//!
//! Ported from `include/basalt/vi_estimator/ba_base.h`,
//! `src/vi_estimator/ba_base.cpp` and `include/basalt/utils/ba_utils.h`.
//! Everything the square-root linearizer needs before it starts building
//! landmark blocks lives here: the two state maps, the landmark database, the
//! per-observation residual with its three Jacobians, the relative pose that is
//! hoisted per (host, target) pair, the error the Levenberg-Marquardt loop
//! accepts or rejects on, and `backup`/`restore`.
//!
//! Three conventions are load-bearing, and each one is a documented deviation of
//! basalt's code from its own papers (papers-part2 §13):
//!
//! * **The residual sign is flipped.** Paper 1 Eq. (8) is `r = z - pi(...)`;
//!   `ba_utils.h:117` computes `res -= kpt_obs`, i.e. `pi(...) - z`. `J^T J` does
//!   not care, `J^T r` does, and basalt compensates by negating the increment at
//!   `sqrt_keypoint_vio.cpp:1450` (`inc = -inc`). The port keeps **both** halves:
//!   [`linearize_point`] returns basalt's sign, and the increment is negated
//!   where basalt negates it (a later stage). Fixing one without the other
//!   inverts the whole optimisation.
//! * **Huber is applied to the raw pixel residual, `1/sigma` afterwards.**
//!   `ba_base.cpp:179-182` compares `res.norm()` against `huber_thresh` in
//!   pixels and only then divides by `obs_std_dev^2`. With the shipped
//!   `vio_obs_huber_thresh = 1.0` px and `vio_obs_std_dev = 0.5` px, the
//!   effective threshold is **2 sigma**, not 1.
//! * **The pose increment is the decoupled, left-multiplied one**
//!   (`imu_types.h:96-99`), which is what `d_res_d_xi` is taken with respect to.
//!
//! Parallelism: none in this stage. `compute_error` is written as a fixed-order
//! fold over per-host-frame partial results, so the `threads` config field can
//! later turn the middle line into a `par_chunks` over the same `Vec` with the
//! same sequential merge and produce the same floating-point sum (decision D31).

use std::collections::BTreeMap;

use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix2x4, Matrix2x6, Matrix3, Matrix4, Matrix4x2, Matrix4x3,
    Matrix6, Vector2, Vector3, Vector4,
};

use crate::calib::Calibration;
use crate::camera::{CameraEnum, CameraError};
use crate::eigen::blas::redux_contiguous;
use crate::eigen::norm3;
use crate::landmark::{Landmark, LandmarkDatabase, LandmarkError, StereographicParam};
use crate::lie::{LieScalar, Se3, So3, c};
use crate::types::{
    AbsOrderMap, CamId, FrameId, LandmarkId, MargLinData, POSE_SIZE, POSE_VEL_BIAS_SIZE,
    PoseStateWithLin, PoseVelBiasStateWithLin, TimeCamId,
};

/// What the bundle-adjustment base refuses to do.
///
/// Each variant replaces a place where C++ aborts, asserts or indexes out of
/// range. The estimator runs inside a released GIL where a panic aborts the
/// process, so none of these may be a panic (decision D32, trap 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BaError {
    /// `getPoseStateWithLin` found the timestamp in neither map and called
    /// `std::abort()` (`ba_base.h:131-142`).
    #[error("no pose or state for frame {t_ns} ns")]
    UnknownFrame {
        /// The timestamp that was looked up.
        t_ns: FrameId,
    },
    /// An image named a camera the calibration does not have; C++ indexes
    /// `calib.T_i_c[cam_id]` and `calib.intrinsics[cam_id]` unchecked
    /// (`ba_base.cpp:154-155`, `:188`).
    #[error("camera {cam_id} is not in the calibration ({camera_count} cameras)")]
    UnknownCamera {
        /// The camera index that was asked for.
        cam_id: CamId,
        /// Cameras the calibration carries.
        camera_count: usize,
    },
    /// The adjacency named a landmark the database does not hold, or a landmark
    /// has no observation in a target the adjacency lists it under; C++ throws
    /// from `at` (`ba_base.cpp:165-166`).
    #[error("landmark {0:?} is missing from the database or from that target")]
    InconsistentLandmark(LandmarkId),
    /// `computeDelta` met a block that is neither a pose nor a full state
    /// (`ba_base.cpp:300`, `BASALT_ASSERT(false)`).
    #[error("frame {frame_id}: block size {size} is neither {POSE_SIZE} nor {POSE_VEL_BIAS_SIZE}")]
    UnexpectedBlockSize {
        /// The frame the ordering names.
        frame_id: FrameId,
        /// The size it was given.
        size: usize,
    },
    /// `computeDelta` met a block whose linearization point is not frozen
    /// (`ba_base.cpp:294`, `:297`). Its delta would be meaningless.
    #[error("frame {frame_id} is in the marginalization ordering but is not linearized")]
    NotLinearized {
        /// The frame the ordering names.
        frame_id: FrameId,
    },
    /// The marginalization prior's matrix does not match its own ordering, or
    /// the destination system is too small (`ba_base.cpp:379`, `:444` assert).
    #[error("marginalization prior has {cols} columns, expected {total_size}")]
    MargPriorSize {
        /// The prior's column count.
        cols: usize,
        /// What its ordering says it should be.
        total_size: usize,
    },
    /// The prior's ordering disagrees with the window's, which C++ asserts
    /// block by block (`ba_base.cpp:383-388`).
    #[error("frame {frame_id} has a different offset in the prior than in the window")]
    MargOrderMismatch {
        /// The frame that disagrees.
        frame_id: FrameId,
    },
    /// A calibration this module cannot project with.
    #[error(transparent)]
    Camera(#[from] CameraError),
    /// A landmark-database refusal.
    #[error(transparent)]
    Landmark(#[from] LandmarkError),
}

// ─── the relative pose, hoisted per (host, target) pair ────────────────────

/// `computeRelPose` (`ba_utils.h:41-78`): the transform that takes a point in
/// the host camera frame to the target camera frame, with its two 6x6 Jacobians.
///
/// The composition is basalt's **decoupled** one (`:49-50`): the rotation is a
/// plain product, but the translation is `R_t^-1 (t_h - t_t)` rather than
/// anything `SE3::inverse` would produce. That is what makes the Jacobians below
/// match the left-multiplied increment of `imu_types.h:96-99`.
pub fn compute_rel_pose<S: LieScalar>(
    t_w_i_h: &Se3<S>,
    t_i_c_h: &Se3<S>,
    t_w_i_t: &Se3<S>,
    t_i_c_t: &Se3<S>,
    d_rel_d_h: Option<&mut Matrix6<S>>,
    d_rel_d_t: Option<&mut Matrix6<S>>,
) -> Se3<S> {
    let tmp2: Se3<S> = t_i_c_t.inverse();

    // `T_t_i_h_i` (`ba_utils.h:48-50`).
    let t_t_i_h_i: Se3<S> = Se3::new(
        t_w_i_t.rotation.inverse() * t_w_i_h.rotation,
        t_w_i_t.rotation.inverse() * (t_w_i_h.translation - t_w_i_t.translation),
    );

    let tmp: Se3<S> = tmp2 * t_t_i_h_i;
    let res: Se3<S> = tmp * *t_i_c_h;

    if let Some(out) = d_rel_d_h {
        // `RR = blkdiag(R, R)` with `R = T_w_i_h.so3().inverse().matrix()`
        // (`ba_utils.h:56-63`).
        let r: Matrix3<S> = t_w_i_h.rotation.inverse().matrix();
        let mut rr: Matrix6<S> = Matrix6::zeros();
        rr.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        rr.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        *out = tmp.adjoint() * rr;
    }

    if let Some(out) = d_rel_d_t {
        // `-T_i_c_t.inverse().Adj() * RR` (`ba_utils.h:67-74`).
        let r: Matrix3<S> = t_w_i_t.rotation.inverse().matrix();
        let mut rr: Matrix6<S> = Matrix6::zeros();
        rr.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        rr.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        *out = -(tmp2.adjoint() * rr);
    }

    res
}

// ─── the reprojection residual ─────────────────────────────────────────────

/// Everything [`linearize_point`] can be asked to fill in besides the residual.
///
/// C++ passes four raw pointers, three of them defaulted to null
/// (`ba_utils.h:80-85`). One struct of options keeps the call sites honest and
/// costs nothing: every field is a borrowed fixed-size matrix, so the residual
/// path allocates nothing.
#[derive(Debug)]
pub struct LinearizePointOut<'a, S: LieScalar> {
    /// `d_res_d_xi` (2x6): the residual against the relative-pose increment.
    pub d_res_d_xi: Option<&'a mut Matrix2x6<S>>,
    /// `d_res_d_p` (2x3): the residual against `[direction(2), inv_dist]`.
    pub d_res_d_p: Option<&'a mut Matrix2x3<S>>,
    /// `proj` (4): `[u, v, inv_depth_in_target, unused]`, for visualisation.
    pub proj: Option<&'a mut Vector4<S>>,
}

impl<S: LieScalar> Default for LinearizePointOut<'_, S> {
    /// All four C++ pointers null: the residual only
    /// (`ba_utils.h:83-85` default arguments).
    fn default() -> Self {
        Self {
            d_res_d_xi: None,
            d_res_d_p: None,
            proj: None,
        }
    }
}

/// `linearizePoint` (`ba_utils.h:80-138`): one observation's residual and, on
/// request, its Jacobians.
///
/// The residual is `pi(T_t_h * q) - z` (`:94-117`) with `q` the homogeneous
/// landmark `[unproject(direction), inv_dist]`. **The sign is basalt's, i.e.
/// flipped relative to Paper 1 Eq. (8)** — see the module docs.
///
/// Returns `false` when the camera rejects the point or the pixel is not finite
/// (`:97-98`), in which case `res` holds whatever the camera model wrote and the
/// caller must ignore it, exactly as C++ does.
pub fn linearize_point<S: LieScalar>(
    kpt_obs: &Vector2<S>,
    kpt_pos: &Landmark<S>,
    t_t_h: &Matrix4<S>,
    cam: &CameraEnum<S>,
    res: &mut Vector2<S>,
    out: &mut LinearizePointOut<'_, S>,
) -> bool {
    // `StereographicParam::unproject(direction, &Jup)` then the inverse distance
    // into the homogeneous slot (`ba_utils.h:89-92`).
    let mut jup: Matrix4x2<S> = Matrix4x2::zeros();
    let mut p_h_3d: Vector4<S> =
        StereographicParam::unproject_with_jacobian(&kpt_pos.direction, &mut jup);
    p_h_3d[3] = kpt_pos.inv_dist;

    let p_t_3d: Vector4<S> = t_t_h * p_h_3d;

    let mut jp: Matrix2x4<S> = Matrix2x4::zeros();
    let mut valid: bool = cam.project_with_jacobian(&p_t_3d, res, &mut jp);
    // `valid &= res.array().isFinite().all()` (`ba_utils.h:98`).
    valid &= res[0].to_f64().is_finite() && res[1].to_f64().is_finite();

    if !valid {
        return false;
    }

    if let Some(proj) = out.proj.as_deref_mut() {
        // `proj.head<2>() = res` and the inverse depth in the target frame
        // (`ba_utils.h:113-116`), before the observation is subtracted.
        proj[0] = res[0];
        proj[1] = res[1];
        proj[2] = p_t_3d[3] / norm3(p_t_3d[0], p_t_3d[1], p_t_3d[2]);
    }

    // `res -= kpt_obs` (`ba_utils.h:117`) — the flipped sign.
    *res -= kpt_obs;

    if let Some(d_res_d_xi) = out.d_res_d_xi.as_deref_mut() {
        // `d_point_d_xi` (4x6, `ba_utils.h:120-123`). The inverse-distance
        // scaling on the translation columns is what the homogeneous `q` costs.
        let mut d_point_d_xi: nalgebra::Matrix4x6<S> = nalgebra::Matrix4x6::zeros();
        let mut ident: Matrix3<S> = Matrix3::identity();
        ident *= kpt_pos.inv_dist;
        d_point_d_xi.fixed_view_mut::<3, 3>(0, 0).copy_from(&ident);
        d_point_d_xi
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-So3::hat(&Vector3::new(p_t_3d[0], p_t_3d[1], p_t_3d[2]))));
        // `row(3).setZero()` (`:123`) — already zero from the constructor.
        *d_res_d_xi = jp * d_point_d_xi;
    }

    if let Some(d_res_d_p) = out.d_res_d_p.as_deref_mut() {
        // `Jpp` (4x3, `ba_utils.h:129-132`).
        let mut jpp: Matrix4x3<S> = Matrix4x3::zeros();
        let top: nalgebra::Matrix3x4<S> = t_t_h.fixed_view::<3, 4>(0, 0).into_owned();
        jpp.fixed_view_mut::<3, 2>(0, 0).copy_from(&(top * jup));
        jpp.set_column(2, &t_t_h.column(3));
        *d_res_d_p = jp * jpp;
    }

    true
}

// ─── triangulation ─────────────────────────────────────────────────────────

/// The implicit-shift sweep budget [`triangulate`] gives its 4x4 SVD.
///
/// nalgebra treats `0` as "iterate until convergence"; a tracker that has to
/// return a frame in a few milliseconds cannot. Four rows converge in a handful
/// of sweeps, so this is a refusal threshold rather than a tuning knob — a
/// matrix that reaches it is degenerate, and `None` is the honest answer.
const SVD_MAX_ITERATIONS: usize = 64;

/// `triangulate(f0, f1, T_0_1)` (`ba_base.h:89-116`): the DLT, returning a
/// homogeneous `[unit direction (3), inverse distance]` in frame 0.
///
/// `f0` and `f1` are the two bearing vectors, `T_0_1` the transform from frame 1
/// to frame 0. The 4x4 `A` is built from the two projection matrices exactly as
/// `:103-107`, the null space comes from the last column of `V`, and the sign is
/// flipped when the result points away from `f0` (`:113`).
///
/// The caller decides what to do with the answer: basalt accepts it only when
/// every coefficient is finite and `0 < inv_dist < 3`
/// (`sqrt_keypoint_vio.cpp:534`), i.e. no further than 1/3 m.
///
/// **Deviations from the C++.** Two, both about what a refusal is.
///
/// The 4x4 null space comes from [`nalgebra::linalg::SVD`] rather than from a
/// port of Eigen's `JacobiSVD` sweep, and it is computed in `f64` whatever `S`
/// is: the DLT rows of a `Vio<f32>` are differences of same-order products, so
/// the smallest singular value is the one quantity in the whole estimator that
/// is built out of cancellation, and its vector is the answer. Promoting costs
/// one 4x4 solve per new landmark and nothing else. The decomposition is
/// `try_new_unordered`, with the smallest singular value found by an explicit
/// scan: the sorting constructor panics on a NaN singular value, and the
/// unsorted one lets the non-finite check below stay the only refusal path.
///
/// On a non-finite input Eigen sets `InvalidInput` and returns
/// with `m_matrixV` never written (`JacobiSVD.h:721-727`); basalt then reads it,
/// which is undefined behaviour. There is no value to reproduce, so the port
/// returns `None` — for that, for a decomposition that does not converge inside
/// its iteration budget, and for a homogeneous vector whose spatial part has no
/// direction to normalise. All three used to come back as an all-NaN or
/// part-infinite vector that the acceptance gate above then rejected, which
/// made a number the control flow.
pub fn triangulate<S: LieScalar>(
    f0: &Vector3<S>,
    f1: &Vector3<S>,
    t_0_1: &Se3<S>,
) -> Option<Vector4<S>> {
    // `P1.setIdentity()`, `P2 = T_0_1.inverse().matrix3x4()` (`ba_base.h:98-100`).
    let p1: nalgebra::Matrix3x4<S> = {
        let mut m: nalgebra::Matrix3x4<S> = nalgebra::Matrix3x4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        m
    };
    let p2: nalgebra::Matrix3x4<S> = t_0_1.inverse().matrix3x4();

    let mut a: Matrix4<S> = Matrix4::zeros();
    a.row_mut(0)
        .copy_from(&(p1.row(2) * f0[0] - p1.row(0) * f0[2]));
    a.row_mut(1)
        .copy_from(&(p1.row(2) * f0[1] - p1.row(1) * f0[2]));
    a.row_mut(2)
        .copy_from(&(p2.row(2) * f1[0] - p2.row(0) * f1[2]));
    a.row_mut(3)
        .copy_from(&(p2.row(2) * f1[1] - p2.row(1) * f1[2]));

    let wide: Matrix4<f64> = a.map(|value| value.to_f64());
    if wide.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // `max_niter` bounds the total implicit-shift sweeps: a 4x4 that has not
    // converged in `SVD_MAX_ITERATIONS` is not going to, and a landmark that
    // does not exist is a better answer than an unbounded loop in the tracker.
    let svd: nalgebra::SVD<f64, nalgebra::U4, nalgebra::U4> =
        nalgebra::SVD::try_new_unordered(wide, false, true, f64::EPSILON, SVD_MAX_ITERATIONS)?;
    if svd.singular_values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // A DLT whose largest singular value is zero carries no constraint at all —
    // two zero bearing vectors build a zero `A` — so *every* direction is a null
    // direction and the one the decomposition happens to return is fabricated.
    // Eigen's sweep refused this by leaving `V` the identity and handing back its
    // last column, `[0, 0, 0, 1]`, whose spatial part then failed to normalise;
    // the rank is the thing that was being tested, so it is tested directly.
    if svd.singular_values.max() <= 0.0 {
        return None;
    }
    // The null vector is the right-singular vector of the *smallest* singular
    // value, and `v_t` holds the right-singular vectors as its **rows**. The
    // decomposition is unordered, so the row is found rather than assumed.
    let mut smallest: usize = 0;
    for i in 1..4 {
        if svd.singular_values[i] < svd.singular_values[smallest] {
            smallest = i;
        }
    }
    let v_t: nalgebra::Matrix4<f64> = svd.v_t?;

    let mut world_point: Vector4<S> = v_t.row(smallest).transpose().map(|value| c::<S>(value));
    let norm: S = norm3(world_point[0], world_point[1], world_point[2]);
    // A homogeneous vector with no spatial part has no direction: dividing by
    // its norm used to hand the caller `[NaN, NaN, NaN, inf]`.
    if norm <= S::zero() {
        return None;
    }
    for i in 0..4 {
        world_point[i] /= norm;
    }

    // `if (f0.dot(worldPoint.head<3>()) < 0) worldPoint *= -1` (`ba_base.h:113`).
    let dot: S = f0[0] * world_point[0] + f0[1] * world_point[1] + f0[2] * world_point[2];
    if dot < S::zero() {
        world_point = -world_point;
    }
    Some(world_point)
}

// ─── the Huber-weighted cost of one observation ───────────────────────────

/// The weight and the cost `computeError` adds for one observation
/// (`ba_base.cpp:179-182`), in the C++'s operation order.
///
/// ```text
/// huber_weight = e < huber_thresh ? 1 : huber_thresh / e
/// obs_weight   = huber_weight / (obs_std_dev * obs_std_dev)
/// cost         = 0.5 * (2 - huber_weight) * obs_weight * res^T * res
/// ```
///
/// `e` is `res.norm()` in **raw pixels**: the Huber comparison happens before
/// the `1/sigma` scaling (papers-part2 §13 D14), so the shipped 1.0 px threshold
/// against a 0.5 px sigma is an effective 2 sigma.
///
/// The last line is not `weight * (rx^2 + ry^2)`. C++'s `*` is left-associative,
/// so the scalar factor multiplies `res.transpose()` — a **row expression whose
/// two coefficients are each scaled** — and only then does the 1x2 by 2x1 product
/// contract it against the unscaled column. Reassociating it to scale the dot
/// product instead moves the last bits: at `res = [10.125, 9.625]` with
/// `huber_thresh = 1` and `obs_std_dev = 0.5` in `f32`, C++ returns
/// `53.879341125488281` and the reassociated form `53.879337310791016`. This is
/// the cost the Levenberg-Marquardt accept test compares against its predicted
/// decrease, so the difference is not cosmetic.
#[inline]
pub fn huber_cost<S: LieScalar>(res: &Vector2<S>, e: S, huber_thresh: S, obs_std_dev: S) -> (S, S) {
    let huber_weight: S = if e < huber_thresh {
        S::one()
    } else {
        huber_thresh / e
    };
    let obs_weight: S = huber_weight / (obs_std_dev * obs_std_dev);
    // `Scalar(0.5) * (2 - huber_weight) * obs_weight` folds left into one
    // scalar, which then scales the row.
    let factor: S = c::<S>(0.5) * (c::<S>(2.0) - huber_weight) * obs_weight;
    let cost: S = (factor * res.x) * res.x + (factor * res.y) * res.y;
    (huber_weight, cost)
}

// ─── the sliding-window state ──────────────────────────────────────────────

/// `BundleAdjustmentBase<Scalar>` (`ba_base.h:42-155`): the window the estimator
/// optimises over.
///
/// `frame_poses` holds keyframes as 6-dof pose blocks, `frame_states` the
/// newest few frames as 15-dof pose/velocity/bias blocks, and `lmdb` the
/// landmarks hosted by live keyframes. A frame is in exactly one of the two
/// maps; [`Self::get_pose_state_with_lin`] hides which.
#[derive(Debug, Clone)]
pub struct BundleAdjustmentBase<S: LieScalar> {
    /// Full states, newest frames (`ba_base.h:144`).
    pub frame_states: BTreeMap<FrameId, PoseVelBiasStateWithLin<S>>,
    /// Pose-only blocks, keyframes (`ba_base.h:145`).
    pub frame_poses: BTreeMap<FrameId, PoseStateWithLin<S>>,
    /// The landmark database (`ba_base.h:148`).
    pub lmdb: LandmarkDatabase<S>,
    /// `vio_obs_std_dev`, the pixel noise the residual is whitened by
    /// (`ba_base.h:150`).
    pub obs_std_dev: S,
    /// `vio_obs_huber_thresh`, in **raw pixels** (`ba_base.h:151`).
    pub huber_thresh: S,
    /// The rig calibration (`ba_base.h:153`).
    pub calib: Calibration<S>,
    /// The projection models, resolved once from `calib.intrinsics`.
    ///
    /// C++ carries a `std::variant` per camera and `std::visit`s it at every
    /// call site (`ba_base.cpp:162-188`); the port resolves the parsed
    /// [`crate::calib::CameraModel`] into a [`CameraEnum`] when the window is
    /// built, so a model this stage cannot project with is an error at
    /// construction rather than inside the residual loop.
    cameras: Vec<CameraEnum<S>>,
}

impl<S: LieScalar> BundleAdjustmentBase<S> {
    /// An empty window over one calibration.
    pub fn new(calib: Calibration<S>, obs_std_dev: S, huber_thresh: S) -> Result<Self, BaError> {
        let cameras: Vec<CameraEnum<S>> = calib
            .intrinsics
            .iter()
            .map(CameraEnum::from_model)
            .collect::<Result<_, CameraError>>()?;
        Ok(Self {
            frame_states: BTreeMap::new(),
            frame_poses: BTreeMap::new(),
            lmdb: LandmarkDatabase::new(),
            obs_std_dev,
            huber_thresh,
            calib,
            cameras,
        })
    }

    /// The projection models, in camera order.
    pub fn cameras(&self) -> &[CameraEnum<S>] {
        &self.cameras
    }

    /// The pose block for a timestamp, `getPoseStateWithLin`
    /// (`ba_base.h:131-142`).
    ///
    /// `frame_poses` is searched first; a hit in `frame_states` is **promoted**
    /// to a pose block, carrying the first six entries of its delta and its
    /// `linearized` flag (`imu_types.h:205-212`). C++ prints to `cerr` and
    /// `std::abort()`s when neither map has it; the port returns
    /// [`BaError::UnknownFrame`].
    pub fn get_pose_state_with_lin(&self, t_ns: FrameId) -> Result<PoseStateWithLin<S>, BaError> {
        if let Some(pose) = self.frame_poses.get(&t_ns) {
            return Ok(*pose);
        }
        match self.frame_states.get(&t_ns) {
            Some(state) => Ok(PoseStateWithLin::from_pose_vel_bias(state)),
            None => Err(BaError::UnknownFrame { t_ns }),
        }
    }

    /// The camera-to-IMU transform and the projection model of one image.
    fn camera_of(&self, tcid: TimeCamId) -> Result<(&Se3<S>, &CameraEnum<S>), BaError> {
        let camera_count: usize = self.cameras.len();
        let t_i_c: &Se3<S> = self
            .calib
            .t_i_c
            .get(tcid.cam_id)
            .ok_or(BaError::UnknownCamera {
                cam_id: tcid.cam_id,
                camera_count,
            })?;
        let cam: &CameraEnum<S> = self
            .cameras
            .get(tcid.cam_id)
            .ok_or(BaError::UnknownCamera {
                cam_id: tcid.cam_id,
                camera_count,
            })?;
        Ok((t_i_c, cam))
    }

    /// `T_t_h` for one (host, target) pair (`ba_base.cpp:148-160`).
    ///
    /// The identity when host and target are the same image, which is why a
    /// landmark hosted and observed in the same frame costs nothing.
    fn rel_pose_matrix(&self, tcid_h: TimeCamId, tcid_t: TimeCamId) -> Result<Matrix4<S>, BaError> {
        if tcid_h == tcid_t {
            return Ok(Matrix4::identity());
        }
        let state_h: PoseStateWithLin<S> = self.get_pose_state_with_lin(tcid_h.frame_id)?;
        let state_t: PoseStateWithLin<S> = self.get_pose_state_with_lin(tcid_t.frame_id)?;
        let (t_i_c_h, _) = self.camera_of(tcid_h)?;
        let (t_i_c_t, _) = self.camera_of(tcid_t)?;
        let rel: Se3<S> =
            compute_rel_pose(state_h.pose(), t_i_c_h, state_t.pose(), t_i_c_t, None, None);
        Ok(rel.matrix())
    }

    /// The Huber-weighted reprojection error over the whole window, and how many
    /// observations contributed, `computeError` (`ba_base.cpp:132-204`).
    ///
    /// Per observation (`:172-185`), with `e = |res|` in **raw pixels**:
    ///
    /// ```text
    /// huber_weight = e < huber_thresh ? 1 : huber_thresh / e
    /// obs_weight   = huber_weight / obs_std_dev^2
    /// error       += 0.5 * (2 - huber_weight) * obs_weight * res^T res
    /// ```
    ///
    /// The Huber comparison happens **before** the `1/sigma` scaling
    /// (papers-part2 §13 D14), so the shipped 1.0 px threshold with a 0.5 px
    /// sigma is an effective 2 sigma.
    ///
    /// With `outliers` given, every observation whose `e` exceeds
    /// `outlier_threshold` is recorded as `(target, e)`, and an observation the
    /// camera rejected as `(target, -1)`; both become `-2` when host and target
    /// are the same image, which is `filterOutliers`' signal to delete the whole
    /// landmark (`:176`, `:184`, `:268`).
    ///
    /// The returned count is the number of observations that produced a residual
    /// at all. C++ does not return it; the port does, because the caller
    /// otherwise cannot tell an error of zero from an empty window.
    ///
    /// Sequential in this stage. The body is a fold over `host_frames` in index
    /// order, so a `par_chunks` with a fixed-order merge is a drop-in that does
    /// not change the sum (decision D31). **That is the only reason
    /// `host_frame_error` is a separate function** — one host frame's
    /// partial sum is what a parallel task would own. It is not inlined here so
    /// the shape stays the C++'s (`ba_base.cpp:141-193` is a TBB lambda), and
    /// D65 freezes the parallel path out of this stage.
    pub fn compute_error(
        &self,
        mut outliers: Option<&mut BTreeMap<LandmarkId, Vec<(TimeCamId, S)>>>,
        outlier_threshold: S,
    ) -> Result<(S, usize), BaError> {
        // `host_frames` (`ba_base.cpp:136-137`), sorted here rather than in
        // `unordered_map` order — see the `landmark` module docs.
        let host_frames: Vec<TimeCamId> = self.lmdb.host_kfs();

        if let Some(map) = outliers.as_deref_mut() {
            map.clear();
        }

        let mut error: S = S::zero();
        let mut num_points: usize = 0;
        for &tcid_h in &host_frames {
            let (host_error, host_points) =
                self.host_frame_error(tcid_h, outliers.as_deref_mut(), outlier_threshold)?;
            error += host_error;
            num_points += host_points;
        }
        Ok((error, num_points))
    }

    /// One host frame's contribution to [`Self::compute_error`]: the body of the
    /// TBB lambda (`ba_base.cpp:141-193`), with its own accumulator so the
    /// fold above sees a fixed number of partial sums.
    fn host_frame_error(
        &self,
        tcid_h: TimeCamId,
        mut outliers: Option<&mut BTreeMap<LandmarkId, Vec<(TimeCamId, S)>>>,
        outlier_threshold: S,
    ) -> Result<(S, usize), BaError> {
        let mut local_error: S = S::zero();
        let mut num_points: usize = 0;
        let Some(targets) = self.lmdb.targets_for_host(tcid_h) else {
            return Ok((local_error, num_points));
        };
        for (&tcid_t, ids) in targets {
            let t_t_h: Matrix4<S> = self.rel_pose_matrix(tcid_h, tcid_t)?;
            let (_, cam) = self.camera_of(tcid_t)?;
            // `tcid_h != tcid_t ? e : -2` (`ba_base.cpp:176`, `:184`).
            let same_image: bool = tcid_h == tcid_t;
            for &kpt_id in ids {
                let kpt_pos: &Landmark<S> = self
                    .lmdb
                    .get_landmark(kpt_id)
                    .ok_or(BaError::InconsistentLandmark(kpt_id))?;
                let kpt_obs: &Vector2<S> = kpt_pos
                    .obs
                    .get(&tcid_t)
                    .ok_or(BaError::InconsistentLandmark(kpt_id))?;

                let mut res: Vector2<S> = Vector2::zeros();
                let valid: bool = linearize_point(
                    kpt_obs,
                    kpt_pos,
                    &t_t_h,
                    cam,
                    &mut res,
                    &mut LinearizePointOut::default(),
                );

                if valid {
                    let e: S = res.norm();
                    if let Some(map) = outliers.as_deref_mut()
                        && e > outlier_threshold
                    {
                        let flag: S = if same_image { c::<S>(-2.0) } else { e };
                        map.entry(kpt_id).or_default().push((tcid_t, flag));
                    }
                    let (_, cost) = huber_cost(&res, e, self.huber_thresh, self.obs_std_dev);
                    local_error += cost;
                    num_points += 1;
                } else if let Some(map) = outliers.as_deref_mut() {
                    let flag: S = if same_image { c::<S>(-2.0) } else { -S::one() };
                    map.entry(kpt_id).or_default().push((tcid_t, flag));
                }
            }
        }
        Ok((local_error, num_points))
    }

    /// Where every landmark projects in the newest frame, `computeProjections`
    /// (`ba_base.cpp:329-372`), one list per camera.
    ///
    /// Each entry is `[u, v, inverse depth in the target camera, landmark id]`:
    /// the residual is evaluated against a **zero** observation (`:363`), so the
    /// first two components are the projection itself, and the fourth slot is
    /// overwritten with the id (`:365`). This is the visualisation payload, not
    /// an optimisation quantity.
    pub fn compute_projections(
        &self,
        last_state_t_ns: FrameId,
    ) -> Result<Vec<Vec<Vector4<S>>>, BaError> {
        let mut data: Vec<Vec<Vector4<S>>> = vec![Vec::new(); self.cameras.len()];
        let zero: Vector2<S> = Vector2::zeros();
        for (&tcid_h, targets) in self.lmdb.observations() {
            for (&tcid_t, ids) in targets {
                if tcid_t.frame_id != last_state_t_ns {
                    continue;
                }
                let t_t_h: Matrix4<S> = self.rel_pose_matrix(tcid_h, tcid_t)?;
                let (_, cam) = self.camera_of(tcid_t)?;
                let camera_count: usize = self.cameras.len();
                let out: &mut Vec<Vector4<S>> =
                    data.get_mut(tcid_t.cam_id).ok_or(BaError::UnknownCamera {
                        cam_id: tcid_t.cam_id,
                        camera_count,
                    })?;
                for &kpt_id in ids {
                    let kpt_pos: &Landmark<S> = self
                        .lmdb
                        .get_landmark(kpt_id)
                        .ok_or(BaError::InconsistentLandmark(kpt_id))?;
                    let mut res: Vector2<S> = Vector2::zeros();
                    let mut proj: Vector4<S> = Vector4::zeros();
                    linearize_point(
                        &zero,
                        kpt_pos,
                        &t_t_h,
                        cam,
                        &mut res,
                        &mut LinearizePointOut {
                            proj: Some(&mut proj),
                            ..LinearizePointOut::default()
                        },
                    );
                    proj[3] = c::<S>(kpt_id.0 as f64);
                    out.push(proj);
                }
            }
        }
        Ok(data)
    }

    /// The stacked deltas of the frames in a marginalization ordering,
    /// `computeDelta` (`ba_base.cpp:288-303`).
    ///
    /// Every block must be frozen at its linearization point, or its delta means
    /// nothing; C++ asserts (`:294`, `:297`), the port returns
    /// [`BaError::NotLinearized`].
    pub fn compute_delta(&self, marg_order: &AbsOrderMap) -> Result<DVector<S>, BaError> {
        // Validate every block **before** allocating. `AbsOrderMap::push` accepts
        // any size that does not overflow the total, so an ordering carrying, say,
        // `usize::MAX` would abort the process in the allocator on the line below
        // instead of returning the typed error four lines later (decision D32).
        for (frame_id, _, size) in marg_order.iter() {
            if size != POSE_SIZE && size != POSE_VEL_BIAS_SIZE {
                return Err(BaError::UnexpectedBlockSize { frame_id, size });
            }
        }
        let mut delta: DVector<S> = DVector::zeros(marg_order.total_size());
        for (frame_id, offset, size) in marg_order.iter() {
            match size {
                POSE_SIZE => {
                    let pose: &PoseStateWithLin<S> = self
                        .frame_poses
                        .get(&frame_id)
                        .ok_or(BaError::UnknownFrame { t_ns: frame_id })?;
                    if !pose.is_linearized() {
                        return Err(BaError::NotLinearized { frame_id });
                    }
                    delta.rows_mut(offset, POSE_SIZE).copy_from(pose.delta());
                }
                POSE_VEL_BIAS_SIZE => {
                    let state: &PoseVelBiasStateWithLin<S> = self
                        .frame_states
                        .get(&frame_id)
                        .ok_or(BaError::UnknownFrame { t_ns: frame_id })?;
                    if !state.is_linearized() {
                        return Err(BaError::NotLinearized { frame_id });
                    }
                    delta
                        .rows_mut(offset, POSE_VEL_BIAS_SIZE)
                        .copy_from(state.delta());
                }
                // Unreachable: the validation pass above rejected every other
                // size before a single row was allocated. Kept for exhaustiveness.
                size => return Err(BaError::UnexpectedBlockSize { frame_id, size }),
            }
        }
        Ok(delta)
    }

    /// The marginalization prior's contribution to `H` and `b`, and its current
    /// cost, `linearizeMargPrior` (`ba_base.cpp:374-439`).
    ///
    /// The prior is a quadratic in the drift since its own linearization point,
    /// `P(x) = 0.5 ‖J (delta + x) + r‖²` (`:390-408`), so linearizing it at
    /// `x = 0` gives Jacobian `J` and residual `J delta + r` — that
    /// re-anchoring is trap 8, and its mirror is
    /// `linearization_abs_qr.cpp:592`. The returned error **drops the constant
    /// `0.5 rᵀr` term** (`:416-419`) and can therefore be negative; it is only
    /// ever compared against itself across an increment.
    ///
    /// C++ asserts that the prior's ordering is a prefix of the window's, block
    /// for block (`:379-388`); the port returns
    /// [`crate::linearize::LinearizeError::MargOrderMismatch`] through the
    /// caller.
    /// The prior's own shape: `H` as wide as its ordering and `b` as long as
    /// `H` is tall.
    ///
    /// C++ asserts only the width (`ba_base.cpp:379`, `:444`) and indexes the
    /// rest; a prior with an empty `b` reaches `mld.b[k]` past its end, which
    /// may not be a panic here (decision D32).
    fn check_marg_prior_shape(mld: &MargLinData<S>) -> Result<(), BaError> {
        let marg_size: usize = mld.order.total_size();
        if mld.h.ncols() != marg_size {
            return Err(BaError::MargPriorSize {
                cols: mld.h.ncols(),
                total_size: marg_size,
            });
        }
        if mld.b.nrows() != mld.h.nrows() {
            return Err(BaError::MargPriorSize {
                cols: mld.b.nrows(),
                total_size: mld.h.nrows(),
            });
        }
        Ok(())
    }

    /// The prior's ordering must be the window's prefix, block for block
    /// (`ba_base.cpp:383-388`).
    ///
    /// Public because the square-root export needs it too: it writes the prior
    /// into the first columns of the stacked system
    /// (`linearization_abs_qr.cpp:587-589`) without checking anything, which
    /// would attach one frame's columns to another.
    pub fn check_marg_prior_order(
        &self,
        mld: &MargLinData<S>,
        aom: &AbsOrderMap,
    ) -> Result<(), BaError> {
        Self::check_marg_prior_shape(mld)?;
        let marg_size: usize = mld.order.total_size();
        for (frame_id, offset, size) in mld.order.iter() {
            match aom.get(frame_id) {
                Some((window_offset, window_size))
                    if window_offset == offset && window_size == size && offset < marg_size => {}
                _ => return Err(BaError::MargOrderMismatch { frame_id }),
            }
        }
        Ok(())
    }

    pub fn linearize_marg_prior(
        &self,
        mld: &MargLinData<S>,
        aom: &AbsOrderMap,
        abs_h: &mut DMatrix<S>,
        abs_b: &mut DVector<S>,
    ) -> Result<S, BaError> {
        let marg_size: usize = mld.order.total_size();
        // `:379`, and the shapes C++ indexes without asserting.
        self.check_marg_prior_order(mld, aom)?;
        if abs_h.nrows() < marg_size || abs_h.ncols() < marg_size || abs_b.nrows() < marg_size {
            return Err(BaError::MargPriorSize {
                cols: abs_h.ncols(),
                total_size: marg_size,
            });
        }

        let delta: DVector<S> = self.compute_delta(&mld.order)?;

        // `:427-431`. C++ takes the squared arm at `:433-437` when the prior
        // is not a square root; the port has no such prior (D68).
        let rows: usize = mld.h.nrows();
        // `H_delta = mld.H * delta`, reused by both `b` and the error.
        let mut h_delta: DVector<S> = DVector::zeros(rows);
        for i in 0..rows {
            let mut acc: S = S::zero();
            for j in 0..marg_size {
                acc += mld.h[(i, j)] * delta[j];
            }
            h_delta[i] = acc;
        }
        for i in 0..marg_size {
            for j in 0..marg_size {
                let mut acc: S = S::zero();
                for k in 0..rows {
                    acc += mld.h[(k, i)] * mld.h[(k, j)];
                }
                abs_h[(i, j)] += acc;
            }
            let mut acc: S = S::zero();
            for k in 0..rows {
                acc += mld.h[(k, i)] * (mld.b[k] + h_delta[k]);
            }
            abs_b[i] += acc;
        }
        // `delta^T H^T (0.5 H delta + b)` (`:431`).
        Ok(prior_error(&h_delta, &h_delta, &mld.b, rows))
    }

    /// The prior's cost at the current state, `computeMargPriorError`
    /// (`ba_base.cpp:441-465`).
    ///
    /// The same expression as [`Self::linearize_marg_prior`]'s error, without
    /// touching `H` or `b`; the constant `0.5 rᵀr` is dropped for the same
    /// reason (`:452-455`), so this can be negative.
    pub fn compute_marg_prior_error(&self, mld: &MargLinData<S>) -> Result<S, BaError> {
        let marg_size: usize = mld.order.total_size();
        Self::check_marg_prior_shape(mld)?;
        let delta: DVector<S> = self.compute_delta(&mld.order)?;
        let rows: usize = mld.h.nrows();
        let mut h_delta: DVector<S> = DVector::zeros(rows);
        for i in 0..rows {
            let mut acc: S = S::zero();
            for j in 0..marg_size {
                acc += mld.h[(i, j)] * delta[j];
            }
            h_delta[i] = acc;
        }
        // `:461`; `:463` is the squared arm, which the port has no prior for
        // (D68).
        Ok(prior_error(&h_delta, &h_delta, &mld.b, rows))
    }

    /// The prior's share of the model cost change,
    /// `computeMargPriorModelCostChange` (`ba_base.cpp:467-528`).
    ///
    /// `l_diff = -(J inc)ᵀ (J delta + r + 0.5 (J inc))` (`:519-522`).
    ///
    /// C++ takes a `marg_scaling` vector and multiplies `H` by it where it
    /// meets `inc` but **not** where it meets `delta` (the asymmetry `:503-507`
    /// spells out, because `delta` was never scaled). The port takes no such
    /// vector: only the Jacobian scaling could produce one and nothing scales
    /// (D68).
    pub fn compute_marg_prior_model_cost_change(
        &self,
        mld: &MargLinData<S>,
        marg_pose_inc: &DVector<S>,
    ) -> Result<S, BaError> {
        let marg_size: usize = mld.order.total_size();
        Self::check_marg_prior_shape(mld)?;
        if marg_pose_inc.nrows() != marg_size {
            return Err(BaError::MargPriorSize {
                cols: marg_pose_inc.nrows(),
                total_size: marg_size,
            });
        }
        let delta: DVector<S> = self.compute_delta(&mld.order)?;

        // `:519-522`; `:524` is the squared arm, which the port has no prior
        // for (D68).
        let rows: usize = mld.h.nrows();
        let mut l_diff: S = S::zero();
        for k in 0..rows {
            let mut b_jdelta: S = S::zero();
            let mut j_inc: S = S::zero();
            for j in 0..marg_size {
                b_jdelta += mld.h[(k, j)] * delta[j];
                j_inc += mld.h[(k, j)] * marg_pose_inc[j];
            }
            b_jdelta += mld.b[k];
            l_diff -= j_inc * (b_jdelta + c::<S>(0.5) * j_inc);
        }
        Ok(l_diff)
    }

    /// Save every state and every landmark parameter, `backup`
    /// (`ba_base.h:118-122`).
    pub fn backup(&mut self) {
        for state in self.frame_states.values_mut() {
            state.backup();
        }
        for pose in self.frame_poses.values_mut() {
            pose.backup();
        }
        self.lmdb.backup();
    }

    /// Undo the last increment everywhere, `restore` (`ba_base.h:124-128`).
    ///
    /// This is what a rejected Levenberg-Marquardt step runs.
    pub fn restore(&mut self) {
        for state in self.frame_states.values_mut() {
            state.restore();
        }
        for pose in self.frame_poses.values_mut() {
            pose.restore();
        }
        self.lmdb.restore();
    }
}

/// `lhsᵀ (½ H delta + b)` over the first `n` coefficients: the prior's cost at
/// the current state, which all four prior-error sites compute.
///
/// The `lhs` is what tells the two forms apart. Square-root
/// (`ba_base.cpp:431`, `:461`) writes `deltaᵀ Hᵀ (½ H delta + b)`, and `deltaᵀ
/// Hᵀ` and `H delta` are the same coefficients, so `h_delta` is passed as both
/// — Eigen evaluates the `ColMajor` `gemv` twice and gets the same bits.
/// Hessian form (`:437`, `:463`) writes `deltaᵀ (½ H delta + b)`, so the `lhs`
/// is `delta` itself.
///
/// The outer `(1×n)·(n×1)` is Eigen's `InnerProduct` in all four, which is
/// `(lhs.transpose().cwiseProduct(rhs)).sum()`
/// (`ProductEvaluators.h`, `generic_product_impl<..., InnerProduct>`), so the
/// fold is [`redux_contiguous`]'s packet tree and not a left fold: the two
/// differ in `f32`, and this value enters `error_total` whose difference across
/// an increment is the LM accept test.
fn prior_error<S: LieScalar>(
    lhs: &DVector<S>,
    h_delta: &DVector<S>,
    b: &DVector<S>,
    n: usize,
) -> S {
    redux_contiguous(n, |i| lhs[i] * (c::<S>(0.5) * h_delta[i] + b[i]))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::types::{PoseVelBiasState, TimeCamId};
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;
    use proptest::prelude::*;

    const MSDMI: &str = include_str!("../tests/fixtures/msdmi_calib.json");
    const MSDMG: &str = include_str!("../tests/fixtures/msdmg_calib.json");

    /// `TestConstants<double>` (`basalt-headers/test/include/test_utils.h:10-14`).
    const EPS_F64: f64 = 1e-8;
    const MAX_NORM_F64: f64 = 1e-3;

    fn tcid(frame_id: i64, cam_id: usize) -> TimeCamId {
        TimeCamId::new(frame_id, cam_id)
    }

    fn calib(text: &str) -> Calibration<f64> {
        Calibration::from_json_str(text).unwrap()
    }

    /// A deterministic stand-in for `Sophus::Vector6d::Random()`, which is
    /// uniform on `[-1, 1]`.
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

    fn random_se3(seed: u64, divisor: f64) -> Se3<f64> {
        let values: Vec<f64> = pseudo_random(seed, 6);
        let tangent: Vector6<f64> =
            Vector6::from_iterator(values.into_iter().map(|value| value / divisor));
        Se3::exp(&tangent)
    }

    /// `test_jacobian` (`basalt-headers/test/include/test_utils.h:22-61`):
    /// central differences at `eps`, compared with `isApprox(max_norm)`, i.e.
    /// `|Jn - Ja| <= max_norm * min(|Jn|, |Ja|)`, with an all-zero special case.
    fn test_jacobian<const R: usize, const C: usize>(
        name: &str,
        analytic: &nalgebra::SMatrix<f64, R, C>,
        f: impl Fn(&nalgebra::SVector<f64, C>) -> nalgebra::SVector<f64, R>,
    ) {
        let mut numeric: nalgebra::SMatrix<f64, R, C> = nalgebra::SMatrix::zeros();
        for i in 0..C {
            let mut inc: nalgebra::SVector<f64, C> = nalgebra::SVector::zeros();
            inc[i] = EPS_F64;
            let plus: nalgebra::SVector<f64, R> = f(&inc);
            let minus: nalgebra::SVector<f64, R> = f(&(-inc));
            numeric.set_column(i, &(plus - minus));
        }
        numeric /= 2.0 * EPS_F64;
        assert!(
            analytic.iter().all(|v| v.is_finite()),
            "{name}: Ja not finite"
        );
        assert!(
            numeric.iter().all(|v| v.is_finite()),
            "{name}: Jn not finite"
        );
        let smaller: f64 = numeric.norm().min(analytic.norm());
        let diff: f64 = (numeric - analytic).norm();
        assert!(
            diff <= MAX_NORM_F64 * smaller || (smaller == 0.0 && diff <= MAX_NORM_F64),
            "{name}: Ja != Jn (diff {diff})\nJa {analytic}\nJn {numeric}"
        );
    }

    // ─── the ported C++ tests ──────────────────────────────────────────────

    /// `RelPoseTest` (`test/src/test_vio.cpp:289-330`), with the deterministic
    /// tangents above in place of `Sophus::Vector6d::Random()`.
    #[test]
    fn rel_pose_jacobians_match_finite_differences() {
        let t_w_i_h: Se3<f64> = random_se3(1, 1.0);
        let t_w_i_t: Se3<f64> = random_se3(2, 1.0);
        let t_i_c_h: Se3<f64> = random_se3(3, 10.0);
        let t_i_c_t: Se3<f64> = random_se3(4, 10.0);

        let mut d_rel_d_h: Matrix6<f64> = Matrix6::zeros();
        let mut d_rel_d_t: Matrix6<f64> = Matrix6::zeros();
        let rel: Se3<f64> = compute_rel_pose(
            &t_w_i_h,
            &t_i_c_h,
            &t_w_i_t,
            &t_i_c_t,
            Some(&mut d_rel_d_h),
            Some(&mut d_rel_d_t),
        );

        test_jacobian("d_rel_d_h", &d_rel_d_h, |x| {
            let mut moved: Se3<f64> = t_w_i_h;
            moved.apply_inc(x);
            let new: Se3<f64> = compute_rel_pose(&moved, &t_i_c_h, &t_w_i_t, &t_i_c_t, None, None);
            (new * rel.inverse()).log_decoupled()
        });
        test_jacobian("d_rel_d_t", &d_rel_d_t, |x| {
            let mut moved: Se3<f64> = t_w_i_t;
            moved.apply_inc(x);
            let new: Se3<f64> = compute_rel_pose(&t_w_i_h, &t_i_c_h, &moved, &t_i_c_t, None, None);
            (new * rel.inverse()).log_decoupled()
        });
    }

    /// `LinearizePointsTest` (`test/src/test_vio.cpp:332-398`).
    ///
    /// The C++ test uses `ExtendedUnifiedCamera`, which this port parses but
    /// does not project with (`camera.rs`); the two shipped reference models —
    /// msd-index cam0 (kb4) and msd-g2 cam0 (radtan8) — stand in, so the test
    /// exercises the calibrations the estimator actually runs on. Everything
    /// else follows the C++ line for line: the observation is manufactured by
    /// projecting the landmark, so the residual is zero at the linearization
    /// point, and both Jacobians go to `test_jacobian` at basalt's tolerances.
    #[test]
    fn linearize_point_jacobians_match_finite_differences() {
        for (name, text) in [("kb4 msdmi cam0", MSDMI), ("radtan8 msdmg cam0", MSDMG)] {
            let calibration: Calibration<f64> = calib(text);
            let cam: CameraEnum<f64> = CameraEnum::from_model(&calibration.intrinsics[0]).unwrap();

            // `cam.unproject(Vector2d::Random() * 50, point3d)` (`:338`).
            let mut point3d: Vector4<f64> = Vector4::zeros();
            let centre: [f64; 4] = cam.focal_and_principal_point();
            assert!(cam.unproject(
                &Vector2::new(centre[2] + 30.0, centre[3] - 20.0),
                &mut point3d
            ));

            let kpt_pos: Landmark<f64> = Landmark::new(
                LandmarkId(0),
                tcid(0, 0),
                StereographicParam::project(&point3d),
                0.123_123_1,
            );

            let t_w_h: Se3<f64> = random_se3(11, 100.0);
            let mut t_w_t: Se3<f64> = random_se3(12, 100.0);
            t_w_t.translation[0] += 0.1;

            let t_t_h_se3: Se3<f64> = t_w_t.inverse() * t_w_h;
            let t_t_h: Matrix4<f64> = t_t_h_se3.matrix();

            // The observation is where the landmark actually lands (`:349-356`).
            let mut p_trans: Vector4<f64> = StereographicParam::unproject(&kpt_pos.direction);
            p_trans[3] = kpt_pos.inv_dist;
            p_trans = t_t_h * p_trans;
            let mut kpt_obs: Vector2<f64> = Vector2::zeros();
            assert!(cam.project(&p_trans, &mut kpt_obs), "{name}: observation");

            let mut res: Vector2<f64> = Vector2::zeros();
            let mut d_res_d_xi: Matrix2x6<f64> = Matrix2x6::zeros();
            let mut d_res_d_p: Matrix2x3<f64> = Matrix2x3::zeros();
            assert!(linearize_point(
                &kpt_obs,
                &kpt_pos,
                &t_t_h,
                &cam,
                &mut res,
                &mut LinearizePointOut {
                    d_res_d_xi: Some(&mut d_res_d_xi),
                    d_res_d_p: Some(&mut d_res_d_p),
                    proj: None,
                },
            ));
            assert_abs_diff_eq!(res.norm(), 0.0, epsilon = 1e-9);

            // `d_res_d_xi` is taken against the **coupled** left-multiplied
            // `se3_expd(x) * T_t_h` of the C++ test (`:370`).
            test_jacobian(&format!("{name} d_res_d_xi"), &d_res_d_xi, |x| {
                let moved: Matrix4<f64> = (Se3::exp(x) * t_t_h_se3).matrix();
                let mut res: Vector2<f64> = Vector2::zeros();
                linearize_point(
                    &kpt_obs,
                    &kpt_pos,
                    &moved,
                    &cam,
                    &mut res,
                    &mut LinearizePointOut::default(),
                );
                res
            });

            test_jacobian(&format!("{name} d_res_d_p"), &d_res_d_p, |x| {
                let mut moved: Landmark<f64> = kpt_pos.clone();
                moved.direction += Vector2::new(x[0], x[1]);
                moved.inv_dist += x[2];
                let mut res: Vector2<f64> = Vector2::zeros();
                linearize_point(
                    &kpt_obs,
                    &moved,
                    &t_t_h,
                    &cam,
                    &mut res,
                    &mut LinearizePointOut::default(),
                );
                res
            });
        }
    }

    // ─── stereographic Jacobians ───────────────────────────────────────────

    #[test]
    fn stereographic_jacobians_match_finite_differences() {
        for &(x, y, z) in &[(0.2, -0.3, 1.0), (-1.0, 0.5, 2.0), (0.0, 0.0, 1.0)] {
            let p: Vector4<f64> = Vector4::new(x, y, z, 1.0);
            let mut d_r_d_p: Matrix2x4<f64> = Matrix2x4::zeros();
            StereographicParam::project_with_jacobian(&p, &mut d_r_d_p);
            test_jacobian("project", &d_r_d_p, |inc| {
                StereographicParam::project(&(p + inc))
            });

            let proj: Vector2<f64> = StereographicParam::project(&p);
            let mut d_u_d_p: Matrix4x2<f64> = Matrix4x2::zeros();
            StereographicParam::unproject_with_jacobian(&proj, &mut d_u_d_p);
            test_jacobian("unproject", &d_u_d_p, |inc| {
                StereographicParam::unproject(&(proj + inc))
            });
        }
    }

    // ─── triangulation ─────────────────────────────────────────────────────

    #[test]
    fn triangulate_recovers_a_known_depth() {
        // A stereo pair with a 10 cm baseline along +x, looking down +z.
        let t_0_1: Se3<f64> = Se3::new(So3::identity(), Vector3::new(0.1, 0.0, 0.0));
        for depth in [0.5f64, 1.0, 3.0, 12.0] {
            let point0: Vector3<f64> = Vector3::new(0.2, -0.1, depth);
            let point1: Vector3<f64> = point0 - Vector3::new(0.1, 0.0, 0.0);
            let f0: Vector3<f64> = point0.normalize();
            let f1: Vector3<f64> = point1.normalize();
            let result: Vector4<f64> = triangulate(&f0, &f1, &t_0_1).unwrap();
            assert_abs_diff_eq!(result.fixed_rows::<3>(0).norm(), 1.0, epsilon = 1e-12);
            // The homogeneous point is `[unit direction, 1/|point|]`.
            let recovered: Vector3<f64> = result.fixed_rows::<3>(0) / result[3];
            assert_abs_diff_eq!(recovered, point0, epsilon = 1e-9);
            assert!(result[3] > 0.0);
        }
    }

    #[test]
    fn triangulate_at_infinity_has_zero_inverse_distance() {
        let t_0_1: Se3<f64> = Se3::new(So3::identity(), Vector3::new(0.1, 0.0, 0.0));
        let f0: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
        let f1: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
        let result: Vector4<f64> = triangulate(&f0, &f1, &t_0_1).unwrap();
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-12);
    }

    /// A refusal is `None`, not four NaNs: Eigen leaves `m_matrixV`
    /// uninitialized on `InvalidInput` and basalt reads it, so there is no
    /// value to reproduce here — only a landmark that does not exist.
    #[test]
    fn a_non_finite_input_is_rejected_instead_of_read_uninitialized() {
        let t_0_1: Se3<f64> = Se3::new(So3::identity(), Vector3::new(0.1, 0.0, 0.0));
        assert_eq!(
            triangulate(
                &Vector3::new(f64::NAN, 0.0, 1.0),
                &Vector3::new(0.0, 0.0, 1.0),
                &t_0_1,
            ),
            None
        );
        // Two zero bearing vectors build a zero `A`: rank zero, so no direction
        // is more null than any other and there is no landmark to report.
        assert_eq!(
            triangulate::<f64>(&Vector3::zeros(), &Vector3::zeros(), &Se3::identity()),
            None
        );
    }

    // ─── the window ────────────────────────────────────────────────────────

    /// Two frames, a stereo rig, and `n` landmarks placed from known 3-D points.
    ///
    /// Both frames are full states, so `get_pose_state_with_lin` has to promote
    /// them; the landmark is hosted in (frame 0, cam 0) and observed in all four
    /// images, with the observation taken from the calibration's own `project`.
    fn a_window(text: &str, points: &[Vector3<f64>], noise: f64) -> BundleAdjustmentBase<f64> {
        let calibration: Calibration<f64> = calib(text);
        let mut ba: BundleAdjustmentBase<f64> =
            BundleAdjustmentBase::new(calibration, 0.5, 1.0).unwrap();

        let pose0: Se3<f64> = Se3::identity();
        let pose1: Se3<f64> = Se3::new(
            So3::exp(&Vector3::new(0.0, 0.02, 0.0)),
            Vector3::new(0.05, 0.01, 0.02),
        );
        for (t_ns, pose) in [(0i64, pose0), (1i64, pose1)] {
            ba.frame_states.insert(
                t_ns,
                PoseVelBiasStateWithLin::new(
                    PoseVelBiasState::new(
                        t_ns,
                        pose,
                        Vector3::zeros(),
                        Vector3::zeros(),
                        Vector3::zeros(),
                    ),
                    false,
                ),
            );
        }

        let host: TimeCamId = tcid(0, 0);
        let cam_count: usize = ba.cameras().len();
        for (i, point_host) in points.iter().enumerate() {
            let id: LandmarkId = LandmarkId(i as u64);
            let mut p4: Vector4<f64> =
                Vector4::new(point_host[0], point_host[1], point_host[2], 1.0);
            p4 /= point_host.norm();
            let direction: Vector2<f64> = StereographicParam::project(&p4);
            let inv_dist: f64 = 1.0 / point_host.norm();
            ba.lmdb
                .add_landmark(id, &Landmark::new(id, host, direction, inv_dist));

            for frame in [0i64, 1] {
                for cam_id in 0..cam_count {
                    let target: TimeCamId = tcid(frame, cam_id);
                    let t_t_h: Matrix4<f64> = ba.rel_pose_matrix(host, target).unwrap();
                    let mut q: Vector4<f64> = StereographicParam::unproject(&direction);
                    q[3] = inv_dist;
                    let in_target: Vector4<f64> = t_t_h * q;
                    let mut pixel: Vector2<f64> = Vector2::zeros();
                    if !ba.cameras()[cam_id].project(&in_target, &mut pixel) {
                        continue;
                    }
                    let jitter: f64 = noise * ((i + cam_id) as f64).sin();
                    ba.lmdb
                        .add_observation(target, id, pixel + Vector2::new(jitter, -jitter))
                        .unwrap();
                }
            }
        }
        ba
    }

    fn synthetic_points() -> Vec<Vector3<f64>> {
        (0..6)
            .map(|i| {
                let f: f64 = f64::from(i);
                Vector3::new(0.2 * (f - 2.5), 0.15 * (f - 3.0), 1.5 + 0.4 * f)
            })
            .collect()
    }

    #[test]
    fn compute_error_is_zero_without_noise() {
        for text in [MSDMI, MSDMG] {
            let ba: BundleAdjustmentBase<f64> = a_window(text, &synthetic_points(), 0.0);
            let (error, num_points) = ba.compute_error(None, 0.0).unwrap();
            assert!(num_points > 0);
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-18);
        }
    }

    /// `compute_error` against a sum built from the landmarks directly, at a
    /// noise level below the Huber threshold and at one well above it.
    ///
    /// The second level is the point: with the 0.4 px jitter this test used to
    /// carry, every residual norm was at most `0.4 sqrt(2) = 0.57` px against a
    /// 1.0 px threshold, so the downweighting branch never ran and replacing it
    /// with a constant weight of 1 still passed. The assertions below count the
    /// downweighted observations and require them.
    #[test]
    fn compute_error_equals_an_independent_huber_sum() {
        for (noise, want_downweighted) in [(0.2f64, false), (4.0f64, true)] {
            let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), noise);
            let (error, num_points) = ba.compute_error(None, 0.0).unwrap();

            // Recomputed from the landmarks, without touching `compute_error`'s
            // machinery: project each landmark into each target and apply
            // `0.5 * (2 - w) * (w / sigma^2) * |r|^2` with `w` the raw-pixel
            // Huber weight (`ba_base.cpp:179-182`).
            let mut want: f64 = 0.0;
            let mut count: usize = 0;
            let mut downweighted: usize = 0;
            let mut largest: f64 = 0.0;
            for lm in ba.lmdb.landmarks() {
                for (&target, observed) in &lm.obs {
                    let t_t_h: Matrix4<f64> = ba.rel_pose_matrix(lm.host_kf_id, target).unwrap();
                    let mut q: Vector4<f64> = StereographicParam::unproject(&lm.direction);
                    q[3] = lm.inv_dist;
                    let mut pixel: Vector2<f64> = Vector2::zeros();
                    assert!(ba.cameras()[target.cam_id].project(&(t_t_h * q), &mut pixel));
                    let r: Vector2<f64> = pixel - observed;
                    let e: f64 = r.norm();
                    largest = largest.max(e);
                    let w: f64 = if e < ba.huber_thresh {
                        1.0
                    } else {
                        downweighted += 1;
                        ba.huber_thresh / e
                    };
                    want += 0.5 * (2.0 - w) * (w / (ba.obs_std_dev * ba.obs_std_dev)) * r.dot(&r);
                    count += 1;
                }
            }
            assert_eq!(num_points, count);
            assert_abs_diff_eq!(error, want, epsilon = 1e-12);
            assert!(want > 0.0);
            assert_eq!(
                downweighted > 0,
                want_downweighted,
                "noise {noise}: {downweighted} of {count} downweighted, largest residual {largest} px"
            );
            if want_downweighted {
                // A meaningful share of them, not one straggler.
                assert!(downweighted * 4 >= count, "only {downweighted} of {count}");
            }
        }
    }

    /// The independent sum above shares the port's `res.norm()`; this one does
    /// not share anything at all with the production Huber branch. Forcing the
    /// weight to 1 changes the total, which is what the old test failed to catch.
    #[test]
    fn the_huber_branch_changes_the_total() {
        let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 4.0);
        let (error, _) = ba.compute_error(None, 0.0).unwrap();
        let mut unweighted: f64 = 0.0;
        for lm in ba.lmdb.landmarks() {
            for (&target, observed) in &lm.obs {
                let t_t_h: Matrix4<f64> = ba.rel_pose_matrix(lm.host_kf_id, target).unwrap();
                let mut q: Vector4<f64> = StereographicParam::unproject(&lm.direction);
                q[3] = lm.inv_dist;
                let mut pixel: Vector2<f64> = Vector2::zeros();
                assert!(ba.cameras()[target.cam_id].project(&(t_t_h * q), &mut pixel));
                let r: Vector2<f64> = pixel - observed;
                // Weight 1 everywhere: the plain squared error, whitened.
                unweighted += 0.5 * r.dot(&r) / (ba.obs_std_dev * ba.obs_std_dev);
            }
        }
        // Huber is a *down*weighting, so the real cost is the smaller one, and
        // by a wide margin at this noise level.
        assert!(error < unweighted * 0.75, "{error} vs {unweighted}");
    }

    #[test]
    fn outliers_are_collected_above_the_threshold() {
        let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.4);
        let mut outliers: BTreeMap<LandmarkId, Vec<(TimeCamId, f64)>> = BTreeMap::new();
        let (_, _) = ba.compute_error(Some(&mut outliers), 0.1).unwrap();
        assert!(!outliers.is_empty());
        for entries in outliers.values() {
            for &(target, flag) in entries {
                // Host and target are the same image only for (frame 0, cam 0).
                if target == tcid(0, 0) {
                    assert_eq!(flag, -2.0);
                } else {
                    assert!(flag > 0.1);
                }
            }
        }
        // A threshold above every residual collects nothing.
        let mut none: BTreeMap<LandmarkId, Vec<(TimeCamId, f64)>> = BTreeMap::new();
        ba.compute_error(Some(&mut none), 1e6).unwrap();
        assert!(none.values().all(|v| v.iter().all(|&(_, f)| f == -2.0)));
    }

    #[test]
    fn compute_projections_reports_the_newest_frame_only() {
        let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        let data: Vec<Vec<Vector4<f64>>> = ba.compute_projections(1).unwrap();
        assert_eq!(data.len(), ba.cameras().len());
        assert!(data.iter().all(|per_cam| !per_cam.is_empty()));
        for per_cam in &data {
            for entry in per_cam {
                assert!(entry[2] > 0.0, "inverse depth in front of the camera");
                assert!(entry[3] >= 0.0, "the fourth slot carries the landmark id");
            }
        }
        // A frame with no observations produces nothing at all.
        let empty: Vec<Vec<Vector4<f64>>> = ba.compute_projections(99).unwrap();
        assert!(empty.iter().all(Vec::is_empty));
    }

    #[test]
    fn an_unknown_frame_is_an_error_not_an_abort() {
        let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        assert_eq!(
            ba.get_pose_state_with_lin(42),
            Err(BaError::UnknownFrame { t_ns: 42 })
        );
        assert!(ba.get_pose_state_with_lin(0).is_ok());
    }

    #[test]
    fn get_pose_state_with_lin_prefers_frame_poses() {
        let mut ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        let marker: Se3<f64> = Se3::new(So3::identity(), Vector3::new(7.0, 8.0, 9.0));
        ba.frame_poses
            .insert(0, PoseStateWithLin::new(0, marker, false));
        let got: PoseStateWithLin<f64> = ba.get_pose_state_with_lin(0).unwrap();
        assert_eq!(got.pose().translation, marker.translation);
    }

    #[test]
    fn backup_and_restore_undo_a_whole_step() {
        let mut ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.2);
        let (before, _) = ba.compute_error(None, 0.0).unwrap();
        ba.backup();
        for state in ba.frame_states.values_mut() {
            state.apply_inc(&crate::types::Vector15::from_element(0.01));
        }
        for lm_id in ba
            .lmdb
            .landmarks()
            .iter()
            .map(|lm| lm.id)
            .collect::<Vec<_>>()
        {
            if let Some(lm) = ba.lmdb.get_landmark_mut(lm_id) {
                lm.direction += Vector2::new(0.05, -0.05);
                lm.inv_dist += 0.01;
            }
        }
        let (moved, _) = ba.compute_error(None, 0.0).unwrap();
        assert!((moved - before).abs() > 1e-9);
        ba.restore();
        let (after, _) = ba.compute_error(None, 0.0).unwrap();
        assert_abs_diff_eq!(after, before, epsilon = 1e-18);
    }

    #[test]
    fn compute_delta_stacks_the_frozen_deltas() {
        let mut ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        let pose: Se3<f64> = Se3::identity();
        let mut block: PoseStateWithLin<f64> = PoseStateWithLin::new(5, pose, false);
        block.set_linearized().unwrap();
        block.apply_inc(&Vector6::from_element(0.1));
        ba.frame_poses.insert(5, block);
        for state in ba.frame_states.values_mut() {
            state.set_linearized().unwrap();
            state.apply_inc(&crate::types::Vector15::from_element(0.2));
        }

        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(5, POSE_SIZE).unwrap();
        order.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        let delta: DVector<f64> = ba.compute_delta(&order).unwrap();
        assert_eq!(delta.len(), POSE_SIZE + POSE_VEL_BIAS_SIZE);
        assert!(delta.rows(0, POSE_SIZE).iter().all(|v| *v == 0.1));
        assert!(
            delta
                .rows(POSE_SIZE, POSE_VEL_BIAS_SIZE)
                .iter()
                .all(|v| *v == 0.2)
        );
    }

    /// `AbsOrderMap::push` accepts any size that does not overflow the total, so
    /// a corrupt ordering can name a block of `usize::MAX`. `compute_delta` must
    /// return the typed error, not abort the process in the allocator: a panic
    /// here happens on a rayon worker inside a released GIL (decision D32).
    #[test]
    fn compute_delta_validates_before_it_allocates() {
        let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        let mut huge: AbsOrderMap = AbsOrderMap::new();
        huge.push(0, usize::MAX).unwrap();
        assert_eq!(
            ba.compute_delta(&huge),
            Err(BaError::UnexpectedBlockSize {
                frame_id: 0,
                size: usize::MAX
            })
        );
        // And a bad block behind a good one is caught before the good one is
        // allocated for, too.
        let mut mixed: AbsOrderMap = AbsOrderMap::new();
        mixed.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        mixed.push(7, usize::MAX - POSE_VEL_BIAS_SIZE).unwrap();
        assert_eq!(
            ba.compute_delta(&mixed),
            Err(BaError::UnexpectedBlockSize {
                frame_id: 7,
                size: usize::MAX - POSE_VEL_BIAS_SIZE
            })
        );
    }

    /// The Huber cost by hand, on residuals chosen so every intermediate is
    /// exactly representable in both precisions — no fixture, no port
    /// arithmetic reused.
    ///
    /// Above the threshold: `res = [3, 4]`, `e = 5`, `huber_thresh = 2.5`,
    /// `obs_std_dev = 0.5`, so `hw = 0.5`, `ow = 0.5 / 0.25 = 2`,
    /// `factor = 0.5 * 1.5 * 2 = 1.5` and `cost = 1.5*9 + 1.5*16 = 37.5`.
    ///
    /// Below it: `res = [0.5, 0.5]`, `e = sqrt(0.5) < 1`, so `hw = 1`,
    /// `ow = 4`, `factor = 2` and `cost = 2 * 0.5 = 1`.
    #[test]
    fn the_huber_cost_matches_a_hand_computation() {
        let above: Vector2<f64> = Vector2::new(3.0, 4.0);
        let (weight, cost) = huber_cost(&above, above.norm(), 2.5, 0.5);
        assert_eq!(weight, 0.5);
        assert_eq!(cost, 37.5);

        let above32: Vector2<f32> = Vector2::new(3.0, 4.0);
        let (weight32, cost32) = huber_cost(&above32, above32.norm(), 2.5, 0.5);
        assert_eq!(weight32, 0.5);
        assert_eq!(cost32, 37.5);

        let below: Vector2<f64> = Vector2::new(0.5, 0.5);
        let (weight, cost) = huber_cost(&below, below.norm(), 1.0, 0.5);
        assert_eq!(weight, 1.0);
        assert_eq!(cost, 1.0);

        let below32: Vector2<f32> = Vector2::new(0.5, 0.5);
        let (weight32, cost32) = huber_cost(&below32, below32.norm(), 1.0, 0.5);
        assert_eq!(weight32, 1.0);
        assert_eq!(cost32, 1.0);

        // The threshold is on the raw pixel norm, before `1/sigma`
        // (papers-part2 §13 D14): a residual of exactly the threshold is *not*
        // downweighted only because the comparison is strict `<`.
        let at: Vector2<f64> = Vector2::new(1.0, 0.0);
        assert_eq!(huber_cost(&at, at.norm(), 1.0, 0.5).0, 1.0);
        let just_over: Vector2<f64> = Vector2::new(1.0 + f64::EPSILON, 0.0);
        assert!(huber_cost(&just_over, just_over.norm(), 1.0, 0.5).0 < 1.0);
    }

    /// The three marginalization-prior helpers agree with each other and with
    /// the algebra written out in `ba_base.cpp:390-419`.
    ///
    /// The prior is a quadratic in the drift since its own linearization point,
    /// so all three have to use the same `delta`, and each squares `J_m` on its
    /// own way to a Hessian.
    #[test]
    fn the_marginalization_prior_helpers_agree() {
        let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };

        // One frozen pose block, drifted from its linearization point.
        let mut estimator: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        estimator.frame_states.clear();
        estimator.frame_poses.clear();
        let delta_applied: Vector6<f64> = Vector6::from_iterator((0..6).map(|_| next() / 100.0));
        let mut pose: PoseStateWithLin<f64> = PoseStateWithLin::new(0, Se3::identity(), false);
        pose.set_linearized().unwrap();
        pose.apply_inc(&delta_applied);
        estimator.frame_poses.insert(0, pose);

        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(0, POSE_SIZE).unwrap();
        let mut aom: AbsOrderMap = AbsOrderMap::new();
        aom.push(0, POSE_SIZE).unwrap();

        // A rectangular `J`: the prior does not have to be square.
        let rows: usize = 4;
        let mut j: DMatrix<f64> = DMatrix::zeros(rows, POSE_SIZE);
        for r in 0..rows {
            for c in 0..POSE_SIZE {
                j[(r, c)] = next();
            }
        }
        let r_vec: DVector<f64> = DVector::from_iterator(rows, (0..rows).map(|_| next()));
        let mld: MargLinData<f64> = MargLinData {
            order: order.clone(),
            h: j.clone(),
            b: r_vec.clone(),
        };

        let delta: DVector<f64> = estimator.compute_delta(&order).unwrap();
        assert_eq!(delta.as_slice(), delta_applied.as_slice());

        let mut h: DMatrix<f64> = DMatrix::zeros(POSE_SIZE, POSE_SIZE);
        let mut b: DVector<f64> = DVector::zeros(POSE_SIZE);
        let error: f64 = estimator
            .linearize_marg_prior(&mld, &aom, &mut h, &mut b)
            .unwrap();

        // `:427-431`.
        let want_h: DMatrix<f64> = j.transpose() * &j;
        let want_b: DVector<f64> = j.transpose() * (&r_vec + &j * &delta);
        let j_delta: DVector<f64> = &j * &delta;
        let want_error: f64 = (j_delta.transpose() * (0.5 * &j_delta + &r_vec))[(0, 0)];
        assert_abs_diff_eq!(h, want_h, epsilon = 1e-12 * want_h.norm());
        assert_abs_diff_eq!(b, want_b, epsilon = 1e-12 * want_b.norm().max(1.0));
        assert_abs_diff_eq!(
            error,
            want_error,
            epsilon = 1e-12 * want_error.abs().max(1.0)
        );

        // `computeMargPriorError` is the same number without touching H or b.
        let error_only: f64 = estimator.compute_marg_prior_error(&mld).unwrap();
        assert_eq!(error_only, error);

        // `computeMargPriorModelCostChange` (`:519-522`).
        let inc: DVector<f64> =
            DVector::from_iterator(POSE_SIZE, (0..POSE_SIZE).map(|_| next() / 10.0));
        let l_diff: f64 = estimator
            .compute_marg_prior_model_cost_change(&mld, &inc)
            .unwrap();
        let j_inc: DVector<f64> = &j * &inc;
        let want_l_diff: f64 = -(j_inc.transpose() * (&j_delta + &r_vec + 0.5 * &j_inc))[(0, 0)];
        assert_abs_diff_eq!(
            l_diff,
            want_l_diff,
            epsilon = 1e-12 * want_l_diff.abs().max(1.0)
        );

        // An ordering the window disagrees with is rejected (`:383-388`).
        let mut wrong: AbsOrderMap = AbsOrderMap::new();
        wrong.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        let mut h3: DMatrix<f64> = DMatrix::zeros(POSE_SIZE, POSE_SIZE);
        let mut b3: DVector<f64> = DVector::zeros(POSE_SIZE);
        assert_eq!(
            estimator
                .linearize_marg_prior(&mld, &wrong, &mut h3, &mut b3)
                .unwrap_err(),
            BaError::MargOrderMismatch { frame_id: 0 }
        );

        // And a prior whose matrix does not match its own ordering (`:379`).
        let ragged: MargLinData<f64> = MargLinData {
            order: order.clone(),
            h: DMatrix::zeros(rows, POSE_SIZE - 1),
            b: r_vec,
        };
        assert!(matches!(
            estimator.compute_marg_prior_error(&ragged).unwrap_err(),
            BaError::MargPriorSize { .. }
        ));

        // A shape C++ indexes without asserting, which would be a panic here
        // (decision D32): a residual that is not as long as `H` is tall.
        let empty_b: MargLinData<f64> = MargLinData {
            order: order.clone(),
            h: j,
            b: DVector::zeros(0),
        };
        for outcome in [
            estimator.compute_marg_prior_error(&empty_b),
            estimator.linearize_marg_prior(
                &empty_b,
                &aom,
                &mut DMatrix::zeros(POSE_SIZE, POSE_SIZE),
                &mut DVector::zeros(POSE_SIZE),
            ),
            estimator.compute_marg_prior_model_cost_change(&empty_b, &inc),
        ] {
            assert!(matches!(outcome, Err(BaError::MargPriorSize { .. })));
        }
    }

    #[test]
    fn compute_delta_refuses_an_unlinearized_block() {
        let ba: BundleAdjustmentBase<f64> = a_window(MSDMI, &synthetic_points(), 0.0);
        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        assert_eq!(
            ba.compute_delta(&order),
            Err(BaError::NotLinearized { frame_id: 0 })
        );
        let mut wrong: AbsOrderMap = AbsOrderMap::new();
        wrong.push(0, 9).unwrap();
        assert_eq!(
            ba.compute_delta(&wrong),
            Err(BaError::UnexpectedBlockSize {
                frame_id: 0,
                size: 9
            })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The stereographic Jacobians are the derivative of the port's own
        /// `project`/`unproject`, in both precisions.
        #[test]
        fn stereographic_jacobians_are_the_derivative(
            x in -2.0f64..2.0,
            y in -2.0f64..2.0,
            z in 0.2f64..3.0,
        ) {
            let p: Vector4<f64> = Vector4::new(x, y, z, 1.0);
            let mut d_r_d_p: Matrix2x4<f64> = Matrix2x4::zeros();
            StereographicParam::project_with_jacobian(&p, &mut d_r_d_p);
            for i in 0..4 {
                let mut inc: Vector4<f64> = Vector4::zeros();
                inc[i] = 1e-7;
                let numeric: Vector2<f64> = (StereographicParam::project(&(p + inc))
                    - StereographicParam::project(&(p - inc)))
                    / 2e-7;
                prop_assert!((numeric - d_r_d_p.column(i)).norm() <= 1e-5 * (1.0 + d_r_d_p.column(i).norm()));
            }

            let proj: Vector2<f64> = StereographicParam::project(&p);
            let mut d_u_d_p: Matrix4x2<f64> = Matrix4x2::zeros();
            StereographicParam::unproject_with_jacobian(&proj, &mut d_u_d_p);
            for i in 0..2 {
                let mut inc: Vector2<f64> = Vector2::zeros();
                inc[i] = 1e-7;
                let numeric: Vector4<f64> = (StereographicParam::unproject(&(proj + inc))
                    - StereographicParam::unproject(&(proj - inc)))
                    / 2e-7;
                prop_assert!((numeric - d_u_d_p.column(i)).norm() <= 1e-5 * (1.0 + d_u_d_p.column(i).norm()));
            }

            // The same in f32, at f32 tolerances (`TestConstants<float>`:
            // eps 1e-2, max_norm 1e-2).
            let p32: Vector4<f32> = Vector4::new(x as f32, y as f32, z as f32, 1.0);
            let mut d32: Matrix2x4<f32> = Matrix2x4::zeros();
            StereographicParam::project_with_jacobian(&p32, &mut d32);
            for i in 0..3 {
                let mut inc: Vector4<f32> = Vector4::zeros();
                inc[i] = 1e-2;
                let numeric: Vector2<f32> = (StereographicParam::project(&(p32 + inc))
                    - StereographicParam::project(&(p32 - inc)))
                    / 2e-2;
                prop_assert!((numeric - d32.column(i)).norm() <= 1e-2 * (1.0 + d32.column(i).norm()));
            }
        }

        /// Triangulation recovers the depth of a point seen by two cameras with
        /// a real baseline, whatever the rotation between them.
        #[test]
        fn triangulate_recovers_synthetic_depths(
            depth in 0.4f64..20.0,
            u in -0.6f64..0.6,
            v in -0.6f64..0.6,
            baseline in 0.05f64..0.5,
            yaw in -0.3f64..0.3,
        ) {
            let t_0_1: Se3<f64> = Se3::new(
                So3::exp(&Vector3::new(0.0, yaw, 0.0)),
                Vector3::new(baseline, 0.02, -0.01),
            );
            let point0: Vector3<f64> = Vector3::new(u * depth, v * depth, depth);
            let point1: Vector3<f64> = t_0_1.inverse() * point0;
            let f0: Vector3<f64> = point0.normalize();
            let f1: Vector3<f64> = point1.normalize();
            let Some(result): Option<Vector4<f64>> = triangulate(&f0, &f1, &t_0_1) else {
                return Err(TestCaseError::fail("the DLT refused a well-conditioned pair"));
            };
            prop_assert!(result[3] > 0.0);
            let recovered: Vector3<f64> = result.fixed_rows::<3>(0) / result[3];
            prop_assert!((recovered - point0).norm() <= 1e-7 * point0.norm());
        }
    }
}
