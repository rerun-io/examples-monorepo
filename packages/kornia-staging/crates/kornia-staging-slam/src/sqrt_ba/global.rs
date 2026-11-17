//! Global square-root bundle adjustment with pose-graph rows (rootba `bal`).
//!
//! Variables: one body pose `T_i_w` per keyframe (decoupled left increment,
//! [`super::apply_left_increment`]) and one world point per landmark. Cameras are
//! fixed: intrinsics and `T_i_c`. Each landmark is a [`LandmarkQr`] block over
//! the poses that observe it: rows are Huber-weighted (`sqrt(w)`) reprojection
//! residuals, landmark columns are Jacobi-scaled, then Householder QR eliminates
//! the landmark. Relative-pose rows ([`super::RelPoseConstraint`]) join the reduced
//! camera system unchanged. LM damping uses [`super::LandmarkDamping`] (bit-exact undo)
//! and `lambda` on the scaled pose columns; the step comes from [`super::pcg`].
//!
//! Speed: the implicit matvec is landmark-parallel into per-block
//! contributions and pose-major in the reduction (a fixed CSR order per pose),
//! so one iteration costs O(nonzeros) and the result does not depend on the
//! thread count. rootba's default reduction zeroes a full pose vector per split
//! (O(landmarks x keyframes) per iteration; phase-1 profile).
//!
//! f32 safety (DESIGN B24). As in rootba: Jacobi scaling, and PCG residual
//! replacement (rootba every 10 iterations, ours every 8). Ours: the
//! block-Jacobi preconditioner factored in f64, a `lambda` floor, the PCG dots,
//! every cost and the model decrease summed in f64 (the per-block matvec stays
//! in `S`, with a fixed 8-lane order), non-finite rows dropped and counted, and
//! the LM/PCG counts logged in [`GbaReport`]. [`GbaProblem::centred`] moves the
//! world origin to the landmark centroid in f64 before the cast.
use std::time::Instant;

use kornia_staging_3d::camera::CameraModelKind;
use kornia_staging_algebra::{
    lie::{RigidTransform, Rotation3},
    Scalar,
};
use nalgebra::{Matrix2x3, Matrix2x6, Matrix6, Vector2, Vector3, Vector6};
use rayon::prelude::*;

use super::{
    apply_left_increment, pcg, relative_pose_error, BlockJacobi, LandmarkDamping, LandmarkQr,
    PcgOptions, PcgSummary, PcgTermination, RelPoseConstraint, SqrtBaError,
};

/// A fixed camera model that projects camera-frame points.
pub trait GbaCamera<S: Scalar>: Sync {
    /// Pixel, validity (in front of the camera and inside the model's domain)
    /// and, when asked, the row-major 2x3 point Jacobian.
    fn project(&self, point: [S; 3], jacobian: Option<&mut [[S; 3]; 2]>) -> ([S; 2], bool);
}

impl<S: Scalar> GbaCamera<S> for CameraModelKind<S> {
    fn project(&self, point: [S; 3], jacobian: Option<&mut [[S; 3]; 2]>) -> ([S; 2], bool) {
        let (pixel, status) = self.project_with_status(point, jacobian);
        (pixel, status.is_ok())
    }
}

/// One camera of the rig: model and extrinsics, both fixed in the GBA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GbaRigCamera<S: Scalar, C> {
    /// Intrinsic model.
    pub model: C,
    /// `T_i_c`.
    pub imu_from_camera: RigidTransform<S>,
}

/// One pixel observation of a landmark.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GbaObservation<S: Scalar> {
    /// Keyframe index.
    pub keyframe: usize,
    /// Rig camera index.
    pub camera: usize,
    /// Observed pixel.
    pub pixel: [S; 2],
}

/// A world point and its observations.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaLandmark<S: Scalar> {
    /// World position.
    pub position: Vector3<S>,
    /// At least two, each (keyframe, camera) at most once.
    pub observations: Vec<GbaObservation<S>>,
}

/// Keyframe poses, landmarks and pose-graph rows of one global BA.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaProblem<S: Scalar> {
    /// `T_i_w` per keyframe.
    pub imu_from_world: Vec<RigidTransform<S>>,
    /// Landmarks with their observations.
    pub landmarks: Vec<GbaLandmark<S>>,
    /// Relative-pose rows (rootba's residual form).
    pub rel_poses: Vec<RelPoseConstraint<S>>,
}

impl<S: Scalar> GbaProblem<S> {
    /// DESIGN §3.1 filters. A keyframe with fewer than `min_obs_per_kf`
    /// observations loses its reprojection rows but keeps its pose-graph rows
    /// (B22: rootba drops the keyframe and its edges). Then every landmark with
    /// fewer than two observations goes. Returns (observations, landmarks) removed.
    pub fn drop_sparse_observations(&mut self, min_obs_per_kf: usize) -> (usize, usize) {
        let mut per_kf = vec![0usize; self.imu_from_world.len()];
        for obs in self.landmarks.iter().flat_map(|l| &l.observations) {
            if let Some(n) = per_kf.get_mut(obs.keyframe) {
                *n += 1;
            }
        }
        let mut dropped_obs = 0;
        for landmark in &mut self.landmarks {
            let before = landmark.observations.len();
            landmark
                .observations
                .retain(|o| per_kf.get(o.keyframe).is_some_and(|&n| n >= min_obs_per_kf));
            dropped_obs += before - landmark.observations.len();
        }
        let before = self.landmarks.len();
        self.landmarks
            .retain(|l| l.observations.len() >= 2 && l.position.iter().all(|v| v.is_finite()));
        (dropped_obs, before - self.landmarks.len())
    }

    /// The problem in another scalar.
    pub fn cast<T: Scalar>(&self) -> GbaProblem<T> {
        GbaProblem {
            imu_from_world: self.imu_from_world.iter().map(|p| p.cast()).collect(),
            landmarks: self
                .landmarks
                .iter()
                .map(|l| GbaLandmark {
                    position: l.position.map(|v| T::from_literal(v.to_f64())),
                    observations: l
                        .observations
                        .iter()
                        .map(|o| GbaObservation {
                            keyframe: o.keyframe,
                            camera: o.camera,
                            pixel: o.pixel.map(|v| T::from_literal(v.to_f64())),
                        })
                        .collect(),
                })
                .collect(),
            rel_poses: self.rel_poses.iter().map(|c| c.cast()).collect(),
        }
    }
}

impl GbaProblem<f64> {
    /// Move the world origin to the landmark centroid (in f64), then cast.
    /// Relative-pose measurements do not change. Returns the centroid for
    /// [`Self::uncentre_from`].
    pub fn centred<T: Scalar>(&self) -> (GbaProblem<T>, Vector3<f64>) {
        let n = self.landmarks.len().max(1) as f64;
        let centroid: Vector3<f64> = self
            .landmarks
            .iter()
            .map(|l| l.position)
            .sum::<Vector3<f64>>()
            / n;
        let mut centred = self.clone();
        for pose in &mut centred.imu_from_world {
            // p_i = R (p_w' + c) + t: the new translation is t + R c.
            pose.translation += pose.rotation * centroid;
        }
        for landmark in &mut centred.landmarks {
            landmark.position -= centroid;
        }
        (centred.cast(), centroid)
    }

    /// Write a solved centred problem back into world coordinates.
    pub fn uncentre_from<T: Scalar>(&mut self, solved: &GbaProblem<T>, centroid: &Vector3<f64>) {
        for (pose, s) in self.imu_from_world.iter_mut().zip(&solved.imu_from_world) {
            let s: RigidTransform<f64> = s.cast();
            *pose = RigidTransform::new(s.rotation, s.translation - s.rotation * *centroid);
        }
        for (landmark, s) in self.landmarks.iter_mut().zip(&solved.landmarks) {
            landmark.position = s.position.map(|v| v.to_f64()) + centroid;
        }
    }
}

/// LM and PCG settings; the defaults are rootba `bal`'s as RootSLAM runs it,
/// except `lambda_min` (B24 floor; rootba 1e-16).
#[derive(Debug, Clone, PartialEq)]
pub struct GbaOptions {
    /// LM attempts, accepted or rejected (rootba `max_num_iterations` 20).
    pub max_iterations: usize,
    /// Inner solver limits.
    pub pcg: PcgOptions,
    /// Huber threshold in pixels.
    pub huber_px: f64,
    /// Initial `lambda` (rootba 1 / 1e4).
    pub initial_lambda: f64,
    /// Floor on `lambda`. 1e-10 sits far below f32 resolution of the unit
    /// scaled diagonal but keeps an unobserved landmark's triangle invertible.
    pub lambda_min: f64,
    /// Stop when `lambda` exceeds this (rootba 1 / 1e-32).
    pub lambda_max: f64,
    /// Stop when `|cost change| <= function_tolerance * cost`.
    pub function_tolerance: f64,
    /// Jacobi scaling `1 / (eps + norm)`; `None` = the scalar's epsilon (rootba).
    pub jacobi_epsilon: Option<f64>,
    /// Keyframes held constant (their scaled columns are zero). rootba fixes none.
    pub fixed_keyframes: Vec<usize>,
}

impl Default for GbaOptions {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            pcg: PcgOptions::default(),
            huber_px: 1.0,
            initial_lambda: 1e-4,
            lambda_min: 1e-10,
            lambda_max: 1e32,
            function_tolerance: 1e-6,
            jacobi_epsilon: None,
            fixed_keyframes: Vec::new(),
        }
    }
}

/// `ERROR_VALID`: Huber cost of valid projections plus `0.5 |r|^2` of the
/// pose-graph rows, summed in f64.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GbaCost {
    /// Cost.
    pub error: f64,
    /// Valid projections plus pose-graph rows.
    pub valid: usize,
    /// Projections rejected by the camera (behind it, outside its domain or non-finite).
    pub invalid: usize,
}

/// Wall time of each stage of one LM attempt.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GbaStageSeconds {
    /// Jacobians, landmark scaling, QR, pose scaling (0 on a retried step).
    pub stage1: f64,
    /// Landmark damping, preconditioner, right-hand side.
    pub stage2: f64,
    /// PCG.
    pub pcg: f64,
    /// Landmark back-substitution and model cost change.
    pub back_substitution: f64,
    /// Cost of the candidate.
    pub evaluation: f64,
}

/// One LM attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GbaIteration {
    /// Damping used.
    pub lambda: f64,
    /// Cost after the step (`None` if the step was not finite).
    pub cost_after: Option<GbaCost>,
    /// Model cost decrease `L(0) - L(step)`.
    pub model_decrease: f64,
    /// Whether the step was kept.
    pub accepted: bool,
    /// PCG count and termination.
    pub pcg: PcgSummary,
    /// Stage timings.
    pub seconds: GbaStageSeconds,
}

/// Why the LM loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbaTermination {
    /// The function tolerance held after an accepted step.
    Converged,
    /// `max_iterations` attempts ran.
    MaxIterations,
    /// `lambda` exceeded `lambda_max`.
    LambdaOverflow,
}

/// Counts and timings of one global BA.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaReport {
    /// Cost before the first step.
    pub initial_cost: GbaCost,
    /// Cost of the returned state.
    pub final_cost: GbaCost,
    /// Every attempt in order.
    pub iterations: Vec<GbaIteration>,
    /// Why the loop stopped.
    pub termination: GbaTermination,
    /// Valid projections whose Jacobian was not finite, dropped from their linearization (summed).
    pub dropped_rows: usize,
    /// Landmark steps that were not finite and were not applied (summed).
    pub nonfinite_landmark_steps: usize,
    /// Preconditioner blocks that needed the diagonal fallback (summed).
    pub preconditioner_fallbacks: usize,
}

impl GbaReport {
    /// PCG iterations over all attempts.
    pub fn pcg_iterations(&self) -> usize {
        self.iterations.iter().map(|i| i.pcg.iterations).sum()
    }
}

/// An invalid global BA input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GbaError {
    /// A landmark has fewer than two observations or repeats a (keyframe, camera).
    #[error("landmark {0} needs at least two distinct (keyframe, camera) observations")]
    Observations(usize),
    /// An index points past the keyframes or cameras.
    #[error("landmark {landmark} references keyframe {keyframe} / camera {camera} out of range")]
    Index {
        /// Landmark.
        landmark: usize,
        /// Keyframe index.
        keyframe: usize,
        /// Camera index.
        camera: usize,
    },
    /// A pose-graph row or fixed keyframe points past the keyframes.
    #[error("keyframe index {0} out of range")]
    Keyframe(usize),
    /// The landmark block layout is invalid.
    #[error(transparent)]
    Layout(#[from] SqrtBaError),
}

struct Camera<'a, S: Scalar, C> {
    model: &'a C,
    cam_from_imu: RigidTransform<S>,
    adjoint: Matrix6<S>,
}

/// One landmark's square-root block and its scratch.
struct Block<S: Scalar> {
    qr: LandmarkQr<S>,
    storage: Vec<S>,
    /// Observation indices sorted by (keyframe, camera), as rootba's map.
    order: Vec<usize>,
    /// Unique keyframes, ascending; block pose slot `s` is `keyframes[s]`.
    keyframes: Vec<usize>,
    /// Pose slot of each sorted observation.
    obs_slot: Vec<usize>,
    jl_scale: [S; 3],
    damping: LandmarkDamping<S>,
    x_red: Vec<S>,
    t: Vec<S>,
    contrib: Vec<S>,
}

impl<S: Scalar> Block<S> {
    fn new(landmark: &GbaLandmark<S>) -> Result<Self, SqrtBaError> {
        let mut order: Vec<usize> = (0..landmark.observations.len()).collect();
        order.sort_by_key(|&i| {
            (
                landmark.observations[i].keyframe,
                landmark.observations[i].camera,
            )
        });
        let mut keyframes: Vec<usize> = Vec::new();
        let obs_slot = order
            .iter()
            .map(|&i| {
                let kf = landmark.observations[i].keyframe;
                if keyframes.last() != Some(&kf) {
                    keyframes.push(kf);
                }
                keyframes.len() - 1
            })
            .collect();
        let pose_cols = 6 * keyframes.len();
        let qr = LandmarkQr::new(order.len(), pose_cols, (0..pose_cols).collect())?;
        Ok(Self {
            storage: vec![S::zero(); qr.rows() * qr.columns()],
            x_red: vec![S::zero(); pose_cols],
            t: vec![S::zero(); qr.rows() - 3],
            contrib: vec![S::zero(); pose_cols],
            order,
            keyframes,
            obs_slot,
            jl_scale: [S::one(); 3],
            damping: LandmarkDamping::default(),
            qr,
        })
    }

    fn rows(&self) -> usize {
        self.qr.rows()
    }

    fn column(&self, c: usize) -> &[S] {
        let rows = self.rows();
        &self.storage[c * rows..(c + 1) * rows]
    }

    fn gather(&mut self, x: &[S]) {
        for (s, &kf) in self.keyframes.iter().enumerate() {
            self.x_red[6 * s..6 * s + 6].copy_from_slice(&x[6 * kf..6 * kf + 6]);
        }
    }

    /// Fill rows, scale the landmark columns, eliminate. Returns dropped rows.
    fn linearize<C: GbaCamera<S>>(
        &mut self,
        landmark: &GbaLandmark<S>,
        poses: &[RigidTransform<S>],
        cameras: &[Camera<'_, S, C>],
        huber: S,
        eps: S,
    ) -> usize {
        self.storage.fill(S::zero());
        self.damping.discard();
        let rows = self.rows();
        let (lm, res_col) = (self.qr.landmark_column(), self.qr.residual_column());
        let mut dropped = 0;
        for (o, &i) in self.order.iter().enumerate() {
            let obs = &landmark.observations[i];
            let Some((res, jp, jl)) = reprojection(
                obs,
                &landmark.position,
                &poses[obs.keyframe],
                &cameras[obs.camera],
                true,
            ) else {
                continue;
            };
            let (Some(jp), Some(jl)) = (jp, jl) else {
                continue;
            };
            if !(jp.iter().all(|v| v.is_finite()) && jl.iter().all(|v| v.is_finite())) {
                dropped += 1;
                continue;
            }
            let e = res.norm();
            let sw = if e < huber {
                S::one()
            } else {
                (huber / e).sqrt()
            };
            let slot = self.obs_slot[o];
            for r in 0..2 {
                let row = 2 * o + r;
                for c in 0..6 {
                    self.storage[(6 * slot + c) * rows + row] = sw * jp[(r, c)];
                }
                for c in 0..3 {
                    self.storage[(lm + c) * rows + row] = sw * jl[(r, c)];
                }
                self.storage[res_col * rows + row] = sw * res[r];
            }
        }
        for k in 0..3 {
            let col = &mut self.storage[(lm + k) * rows..(lm + k + 1) * rows - 3];
            let norm = col
                .iter()
                .map(|v| *v * *v)
                .fold(S::zero(), |a, b| a + b)
                .sqrt();
            let scale = S::one() / (eps + norm);
            col.iter_mut().for_each(|v| *v *= scale);
            self.jl_scale[k] = scale;
        }
        self.qr.eliminate_householder_unchecked(&mut self.storage);
        dropped
    }

    /// Multiply the pose columns by the pose Jacobi scale.
    fn scale_poses(&mut self, scale: &[S]) {
        let rows = self.rows();
        for (s, &kf) in self.keyframes.iter().enumerate() {
            for j in 0..6 {
                let f = scale[6 * kf + j];
                self.storage[(6 * s + j) * rows..(6 * s + j + 1) * rows]
                    .iter_mut()
                    .for_each(|v| *v *= f);
            }
        }
    }

    /// The six `Q2^T J_p` columns (rows `3..`) of pose slot `s`.
    fn slot_columns(&self, s: usize) -> [&[S]; 6] {
        let rows = self.rows();
        std::array::from_fn(|j| &self.storage[(6 * s + j) * rows + 3..(6 * s + j + 1) * rows])
    }

    /// `contrib = (Q2^T J_p)^T (Q2^T J_p) x_red`, local to this block: column
    /// axpys, then 8-lane dot products (a fixed order, independent of threads).
    fn matvec(&mut self, x: &[S]) {
        self.gather(x);
        let rows = self.rows();
        let m = rows - 3;
        let mut t = std::mem::take(&mut self.t);
        t.fill(S::zero());
        for (c, &xc) in self.x_red.iter().enumerate() {
            if xc != S::zero() {
                let col = &self.storage[c * rows + 3..c * rows + 3 + m];
                for (tr, v) in t.iter_mut().zip(col) {
                    *tr += *v * xc;
                }
            }
        }
        for (c, out) in self.contrib.iter_mut().enumerate() {
            *out = dot_lanes(&self.storage[c * rows + 3..c * rows + 3 + m], &t);
        }
        self.t = t;
    }

    /// Landmark step from the scaled pose step, model cost decrease (f64), and
    /// whether the step is finite. Leaves the block undamped.
    fn back_substitute(&mut self, x: &[S]) -> (Vector3<S>, f64, bool) {
        self.gather(x);
        let rows = self.rows();
        let (lm, res_col) = (self.qr.landmark_column(), self.qr.residual_column());
        let pose_cols = self.x_red.len();
        let apply_jp = |storage: &[S], x_red: &[S], r: usize| {
            (0..pose_cols).fold(S::zero(), |a, c| a + storage[c * rows + r] * x_red[c])
        };
        let mut rhs = [S::zero(); 3];
        for (r, v) in rhs.iter_mut().enumerate() {
            *v = self.storage[res_col * rows + r] + apply_jp(&self.storage, &self.x_red, r);
        }
        let mut inc = Vector3::zeros();
        for r in (0..3).rev() {
            let mut acc = rhs[r];
            for k in r + 1..3 {
                acc -= self.storage[(lm + k) * rows + r] * inc[k];
            }
            inc[r] = acc / self.storage[(lm + r) * rows + r];
        }
        inc = -inc;
        self.damping
            .undo(&mut self.storage, rows, self.qr.columns());
        let mut l_diff = 0.0f64;
        for r in 0..rows - 3 {
            let mut qj = apply_jp(&self.storage, &self.x_red, r);
            if r < 3 {
                for k in r..3 {
                    qj += self.storage[(lm + k) * rows + r] * inc[k];
                }
            }
            let qj = qj.to_f64();
            l_diff -= qj * (0.5 * qj + self.storage[res_col * rows + r].to_f64());
        }
        let step = Vector3::new(
            inc[0] * self.jl_scale[0],
            inc[1] * self.jl_scale[1],
            inc[2] * self.jl_scale[2],
        );
        let finite = step.iter().all(|v| v.is_finite()) && l_diff.is_finite();
        (step, l_diff, finite)
    }
}

/// Dot product over eight fixed lanes, then a fixed pairwise sum and the tail.
#[inline]
fn dot_lanes<S: Scalar>(a: &[S], b: &[S]) -> S {
    debug_assert_eq!(a.len(), b.len());
    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    let mut acc = [S::zero(); 8];
    for (ca, cb) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += ca[i] * cb[i];
        }
    }
    let mut sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += *x * *y;
    }
    sum
}

type Reprojection<S> = (Vector2<S>, Option<Matrix2x6<S>>, Option<Matrix2x3<S>>);

/// Residual (projection minus observation) and, when asked, the Jacobians
/// against the body pose increment and the world point; `None` if invalid.
fn reprojection<S: Scalar, C: GbaCamera<S>>(
    obs: &GbaObservation<S>,
    point: &Vector3<S>,
    imu_from_world: &RigidTransform<S>,
    camera: &Camera<'_, S, C>,
    jacobians: bool,
) -> Option<Reprojection<S>> {
    let cam_from_world: RigidTransform<S> = camera.cam_from_imu * *imu_from_world;
    let p_c: Vector3<S> = cam_from_world * *point;
    let mut jp = [[S::zero(); 3]; 2];
    let (pixel, valid) = camera
        .model
        .project([p_c.x, p_c.y, p_c.z], jacobians.then_some(&mut jp));
    let res = Vector2::new(pixel[0] - obs.pixel[0], pixel[1] - obs.pixel[1]);
    if !valid || !res.iter().all(|v| v.is_finite()) {
        return None;
    }
    if !jacobians {
        return Some((res, None, None));
    }
    let jp = Matrix2x3::from_row_slice(jp.as_flattened());
    let mut d_cam = Matrix2x6::zeros();
    d_cam.fixed_view_mut::<2, 3>(0, 0).copy_from(&jp);
    d_cam
        .fixed_view_mut::<2, 3>(0, 3)
        .copy_from(&(-(jp * Rotation3::hat(&p_c))));
    // T_c_w <- T_c_i expd(xi) T_i_w = expd(Ad(T_c_i) xi) T_c_w.
    let d_pose = d_cam * camera.adjoint;
    let d_point = jp * cam_from_world.rotation.matrix();
    Some((res, Some(d_pose), Some(d_point)))
}

fn huber_cost(res_sq: f64, huber: f64) -> f64 {
    let w = if res_sq < huber * huber {
        1.0
    } else {
        huber / res_sq.sqrt()
    };
    0.5 * (2.0 - w) * w * res_sq
}

fn par_map<T: Send, R: Send>(
    pool: Option<&rayon::ThreadPool>,
    items: &mut [T],
    f: impl Fn(&mut T) -> R + Sync + Send,
) -> Vec<R> {
    match pool {
        Some(pool) => pool.install(|| items.par_iter_mut().with_min_len(16).map(f).collect()),
        None => items.iter_mut().map(f).collect(),
    }
}

/// Fill `out` (six values per keyframe) pose by pose, in a fixed entry order.
fn par_poses<S: Scalar>(
    pool: Option<&rayon::ThreadPool>,
    out: &mut [S],
    f: impl Fn(usize, &mut [S]) + Sync + Send,
) {
    match pool {
        Some(pool) => pool.install(|| out.par_chunks_mut(6).enumerate().for_each(|(k, o)| f(k, o))),
        None => out.chunks_mut(6).enumerate().for_each(|(k, o)| f(k, o)),
    }
}

/// Linearized pose-graph row: whitened residual and `[J_a | J_b]`, scaled in place.
struct RelRow<S: Scalar> {
    a: usize,
    b: usize,
    r: Vector6<S>,
    ja: Matrix6<S>,
    jb: Matrix6<S>,
}

fn rel_cost<S: Scalar>(c: &RelPoseConstraint<S>, poses: &[RigidTransform<S>]) -> f64 {
    let r = c.sqrt_info
        * relative_pose_error(
            &c.a_from_b,
            &poses[c.frame_a],
            &poses[c.frame_b],
            None,
            None,
        );
    0.5 * r.iter().map(|v| v.to_f64() * v.to_f64()).sum::<f64>()
}

struct Solver<'a, S: Scalar, C> {
    cameras: Vec<Camera<'a, S, C>>,
    blocks: Vec<Block<S>>,
    /// Per keyframe: (block, slot) entries in landmark order.
    pose_ptr: Vec<usize>,
    pose_entries: Vec<(usize, usize)>,
    rel_rows: Vec<RelRow<S>>,
    scale: Vec<S>,
    fixed: Vec<bool>,
    huber: S,
    eps: S,
}

impl<S: Scalar, C: GbaCamera<S>> Solver<'_, S, C> {
    fn evaluate(
        &self,
        problem_landmarks: &[GbaLandmark<S>],
        positions: Option<&[Vector3<S>]>,
        poses: &[RigidTransform<S>],
        rel: &[RelPoseConstraint<S>],
        pool: Option<&rayon::ThreadPool>,
    ) -> GbaCost {
        let huber = self.huber.to_f64();
        let per_landmark = |(i, landmark): (usize, &GbaLandmark<S>)| {
            let point = positions.map_or(landmark.position, |p| p[i]);
            let (mut error, mut valid, mut invalid) = (0.0f64, 0usize, 0usize);
            for obs in &landmark.observations {
                match reprojection(
                    obs,
                    &point,
                    &poses[obs.keyframe],
                    &self.cameras[obs.camera],
                    false,
                ) {
                    Some((res, _, _)) => {
                        error +=
                            huber_cost(res.iter().map(|v| v.to_f64() * v.to_f64()).sum(), huber);
                        valid += 1;
                    }
                    None => invalid += 1,
                }
            }
            (error, valid, invalid)
        };
        let parts: Vec<(f64, usize, usize)> = match pool {
            Some(pool) => pool.install(|| {
                problem_landmarks
                    .par_iter()
                    .enumerate()
                    .with_min_len(64)
                    .map(per_landmark)
                    .collect()
            }),
            None => problem_landmarks
                .iter()
                .enumerate()
                .map(per_landmark)
                .collect(),
        };
        let mut cost = GbaCost::default();
        for (e, v, i) in parts {
            cost.error += e;
            cost.valid += v;
            cost.invalid += i;
        }
        for c in rel {
            cost.error += rel_cost(c, poses);
            cost.valid += 1;
        }
        cost
    }

    /// Stage 1 at the current state. Returns dropped rows.
    fn linearize(&mut self, problem: &GbaProblem<S>, pool: Option<&rayon::ThreadPool>) -> usize {
        let (cameras, huber, eps) = (&self.cameras, self.huber, self.eps);
        let poses = &problem.imu_from_world;
        let mut indexed: Vec<(&mut Block<S>, &GbaLandmark<S>)> =
            self.blocks.iter_mut().zip(&problem.landmarks).collect();
        let dropped: usize = par_map(pool, &mut indexed, |(block, landmark)| {
            block.linearize(landmark, poses, cameras, huber, eps)
        })
        .into_iter()
        .sum();
        self.rel_rows = problem
            .rel_poses
            .iter()
            .map(|c| {
                let (mut ja, mut jb) = (Matrix6::zeros(), Matrix6::zeros());
                let r = relative_pose_error(
                    &c.a_from_b,
                    &poses[c.frame_a],
                    &poses[c.frame_b],
                    Some(&mut ja),
                    Some(&mut jb),
                );
                RelRow {
                    a: c.frame_a,
                    b: c.frame_b,
                    r: c.sqrt_info * r,
                    ja: c.sqrt_info * ja,
                    jb: c.sqrt_info * jb,
                }
            })
            .collect();
        // Pose Jacobi scale from the column norms (QR keeps them; damping rows are zero).
        let mut diag2 = vec![S::zero(); self.scale.len()];
        let blocks = &self.blocks;
        let (ptr, entries) = (&self.pose_ptr, &self.pose_entries);
        par_poses(pool, &mut diag2, |k, out| {
            for &(b, s) in &entries[ptr[k]..ptr[k + 1]] {
                let block = &blocks[b];
                let rows = block.rows();
                for (j, o) in out.iter_mut().enumerate() {
                    let col = &block.column(6 * s + j)[..rows - 3];
                    *o += col.iter().fold(S::zero(), |a, v| a + *v * *v);
                }
            }
        });
        for row in &self.rel_rows {
            for j in 0..6 {
                diag2[6 * row.a + j] += row.ja.column(j).norm_squared();
                diag2[6 * row.b + j] += row.jb.column(j).norm_squared();
            }
        }
        for (i, (s, d)) in self.scale.iter_mut().zip(&diag2).enumerate() {
            *s = if self.fixed[i / 6] {
                S::zero()
            } else {
                S::one() / (eps + d.sqrt())
            };
        }
        let scale = &self.scale;
        par_map(pool, &mut self.blocks, |block| block.scale_poses(scale));
        for row in &mut self.rel_rows {
            for j in 0..6 {
                let (sa, sb) = (scale[6 * row.a + j], scale[6 * row.b + j]);
                row.ja.column_mut(j).iter_mut().for_each(|v| *v *= sa);
                row.jb.column_mut(j).iter_mut().for_each(|v| *v *= sb);
            }
        }
        dropped
    }

    /// Stage 2: damp, then the block-Jacobi preconditioner and `b = J^T r`.
    fn damp(&mut self, lambda: S, pool: Option<&rayon::ThreadPool>) -> (BlockJacobi<S>, Vec<S>) {
        par_map(pool, &mut self.blocks, |block| {
            let (rows, cols, lm) = (block.rows(), block.qr.columns(), block.qr.landmark_column());
            block
                .damping
                .set(&mut block.storage, rows, cols, lm, lambda);
        });
        let n = self.scale.len();
        let mut precond: Vec<Matrix6<f64>> = vec![Matrix6::identity() * lambda.to_f64(); n / 6];
        let mut b = vec![S::zero(); n];
        let blocks = &self.blocks;
        let (ptr, entries) = (&self.pose_ptr, &self.pose_entries);
        let fill = |k: usize, h: &mut Matrix6<f64>, bk: &mut [S]| {
            let mut hb = [0.0f64; 21];
            let mut bb = [0.0f64; 6];
            for &(bi, s) in &entries[ptr[k]..ptr[k + 1]] {
                let block = &blocks[bi];
                let res = &block.column(block.qr.residual_column())[3..];
                let cols = block.slot_columns(s);
                for (r, &rr) in res.iter().enumerate() {
                    let v: [f64; 6] = std::array::from_fn(|j| cols[j][r].to_f64());
                    let rr = rr.to_f64();
                    let mut n = 0;
                    for i in 0..6 {
                        bb[i] += v[i] * rr;
                        for j in i..6 {
                            hb[n] += v[i] * v[j];
                            n += 1;
                        }
                    }
                }
            }
            let mut n = 0;
            for i in 0..6 {
                bk[i] += S::from_literal(bb[i]);
                for j in i..6 {
                    h[(i, j)] += hb[n];
                    n += 1;
                }
            }
        };
        match pool {
            Some(pool) => pool.install(|| {
                precond
                    .par_iter_mut()
                    .zip(b.par_chunks_mut(6))
                    .enumerate()
                    .for_each(|(k, (h, bk))| fill(k, h, bk))
            }),
            None => precond
                .iter_mut()
                .zip(b.chunks_mut(6))
                .enumerate()
                .for_each(|(k, (h, bk))| fill(k, h, bk)),
        }
        for h in &mut precond {
            for i in 0..6 {
                for j in 0..i {
                    h[(i, j)] = h[(j, i)];
                }
            }
        }
        for row in &self.rel_rows {
            for (k, j) in [(row.a, &row.ja), (row.b, &row.jb)] {
                let jt_r = j.transpose() * row.r;
                for i in 0..6 {
                    b[6 * k + i] += jt_r[i];
                }
                precond[k] += (j.transpose() * j).map(|v| v.to_f64());
            }
        }
        (BlockJacobi::from_blocks(&precond), b)
    }

    /// `y = (J^T J + lambda I) x` over the reduced system.
    fn matvec(&mut self, x: &[S], y: &mut [S], lambda: S, pool: Option<&rayon::ThreadPool>) {
        par_map(pool, &mut self.blocks, |block| block.matvec(x));
        let blocks = &self.blocks;
        let (ptr, entries) = (&self.pose_ptr, &self.pose_entries);
        par_poses(pool, y, |k, out| {
            for (o, xi) in out.iter_mut().zip(&x[6 * k..6 * k + 6]) {
                *o = lambda * *xi;
            }
            for &(b, s) in &entries[ptr[k]..ptr[k + 1]] {
                for (o, v) in out.iter_mut().zip(&blocks[b].contrib[6 * s..6 * s + 6]) {
                    *o += *v;
                }
            }
        });
        for row in &self.rel_rows {
            let jx = row.ja * Vector6::from_column_slice(&x[6 * row.a..6 * row.a + 6])
                + row.jb * Vector6::from_column_slice(&x[6 * row.b..6 * row.b + 6]);
            for (k, j) in [(row.a, &row.ja), (row.b, &row.jb)] {
                let v = j.transpose() * jx;
                for i in 0..6 {
                    y[6 * k + i] += v[i];
                }
            }
        }
    }
}

/// Run the global BA in place. Cameras stay fixed; `rig[c]` is camera `c`.
///
/// # Errors
/// Rejects out-of-range indices and landmarks with fewer than two distinct
/// observations ([`GbaProblem::drop_sparse_observations`] removes them).
pub fn solve_global<S: Scalar, C: GbaCamera<S>>(
    problem: &mut GbaProblem<S>,
    rig: &[GbaRigCamera<S, C>],
    options: &GbaOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<GbaReport, GbaError> {
    let n_kf = problem.imu_from_world.len();
    for (l, landmark) in problem.landmarks.iter().enumerate() {
        for o in &landmark.observations {
            if o.keyframe >= n_kf || o.camera >= rig.len() {
                return Err(GbaError::Index {
                    landmark: l,
                    keyframe: o.keyframe,
                    camera: o.camera,
                });
            }
        }
        let mut keys: Vec<_> = landmark
            .observations
            .iter()
            .map(|o| (o.keyframe, o.camera))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() < 2 || keys.len() != landmark.observations.len() {
            return Err(GbaError::Observations(l));
        }
    }
    for c in &problem.rel_poses {
        for k in [c.frame_a, c.frame_b] {
            if k >= n_kf {
                return Err(GbaError::Keyframe(k));
            }
        }
    }
    let mut fixed = vec![false; n_kf];
    for &k in &options.fixed_keyframes {
        *fixed.get_mut(k).ok_or(GbaError::Keyframe(k))? = true;
    }
    let blocks = problem
        .landmarks
        .iter()
        .map(Block::new)
        .collect::<Result<Vec<_>, _>>()?;
    let mut counts = vec![0usize; n_kf + 1];
    for block in &blocks {
        for &kf in &block.keyframes {
            counts[kf + 1] += 1;
        }
    }
    let mut pose_ptr = counts.clone();
    for k in 0..n_kf {
        pose_ptr[k + 1] += pose_ptr[k];
    }
    let mut cursor = pose_ptr.clone();
    let mut pose_entries = vec![(0, 0); pose_ptr[n_kf]];
    for (b, block) in blocks.iter().enumerate() {
        for (s, &kf) in block.keyframes.iter().enumerate() {
            pose_entries[cursor[kf]] = (b, s);
            cursor[kf] += 1;
        }
    }
    let eps = options
        .jacobi_epsilon
        .map_or(S::default_epsilon(), S::from_literal);
    let mut solver = Solver {
        cameras: rig
            .iter()
            .map(|c| {
                let cam_from_imu = c.imu_from_camera.inverse();
                Camera {
                    model: &c.model,
                    adjoint: cam_from_imu.adjoint(),
                    cam_from_imu,
                }
            })
            .collect(),
        blocks,
        pose_ptr,
        pose_entries,
        rel_rows: Vec::new(),
        scale: vec![S::one(); 6 * n_kf],
        fixed,
        huber: S::from_literal(options.huber_px),
        eps,
    };

    let initial_cost = solver.evaluate(
        &problem.landmarks,
        None,
        &problem.imu_from_world,
        &problem.rel_poses,
        pool,
    );
    let mut report = GbaReport {
        initial_cost,
        final_cost: initial_cost,
        iterations: Vec::new(),
        termination: GbaTermination::MaxIterations,
        dropped_rows: 0,
        nonfinite_landmark_steps: 0,
        preconditioner_fallbacks: 0,
    };
    let mut cost = initial_cost;
    let mut lambda = options.initial_lambda.max(options.lambda_min);
    let mut vee = 2.0f64;
    let n = 6 * n_kf;
    let mut x = vec![S::zero(); n];
    let mut positions: Vec<Vector3<S>> = Vec::with_capacity(problem.landmarks.len());
    'outer: while report.iterations.len() < options.max_iterations {
        let t1 = Instant::now();
        report.dropped_rows += solver.linearize(problem, pool);
        let mut stage1 = t1.elapsed().as_secs_f64();
        loop {
            let mut seconds = GbaStageSeconds {
                stage1,
                ..Default::default()
            };
            stage1 = 0.0;
            let lambda_s = S::from_literal(lambda);
            let t = Instant::now();
            let (precond, b) = solver.damp(lambda_s, pool);
            report.preconditioner_fallbacks += precond.fallbacks();
            seconds.stage2 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let summary = pcg(
                |v, out| solver.matvec(v, out, lambda_s, pool),
                &precond,
                &b,
                &mut x,
                &options.pcg,
            );
            seconds.pcg = t.elapsed().as_secs_f64();
            // We solved H x = J^T r; the step is -x (scaled).
            x.iter_mut().for_each(|v| *v = -*v);
            let mut iteration = GbaIteration {
                lambda,
                cost_after: None,
                model_decrease: f64::NAN,
                accepted: false,
                pcg: summary,
                seconds,
            };
            let step_finite = x.iter().all(|v| v.is_finite())
                && summary.termination != PcgTermination::NumericalFailure;
            if step_finite {
                let t = Instant::now();
                let steps = par_map(pool, &mut solver.blocks, |block| block.back_substitute(&x));
                let mut l_diff = 0.0f64;
                positions.clear();
                for ((step, diff, finite), landmark) in steps.into_iter().zip(&problem.landmarks) {
                    if finite {
                        l_diff += diff;
                        positions.push(landmark.position + step);
                    } else {
                        report.nonfinite_landmark_steps += 1;
                        positions.push(landmark.position);
                    }
                }
                for row in &solver.rel_rows {
                    let jinc = row.ja * Vector6::from_column_slice(&x[6 * row.a..6 * row.a + 6])
                        + row.jb * Vector6::from_column_slice(&x[6 * row.b..6 * row.b + 6]);
                    l_diff -= jinc
                        .iter()
                        .zip(row.r.iter())
                        .map(|(j, r)| j.to_f64() * (0.5 * j.to_f64() + r.to_f64()))
                        .sum::<f64>();
                }
                let poses: Vec<RigidTransform<S>> = problem
                    .imu_from_world
                    .iter()
                    .enumerate()
                    .map(|(k, pose)| {
                        let inc = Vector6::from_fn(|i, _| x[6 * k + i] * solver.scale[6 * k + i]);
                        if solver.fixed[k] {
                            *pose
                        } else {
                            apply_left_increment(pose, &inc)
                        }
                    })
                    .collect();
                iteration.seconds.back_substitution = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let new_cost = solver.evaluate(
                    &problem.landmarks,
                    Some(&positions),
                    &poses,
                    &problem.rel_poses,
                    pool,
                );
                iteration.seconds.evaluation = t.elapsed().as_secs_f64();
                iteration.cost_after = Some(new_cost);
                iteration.model_decrease = l_diff;
                let f_diff = cost.error - new_cost.error;
                if l_diff.is_finite()
                    && new_cost.error.is_finite()
                    && l_diff > 0.0
                    && f_diff / l_diff > 0.0
                {
                    let rho = f_diff / l_diff;
                    lambda = (lambda * (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3)))
                        .max(options.lambda_min);
                    vee = 2.0;
                    problem.imu_from_world = poses;
                    for (landmark, p) in problem.landmarks.iter_mut().zip(&positions) {
                        landmark.position = *p;
                    }
                    cost = new_cost;
                    iteration.accepted = true;
                    report.iterations.push(iteration);
                    if f_diff.abs() <= options.function_tolerance * cost.error {
                        report.termination = GbaTermination::Converged;
                        break 'outer;
                    }
                    continue 'outer;
                }
            }
            report.iterations.push(iteration);
            lambda *= vee;
            vee *= 2.0;
            if lambda > options.lambda_max {
                report.termination = GbaTermination::LambdaOverflow;
                break 'outer;
            }
            if report.iterations.len() >= options.max_iterations {
                break 'outer;
            }
        }
    }
    report.final_cost = cost;
    Ok(report)
}

#[cfg(test)]
#[path = "global_tests.rs"]
mod tests;
