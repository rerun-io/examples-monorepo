//! The global BA at the end of a sequence (DESIGN §3.1, §8; PR d2).
//!
//! [`MapBackend::finish`] turns the persistent keyframe map into a
//! [`kornia_staging_slam::sqrt_ba`] global problem ([`map_problem`]), solves it
//! ([`solve`]: f64 centring, then the f32 or f64 square-root BA), takes the
//! free gauge out ([`remove_gauge`]) and writes the result back into the map
//! ([`apply`]). RootSLAM runs the same BA as a separate process on an exported
//! `map.json` (rootba `bal`, B25); here it runs in process, on the map the
//! window built.
//!
//! The problem, as `bal` runs it:
//! - one `T_i_w` per map keyframe that has a row; cameras are fixed;
//! - one world point per landmark with at least two observations and a finite
//!   position (RootSLAM's export rule, `map_database.cpp:934-944`), computed once
//!   from its host-relative form;
//! - a keyframe with fewer than `min_obs_per_kf` observations loses its
//!   reprojection rows and keeps its pose-graph rows (B22: rootba drops it);
//! - one rel-pose row per pose-graph edge (spanning tree, loop closures,
//!   high covisibility: what RootSLAM writes to `pose_graph.json`), measured from
//!   the map's own poses, sigmas 0.05 m / 0.01 rad (B21);
//! - Huber 1 px, `ERROR_VALID`, 20 LM attempts, no pose held constant.
//!
//! Neither row kind sees roll, pitch or the origin, so the free gauge lets the
//! solution turn out of the VIO's gravity frame (up to 0.9 deg on MG). After the
//! solve, one rigid transform brings it back onto the input poses (DESIGN §3.1,
//! d2 decision); the cost does not change.
//!
//! The window never reads the map, so the raw VIO poses do not change.

use std::collections::BTreeMap;
use std::time::Instant;

use kornia_staging_3d::camera::{
    BrownConrady, CameraModelKind, Fisheye624, KannalaBrandt4, Pinhole,
};
use kornia_staging_slam::sqrt_ba::{
    GbaError, GbaLandmark, GbaObservation, GbaOptions, GbaProblem, GbaReport, GbaRigCamera,
    PcgOptions, RelPoseConstraint, solve_global,
};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3, Vector4};

use super::{KeyframeMap, MapBackend, PoseGraphEdge};

use crate::config::VioConfig;
use crate::landmark::StereographicParam;
use crate::lie::{Se3, So3};
use crate::types::{FrameId, LandmarkId};
/// The solver's report types, for callers that read a [`GbaRun`].
pub use kornia_staging_slam::sqrt_ba::{GbaCost, GbaIteration, GbaStageSeconds, GbaTermination};

/// The rig as the global BA sees it: each camera's model and `T_i_c`, fixed.
/// The f32 lane casts them in [`solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct GbaRig {
    /// Camera models and extrinsics.
    pub cameras: Vec<GbaRigCamera<f64, CameraModelKind<f64>>>,
}

impl GbaRig {
    /// A rig from its staged camera models and extrinsics; `None` when the
    /// lists differ in length or a model does not survive the cast to f32.
    pub fn new(models: Vec<CameraModelKind<f64>>, t_i_c: &[Se3<f64>]) -> Option<Self> {
        let cameras: Vec<GbaRigCamera<f64, CameraModelKind<f64>>> = models
            .into_iter()
            .zip(t_i_c)
            .map(|(model, t)| GbaRigCamera {
                model,
                imu_from_camera: *t,
            })
            .collect();
        (cameras.len() == t_i_c.len() && cameras.iter().all(|c| cast_model(&c.model).is_some()))
            .then_some(Self { cameras })
    }
}

/// A staged camera model in f32.
fn cast_model(model: &CameraModelKind<f64>) -> Option<CameraModelKind<f32>> {
    let narrow = |v: f64| v as f32;
    Some(match model {
        CameraModelKind::Pinhole(m) => {
            CameraModelKind::Pinhole(Pinhole::new(m.params().map(narrow)).ok()?)
        }
        CameraModelKind::Kb4(m) => {
            CameraModelKind::Kb4(KannalaBrandt4::new(m.params().map(narrow)).ok()?)
        }
        CameraModelKind::BrownConrady(m) => CameraModelKind::BrownConrady(
            BrownConrady::new(m.params().map(narrow), m.valid_radius().map(narrow)).ok()?,
        ),
        CameraModelKind::Fisheye624(m) => {
            CameraModelKind::Fisheye624(Fisheye624::new(m.params().map(narrow)).ok()?)
        }
    })
}

/// The LM options from `backend.gba_*`: rootba `bal`'s, no pose held constant.
pub fn gba_options(config: &VioConfig) -> GbaOptions {
    GbaOptions {
        max_iterations: config.gba_max_iterations,
        pcg: PcgOptions {
            max_iterations: config.gba_cg_max_iterations,
            eta: config.gba_cg_eta,
        },
        huber_px: config.gba_huber_px,
        ..GbaOptions::default()
    }
}

/// What [`map_problem`] kept and left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GbaCounts {
    /// Keyframe variables (every map keyframe with a row).
    pub keyframes: usize,
    /// Keyframes under `min_obs_per_kf`: pose-graph rows only (B22).
    pub keyframes_without_reprojection: usize,
    /// Landmarks in the problem.
    pub landmarks: usize,
    /// Reprojection observations in the problem.
    pub observations: usize,
    /// Pose-graph rows.
    pub pose_graph_rows: usize,
    /// Observations of keyframes under `min_obs_per_kf`.
    pub dropped_observations: usize,
    /// Landmarks without a finite world position (at infinity).
    pub nonfinite_landmarks: usize,
}

/// The map as one global BA problem, and the ids behind its indices.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaInput {
    /// Map keyframe of each pose variable, in id order.
    pub keyframes: Vec<FrameId>,
    /// Map landmark of each problem landmark, in id order.
    pub landmarks: Vec<LandmarkId>,
    /// The problem in f64 world coordinates.
    pub problem: GbaProblem<f64>,
    /// What was kept and left out.
    pub counts: GbaCounts,
}

/// One global BA over the map.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaRun {
    /// What the problem held.
    pub counts: GbaCounts,
    /// The LM and PCG report (cost per attempt, CG iterations, stage times).
    pub report: GbaReport,
    /// The rigid transform [`remove_gauge`] applied to the solution.
    pub gauge: Se3<f64>,
    /// Wall seconds: map to problem.
    pub build_s: f64,
    /// Wall seconds: centring, cast, LM, uncentring and gauge.
    pub solve_s: f64,
    /// Wall seconds: problem back into the map.
    pub apply_s: f64,
    /// Where the PCG ran (the CPU when `backend.gba_use_double`: the device
    /// lane is f32 only).
    pub lane: crate::Backend,
    /// Worker threads used.
    pub threads: usize,
    /// Whether it ran in f64.
    pub use_double: bool,
}

/// The map's keyframes at the end of a sequence (`Vio::finish_map`).
#[derive(Debug, Clone, PartialEq)]
pub struct MapResult {
    /// Map keyframes, oldest first.
    pub keyframe_t_ns: Vec<FrameId>,
    /// `T_w_i` as the window left them (before the GBA).
    pub keyframe_poses_vio: Vec<Se3<f64>>,
    /// `T_w_i` after the GBA (the same without it).
    pub keyframe_poses: Vec<Se3<f64>>,
    /// The GBA, when it ran.
    pub gba: Option<GbaRun>,
}

/// Why the global BA did not run.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GbaDriverError {
    /// The map has no camera models (a model the GBA cannot project).
    #[error(
        "the global BA needs the rig's camera models, and this rig has a model it cannot project"
    )]
    NoRig,
    /// The solver refused the problem.
    #[error(transparent)]
    Solve(#[from] GbaError),
    /// The worker pool could not start.
    #[error("could not start the global BA's threads: {0}")]
    Threads(String),
    /// The GPU lane was chosen and cannot start (a lane that fails while
    /// solving is [`GbaError::Lane`]).
    #[error("the global BA's GPU lane cannot start: {0}")]
    GpuLane(String),
}

/// A world point as the window stores it: stereographic direction and
/// inverse distance in the camera `world_from_camera` places. `None` for a
/// point at the camera centre or a non-finite result.
pub fn host_relative(
    world_from_camera: &Se3<f64>,
    point: &Vector3<f64>,
) -> Option<(Vector2<f64>, f64)> {
    let p_c: Vector3<f64> = world_from_camera.inverse() * *point;
    let distance: f64 = p_c.norm();
    let direction: Vector2<f64> =
        StereographicParam::project(&Vector4::new(p_c.x, p_c.y, p_c.z, 0.0));
    let inv_dist: f64 = 1.0 / distance;
    (distance > 0.0 && inv_dist.is_finite() && direction.iter().all(|v| v.is_finite()))
        .then_some((direction, inv_dist))
}

/// The world position of a host-relative landmark.
fn world_point(
    world_from_camera: &Se3<f64>,
    direction: &Vector2<f64>,
    inv_dist: f64,
) -> Vector3<f64> {
    let bearing: Vector4<f64> = StereographicParam::unproject(direction);
    *world_from_camera * (Vector3::new(bearing[0], bearing[1], bearing[2]) / inv_dist)
}

/// A landmark in world coordinates with its (keyframe, camera, pixel) observations.
type WorldLandmark = (LandmarkId, Vector3<f64>, Vec<(FrameId, usize, [f64; 2])>);

/// The map as a global BA problem (DESIGN §3.1). `edges` are the pose-graph
/// edges; an edge to a keyframe the map does not hold is skipped.
pub fn map_problem(
    map: &KeyframeMap,
    edges: &[PoseGraphEdge],
    t_i_c: &[Se3<f64>],
    config: &VioConfig,
) -> GbaInput {
    let keyframes = map.keyframes();
    let mut counts = GbaCounts::default();
    // Landmarks in world coordinates, once, with their keyframe observations.
    let mut world: Vec<WorldLandmark> = Vec::new();
    for (&id, landmark) in map.landmarks() {
        let (Some(host), Some(extrinsic)) = (
            keyframes.get(&landmark.host.frame_id),
            t_i_c.get(landmark.host.cam_id),
        ) else {
            continue;
        };
        let position: Vector3<f64> = world_point(
            &(host.pose * *extrinsic),
            &landmark.direction,
            landmark.inv_dist,
        );
        if !position.iter().all(|v| v.is_finite()) {
            counts.nonfinite_landmarks += 1;
            continue;
        }
        let observations: Vec<(FrameId, usize, [f64; 2])> = landmark
            .obs
            .iter()
            .filter(|(image, _)| {
                keyframes.contains_key(&image.frame_id) && image.cam_id < t_i_c.len()
            })
            .map(|(image, pixel)| {
                (
                    image.frame_id,
                    image.cam_id,
                    [f64::from(pixel.x), f64::from(pixel.y)],
                )
            })
            .collect();
        if observations.len() >= 2 {
            world.push((id, position, observations));
        }
    }
    // B22: a keyframe under `min_obs_per_kf` loses its reprojection rows.
    let mut per_keyframe: BTreeMap<FrameId, usize> = BTreeMap::new();
    for (_, _, observations) in &world {
        for (frame, _, _) in observations {
            *per_keyframe.entry(*frame).or_insert(0) += 1;
        }
    }
    let sparse = |frame: &FrameId| {
        per_keyframe
            .get(frame)
            .is_some_and(|&n| n < config.gba_min_obs_per_kf)
    };
    counts.keyframes_without_reprojection =
        per_keyframe.keys().filter(|frame| sparse(frame)).count();
    for (_, _, observations) in &mut world {
        let before: usize = observations.len();
        observations.retain(|(frame, _, _)| !sparse(frame));
        counts.dropped_observations += before - observations.len();
    }
    world.retain(|(_, _, observations)| observations.len() >= 2);
    let edges: Vec<&PoseGraphEdge> = edges
        .iter()
        .filter(|e| {
            e.from != e.to && keyframes.contains_key(&e.from) && keyframes.contains_key(&e.to)
        })
        .collect();
    // A keyframe is a variable when a row touches it.
    let mut used: std::collections::BTreeSet<FrameId> =
        edges.iter().flat_map(|e| [e.from, e.to]).collect();
    used.extend(
        world
            .iter()
            .flat_map(|(_, _, obs)| obs.iter().map(|(frame, _, _)| *frame)),
    );
    let index: BTreeMap<FrameId, usize> = used.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let pose = |id: FrameId| keyframes[&id].pose;
    let rel_poses: Vec<RelPoseConstraint<f64>> = edges
        .iter()
        .map(|e| {
            RelPoseConstraint::with_sigmas(
                index[&e.from],
                index[&e.to],
                pose(e.from).inverse() * pose(e.to),
                config.gba_pose_graph_sigma_translation,
                config.gba_pose_graph_sigma_rotation,
            )
        })
        .collect();
    let landmarks: Vec<GbaLandmark<f64>> = world
        .iter()
        .map(|(_, position, observations)| GbaLandmark {
            position: *position,
            observations: observations
                .iter()
                .map(|&(frame, camera, pixel)| GbaObservation {
                    keyframe: index[&frame],
                    camera,
                    pixel,
                })
                .collect(),
        })
        .collect();
    counts.keyframes = used.len();
    counts.landmarks = landmarks.len();
    counts.observations = landmarks.iter().map(|l| l.observations.len()).sum();
    counts.pose_graph_rows = rel_poses.len();
    GbaInput {
        problem: GbaProblem {
            imu_from_world: used.iter().map(|&id| pose(id).inverse()).collect(),
            landmarks,
            rel_poses,
        },
        keyframes: used.into_iter().collect(),
        landmarks: world.into_iter().map(|(id, _, _)| id).collect(),
        counts,
    }
}

/// Centre in f64 on the landmark centroid, solve in f32 (or f64 with
/// `backend.gba_use_double`), uncentre (the `norm` fix: RootSLAM's CI runs
/// `--no-normalize`). The result is in the solver's free gauge, as rootba
/// `bal` returns it; [`remove_gauge`] takes that out.
///
/// # Errors
/// The worker pool cannot start, a camera model does not cast to f32, or the
/// solver refuses the problem.
pub fn solve(
    input: &GbaInput,
    rig: &GbaRig,
    config: &VioConfig,
    lane: crate::Backend,
) -> Result<(GbaProblem<f64>, GbaReport), GbaDriverError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.gba_threads.get())
        .thread_name(|i| format!("slam-rs-gba-{i}"))
        .build()
        .map_err(|e| GbaDriverError::Threads(e.to_string()))?;
    let options: GbaOptions = gba_options(config);
    let mut problem: GbaProblem<f64> = input.problem.clone();
    let report: GbaReport = if config.gba_use_double {
        let (mut centred, centroid) = input.problem.centred::<f64>();
        let report = solve_global(&mut centred, &rig.cameras, &options, Some(&pool))?;
        problem.uncentre_from(&centred, &centroid);
        report
    } else {
        let cameras: Vec<GbaRigCamera<f32, CameraModelKind<f32>>> = rig
            .cameras
            .iter()
            .map(|c| {
                Some(GbaRigCamera {
                    model: cast_model(&c.model)?,
                    imu_from_camera: c.imu_from_camera.cast(),
                })
            })
            .collect::<Option<_>>()
            .ok_or(GbaDriverError::NoRig)?;
        let (mut centred, centroid) = input.problem.centred::<f32>();
        let report = match lane {
            crate::Backend::Cpu => solve_global(&mut centred, &cameras, &options, Some(&pool))?,
            crate::Backend::Gpu => solve_on_device(&mut centred, &cameras, &options, &pool)?,
        };
        problem.uncentre_from(&centred, &centroid);
        report
    };
    Ok((problem, report))
}

/// The f32 solve with the PCG on the device (`kornia_staging_gpu::linalg`).
#[cfg(feature = "gpu-wgpu")]
fn solve_on_device(
    problem: &mut GbaProblem<f32>,
    cameras: &[GbaRigCamera<f32, CameraModelKind<f32>>],
    options: &GbaOptions,
    pool: &rayon::ThreadPool,
) -> Result<GbaReport, GbaDriverError> {
    let client = kornia_staging_gpu::runtime::gpu_client()
        .map_err(|error| GbaDriverError::GpuLane(error.to_string()))?;
    let mut device = kornia_staging_gpu::linalg::GpuPcg::new(client);
    Ok(kornia_staging_slam::sqrt_ba::solve_global_with(
        problem,
        cameras,
        options,
        Some(pool),
        Some(&mut device),
    )?)
}

/// A build without the GPU lane: `select_backend` never chooses it.
#[cfg(not(feature = "gpu-wgpu"))]
fn solve_on_device(
    _: &mut GbaProblem<f32>,
    _: &[GbaRigCamera<f32, CameraModelKind<f32>>],
    _: &GbaOptions,
    _: &rayon::ThreadPool,
) -> Result<GbaReport, GbaDriverError> {
    Err(GbaDriverError::GpuLane(
        crate::VioError::GpuUnavailable.to_string(),
    ))
}

/// Take the free gauge out of a solution: the rigid transform `G` that best
/// maps the solved `T_w_i` onto the input ones (least squares over the
/// positions, metres, and the rotation columns, unit vectors; Kabsch), applied
/// to every solved pose and landmark. Rigid, so no residual and no cost
/// changes. Returns `G`.
pub fn remove_gauge(input: &GbaProblem<f64>, solved: &mut GbaProblem<f64>) -> Se3<f64> {
    let poses = |p: &GbaProblem<f64>| -> Vec<Se3<f64>> {
        p.imu_from_world.iter().map(Se3::inverse).collect()
    };
    let (from, to) = (poses(solved), poses(input));
    if from.is_empty() {
        return Se3::identity();
    }
    let n = from.len() as f64;
    let mean_from: Vector3<f64> = from.iter().map(|p| p.translation).sum::<Vector3<f64>>() / n;
    let mean_to: Vector3<f64> = to.iter().map(|p| p.translation).sum::<Vector3<f64>>() / n;
    let mut h: Matrix3<f64> = Matrix3::zeros();
    for (a, b) in from.iter().zip(&to) {
        h += (a.translation - mean_from) * (b.translation - mean_to).transpose()
            + a.rotation.matrix() * b.rotation.matrix().transpose();
    }
    let svd = h.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return Se3::identity();
    };
    let mut d: Matrix3<f64> = Matrix3::identity();
    d[(2, 2)] = (v_t.transpose() * u.transpose()).determinant().signum();
    let r: Matrix3<f64> = v_t.transpose() * d * u.transpose();
    let rotation = So3::from_unit_quaternion(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(r),
    ));
    let gauge = Se3::new(rotation, mean_to - rotation * mean_from);
    let inverse: Se3<f64> = gauge.inverse();
    for pose in &mut solved.imu_from_world {
        *pose = *pose * inverse;
    }
    for landmark in &mut solved.landmarks {
        landmark.position = gauge * landmark.position;
    }
    gauge
}

/// Write a solution into the map. Each keyframe moves by the GBA's delta,
/// `T_new = T_gba T_input^-1 T_now` (B11), so a pose the map changed since the
/// problem was built keeps that change. Each solved landmark is stored relative
/// to its host's solved pose, so it moves with its host the same way; a host
/// outside the problem keeps its current pose. Unsolved landmarks ride on their
/// host.
pub fn apply(
    map: &mut KeyframeMap,
    input: &GbaInput,
    solved: &GbaProblem<f64>,
    t_i_c: &[Se3<f64>],
) {
    for ((&id, input_pose), solved_pose) in input
        .keyframes
        .iter()
        .zip(&input.problem.imu_from_world)
        .zip(&solved.imu_from_world)
    {
        let Some(now) = map.keyframe(id).map(|k| k.pose) else {
            continue;
        };
        let delta: Se3<f64> = solved_pose.inverse() * *input_pose;
        map.upsert_keyframe(id, delta * now, None);
    }
    for (&id, landmark) in input.landmarks.iter().zip(&solved.landmarks) {
        let Some(host) = map.landmark(id).map(|l| l.host) else {
            continue;
        };
        let (Some(keyframe), Some(extrinsic)) =
            (map.keyframe(host.frame_id), t_i_c.get(host.cam_id))
        else {
            continue;
        };
        let host_pose: Se3<f64> = input
            .keyframes
            .binary_search(&host.frame_id)
            .map_or(keyframe.pose, |k| solved.imu_from_world[k].inverse());
        if let Some((direction, inv_dist)) =
            host_relative(&(host_pose * *extrinsic), &landmark.position)
        {
            map.set_landmark_position(id, direction, inv_dist);
        }
    }
}

impl MapBackend {
    /// Where the global BA's PCG runs (CPU by default).
    #[must_use]
    pub fn with_gba_lane(mut self, lane: crate::Backend) -> Self {
        self.gba_lane = lane;
        self
    }

    /// The rig the global BA projects through.
    #[must_use]
    pub fn with_gba_rig(mut self, rig: Option<GbaRig>) -> Self {
        self.gba_rig = rig;
        self
    }

    /// The map's keyframes at the end of a sequence, after a global BA when
    /// `run_gba`. The pose-graph rows are the graph's
    /// [`super::CovisibilityGraph::pose_graph_edges`] at the high-covisibility
    /// weight.
    ///
    /// # Errors
    /// [`GbaDriverError::NoRig`] when `run_gba` and the map has no camera
    /// models; the solver's refusal.
    pub fn finish(
        &mut self,
        run_gba: bool,
        config: &VioConfig,
    ) -> Result<MapResult, GbaDriverError> {
        let keyframe_t_ns: Vec<FrameId> = self.map.keyframes().keys().copied().collect();
        let keyframe_poses_vio: Vec<Se3<f64>> =
            self.map.keyframes().values().map(|k| k.pose).collect();
        if !run_gba {
            return Ok(MapResult {
                keyframe_t_ns,
                keyframe_poses: keyframe_poses_vio.clone(),
                keyframe_poses_vio,
                gba: None,
            });
        }
        let rig: &GbaRig = self.gba_rig.as_ref().ok_or(GbaDriverError::NoRig)?;
        let started = Instant::now();
        let edges: Vec<PoseGraphEdge> =
            self.graph.pose_graph_edges(self.high_covisibility_weight());
        let input: GbaInput = map_problem(&self.map, &edges, &self.t_i_c, config);
        let build_s: f64 = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let (mut solved, report) = solve(&input, rig, config, self.gba_lane)?;
        let gauge: Se3<f64> = remove_gauge(&input.problem, &mut solved);
        let solve_s: f64 = started.elapsed().as_secs_f64();
        let started = Instant::now();
        apply(&mut self.map, &input, &solved, &self.t_i_c);
        let apply_s: f64 = started.elapsed().as_secs_f64();
        Ok(MapResult {
            keyframe_t_ns,
            keyframe_poses_vio,
            keyframe_poses: self.map.keyframes().values().map(|k| k.pose).collect(),
            gba: Some(GbaRun {
                counts: input.counts,
                report,
                gauge,
                build_s,
                solve_s,
                apply_s,
                threads: config.gba_threads.get(),
                use_double: config.gba_use_double,
                lane: if config.gba_use_double {
                    crate::Backend::Cpu
                } else {
                    self.gba_lane
                },
            }),
        })
    }
}

#[cfg(test)]
#[path = "gba_tests.rs"]
mod tests;
