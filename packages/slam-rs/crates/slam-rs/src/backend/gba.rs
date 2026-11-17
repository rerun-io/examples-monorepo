//! The global BA at the end of a sequence (DESIGN §3.1, §8; PR d2).
//!
//! [`MapBackend::finish`] turns the persistent keyframe map into a
//! [`kornia_staging_slam::sqrt_ba`] global problem ([`map_problem`]), solves it
//! ([`solve`]: f64 centring, then the f32 or f64 square-root BA), and writes the
//! result back into the map ([`apply`]). RootSLAM runs the same BA as a separate
//! process on an exported `map.json` (rootba `bal`, B25); here it runs in
//! process, on the map the window built.
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
//! The window never reads the map, so the raw VIO poses do not change.

use std::collections::BTreeMap;
use std::time::Instant;

use kornia_staging_3d::camera::{
    BrownConrady, CameraModelKind, Fisheye624, KannalaBrandt4, Pinhole,
};
use kornia_staging_algebra::Scalar;
use kornia_staging_slam::sqrt_ba::{
    GbaError, GbaLandmark, GbaObservation, GbaOptions, GbaProblem, GbaReport, GbaRigCamera,
    PcgOptions, RelPoseConstraint, solve_global,
};
use nalgebra::{Vector2, Vector3, Vector4};

use super::{KeyframeMap, MapBackend, PoseGraphEdge};

use crate::config::VioConfig;
use crate::landmark::StereographicParam;
use crate::lie::Se3;
use crate::types::{FrameId, LandmarkId};
/// The solver's report types, for callers that read a [`GbaRun`].
pub use kornia_staging_slam::sqrt_ba::{GbaCost, GbaIteration, GbaStageSeconds, GbaTermination};

/// The rig as the global BA sees it: the camera models in both scalars and
/// `T_i_c`, all fixed.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaRig {
    /// Camera models and extrinsics, f64 lane.
    pub f64: Vec<GbaRigCamera<f64, CameraModelKind<f64>>>,
    /// The same in f32 (the default lane).
    pub f32: Vec<GbaRigCamera<f32, CameraModelKind<f32>>>,
    /// `T_i_c` per camera.
    pub t_i_c: Vec<Se3<f64>>,
}

impl GbaRig {
    /// A rig from its staged camera models and extrinsics; `None` when the
    /// lists differ in length or a model does not survive the cast to f32.
    pub fn new(models: Vec<CameraModelKind<f64>>, t_i_c: Vec<Se3<f64>>) -> Option<Self> {
        if models.len() != t_i_c.len() {
            return None;
        }
        let f32: Vec<GbaRigCamera<f32, CameraModelKind<f32>>> = models
            .iter()
            .zip(&t_i_c)
            .map(|(model, t)| {
                Some(GbaRigCamera {
                    model: cast_model(model)?,
                    imu_from_camera: t.cast(),
                })
            })
            .collect::<Option<_>>()?;
        let f64 = models
            .into_iter()
            .zip(&t_i_c)
            .map(|(model, t)| GbaRigCamera {
                model,
                imu_from_camera: *t,
            })
            .collect();
        Some(Self { f64, f32, t_i_c })
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

/// The global BA's knobs (`backend.gba_*`).
#[derive(Debug, Clone, PartialEq)]
pub struct GbaSettings {
    /// LM attempts, accepted or rejected.
    pub max_iterations: usize,
    /// PCG iteration cap per attempt.
    pub cg_max_iterations: usize,
    /// PCG forcing sequence.
    pub cg_eta: f64,
    /// Huber threshold, pixels.
    pub huber_px: f64,
    /// Fewest observations for a keyframe's reprojection rows (B22).
    pub min_obs_per_kf: usize,
    /// Pose-graph translation sigma, metres.
    pub sigma_translation: f64,
    /// Pose-graph rotation sigma, radians.
    pub sigma_rotation: f64,
    /// Worker threads.
    pub threads: usize,
    /// Solve in f64 instead of f32.
    pub use_double: bool,
}

impl Default for GbaSettings {
    fn default() -> Self {
        Self::from(&VioConfig::default())
    }
}

impl From<&VioConfig> for GbaSettings {
    fn from(config: &VioConfig) -> Self {
        Self {
            max_iterations: config.gba_max_iterations,
            cg_max_iterations: config.gba_cg_max_iterations,
            cg_eta: config.gba_cg_eta,
            huber_px: config.gba_huber_px,
            min_obs_per_kf: config.gba_min_obs_per_kf,
            sigma_translation: config.gba_pose_graph_sigma_translation,
            sigma_rotation: config.gba_pose_graph_sigma_rotation,
            threads: config.gba_threads.get(),
            use_double: config.gba_use_double,
        }
    }
}

impl GbaSettings {
    /// The LM options: rootba `bal`'s, with no pose held constant.
    pub fn options(&self) -> GbaOptions {
        GbaOptions {
            max_iterations: self.max_iterations,
            pcg: PcgOptions {
                max_iterations: self.cg_max_iterations,
                eta: self.cg_eta,
            },
            huber_px: self.huber_px,
            ..GbaOptions::default()
        }
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

/// Wall time of the three driver stages, seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GbaSeconds {
    /// Map to problem.
    pub build: f64,
    /// Centring, cast, LM, uncentring.
    pub solve: f64,
    /// Problem back into the map.
    pub apply: f64,
}

/// One global BA over the map.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaRun {
    /// What the problem held.
    pub counts: GbaCounts,
    /// The LM and PCG report (cost per attempt, CG iterations, stage times).
    pub report: GbaReport,
    /// Driver stage walls.
    pub seconds: GbaSeconds,
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
}

/// A solved problem in world coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct GbaSolution {
    /// The problem after the LM loop, uncentred.
    pub problem: GbaProblem<f64>,
    /// The LM and PCG report.
    pub report: GbaReport,
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
    settings: &GbaSettings,
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
            .is_some_and(|&n| n < settings.min_obs_per_kf)
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
                settings.sigma_translation,
                settings.sigma_rotation,
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

/// Centre in f64 on the landmark centroid, solve in the settings' scalar,
/// uncentre (the `norm` fix: RootSLAM's CI runs `--no-normalize`).
///
/// # Errors
/// The worker pool cannot start, or the solver refuses the problem.
pub fn solve(
    input: &GbaInput,
    rig: &GbaRig,
    settings: &GbaSettings,
) -> Result<GbaSolution, GbaDriverError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads)
        .thread_name(|i| format!("slam-rs-gba-{i}"))
        .build()
        .map_err(|e| GbaDriverError::Threads(e.to_string()))?;
    fn run<S: Scalar>(
        input: &GbaInput,
        rig: &[GbaRigCamera<S, CameraModelKind<S>>],
        options: &GbaOptions,
        pool: &rayon::ThreadPool,
    ) -> Result<GbaSolution, GbaDriverError> {
        let (mut centred, centroid) = input.problem.centred::<S>();
        let report: GbaReport = solve_global(&mut centred, rig, options, Some(pool))?;
        let mut problem: GbaProblem<f64> = input.problem.clone();
        problem.uncentre_from(&centred, &centroid);
        Ok(GbaSolution { problem, report })
    }
    let options: GbaOptions = settings.options();
    if settings.use_double {
        run(input, &rig.f64, &options, &pool)
    } else {
        run(input, &rig.f32, &options, &pool)
    }
}

/// Write a solution into the map. Each keyframe moves by the GBA's delta,
/// `T_new = T_gba T_input^-1 T_now` (B11), so a pose the map changed since the
/// problem was built keeps that change. Each solved landmark is stored at its
/// solved world point, relative to its host's new pose; the others ride on
/// their host.
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
        if let Some((direction, inv_dist)) =
            host_relative(&(keyframe.pose * *extrinsic), &landmark.position)
        {
            map.set_landmark_position(id, direction, inv_dist);
        }
    }
}

impl MapBackend {
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
        settings: &GbaSettings,
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
        let input: GbaInput = map_problem(&self.map, &edges, &rig.t_i_c, settings);
        let build: f64 = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let solution: GbaSolution = solve(&input, rig, settings)?;
        let solve_s: f64 = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let t_i_c: Vec<Se3<f64>> = rig.t_i_c.clone();
        apply(&mut self.map, &input, &solution.problem, &t_i_c);
        let apply_s: f64 = started.elapsed().as_secs_f64();
        Ok(MapResult {
            keyframe_t_ns,
            keyframe_poses_vio,
            keyframe_poses: self.map.keyframes().values().map(|k| k.pose).collect(),
            gba: Some(GbaRun {
                counts: input.counts,
                report: solution.report,
                seconds: GbaSeconds {
                    build,
                    solve: solve_s,
                    apply: apply_s,
                },
                threads: settings.threads,
                use_double: settings.use_double,
            }),
        })
    }
}

#[cfg(test)]
#[path = "gba_tests.rs"]
mod tests;
