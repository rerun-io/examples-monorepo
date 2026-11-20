//! VIO configuration shared by the Rust core and Python entry points.
//! Missing fields retain their defaults; unknown fields are errors.
//! The JSON object is wrapped in `value0`, with `config.` and `port.` keys.

use serde::{Deserialize, Serialize};

#[path = "backend/loop_closing/config.rs"]
mod loop_closing;
pub use loop_closing::{BriefTests, LoopClosingConfig, LoopQueryCameras};

/// Which linearization the estimator runs.
///
/// Only [`LinearizationType::AbsQr`] is ported (decision D13); the other two
/// parse so a config that names them is still readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearizationType {
    /// The square-root absolute-QR path, the only one V0 implements.
    #[serde(rename = "ABS_QR")]
    AbsQr,
    /// Absolute Schur complement.
    #[serde(rename = "ABS_SC")]
    AbsSc,
    /// Relative Schur complement.
    #[serde(rename = "REL_SC")]
    RelSc,
}

/// How the frontend guesses where a feature moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchingGuessType {
    /// Start from the same pixel.
    #[serde(rename = "SAME_PIXEL")]
    SamePixel,
    /// Reproject at `optical_flow_matching_default_depth`.
    #[serde(rename = "REPROJ_FIX_DEPTH")]
    ReprojFixDepth,
    /// Reproject at the depth the estimator fed back.
    #[serde(rename = "REPROJ_AVG_DEPTH")]
    ReprojAvgDepth,
}

/// Which keyframe gets marginalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyframeMargCriteria {
    /// The shared-feature ratio rule.
    #[serde(rename = "KF_MARG_DEFAULT")]
    Default,
    /// The forward-vector eviction rule.
    #[serde(rename = "KF_MARG_FORWARD_VECTOR")]
    ForwardVector,
}

/// How the estimator gets its first state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VioBootstrap {
    /// First-frameset stereo landmarks, gravity from one accelerometer
    /// sample, zero velocity. Today's behaviour, bit for bit.
    #[default]
    #[serde(rename = "stereo")]
    Stereo,
    /// Solve velocity, gravity and biases from a moving window of tracks and
    /// IMU (`kornia_staging_slam::init::inertial`), then start a fresh
    /// estimator at the window's first frameset and replay the window.
    #[serde(rename = "inertial")]
    Inertial,
}

/// How the back end runs beside tracking (`backend.mode`, DESIGN §2).
///
/// Both modes run the same [`crate::backend::MapBackend`] steps in the same
/// order, and neither changes a VIO pose: the window never reads the map.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendMode {
    /// Inline: tracking waits for each step, every loop query runs, and runs
    /// repeat (RootSLAM `--deterministic 1`). For scoring and gates.
    #[default]
    Offline,
    /// On its own thread: tracking hands stamps over and never waits; when the
    /// back end falls behind it drops the oldest queued loop queries and counts
    /// them. For live use.
    Realtime,
}

/// Something that went wrong reading a config.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The text is not the JSON the struct expects.
    #[error("could not parse the vio config: {0}")]
    Parse(#[from] serde_json::Error),
    /// A value outside the range its key allows.
    #[error("{key} = {value} is out of range: it must be {range}")]
    OutOfRange {
        /// The config key.
        key: &'static str,
        /// The value read.
        value: f64,
        /// The allowed range, in words.
        range: &'static str,
    },
}

/// The JSON wrapper: one object under `value0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Value0<T> {
    #[serde(rename = "value0")]
    value0: T,
}

/// VIO configuration with explicit scalar widths for numerical fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VioConfig {
    // ── frontend ────────────────────────────────────────────────────────
    /// Which optical-flow implementation runs; only `frame_to_frame` is ported.
    #[serde(rename = "config.optical_flow_type")]
    pub optical_flow_type: String,
    /// Detection grid cell size in pixels.
    #[serde(rename = "config.optical_flow_detection_grid_size")]
    pub optical_flow_detection_grid_size: i32,
    /// Features kept per grid cell.
    #[serde(rename = "config.optical_flow_detection_num_points_cell")]
    pub optical_flow_detection_num_points_cell: i32,
    /// Lowest FAST threshold on the ladder.
    #[serde(rename = "config.optical_flow_detection_min_threshold")]
    pub optical_flow_detection_min_threshold: i32,
    /// Highest FAST threshold on the ladder.
    #[serde(rename = "config.optical_flow_detection_max_threshold")]
    pub optical_flow_detection_max_threshold: i32,
    /// Also detect in the part of each camera outside camera 0's frustum.
    #[serde(rename = "config.optical_flow_detection_nonoverlap")]
    pub optical_flow_detection_nonoverlap: bool,
    /// Match camera 0's new keypoints into the other cameras and filter them
    /// epipolarly. `false` turns every cross-camera match off: each camera then
    /// detects on its whole image (its own masks aside), whatever
    /// `optical_flow_detection_nonoverlap` says, and no keypoint id is shared
    /// between cameras. A pair of cameras with a common centre has no epipolar
    /// geometry and is never matched, whatever this says.
    #[serde(rename = "config.optical_flow_stereo_matching")]
    pub optical_flow_stereo_matching: bool,
    /// Let the GPU frontend's one-wait stereo stage (device stereo guesses,
    /// one read per frameset) run for a rig whose cameras share one other
    /// device model (KB4, double sphere, EUCM, equirectangular), not only for
    /// Brown8 rigs. The device guesses of those models differ from the host's
    /// by up to 1e-2 px, so this changes their GPU-lane results. `false` (the
    /// default) keeps existing rigs' GPU results as they were: only Brown8 rigs
    /// take the one-wait stage.
    #[serde(rename = "config.optical_flow_gpu_one_wait_any_model")]
    pub optical_flow_gpu_one_wait_any_model: bool,
    /// Forward-backward consistency gate, in squared pixels.
    #[serde(rename = "config.optical_flow_max_recovered_dist2")]
    pub optical_flow_max_recovered_dist2: f32,
    /// Which sample pattern the patch uses; 51 means the 52-tap pattern.
    #[serde(rename = "config.optical_flow_pattern")]
    pub optical_flow_pattern: i32,
    /// Gauss-Newton iterations per pyramid level.
    #[serde(rename = "config.optical_flow_max_iterations")]
    pub optical_flow_max_iterations: i32,
    /// Pyramid levels above the base, so 3 means four levels.
    #[serde(rename = "config.optical_flow_levels")]
    pub optical_flow_levels: i32,
    /// Stereo epipolar outlier gate.
    #[serde(rename = "config.optical_flow_epipolar_error")]
    pub optical_flow_epipolar_error: f32,
    /// Frontend-to-backend decimation; 1 means every frame.
    #[serde(rename = "config.optical_flow_skip_frames")]
    pub optical_flow_skip_frames: i32,
    /// Where the tracker starts its search.
    #[serde(rename = "config.optical_flow_matching_guess_type")]
    pub optical_flow_matching_guess_type: MatchingGuessType,
    /// Depth used before the estimator reports an average, in metres.
    #[serde(rename = "config.optical_flow_matching_default_depth")]
    pub optical_flow_matching_default_depth: f32,
    /// Circular mask radius that hides a fisheye's black corners; 0 disables it.
    #[serde(rename = "config.optical_flow_image_safe_radius")]
    pub optical_flow_image_safe_radius: f32,
    /// Whether the recall subsystem runs; off in every shipped config.
    #[serde(rename = "config.optical_flow_recall_enable")]
    pub optical_flow_recall_enable: bool,
    /// Recall in every camera rather than camera 0 only.
    #[serde(rename = "config.optical_flow_recall_all_cams")]
    pub optical_flow_recall_all_cams: bool,
    /// Whether recall respects the per-cell feature limit.
    #[serde(rename = "config.optical_flow_recall_num_points_cell")]
    pub optical_flow_recall_num_points_cell: bool,
    /// Recall even for features that are already tracked.
    #[serde(rename = "config.optical_flow_recall_over_tracking")]
    pub optical_flow_recall_over_tracking: bool,
    /// Re-anchor a patch when it is recalled.
    #[serde(rename = "config.optical_flow_recall_update_patch_viewpoint")]
    pub optical_flow_recall_update_patch_viewpoint: bool,
    /// Recall distance cap, as a percentage of image width.
    #[serde(rename = "config.optical_flow_recall_max_patch_dist")]
    pub optical_flow_recall_max_patch_dist: f32,
    /// Per-level residual cap for accepting a recall.
    #[serde(rename = "config.optical_flow_recall_max_patch_norms")]
    pub optical_flow_recall_max_patch_norms: Vec<f32>,

    // ── the port's own knobs ────────────────────────────────────────────
    /// Survivor fraction below which a frameset detects; `0` detects always (D75).
    ///
    /// Zero detects on every frameset. Above zero, detect when camera 0 retains
    /// fewer than this fraction of the keypoints from the last detection.
    /// Non-finite or non-positive values use zero.
    #[serde(rename = "port.redetect_survivor_ratio")]
    pub port_redetect_survivor_ratio: f32,
    /// LM steps for a non-keyframe update; zero solves the whole window each frameset (D76).
    /// Above zero, non-keyframes solve only the newest state's 15 unknowns against
    /// fixed landmarks and its IMU factor. Keyframes still run the joint solve.
    /// The iteration cap is inclusive and shared with backtracking: five permits six
    /// trials. `configs/profiles/fast.json` selects this schedule through the `port.` key.
    #[serde(rename = "port.frame_update_max_iterations")]
    pub port_frame_update_max_iterations: i32,
    /// Defer a keyframe's joint solve beside the next frameset's frontend (D84).
    ///
    /// Off solves a keyframe frameset's window before the frameset returns. On,
    /// and with [`Self::port_frame_update_max_iterations`] above zero, a
    /// keyframe frameset first gets the newest-state update every other
    /// frameset gets, returns that pose, and leaves triangulation, the joint
    /// solve and the marginalization pending; [`crate::Vio::track`] runs them on
    /// a second thread while the next frameset's frontend runs, and the
    /// estimator finishes them before it touches the window again. The
    /// frontend's prediction for that next frameset comes from the updated, not
    /// the jointly solved, state. Deterministic: no timing enters a decision.
    #[serde(rename = "port.keyframe_solve_deferred")]
    pub port_keyframe_solve_deferred: bool,
    /// Run the estimator one frameset behind the frontend (M7, VkVIO's overlap).
    ///
    /// Off, [`crate::Vio::track`] runs frameset t's frontend and then its
    /// estimator, and returns t's pose. On, it runs frameset t's frontend on
    /// the calling thread while frameset t-1's estimator runs on a second
    /// thread, and returns t-1's pose; [`crate::Vio::flush`] returns the last
    /// one. The frontend's prediction for t then starts from the newest state
    /// it has, t-2's, propagated over [t-2, t] by its own IMU preintegration,
    /// and the depth guess it reprojects with is one frameset older too.
    /// Deterministic: the lag is structural, no timing enters a decision.
    #[serde(rename = "port.frontend_lag")]
    pub port_frontend_lag: bool,
    /// Stop a KLT level after a valid update when both the translation norm and
    /// `abs(theta) * 4` are below this many level pixels. `None` keeps every step.
    /// Supported by the CPU tracker and fused GPU kernels; must be finite and positive.
    #[serde(rename = "port.klt_exit_step_px")]
    pub port_klt_exit_step_px: Option<f32>,

    // ── estimator ───────────────────────────────────────────────────────
    /// Which linearization runs.
    #[serde(rename = "config.vio_linearization_type")]
    pub vio_linearization_type: LinearizationType,
    /// Square-root marginalization prior rather than the Hessian form.
    #[serde(rename = "config.vio_sqrt_marg")]
    pub vio_sqrt_marg: bool,
    /// Full pose-velocity-bias states in the sliding window.
    #[serde(rename = "config.vio_max_states")]
    pub vio_max_states: i32,
    /// Pose-only keyframes kept alongside them.
    #[serde(rename = "config.vio_max_kfs")]
    pub vio_max_kfs: i32,
    /// Rate limit on keyframe creation, in frames.
    #[serde(rename = "config.vio_min_frames_after_kf")]
    pub vio_min_frames_after_kf: i32,
    /// Tracked-keypoint ratio below which a new keyframe is made.
    #[serde(rename = "config.vio_new_kf_keypoints_thresh")]
    pub vio_new_kf_keypoints_thresh: f32,
    /// Parsed compatibility field for estimator debug output; currently unused (D68).
    #[serde(rename = "config.vio_debug")]
    pub vio_debug: bool,
    /// Nullspace and eigenvalue logging. Parsed and read by nothing, for the
    /// same reason as [`Self::vio_debug`].
    #[serde(rename = "config.vio_extended_logging")]
    pub vio_extended_logging: bool,
    /// Reprojection standard deviation in pixels; the residual weight is its
    /// inverse.
    #[serde(rename = "config.vio_obs_std_dev")]
    pub vio_obs_std_dev: f64,
    /// Huber threshold on the reprojection residual, in pixels.
    #[serde(rename = "config.vio_obs_huber_thresh")]
    pub vio_obs_huber_thresh: f64,
    /// Minimum baseline before a landmark is triangulated, in metres.
    #[serde(rename = "config.vio_min_triangulation_dist")]
    pub vio_min_triangulation_dist: f64,
    /// Levenberg-Marquardt iteration budget per frame.
    #[serde(rename = "config.vio_max_iterations")]
    pub vio_max_iterations: i32,
    /// Drop frames when the estimator falls behind; forced off for replay.
    #[serde(rename = "config.vio_enforce_realtime")]
    pub vio_enforce_realtime: bool,
    /// Inert on the ABS_QR path: `SqrtKeypointVioEstimator` is unconditionally
    /// LM-damped and never reads this field.
    #[serde(rename = "config.vio_use_lm")]
    pub vio_use_lm: bool,
    /// Initial LM damping.
    #[serde(rename = "config.vio_lm_lambda_initial")]
    pub vio_lm_lambda_initial: f64,
    /// Damping floor.
    #[serde(rename = "config.vio_lm_lambda_min")]
    pub vio_lm_lambda_min: f64,
    /// Damping ceiling; exceeding it abandons the frame's optimization.
    #[serde(rename = "config.vio_lm_lambda_max")]
    pub vio_lm_lambda_max: f64,
    /// Inert: Jacobian scaling is absent from the estimator (D68).
    #[serde(rename = "config.vio_scale_jacobian")]
    pub vio_scale_jacobian: bool,
    /// Gauge prior weight on the first pose.
    #[serde(rename = "config.vio_init_pose_weight")]
    pub vio_init_pose_weight: f64,
    /// Prior weight on the initial accelerometer bias.
    #[serde(rename = "config.vio_init_ba_weight")]
    pub vio_init_ba_weight: f64,
    /// Prior weight on the initial gyroscope bias.
    #[serde(rename = "config.vio_init_bg_weight")]
    pub vio_init_bg_weight: f64,
    /// Marginalize a landmark as soon as its track is lost.
    #[serde(rename = "config.vio_marg_lost_landmarks")]
    pub vio_marg_lost_landmarks: bool,
    /// Hold long-term keyframes fixed.
    #[serde(rename = "config.vio_fix_long_term_keyframes")]
    pub vio_fix_long_term_keyframes: bool,
    /// Shared-feature ratio that picks the keyframe to marginalize.
    #[serde(rename = "config.vio_kf_marg_feature_ratio")]
    pub vio_kf_marg_feature_ratio: f64,
    /// Which keyframe-removal rule applies.
    #[serde(rename = "config.vio_kf_marg_criteria")]
    pub vio_kf_marg_criteria: KeyframeMargCriteria,
    /// How the first state is found; see [`VioBootstrap`].
    #[serde(rename = "config.vio_bootstrap")]
    pub vio_bootstrap: VioBootstrap,

    // ── the persistent map (back end, cargo feature `backend`) ──────────
    // One default set for every dataset (DESIGN §8). Keys RootSLAM also has
    // keep its `config.` names; keys of this port are under `backend.`.
    /// Build the persistent keyframe map beside the window. Read only by builds
    /// with the `backend` feature; the raw VIO poses are the same either way.
    /// On by default (loops, PGO and the global BA give the final trajectory);
    /// a run that scores only the raw VIO, or a device that cannot spare the
    /// work, turns it off. `backend.mode` says where it runs; Realtime never
    /// makes tracking wait. Builds without the feature (robocap-live, wasm,
    /// iOS, embedded) ignore it.
    #[serde(rename = "backend.enabled")]
    pub backend_enabled: bool,
    /// Where the back end runs: inline (`offline`) or on its own thread
    /// (`realtime`).
    #[serde(rename = "backend.mode")]
    pub backend_mode: BackendMode,
    /// Realtime only: loop queries the back end keeps waiting while it is
    /// behind. One more drops the oldest (counted). 4 is about one second of
    /// keyframes at MSD's keyframe rate. Map stamps are never dropped. Zero is
    /// refused: it would drop every loop query and silently turn loop closing
    /// off.
    #[serde(rename = "backend.realtime_max_queued_queries")]
    pub backend_realtime_max_queued_queries: std::num::NonZeroUsize,
    /// Cull keyframes: redundant ones as they leave the window, and the
    /// lowest-scored ones in batches when the map is over its budget.
    #[serde(rename = "config.map_enable_culling")]
    pub map_enable_culling: bool,
    /// A keyframe leaving the window is culled when at least this fraction of
    /// the landmarks it sees are seen by more than three keyframes.
    #[serde(rename = "config.map_max_redundancy_score")]
    pub map_max_redundancy_score: f64,
    /// Leave the landmarks a keyframe hosts out of its redundancy score.
    #[serde(rename = "config.map_redundancy_exclude_hosted_lms")]
    pub map_redundancy_exclude_hosted_lms: bool,
    /// Keyframe budget of the map; above it, [`Self::map_batch_cull_size`]
    /// keyframes are culled at once.
    #[serde(rename = "config.map_max_keyframes")]
    pub map_max_keyframes: usize,
    /// Keyframes culled per batch once the map exceeds its budget.
    #[serde(rename = "config.map_batch_cull_size")]
    pub map_batch_cull_size: usize,
    /// Shared landmarks at which a covisibility edge is "high": it lowers the
    /// culling graph score and becomes a pose-graph edge. RootSLAM counts
    /// observations here and ships one value per dataset (MG 100, MI 70, MO 50);
    /// this port counts landmarks. 96 gives the high edges per map keyframe that
    /// RootSLAM's thresholds give on its loop-closing-off runs where the keyframe
    /// budget does not bind: MGO07 1.04 (RootSLAM 1.07), MGO13 0.30 (0.30). On
    /// MIPB08 the budget binds, and the batch cull takes keyframes with many
    /// high edges first, so the count feeds back: 0.009 at 96 against 0.026 (92
    /// gives 0.028 on that map, but 1.19 and 0.36 on MG). No multiple of the
    /// median landmarks per keyframe fits: it would be 0.65 on MGO07, 1.85 on
    /// MGO13 and 1.06 on MIPB08 (a1 review, 2026-11-17). EuRoC and TUM-VI set
    /// their own (Pablo, option a: loop settings per dataset): RootSLAM's 120 and
    /// 150 observations at a1's MG ratio 96 / 100 give 115 and 144. That ratio
    /// is carried over, not re-derived from those datasets' high edges per
    /// keyframe.
    #[serde(rename = "config.loop_closing_pgo_min_covisibility_weight")]
    pub loop_closing_pgo_min_covisibility_weight: u32,
    /// Loop detection, verification and fusion (PR b2, DESIGN §3 rows 7-11).
    /// Read only with the `backend` feature and `backend.enabled`.
    #[serde(rename = "backend.loop_closing")]
    pub backend_loop_closing: LoopClosingConfig,

    // ── the global BA at the end of a sequence (`Vio::finish_map`, PR d2) ──
    // rootba `bal` as RootSLAM's CI runs it (`run_sequence.sh`, DESIGN §3.1).
    /// LM attempts, accepted or rejected (rootba `max_num_iterations`).
    #[serde(rename = "backend.gba_max_iterations")]
    pub gba_max_iterations: usize,
    /// PCG iteration cap per LM attempt.
    #[serde(rename = "backend.gba_cg_max_iterations")]
    pub gba_cg_max_iterations: usize,
    /// PCG forcing sequence `eta`.
    #[serde(rename = "backend.gba_cg_eta")]
    pub gba_cg_eta: f64,
    /// Huber threshold of the reprojection rows, pixels.
    #[serde(rename = "backend.gba_huber_px")]
    pub gba_huber_px: f64,
    /// A keyframe with fewer observations keeps its pose-graph rows and loses
    /// its reprojection rows (B22: rootba `--min-obs-per-kf` drops the keyframe).
    #[serde(rename = "backend.gba_min_obs_per_kf")]
    pub gba_min_obs_per_kf: usize,
    /// Translation sigma of the pose-graph rows, metres (B21).
    #[serde(rename = "backend.gba_pose_graph_sigma_translation")]
    pub gba_pose_graph_sigma_translation: f64,
    /// Rotation sigma of the pose-graph rows, radians (B21).
    #[serde(rename = "backend.gba_pose_graph_sigma_rotation")]
    pub gba_pose_graph_sigma_rotation: f64,
    /// Threads of the global BA (the two-core budget, DESIGN §2).
    #[serde(rename = "backend.gba_threads")]
    pub gba_threads: std::num::NonZeroUsize,
    /// Solve in f64 instead of f32 (rootba `--use-double`; a debug lane).
    #[serde(rename = "backend.gba_use_double")]
    pub gba_use_double: bool,
}

impl Default for VioConfig {
    /// `VioConfig::VioConfig()`.
    fn default() -> Self {
        Self {
            optical_flow_type: "frame_to_frame".to_owned(),
            optical_flow_detection_grid_size: 50,
            optical_flow_detection_num_points_cell: 1,
            optical_flow_detection_min_threshold: 5,
            optical_flow_detection_max_threshold: 40,
            optical_flow_detection_nonoverlap: true,
            optical_flow_stereo_matching: true,
            optical_flow_gpu_one_wait_any_model: false,
            optical_flow_max_recovered_dist2: 0.04,
            optical_flow_pattern: 51,
            optical_flow_max_iterations: 5,
            optical_flow_levels: 3,
            optical_flow_epipolar_error: 0.005,
            optical_flow_skip_frames: 1,
            optical_flow_matching_guess_type: MatchingGuessType::ReprojAvgDepth,
            optical_flow_matching_default_depth: 2.0,
            optical_flow_image_safe_radius: 0.0,
            optical_flow_recall_enable: false,
            optical_flow_recall_all_cams: false,
            optical_flow_recall_num_points_cell: true,
            optical_flow_recall_over_tracking: false,
            optical_flow_recall_update_patch_viewpoint: false,
            optical_flow_recall_max_patch_dist: 3.0,
            optical_flow_recall_max_patch_norms: vec![1.74, 0.96, 0.99, 0.44],

            port_redetect_survivor_ratio: 0.0,
            port_frame_update_max_iterations: 0,
            port_keyframe_solve_deferred: false,
            port_frontend_lag: false,
            port_klt_exit_step_px: None,

            vio_linearization_type: LinearizationType::AbsQr,
            vio_sqrt_marg: true,
            vio_max_states: 3,
            vio_max_kfs: 7,
            vio_min_frames_after_kf: 5,
            vio_new_kf_keypoints_thresh: 0.7,
            vio_debug: false,
            vio_extended_logging: false,
            vio_obs_std_dev: 0.5,
            vio_obs_huber_thresh: 1.0,
            vio_min_triangulation_dist: 0.05,
            vio_max_iterations: 7,
            vio_enforce_realtime: false,
            vio_use_lm: true,
            vio_lm_lambda_initial: 1e-4,
            vio_lm_lambda_min: 1e-6,
            vio_lm_lambda_max: 1e2,
            vio_scale_jacobian: false,
            vio_init_pose_weight: 1e8,
            vio_init_ba_weight: 1e1,
            vio_init_bg_weight: 1e2,
            vio_marg_lost_landmarks: false,
            vio_fix_long_term_keyframes: false,
            vio_kf_marg_feature_ratio: 0.1,
            vio_kf_marg_criteria: KeyframeMargCriteria::Default,
            vio_bootstrap: VioBootstrap::Stereo,

            backend_enabled: true,
            backend_mode: BackendMode::Offline,
            backend_realtime_max_queued_queries: std::num::NonZeroUsize::MIN.saturating_add(3),
            map_enable_culling: true,
            map_max_redundancy_score: 0.9,
            map_redundancy_exclude_hosted_lms: false,
            map_max_keyframes: 1500,
            map_batch_cull_size: 30,
            loop_closing_pgo_min_covisibility_weight: 96,
            backend_loop_closing: LoopClosingConfig::default(),
            gba_max_iterations: 20,
            gba_cg_max_iterations: 500,
            gba_cg_eta: 0.1,
            gba_huber_px: 1.0,
            gba_min_obs_per_kf: 10,
            gba_pose_graph_sigma_translation: 0.05,
            gba_pose_graph_sigma_rotation: 0.01,
            gba_threads: std::num::NonZeroUsize::MIN.saturating_add(1),
            gba_use_double: false,
        }
    }
}

impl VioConfig {
    /// Read a wrapped config. Missing keys keep their defaults; unknown keys fail.
    ///
    /// Loop-closing knobs are flat keys like every other one,
    /// `backend.loop_closing.<knob>`, so a profile overlay sets one knob and
    /// keeps the dataset's others; they are folded into the
    /// `backend.loop_closing` section here (a flat key wins over the same knob
    /// in a written-out section).
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        const SECTION: &str = "backend.loop_closing";
        let mut document: Value0<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_str(text)?;
        let values = &mut document.value0;
        let flat: Vec<String> = values
            .keys()
            .filter(|key| key.starts_with("backend.loop_closing."))
            .cloned()
            .collect();
        let section_is_object: bool = values.get(SECTION).is_none_or(serde_json::Value::is_object);
        // A section that is not an object stays, and its flat keys fail as unknown.
        if !flat.is_empty() && section_is_object {
            let mut knobs: serde_json::Map<String, serde_json::Value> = match values.remove(SECTION)
            {
                Some(serde_json::Value::Object(knobs)) => knobs,
                _ => serde_json::Map::new(),
            };
            for key in flat {
                if let Some(value) = values.remove(&key) {
                    knobs.insert(key[SECTION.len() + 1..].to_string(), value);
                }
            }
            values.insert(SECTION.to_string(), serde_json::Value::Object(knobs));
        }
        let config: Self = serde_json::from_value(serde_json::Value::Object(document.value0))?;
        config.check_ranges()
    }

    /// Refuse values the global BA cannot run with (a sigma <= 0 gives an
    /// infinite sqrt-info; a Huber threshold <= 0 or `eta` outside (0, 1)
    /// breaks the LM loop without a word).
    fn check_ranges(self) -> Result<Self, ConfigError> {
        let out = |key, value, range| Err(ConfigError::OutOfRange { key, value, range });
        let positive: [(&'static str, f64); 4] = [
            ("backend.gba_max_iterations", self.gba_max_iterations as f64),
            ("backend.gba_huber_px", self.gba_huber_px),
            (
                "backend.gba_pose_graph_sigma_translation",
                self.gba_pose_graph_sigma_translation,
            ),
            (
                "backend.gba_pose_graph_sigma_rotation",
                self.gba_pose_graph_sigma_rotation,
            ),
        ];
        if let Some(&(key, value)) = positive.iter().find(|(_, v)| v.is_nan() || *v <= 0.0) {
            return out(key, value, "> 0");
        }
        let eta: f64 = self.gba_cg_eta;
        if eta.is_nan() || eta <= 0.0 || eta >= 1.0 {
            return out("backend.gba_cg_eta", eta, "in (0, 1)");
        }
        if let Some(threads) = self.backend_loop_closing.threads
            && threads.get() > crate::frontend::parallel::MAX_THREADS
        {
            return out(
                "backend.loop_closing.threads",
                threads.get() as f64,
                "in 1..=1024",
            );
        }
        Ok(self)
    }

    /// Apply a flat profile overlay to a wrapped config, validating keys and values.
    pub fn with_overlay(base: &str, overlay: &str) -> Result<Self, ConfigError> {
        let mut document: Value0<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_str(base)?;
        let fields: serde_json::Map<String, serde_json::Value> = serde_json::from_str(overlay)?;
        document.value0.extend(fields);
        Self::from_json_str(&serde_json::to_string(&document)?)
    }

    /// The same config with the back end off (`None`, `backend.mode` kept) or
    /// on in `mode`.
    #[must_use]
    pub fn with_backend(&self, mode: Option<BackendMode>) -> Self {
        let mut config: Self = self.clone();
        config.backend_enabled = mode.is_some();
        if let Some(mode) = mode {
            config.backend_mode = mode;
        }
        config
    }

    /// Write configuration JSON with its `value0` wrapper.
    pub fn to_json_string(&self) -> Result<String, ConfigError> {
        Ok(serde_json::to_string_pretty(&Value0 { value0: self })?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn an_unknown_key_is_refused() {
        let text = r#"{"value0":{"config.not_a_real_field":3}}"#;
        let error = VioConfig::from_json_str(text).unwrap_err();
        assert!(error.to_string().contains("config.not_a_real_field"));
    }

    #[test]
    fn overlays_use_the_config_schema_and_preserve_other_fields() {
        let config = VioConfig::with_overlay(
            MSDMO_JSON,
            include_str!("../../../configs/profiles/fast.json"),
        )
        .unwrap();
        assert_eq!(config.port_klt_exit_step_px, Some(0.05));
        assert_eq!(config.optical_flow_image_safe_radius, 388.0);
        assert!(
            VioConfig::with_overlay(r#"{"value0":{}}"#, r#"{"port.frontend_lag":true}"#)
                .unwrap()
                .port_frontend_lag
        );
        for overlay in [
            r#"{"port.typo":true}"#,
            r#"{"port.frontend_lag":"yes"}"#,
            "[]",
        ] {
            assert!(VioConfig::with_overlay(MSDMO_JSON, overlay).is_err());
        }
    }

    #[test]
    fn klt_exit_is_opt_in_and_round_trips() {
        for threshold in ["null", "0.01", "0.03", "0.05"] {
            let text = format!(r#"{{"value0":{{"port.klt_exit_step_px":{threshold}}}}}"#);
            let config = VioConfig::from_json_str(&text).unwrap();
            let written: serde_json::Value =
                serde_json::from_str(&config.to_json_string().unwrap()).unwrap();
            let expected: serde_json::Value = serde_json::from_str(threshold).unwrap();
            let actual = &written["value0"]["port.klt_exit_step_px"];
            if let Some(expected) = expected.as_f64() {
                assert!((actual.as_f64().unwrap() - expected).abs() < 1e-8);
            } else {
                assert!(actual.is_null());
            }
        }
        let defaults: serde_json::Value =
            serde_json::from_str(&VioConfig::default().to_json_string().unwrap()).unwrap();
        assert!(
            defaults["value0"]
                .get("port.klt_exit_step_px")
                .unwrap()
                .is_null()
        );
    }

    const MSDMI_JSON: &str = include_str!("../../../configs/msdmi_config.json");
    const MSDMG_JSON: &str = include_str!("../../../configs/msdmg_config.json");
    const MSDMO_JSON: &str = include_str!("../../../configs/msdmo_config.json");
    const EUROC_JSON: &str = include_str!("../../../configs/euroc_config.json");
    const TUMVI_JSON: &str = include_str!("../../../configs/tumvi_config.json");

    fn every_fixture() -> [(&'static str, &'static str); 3] {
        [
            ("msdmi_config.json", MSDMI_JSON),
            ("msdmg_config.json", MSDMG_JSON),
            ("msdmo_config.json", MSDMO_JSON),
        ]
    }

    #[test]
    fn every_shipped_config_parses() {
        for (name, text) in every_fixture() {
            let config: VioConfig = VioConfig::from_json_str(text)
                .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
            assert_eq!(config.optical_flow_type, "frame_to_frame");
            assert_eq!(config.vio_linearization_type, LinearizationType::AbsQr);
            assert_eq!(
                config.optical_flow_matching_guess_type,
                MatchingGuessType::ReprojAvgDepth
            );
            assert_eq!(config.vio_kf_marg_criteria, KeyframeMargCriteria::Default);
        }
    }

    /// The one field the three MSD configs disagree on: the lens-circle mask.
    #[test]
    fn the_msd_configs_differ_only_in_the_safe_radius() {
        let index: VioConfig = VioConfig::from_json_str(MSDMI_JSON).unwrap();
        let g2: VioConfig = VioConfig::from_json_str(MSDMG_JSON).unwrap();
        let odyssey: VioConfig = VioConfig::from_json_str(MSDMO_JSON).unwrap();

        assert_eq!(index.optical_flow_image_safe_radius, 472.0);
        assert_eq!(g2.optical_flow_image_safe_radius, 340.0);
        // The RoboCap driver points at this file (`python/robocap_vit.toml`).
        assert_eq!(odyssey.optical_flow_image_safe_radius, 388.0);

        let mut normalised: VioConfig = g2.clone();
        normalised.optical_flow_image_safe_radius = index.optical_flow_image_safe_radius;
        assert_eq!(normalised, index);

        let mut normalised: VioConfig = odyssey.clone();
        normalised.optical_flow_image_safe_radius = index.optical_flow_image_safe_radius;
        assert_eq!(normalised, index);
    }

    #[test]
    fn removed_gpu_kernel_selector_is_rejected() {
        assert!(
            VioConfig::from_json_str(r#"{"value0":{"port.gpu_klt_kernel":"butterfly16"}}"#)
                .is_err()
        );
    }

    /// Even disabled port knobs are written explicitly.
    #[test]
    fn the_port_knobs_are_written_in_every_shipped_config() {
        for (_, text) in every_fixture() {
            let config = VioConfig::from_json_str(text).unwrap();
            let written: serde_json::Value =
                serde_json::from_str(&config.to_json_string().unwrap()).unwrap();
            assert_eq!(written["value0"]["port.redetect_survivor_ratio"], 0.0);
            assert_eq!(written["value0"]["port.frame_update_max_iterations"], 0);
            assert_eq!(written["value0"]["port.keyframe_solve_deferred"], false);
            assert_eq!(written["value0"]["port.frontend_lag"], false);
        }
    }

    /// What `configs/profiles/fast.json` does to a vendored config: one key
    /// added, everything else untouched, and it survives a round trip.
    #[test]
    fn the_port_knobs_survive_the_round_trip_when_a_profile_sets_them() {
        let base: VioConfig = VioConfig::from_json_str(MSDMI_JSON).unwrap();
        let overlaid: String = MSDMI_JSON.replace(
            "\"value0\": {",
            "\"value0\": {\"port.redetect_survivor_ratio\": 0.7, \"port.frame_update_max_iterations\": 2,",
        );
        let config: VioConfig = VioConfig::from_json_str(&overlaid).unwrap();
        assert_eq!(config.port_redetect_survivor_ratio, 0.7);
        assert_eq!(config.port_frame_update_max_iterations, 2);

        let mut normalised: VioConfig = config.clone();
        normalised.port_redetect_survivor_ratio = 0.0;
        normalised.port_frame_update_max_iterations = 0;
        assert_eq!(normalised, base, "the overlay moved something else");

        let written: String = config.to_json_string().unwrap();
        assert!(written.contains("port.redetect_survivor_ratio"));
        assert!(written.contains("port.frame_update_max_iterations"));
        assert_eq!(VioConfig::from_json_str(&written).unwrap(), config);
    }

    #[test]
    fn a_config_round_trips_through_serde() {
        for (name, text) in every_fixture() {
            let once: VioConfig = VioConfig::from_json_str(text).unwrap();
            let written: String = once.to_json_string().unwrap();
            let twice: VioConfig = VioConfig::from_json_str(&written).unwrap();
            assert_eq!(once, twice, "{name} did not round trip");
        }
    }

    /// One default set for the persistent map (DESIGN §8, phase1/kf-density.md):
    /// RootSLAM's MSD values where they are one value, and one shared-landmark
    /// covisibility weight instead of a per-dataset observation count. The
    /// loop gates are the exception (Pablo, option a): each dataset config sets
    /// RootSLAM's own values (the tests below); the MSD configs set none.
    #[test]
    fn the_map_keys_default_to_one_set_and_take_rootslam_names() {
        let defaults = VioConfig::default();
        assert!(defaults.backend_enabled);
        assert!(defaults.map_enable_culling);
        assert_eq!(defaults.map_max_redundancy_score, 0.9);
        assert!(!defaults.map_redundancy_exclude_hosted_lms);
        assert_eq!(defaults.map_max_keyframes, 1500);
        assert_eq!(defaults.map_batch_cull_size, 30);
        assert_eq!(defaults.loop_closing_pgo_min_covisibility_weight, 96);
        // Every map key of every fixture is the default; `backend.enabled` may differ.
        let map_keys = |c: &VioConfig| {
            (
                c.map_enable_culling,
                c.map_max_redundancy_score,
                c.map_redundancy_exclude_hosted_lms,
                c.map_max_keyframes,
                c.map_batch_cull_size,
                c.loop_closing_pgo_min_covisibility_weight,
            )
        };
        for (name, text) in every_fixture() {
            let config = VioConfig::from_json_str(text).unwrap();
            assert_eq!(
                map_keys(&config),
                map_keys(&defaults),
                "{name} sets a map key"
            );
        }

        let parity = VioConfig::from_json_str(
            r#"{"value0":{"backend.enabled":true,"config.map_max_keyframes":3000,
            "config.map_batch_cull_size":10,"config.map_enable_culling":false,
            "config.loop_closing_pgo_min_covisibility_weight":100}}"#,
        )
        .unwrap();
        assert!(parity.backend_enabled);
        assert_eq!(parity.map_max_keyframes, 3000);
        assert_eq!(parity.map_batch_cull_size, 10);
        assert!(!parity.map_enable_culling);
        assert_eq!(parity.loop_closing_pgo_min_covisibility_weight, 100);
        assert_eq!(
            VioConfig::from_json_str(&parity.to_json_string().unwrap()).unwrap(),
            parity
        );
        // RootSLAM keys with no reader here stay refused (B20).
        for dead in [
            "config.map_covisibility_min_weight",
            "config.map_covisibility_max_size",
            "config.map_debug",
        ] {
            let text = format!(r#"{{"value0":{{"{dead}":3}}}}"#);
            assert!(VioConfig::from_json_str(&text).is_err(), "{dead} parsed");
        }
    }

    /// RootSLAM's loop gates for one dataset, mapped as for MSD: the PnP
    /// minimums as they are, the inlier ratio on the final PnP (B1), RANSAC
    /// adaptive (RootSLAM's count is the fixed-schedule cap, which only
    /// `rootslam-parity` turns on), and the covisibility weight in landmarks.
    fn assert_loop_gates(text: &str, gates: (usize, usize, f64), covisibility: u32) {
        let config = VioConfig::from_json_str(text).unwrap();
        let loops = &config.backend_loop_closing;
        assert_eq!(
            (
                loops.pnp_min_inliers,
                loops.reprojected_pnp_min_inliers,
                loops.pnp_inliers_ratio
            ),
            gates
        );
        assert_eq!(loops.pnp_ransac_fixed_iterations, None, "adaptive");
        assert_eq!(
            LoopClosingConfig {
                pnp_min_inliers: 10,
                reprojected_pnp_min_inliers: 20,
                pnp_inliers_ratio: 0.5,
                ..loops.clone()
            },
            LoopClosingConfig::default(),
            "only the gates differ"
        );
        assert_eq!(
            config.loop_closing_pgo_min_covisibility_weight,
            covisibility
        );
    }

    /// RootSLAM msdmg / msdmi / msdmo: 10 / 20 / 0.5, 100 observations -> 96.
    #[test]
    fn msd_loop_gates_are_rootslams_msd_values() {
        for (_, text) in every_fixture() {
            assert_loop_gates(text, (10, 20, 0.5), 96);
        }
    }

    /// RootSLAM euroc: 10 / 30 / 0.7, 120 observations -> 115.
    #[test]
    fn euroc_loop_gates_are_rootslams_euroc_values() {
        assert_loop_gates(EUROC_JSON, (10, 30, 0.7), 115);
    }

    /// RootSLAM tumvi_512: 20 / 30 / 0.6, 150 observations -> 144.
    #[test]
    fn tumvi_loop_gates_are_rootslams_tumvi_512_values() {
        assert_loop_gates(TUMVI_JSON, (20, 30, 0.6), 144);
    }

    /// The parity profile overlays its loop knobs one by one: a dataset keeps
    /// the gates the profile does not set (RootSLAM's euroc 10 / 30, tumvi_512
    /// 20 / 30), where a whole-section overlay ran MSD's gates everywhere.
    #[test]
    fn the_parity_profile_keeps_each_datasets_loop_gates() {
        let parity = include_str!("../../../configs/profiles/rootslam-parity.json");
        for (text, gates) in [(EUROC_JSON, (10, 30)), (TUMVI_JSON, (20, 30))] {
            let loops = VioConfig::with_overlay(text, parity)
                .unwrap()
                .backend_loop_closing;
            assert_eq!(
                (loops.pnp_min_inliers, loops.reprojected_pnp_min_inliers),
                gates
            );
            assert_eq!(loops.pnp_inliers_ratio, 0.0, "the profile's own knob");
        }
    }

    /// The global BA keys (PR d2) default to rootba `bal` as RootSLAM's CI runs
    /// it, two threads and f32, and a dataset config can override each.
    #[test]
    fn the_gba_keys_default_to_rootba_bal() {
        let defaults = VioConfig::default();
        assert_eq!(defaults.gba_max_iterations, 20);
        assert_eq!(defaults.gba_cg_max_iterations, 500);
        assert_eq!(defaults.gba_cg_eta, 0.1);
        assert_eq!(defaults.gba_huber_px, 1.0);
        assert_eq!(defaults.gba_min_obs_per_kf, 10);
        assert_eq!(defaults.gba_pose_graph_sigma_translation, 0.05);
        assert_eq!(defaults.gba_pose_graph_sigma_rotation, 0.01);
        assert_eq!(defaults.gba_threads.get(), 2);
        assert!(!defaults.gba_use_double);
        let parsed = VioConfig::from_json_str(
            r#"{"value0":{"backend.gba_max_iterations":5,"backend.gba_threads":8,
            "backend.gba_use_double":true,"backend.gba_min_obs_per_kf":20}}"#,
        )
        .unwrap();
        assert_eq!(parsed.gba_max_iterations, 5);
        assert_eq!(parsed.gba_threads.get(), 8);
        assert!(parsed.gba_use_double);
        assert_eq!(parsed.gba_min_obs_per_kf, 20);
        assert!(VioConfig::from_json_str(r#"{"value0":{"backend.gba_threads":0}}"#).is_err());
        // Out of range: refused at load, naming the key (rootba checks its
        // Huber threshold too). A sigma <= 0 gives an infinite sqrt-info.
        for (key, value) in [
            ("backend.gba_max_iterations", "0"),
            ("backend.gba_cg_eta", "0.0"),
            ("backend.gba_cg_eta", "1.0"),
            ("backend.gba_huber_px", "0.0"),
            ("backend.gba_pose_graph_sigma_translation", "0.0"),
            ("backend.gba_pose_graph_sigma_rotation", "-0.01"),
        ] {
            let text = format!(r#"{{"value0":{{"{key}":{value}}}}}"#);
            let error = VioConfig::from_json_str(&text).unwrap_err();
            assert!(error.to_string().contains(key), "{text}: {error}");
        }
    }

    /// `backend.mode` (DESIGN §2, §8): Offline by default, lowercase names, and
    /// an unknown mode refused.
    #[test]
    fn the_back_end_mode_defaults_to_offline_and_reads_lowercase_names() {
        let defaults = VioConfig::default();
        assert_eq!(defaults.backend_mode, BackendMode::Offline);
        assert_eq!(defaults.backend_realtime_max_queued_queries.get(), 4);
        let realtime = VioConfig::from_json_str(
            r#"{"value0":{"backend.mode":"realtime","backend.realtime_max_queued_queries":2}}"#,
        )
        .unwrap();
        assert_eq!(realtime.backend_mode, BackendMode::Realtime);
        assert_eq!(realtime.backend_realtime_max_queued_queries.get(), 2);
        let written: String = realtime.to_json_string().unwrap();
        assert!(written.contains(r#""realtime""#), "{written}");
        assert_eq!(VioConfig::from_json_str(&written).unwrap(), realtime);
        assert!(VioConfig::from_json_str(r#"{"value0":{"backend.mode":"Realtime"}}"#).is_err());
        assert!(VioConfig::from_json_str(r#"{"value0":{"backend.mode":"threaded"}}"#).is_err());
        let zero =
            VioConfig::from_json_str(r#"{"value0":{"backend.realtime_max_queued_queries":0}}"#)
                .unwrap_err();
        assert!(zero.to_string().contains("nonzero"), "{zero}");
    }

    /// `with_backend` sets the two back-end keys through the type, every
    /// other key unchanged (the bindings use it instead of editing JSON).
    #[test]
    fn with_backend_sets_only_the_back_end_keys() {
        let base = VioConfig {
            vio_max_iterations: 3,
            backend_mode: BackendMode::Realtime,
            ..VioConfig::default()
        };
        let off = base.with_backend(None);
        assert!(!off.backend_enabled);
        assert_eq!(
            off.backend_mode,
            BackendMode::Realtime,
            "off keeps the mode"
        );
        let offline = base.with_backend(Some(BackendMode::Offline));
        assert!(offline.backend_enabled);
        assert_eq!(offline.backend_mode, BackendMode::Offline);
        assert_eq!(
            VioConfig {
                backend_enabled: base.backend_enabled,
                backend_mode: base.backend_mode,
                ..offline
            },
            base
        );
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(VioConfig::from_json_str("{").is_err());
        assert!(VioConfig::from_json_str(r#"{"value0": {"config.vio_max_kfs": "no"}}"#).is_err());
    }
}
