//! VIO configuration shared by the Rust core and Python entry points.
//! Missing fields retain their defaults; unknown fields are errors.
//! The JSON object is wrapped in `value0`, with `config.` and `port.` keys.

use serde::{Deserialize, Serialize};

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

    // ── the persistent map (back end, cargo feature `backend`) ──────────
    // One default set for every dataset (DESIGN §8). Keys RootSLAM also has
    // keep its `config.` names; keys of this port are under `backend.`.
    /// Build the persistent keyframe map beside the window. Read only by builds
    /// with the `backend` feature; the raw VIO poses are the same either way.
    /// Off by default: opt in until loop closing (PR b2) gives the map a use
    /// (a2 review). `backend.mode` says where it runs; Realtime never makes
    /// tracking wait. Builds without the feature ignore it.
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
    /// MGO13 and 1.06 on MIPB08 (a1 review, 2026-11-17).
    #[serde(rename = "config.loop_closing_pgo_min_covisibility_weight")]
    pub loop_closing_pgo_min_covisibility_weight: u32,
    /// Loop detection, verification and fusion (PR b2, DESIGN §3 rows 7-11).
    /// Read only with the `backend` feature and `backend.enabled`.
    #[serde(rename = "backend.loop_closing")]
    pub backend_loop_closing: LoopClosingConfig,
}

/// Which cameras of the new keyframe query the bag-of-words database (B5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopQueryCameras {
    /// Every camera's bag queries; a keyframe scores its best (query camera,
    /// entry camera) pair.
    #[default]
    All,
    /// Camera 0 only, as RootSLAM (`rootslam-parity`).
    Cam0,
}

/// How the BRIEF tests read the image (B16).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BriefTests {
    /// Compare the sums of the 5x5 boxes around the two points.
    #[default]
    Box,
    /// Compare raw pixels, bit-identical to Basalt (`rootslam-parity`).
    Raw,
}

/// The `backend.loop_closing` section: RootSLAM's `config.loop_closing_*` MSD
/// values, with each RootSLAM bug the port fixes at its fixed behaviour (DESIGN
/// §4). `configs/profiles/rootslam-parity.json` sets the knobs (B1, B5, B15,
/// B16) to RootSLAM's behaviour. Missing keys keep these values; unknown keys
/// are refused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoopClosingConfig {
    /// Look for loops at every new keyframe (`enable_loop_closing`).
    pub enabled: bool,
    /// Descriptor bits per HashBoW word (`bow_num_bits`).
    pub bow_num_bits: usize,
    /// Candidate keyframes kept from the query, best first (`num_frames_to_match`).
    pub num_frames_to_match: usize,
    /// Least L1 score of a candidate (`frames_to_match_threshold`).
    pub frames_to_match_threshold: f64,
    /// Candidates must be at least this much older, in seconds (`frame_time_margin_s`).
    pub frame_time_margin_s: f64,
    /// Which cameras query (B5).
    pub query_cameras: LoopQueryCameras,
    /// BRIEF test reading (B16).
    pub brief_tests: BriefTests,
    /// A descriptor match needs a Hamming distance below this (`max_hamming_distance`).
    pub max_hamming_distance: u32,
    /// Second-best distance at least this times the best (`second_best_test_ratio`);
    /// ties never match (B17).
    pub second_best_test_ratio: f64,
    /// Least keyframe-to-keyframe matches before the island is read (`min_initial_matches`).
    pub min_initial_matches: usize,
    /// Covisibility neighbours of the candidate in its island (`island_size`).
    pub island_size: usize,
    /// Least current-to-island matches (`min_island_matches`).
    pub min_island_matches: usize,
    /// GP3P inlier threshold on `1 - cos` (`pnp_ransac_threshold`).
    pub pnp_ransac_threshold: f64,
    /// `None`: adaptive RANSAC (p = 0.999, at most 500 draws, B15). `Some(n)`:
    /// OpenGV's schedule capped at `n` (`pnp_ransac_iterations`, parity 35).
    pub pnp_ransac_fixed_iterations: Option<u32>,
    /// Least PnP inliers (`pnp_min_inliers`).
    pub pnp_min_inliers: usize,
    /// Least inliers over the matches fed to that PnP (`pnp_inliers_ratio`).
    /// RootSLAM casts it to an integer, so its 0.5 acts as 0 (B1); parity 0.0.
    pub pnp_inliers_ratio: f64,
    /// Search for unmatched landmarks of the candidate around their
    /// reprojection, then verify again (`reproject_landmarks`).
    pub reproject_landmarks: bool,
    /// FAST threshold in the search window, on 8-bit intensities (`fast_threshold`).
    pub fast_threshold: u8,
    /// Side of the search window in pixels (`fast_grid_size`).
    pub fast_grid_size: u32,
    /// Hamming bound of a redetected match (`redetect_max_hamming_distance`).
    pub redetect_max_hamming_distance: u32,
    /// Ratio test of a redetected match (`redetect_second_best_test_ratio`).
    pub redetect_second_best_test_ratio: f64,
    /// Least PnP inliers after the reprojection pass (`reprojected_pnp_min_inliers`).
    pub reprojected_pnp_min_inliers: usize,
    /// A verified loop that would move the keyframe at least this far, in
    /// metres, waits for the pose-graph optimization; a smaller one fuses at
    /// once (`min_drift_reduction`).
    pub min_drift_reduction_m: f64,
    /// The same rule for rotation, in degrees (`min_rot_reduction_deg`).
    pub min_rot_reduction_deg: f64,
}

impl Default for LoopClosingConfig {
    /// RootSLAM's `msdmg_config.json` values; B1 at 0.5, B5 all cameras, B15
    /// adaptive, B16 box tests.
    fn default() -> Self {
        Self {
            enabled: true,
            bow_num_bits: 16,
            num_frames_to_match: 10,
            frames_to_match_threshold: 0.02,
            frame_time_margin_s: 20.0,
            query_cameras: LoopQueryCameras::All,
            brief_tests: BriefTests::Box,
            max_hamming_distance: 70,
            second_best_test_ratio: 1.2,
            min_initial_matches: 10,
            island_size: 4,
            min_island_matches: 20,
            pnp_ransac_threshold: 5e-5,
            pnp_ransac_fixed_iterations: None,
            pnp_min_inliers: 10,
            pnp_inliers_ratio: 0.5,
            reproject_landmarks: true,
            fast_threshold: 15,
            fast_grid_size: 16,
            redetect_max_hamming_distance: 70,
            redetect_second_best_test_ratio: 1.0,
            reprojected_pnp_min_inliers: 20,
            min_drift_reduction_m: 0.1,
            min_rot_reduction_deg: 2.0,
        }
    }
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

            backend_enabled: false,
            backend_mode: BackendMode::Offline,
            backend_realtime_max_queued_queries: std::num::NonZeroUsize::MIN.saturating_add(3),
            map_enable_culling: true,
            map_max_redundancy_score: 0.9,
            map_redundancy_exclude_hosted_lms: false,
            map_max_keyframes: 1500,
            map_batch_cull_size: 30,
            loop_closing_pgo_min_covisibility_weight: 96,
            backend_loop_closing: LoopClosingConfig::default(),
        }
    }
}

impl VioConfig {
    /// Read a wrapped config. Missing keys keep their defaults; unknown keys fail.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let wrapper: Value0<Self> = serde_json::from_str(text)?;
        Ok(wrapper.value0)
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
    /// covisibility weight instead of a per-dataset observation count.
    #[test]
    fn the_map_keys_default_to_one_set_and_take_rootslam_names() {
        let defaults = VioConfig::default();
        assert!(!defaults.backend_enabled);
        assert!(defaults.map_enable_culling);
        assert_eq!(defaults.map_max_redundancy_score, 0.9);
        assert!(!defaults.map_redundancy_exclude_hosted_lms);
        assert_eq!(defaults.map_max_keyframes, 1500);
        assert_eq!(defaults.map_batch_cull_size, 30);
        assert_eq!(defaults.loop_closing_pgo_min_covisibility_weight, 96);
        for (name, text) in every_fixture() {
            let config = VioConfig::from_json_str(text).unwrap();
            let mut normalised = config.clone();
            normalised.backend_enabled = defaults.backend_enabled;
            assert_eq!(config, normalised, "{name} sets a map key");
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

    /// `backend.loop_closing` (PR b2): RootSLAM's MSD values with the fixed
    /// behaviour of B1, B5, B15 and B16 by default; the `rootslam-parity`
    /// profile sets those four to RootSLAM's; a partial section keeps the other
    /// defaults; unknown keys are refused.
    #[test]
    fn loop_closing_defaults_fix_the_bugs_and_the_parity_profile_restores_them() {
        let defaults = VioConfig::default().backend_loop_closing;
        assert!(defaults.enabled);
        assert_eq!(defaults.pnp_inliers_ratio, 0.5);
        assert_eq!(defaults.query_cameras, LoopQueryCameras::All);
        assert_eq!(defaults.pnp_ransac_fixed_iterations, None);
        assert_eq!(defaults.brief_tests, BriefTests::Box);
        assert_eq!(
            (
                defaults.island_size,
                defaults.min_island_matches,
                defaults.fast_grid_size
            ),
            (4, 20, 16)
        );
        let parity = VioConfig::with_overlay(
            r#"{"value0":{}}"#,
            include_str!("../../../configs/profiles/rootslam-parity.json"),
        )
        .unwrap();
        let loops = &parity.backend_loop_closing;
        assert_eq!(loops.pnp_inliers_ratio, 0.0);
        assert_eq!(loops.query_cameras, LoopQueryCameras::Cam0);
        assert_eq!(loops.pnp_ransac_fixed_iterations, Some(35));
        assert_eq!(loops.brief_tests, BriefTests::Raw);
        let fast = VioConfig::with_overlay(
            r#"{"value0":{}}"#,
            include_str!("../../../configs/profiles/fast.json"),
        )
        .unwrap();
        assert_eq!(
            VioConfig {
                backend_loop_closing: fast.backend_loop_closing.clone(),
                ..parity.clone()
            },
            fast,
            "parity = fast + the loop-closing knobs"
        );
        assert_eq!(
            LoopClosingConfig {
                pnp_inliers_ratio: 0.5,
                query_cameras: LoopQueryCameras::All,
                pnp_ransac_fixed_iterations: None,
                brief_tests: BriefTests::Box,
                ..loops.clone()
            },
            defaults
        );
        assert!(
            VioConfig::from_json_str(r#"{"value0":{"backend.loop_closing":{"island":3}}}"#)
                .is_err()
        );
        assert_eq!(
            VioConfig::from_json_str(&parity.to_json_string().unwrap()).unwrap(),
            parity
        );
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
                backend_enabled: false,
                backend_mode: BackendMode::Realtime,
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
