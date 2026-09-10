//! basalt's own VIO configuration, read with serde.
//!
//! One file drives both the C++ reference run and the Rust port (decision D18),
//! so this deserializes `data/default_config.json` and `data/msd/*_config.json`
//! unmodified. The on-disk shape is cereal's: a `{"value0": {...}}` wrapper
//! whose keys are prefixed `config.` (`src/utils/vio_config.cpp:131-180`).
//!
//! Every field the ABS_QR path reads is modelled. The seventeen
//! `config.mapper_*` keys every shipped file carries are not: D13 puts the
//! mapper out of scope, nothing here reads them, and they round-trip through
//! [`VioConfig::unknown`] like any other key the struct does not model.
//!
//! ## Unknown keys are warned about, never fatal
//!
//! All four shipped JSONs carry `config.vio_outlier_threshold`,
//! `config.vio_filter_iteration`, `config.vio_lm_landmark_damping_variant` and
//! `config.vio_lm_pose_damping_variant`, which the C++ struct commented out
//! (`vio_config.h:84-85`, `vio_config.cpp:86-87,96-97`). Cereal ignores them;
//! `#[serde(deny_unknown_fields)]` would reject every reference config, so
//! unknown keys are collected and logged instead. The `mapper_*` block is
//! collected the same way but **not** warned about: it is out of scope by
//! decision, and listing seventeen expected keys would bury the one that is a
//! typo.
//!
//! ## `Default` is the C++ constructor, not `default_config.json`
//!
//! [`VioConfig::default`] reproduces `VioConfig::VioConfig()`
//! (`src/utils/vio_config.cpp:47-128`), because that is what basalt uses for any
//! key a JSON omits. The shipped `default_config.json` is *not* the same file:
//! it sets `vio_marg_lost_landmarks` to `true` where the constructor says
//! `false` (`vio_config.cpp:105`), and its
//! `optical_flow_recall_max_patch_norms` are (nearly) a quarter of the
//! constructor's. Both discrepancies are pinned by a test rather than papered
//! over — see `config_default_json_disagrees_with_the_cpp_constructor`.
//!
//! ## The fixtures
//!
//! The three MSD configs the tests parse are the package's own
//! `configs/*.json`, the files `reference_segments.toml` names and the C++
//! reference runs were driven with; `tests/fixtures/` holds only
//! `default_config.json`, which no lane runs.
//! `msdmi` (Valve Index) and `msdmg` (HP Reverb G2) are the reference datasets;
//! `msdmo` (Samsung Odyssey+) is here because the RoboCap driver reuses it —
//! `python/robocap_vit.toml:8` sets `config-path="data/msd/msdmo_config.json"`,
//! so the RoboCap gate ran with the Odyssey+ config and its
//! `optical_flow_image_safe_radius` of 388, not a RoboCap-specific one. The
//! three MSD files differ from each other in that one field alone.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Which linearization the estimator runs (`vio_config.h:42`).
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

/// How the frontend guesses where a feature moved (`vio_config.h:43`).
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

/// Which keyframe gets marginalized (`vio_config.h:44`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyframeMargCriteria {
    /// The shared-feature ratio rule (`sqrt_keypoint_vio.cpp:814`).
    #[serde(rename = "KF_MARG_DEFAULT")]
    Default,
    /// The fork's forward-vector rule (`sqrt_keypoint_vio.cpp:770`).
    #[serde(rename = "KF_MARG_FORWARD_VECTOR")]
    ForwardVector,
}

/// Something that went wrong reading a config.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The text is not the JSON the struct expects.
    #[error("could not parse the vio config: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Whether [`VioConfig::port_redetect_survivor_ratio`] is at its off value.
///
/// The port's own key is skipped when it is off, so a basalt document still
/// round-trips to exactly the keys it arrived with and a C++ run reading a
/// config the port wrote back never meets a key cereal has no field for.
#[expect(clippy::trivially_copy_pass_by_ref, reason = "serde's predicate shape")]
fn redetect_is_off(ratio: &f32) -> bool {
    *ratio == 0.0
}

/// cereal's outer wrapper: every basalt JSON is one object under `value0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Value0<T> {
    #[serde(rename = "value0")]
    value0: T,
}

/// basalt's `VioConfig` (`include/basalt/utils/vio_config.h:46-128`).
///
/// The scalar widths follow the C++ exactly (`float` vs `double` vs `int`), so a
/// value that is `float` there cannot silently gain precision here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
    /// The one key here that basalt has no counterpart for, which is why it is
    /// spelled `port.` instead of `config.`: the vendored `configs/*.json` are
    /// the files the C++ reference runs read, so nothing writes a key into them
    /// that the C++ never saw, and this one is carried by a profile overlay
    /// alone (`configs/profiles/fast.json`). `0.0` — the value every basalt file
    /// leaves it at, and [`VioConfig::default`]'s — is basalt's own behaviour:
    /// `addPoints` on every frameset. Above zero the frameset detects only once
    /// camera 0 holds fewer than this fraction of the keypoints the last
    /// detecting frameset left it with, which is cuVSLAM's rule. Anything not
    /// finite and above zero reads as `0`, so a garbled value detects rather
    /// than silently stopping.
    #[serde(
        rename = "port.redetect_survivor_ratio",
        skip_serializing_if = "redetect_is_off"
    )]
    pub port_redetect_survivor_ratio: f32,

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
    /// Estimator debug output. Parsed because every shipped JSON carries it,
    /// and read by nothing: in C++ it gates console prints and the nullspace
    /// and eigenvalue diagnostics, which the port does not carry (D68).
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
    /// Inert: the Jacobian-scaling code it would select is commented out
    /// (`sqrt_keypoint_vio.cpp:1211-1212`), matching the paper's statement that
    /// scaling is skipped.
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

    /// Keys the struct does not model, kept so a round trip loses nothing.
    ///
    /// Every shipped file carries seventeen `config.mapper_*` keys and the four
    /// the C++ struct commented out; none of them reaches a VIO decision (D13
    /// puts the mapper out of scope), so they live here rather than as fields.
    /// `to_json_string` writes them back unchanged, which is what keeps the
    /// round trip lossless.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

impl Default for VioConfig {
    /// `VioConfig::VioConfig()` (`src/utils/vio_config.cpp:47-128`).
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

            unknown: BTreeMap::new(),
        }
    }
}

impl VioConfig {
    /// Read one of basalt's config files.
    ///
    /// Missing keys keep their [`VioConfig::default`] value, as cereal does when
    /// it loads onto a default-constructed struct; unknown keys are collected
    /// into [`VioConfig::unknown`] and logged once at warning level.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let wrapper: Value0<Self> = serde_json::from_str(text)?;
        let config: Self = wrapper.value0;
        // The `config.mapper_*` block is out of scope by decision, not by
        // oversight, so it is not what this warning is for: it would bury the
        // one key that is a typo under seventeen that are expected.
        let names: Vec<&str> = config
            .unknown
            .keys()
            .map(String::as_str)
            .filter(|name| !name.starts_with("config.mapper_"))
            .collect();
        if !names.is_empty() {
            log::warn!(
                "vio config: ignoring {} unmodelled key(s): {}",
                names.len(),
                names.join(", ")
            );
        }
        Ok(config)
    }

    /// Write the config back in basalt's shape, wrapper and all.
    pub fn to_json_string(&self) -> Result<String, ConfigError> {
        Ok(serde_json::to_string_pretty(&Value0 { value0: self })?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const DEFAULT_JSON: &str = include_str!("../tests/fixtures/default_config.json");
    const MSDMI_JSON: &str = include_str!("../../../configs/msdmi_config.json");
    const MSDMG_JSON: &str = include_str!("../../../configs/msdmg_config.json");
    const MSDMO_JSON: &str = include_str!("../../../configs/msdmo_config.json");

    fn every_fixture() -> [(&'static str, &'static str); 4] {
        [
            ("default_config.json", DEFAULT_JSON),
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
        // The RoboCap driver points at this file (`python/robocap_vit.toml:8`).
        assert_eq!(odyssey.optical_flow_image_safe_radius, 388.0);

        let mut normalised: VioConfig = g2.clone();
        normalised.optical_flow_image_safe_radius = index.optical_flow_image_safe_radius;
        assert_eq!(normalised, index);

        let mut normalised: VioConfig = odyssey.clone();
        normalised.optical_flow_image_safe_radius = index.optical_flow_image_safe_radius;
        assert_eq!(normalised, index);
    }

    /// `Default` follows the C++ constructor, so it must differ from
    /// `default_config.json` in exactly two fields
    /// (`vio_config.cpp:71,105` vs `default_config.json:24-29,55`).
    #[test]
    fn config_default_json_disagrees_with_the_cpp_constructor() {
        let shipped: VioConfig = VioConfig::from_json_str(DEFAULT_JSON).unwrap();
        let constructed: VioConfig = VioConfig::default();

        assert!(!constructed.vio_marg_lost_landmarks);
        assert!(shipped.vio_marg_lost_landmarks);

        assert_eq!(
            constructed.optical_flow_recall_max_patch_norms,
            vec![1.74, 0.96, 0.99, 0.44]
        );
        assert_eq!(
            shipped.optical_flow_recall_max_patch_norms,
            vec![0.435, 0.24, 0.24, 0.11]
        );
        // Three of the four are exactly a quarter of the constructor's value;
        // the third is 0.24 where a quarter of 0.99 would be 0.2475. Pinned so
        // the near-pattern is never "tidied" into an exact one.
        let quarters: Vec<f32> = constructed
            .optical_flow_recall_max_patch_norms
            .iter()
            .map(|v| v / 4.0)
            .collect();
        assert_eq!(quarters, vec![0.435, 0.24, 0.2475, 0.11]);
        assert!(
            (shipped.optical_flow_recall_max_patch_norms[2] - quarters[2]).abs() > 1e-4,
            "the third recall norm is the one that is not a quarter"
        );

        // Nothing else moves: patching those two fields makes them equal, up to
        // the unknown keys the JSON carries and the constructor cannot.
        let mut patched: VioConfig = constructed;
        patched.vio_marg_lost_landmarks = true;
        patched.optical_flow_recall_max_patch_norms = vec![0.435, 0.24, 0.24, 0.11];
        patched.unknown = shipped.unknown.clone();
        assert_eq!(patched, shipped);
    }

    /// What every shipped file carries that the struct does not model: the four
    /// keys the C++ struct commented out (`vio_config.cpp:86-87,96-97`) and the
    /// seventeen `config.mapper_*` keys D13 puts out of scope. Neither may be
    /// fatal, and the list is exact so a *new* unmodelled key is a red test.
    #[test]
    fn the_unmodelled_keys_are_tolerated_and_listed() {
        let expected: [&str; 21] = [
            "config.mapper_bow_num_bits",
            "config.mapper_detection_num_points",
            "config.mapper_frames_to_match_threshold",
            "config.mapper_lm_lambda_max",
            "config.mapper_lm_lambda_min",
            "config.mapper_max_hamming_distance",
            "config.mapper_min_matches",
            "config.mapper_min_track_length",
            "config.mapper_min_triangulation_dist",
            "config.mapper_no_factor_weights",
            "config.mapper_num_frames_to_match",
            "config.mapper_obs_huber_thresh",
            "config.mapper_obs_std_dev",
            "config.mapper_ransac_threshold",
            "config.mapper_second_best_test_ratio",
            "config.mapper_use_factors",
            "config.mapper_use_lm",
            "config.vio_filter_iteration",
            "config.vio_lm_landmark_damping_variant",
            "config.vio_lm_pose_damping_variant",
            "config.vio_outlier_threshold",
        ];
        for (name, text) in every_fixture() {
            let config: VioConfig = VioConfig::from_json_str(text).unwrap();
            let seen: Vec<&str> = config.unknown.keys().map(String::as_str).collect();
            assert_eq!(seen, expected, "{name} carries different unknown keys");
        }
    }

    /// A `mapper_*` value survives the round trip byte for byte, which is what
    /// makes dropping the seventeen fields free: `_core.to_json_string`
    /// (`slam-rs-py/src/lib.rs`) writes a config back for a C++ run to read.
    #[test]
    fn a_mapper_key_survives_the_round_trip() {
        let config: VioConfig = VioConfig::from_json_str(MSDMI_JSON).unwrap();
        let Some(value) = config.unknown.get("config.mapper_ransac_threshold") else {
            panic!("the shipped file carries config.mapper_ransac_threshold");
        };
        assert_eq!(value.as_f64(), Some(5e-5));

        let text: String = config.to_json_string().unwrap();
        let reread: VioConfig = VioConfig::from_json_str(&text).unwrap();
        assert_eq!(reread, config, "every key, modelled or not");
        assert_eq!(
            reread.unknown.get("config.mapper_ransac_threshold"),
            Some(value)
        );
    }

    /// The port's own key is off in every shipped file, and off it is invisible:
    /// a basalt document round-trips to exactly the keys it arrived with, so a
    /// C++ run reading a config the port wrote back never meets it (D75).
    #[test]
    fn the_redetect_knob_is_off_and_unwritten_in_every_shipped_config() {
        for (name, text) in every_fixture() {
            let config: VioConfig = VioConfig::from_json_str(text).unwrap();
            assert_eq!(config.port_redetect_survivor_ratio, 0.0, "{name}");
            let written: String = config.to_json_string().unwrap();
            assert!(
                !written.contains("port.redetect_survivor_ratio"),
                "{name} wrote the port key back into a basalt document"
            );
            // Not an unknown key either: it is modelled, so a file that does
            // carry it is not merely tolerated.
            assert!(!config.unknown.contains_key("port.redetect_survivor_ratio"));
        }
    }

    /// What `configs/profiles/fast.json` does to a vendored config: one key
    /// added, everything else untouched, and it survives a round trip.
    #[test]
    fn the_redetect_knob_survives_the_round_trip_when_a_profile_sets_it() {
        let base: VioConfig = VioConfig::from_json_str(MSDMI_JSON).unwrap();
        let overlaid: String = MSDMI_JSON.replace(
            "\"value0\": {",
            "\"value0\": {\"port.redetect_survivor_ratio\": 0.5,",
        );
        let config: VioConfig = VioConfig::from_json_str(&overlaid).unwrap();
        assert_eq!(config.port_redetect_survivor_ratio, 0.5);

        let mut normalised: VioConfig = config.clone();
        normalised.port_redetect_survivor_ratio = 0.0;
        assert_eq!(normalised, base, "the overlay moved something else");

        let written: String = config.to_json_string().unwrap();
        assert!(written.contains("port.redetect_survivor_ratio"));
        assert_eq!(VioConfig::from_json_str(&written).unwrap(), config);
    }

    /// A key nobody has ever heard of is warned about, not rejected.
    #[test]
    fn an_invented_key_is_ignored() {
        let text: &str = r#"{"value0": {"config.vio_max_kfs": 9, "config.not_a_real_field": 3}}"#;
        let config: VioConfig = VioConfig::from_json_str(text).unwrap();
        assert_eq!(config.vio_max_kfs, 9);
        // Everything absent falls back to the constructor value.
        assert_eq!(config.vio_max_states, 3);
        assert_eq!(
            config.unknown.keys().collect::<Vec<_>>(),
            ["config.not_a_real_field"]
        );
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

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(VioConfig::from_json_str("{").is_err());
        assert!(VioConfig::from_json_str(r#"{"value0": {"config.vio_max_kfs": "no"}}"#).is_err());
    }
}
