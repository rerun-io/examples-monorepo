//! The `backend.loop_closing` section of [`VioConfig`] (PR b2). It lives beside
//! loop closing; `config.rs` includes it in every build, with or without the
//! `backend` feature, and re-exports it.

use serde::{Deserialize, Serialize};

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
/// B16, the PGO's B12) to RootSLAM's behaviour. Missing keys keep these values;
/// unknown keys are refused.
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
    /// Run the pose-graph optimization on a loop at or above the drift gate
    /// (PR c2); off, such a loop changes nothing.
    pub pgo: bool,
    /// `[translation m, rotation rad]` sigma of every pose-graph edge (B12).
    /// RootSLAM weighs every edge with I6 (1 m counts as 1 rad); parity `[1, 1]`.
    pub pgo_edge_sigmas: [f64; 2],
    /// Workers that describe keyframes and screen candidates. `None`: Rayon's
    /// global pool. A device with a fixed core budget sets it (RoboCap: 2).
    pub threads: Option<std::num::NonZeroUsize>,
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
            pgo: true,
            pgo_edge_sigmas: [0.05, 0.01],
            threads: None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::config::VioConfig;

    #[test]
    fn threads_default_to_the_global_pool_and_take_a_flat_key() {
        assert_eq!(VioConfig::default().backend_loop_closing.threads, None);
        let capped =
            VioConfig::from_json_str(r#"{"value0":{"backend.loop_closing.threads":2}}"#).unwrap();
        assert_eq!(
            capped
                .backend_loop_closing
                .threads
                .map(std::num::NonZeroUsize::get),
            Some(2)
        );
        for refused in [0, 1025] {
            let text = format!(r#"{{"value0":{{"backend.loop_closing.threads":{refused}}}}}"#);
            assert!(VioConfig::from_json_str(&text).is_err(), "{refused}");
        }
    }

    /// `backend.loop_closing` (PR b2): RootSLAM's MSD values with the fixed
    /// behaviour of B1, B5, B15, B16 and the PGO's B12 by default; the
    /// `rootslam-parity` profile sets those to RootSLAM's; a partial section
    /// keeps the other defaults; unknown keys are refused.
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
            include_str!("../../../../../configs/profiles/rootslam-parity.json"),
        )
        .unwrap();
        let loops = &parity.backend_loop_closing;
        assert_eq!(loops.pnp_inliers_ratio, 0.0);
        assert_eq!(loops.query_cameras, LoopQueryCameras::Cam0);
        assert_eq!(loops.pnp_ransac_fixed_iterations, Some(35));
        assert_eq!(loops.brief_tests, BriefTests::Raw);
        assert_eq!(loops.pgo_edge_sigmas, [1.0, 1.0]);
        assert_eq!(
            (defaults.pgo, defaults.pgo_edge_sigmas),
            (true, [0.05, 0.01])
        );
        let fast = VioConfig::with_overlay(
            r#"{"value0":{}}"#,
            include_str!("../../../../../configs/profiles/fast.json"),
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
                pgo_edge_sigmas: [0.05, 0.01],
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
}
