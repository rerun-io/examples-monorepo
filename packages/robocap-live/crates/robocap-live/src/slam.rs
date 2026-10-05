//! slam-rs VIO on the four SLAM cameras (`[4, 0, 1, 5]` = left, left_front, right_front, right) at 640x360 + IMU0, set up as
//! PR #270's live adapter (`robocap-recorder/src/live_slam.rs` at 271ce643) with the cap's factory calibration
//! ([`crate::capture::Cap::slam_calibration`], chosen by the rig's `device`), [`MSDMO_CONFIG`] with a [`SlamProfile`] overlay
//! (`live` by default), and the CPU frontend with [`SlamConfig::frontend_threads`] threads.
//!
//! # Frames and conventions
//!
//! - slam-rs's rig frame IS the IMU0 frame (the calibration's extrinsics are `T_imu_cam`), and DataForge's RoboCap rig frame
//!   `/world/rig_00` is the same frame (`dataforge/datasets/robocap.py`: "The rig frame *is* dev0's IMU frame"). So slam-rs's
//!   `world_from_rig` is directly `world_from_rig` for `/world/rig_00` and for the hands' rig extrinsics (`Rig::cam_from_rig`).
//! - slam-rs's world frame: z opposite gravity (the first accel sample at or after the first frameset is rotated onto +z by
//!   the minimal rotation, so yaw is the IMU's heading then), origin at the rig position at initialisation, metres. After a
//!   reset (a gap of [`SlamConfig::reset_gap_ns`] in time, e.g. a replay loop) a new world starts.
//! - Times are integer nanoseconds on one clock for frames and IMU (CLOCK_MONOTONIC live; the dump's times in replay); the
//!   calibration's `cam_time_offset_ns` is 0.

use std::time::Instant;

use kornia_image::Image;
use nalgebra::Isometry3;
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::{ImageView, Vio, VioStatus};

use crate::frame::{ImuSample, SLAM_CAMERAS, SMALL_SIZE, isometry_from_array, isometry_from_matrix};

/// Cap A's 4-camera calibration at 640x360 (Basalt JSON, KB4, `T_imu_cam`), as PR #270 embeds it.
pub const CAP_A_CALIBRATION: &str = include_str!("../../../../slam-rs/configs/robocap_calib_downscale3.json");
/// Cap B's, made the same way from its factory Kalibr set (`fe6fede545c972fa`): `T_imu_cam = inv(T_cam_imu)`, `f / 3`,
/// `c' = (c + 0.5) / 3 - 0.5`, the `imu_mid_0` noise model. The same conversion rebuilds [`CAP_A_CALIBRATION`] exactly.
pub const CAP_B_CALIBRATION: &str = include_str!("../../../../slam-rs/configs/robocap_cap_b_calib_downscale3.json");
/// The VIO configuration PR #270's live adapter starts from.
pub const MSDMO_CONFIG: &str = include_str!("../../../../slam-rs/configs/msdmo_config.json");

/// Errors of the SLAM stage.
#[derive(Debug, thiserror::Error)]
pub enum SlamError {
    /// slam-rs refused the configuration, calibration or a frameset.
    #[error("slam-rs: {0}")]
    Vio(#[from] slam_rs::VioError),
    /// The calibration or configuration JSON did not parse.
    #[error("slam-rs configuration: {0}")]
    Config(String),
    /// A frameset lacked a SLAM camera or had the wrong size.
    #[error("{0}")]
    Input(String),
}

/// Where world_from_rig comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlamMode {
    /// slam-rs VIO.
    On,
    /// Identity, never `ok`.
    Off,
    /// The dump's `reference_world_from_rig.bin` (parity runs).
    Reference,
}

/// Settings of the SLAM stage.
#[derive(Clone, Debug)]
pub struct SlamConfig {
    /// Target rate: a frameset is selected when it is at least one period (minus [`SlamConfig::rate_tolerance_ns`]) after the
    /// last selected one.
    pub hz: f64,
    /// Slack on the period, so 30 Hz takes every frameset of a 30 fps stream with jitter.
    pub rate_tolerance_ns: i64,
    /// slam-rs frontend threads.
    pub frontend_threads: usize,
    /// A gap this long between selected framesets restarts the estimator (new world).
    pub reset_gap_ns: i64,
    /// The VIO profile ([`SlamProfile::Live`] by default).
    pub profile: SlamProfile,
    /// VIO configuration keys set after the profile, e.g. `("config.optical_flow_max_iterations", 4)`.
    pub overrides: Vec<(String, serde_json::Value)>,
    /// The 4-camera 640x360 Basalt calibration (JSON text): [`CAP_A_CALIBRATION`] by default.
    pub calibration: String,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            hz: 30.0,
            rate_tolerance_ns: 4_000_000,
            frontend_threads: 2,
            reset_gap_ns: 250_000_000,
            profile: SlamProfile::Live,
            overrides: Vec::new(),
            calibration: CAP_A_CALIBRATION.to_string(),
        }
    }
}

pub use robocap_types::{SlamPose, SlamStages, SlamStatus};

/// Picks framesets for the target rate by their timestamps (deterministic in lossless replay).
#[derive(Clone, Copy, Debug)]
pub struct RateSelector {
    period_ns: i64,
    tolerance_ns: i64,
    last_ns: Option<i64>,
}

impl RateSelector {
    /// A selector for `hz` (non-positive or infinite = every frameset).
    pub fn new(hz: f64, tolerance_ns: i64) -> Self {
        let period_ns = if hz.is_finite() && hz > 0.0 { (1e9 / hz) as i64 } else { 0 };
        Self { period_ns, tolerance_ns, last_ns: None }
    }

    /// Whether a frameset at `t_ns` is due; selecting it is a separate [`RateSelector::selected`] call.
    pub fn due(&self, t_ns: i64) -> bool {
        self.last_ns.is_none_or(|last| t_ns - last >= self.period_ns - self.tolerance_ns)
    }

    /// Record that the frameset at `t_ns` was taken.
    pub fn selected(&mut self, t_ns: i64) {
        self.last_ns = Some(t_ns);
    }
}

/// slam-rs VIO with PR #270's live configuration and its visual-support rule.
pub struct SlamEstimator {
    vio: Vio<f32>,
    calibration: Calibration<f64>,
    config: VioConfig,
    threads: usize,
    /// IMU samples dropped because they did not follow the previous one.
    pub imu_unordered: u64,
    /// Estimator restarts.
    pub resets: u64,
}

/// Which VIO profile the SLAM stage runs: a checked-in overlay of VIO configuration keys (`slam-rs/configs/profiles/*.json`),
/// applied over [`MSDMO_CONFIG`] through the same path as [`SlamConfig::overrides`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlamProfile {
    /// PR #270's `fast.json`.
    Fast,
    /// `live.json`, the runtime default: `fast`, then a keyframe's joint solve runs beside the next frameset's
    /// frontend (slam-rs D84) and takes at most 5 LM iterations. Inside every accuracy band. Full s66, lossless, against
    /// `fast`: 1333 tracked framesets both, ATE 6.15 vs 8.73 mm, RPE over 30 framesets 5.6 mm / 0.17 deg vs 8.3 mm / 0.20 deg;
    /// the 10 s clip 167 both, 3.23 vs 3.15 mm. Cap B, full system in realtime (60 s): 29.3 Hz while tracking with the source
    /// at 29.4/s, compute mean 26.0 / p95 54.2 ms (fast: 25.4 Hz, 32.3 / 73.6 ms).
    #[default]
    Live,
    /// `live30.json`, opt-in: `live` with a 70 px detection grid (about 40 % less frontend work). Full s66: 1333
    /// tracked, ATE 6.93 mm; Cap B full system: 29.6 Hz while tracking (= the source), compute mean 18.9 / p95 35.7 ms. Outside
    /// the 10 s clip's band: ATE 5.00 mm (band 3.8 mm), and one of seven 15 s cold-start sub-clips starts tracking 2.3 s later.
    Live30,
}

impl SlamProfile {
    /// The profile's overlay: a JSON object of VIO configuration keys (as in [`SlamConfig::overrides`]) and values.
    pub fn overlay(self) -> &'static str {
        match self {
            Self::Fast => include_str!("../../../../slam-rs/configs/profiles/fast.json"),
            Self::Live => include_str!("../../../../slam-rs/configs/profiles/live.json"),
            Self::Live30 => include_str!("../../../../slam-rs/configs/profiles/live30.json"),
        }
    }

    /// The profile's name (`fast`, `live`, `live30`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Live => "live",
            Self::Live30 => "live30",
        }
    }
}

impl std::str::FromStr for SlamProfile {
    type Err = SlamError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "fast" => Ok(Self::Fast),
            "live" => Ok(Self::Live),
            "live30" => Ok(Self::Live30),
            other => Err(SlamError::Config(format!("unknown SLAM profile {other} (fast, live, live30)"))),
        }
    }
}

/// One `KEY=VALUE` VIO configuration override (`robocap-live --slam-set`, `slam_bench --set`): the value as JSON, or as a
/// string when it is not JSON. The key is resolved when the overrides are applied ([`SlamEstimator::with_profile`]).
///
/// # Errors
///
/// [`SlamError::Config`] without an `=`.
pub fn parse_override(setting: &str) -> Result<(String, serde_json::Value), SlamError> {
    let (key, value) = setting.split_once('=').ok_or_else(|| SlamError::Config(format!("{setting}: expected KEY=VALUE")))?;
    let value = serde_json::from_str(value).unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
    Ok((key.to_string(), value))
}

/// [`MSDMO_CONFIG`] with `profile`'s overlay and then `overrides` applied.
fn profile_config(profile: SlamProfile, overrides: &[(String, serde_json::Value)]) -> Result<VioConfig, SlamError> {
    let config_error = |e: &dyn std::fmt::Display| SlamError::Config(e.to_string());
    let overlay: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(profile.overlay()).map_err(|e| SlamError::Config(format!("profile {}: {e}", profile.as_str())))?;
    let text = VioConfig::from_json_str(MSDMO_CONFIG).and_then(|config| config.to_json_string()).map_err(|e| config_error(&e))?;
    let mut value: serde_json::Value = serde_json::from_str(&text).map_err(|e| config_error(&e))?;
    let fields = value.get_mut("value0").and_then(serde_json::Value::as_object_mut).ok_or_else(|| SlamError::Config("no value0".into()))?;
    for (key, setting) in overlay.iter().chain(overrides.iter().map(|(key, setting)| (key, setting))) {
        // The key as given (`config.x`, `port.x`), bare (`x` = `config.x`), or `config.port.x` (= `port.x`).
        let candidates = [key.clone(), format!("config.{key}"), key.strip_prefix("config.").unwrap_or(key).to_string()];
        let key = candidates.into_iter().find(|key| fields.contains_key(key)).ok_or_else(|| SlamError::Config(format!("unknown VIO configuration key {key}")))?;
        fields.insert(key, setting.clone());
    }
    VioConfig::from_json_str(&value.to_string()).map_err(|e| config_error(&e))
}

impl SlamEstimator {
    /// A 4-camera 640x360 Basalt calibration (`calibration_json`, e.g. [`CAP_A_CALIBRATION`]) with [`MSDMO_CONFIG`], then
    /// `profile`'s overlay, then `overrides` (keys as in the Basalt JSON's `value0`, e.g. `config.optical_flow_max_iterations` or
    /// `port.keyframe_solve_deferred`; a bare key is a `config.` one, and `config.port.X` is read as `port.X`), with `threads`
    /// frontend threads. Build it on the thread that will run it, after pinning that thread: the frontend's rayon pool inherits
    /// its CPU affinity.
    ///
    /// # Errors
    ///
    /// [`SlamError::Config`] for a calibration that does not parse, an unknown key, or a value slam-rs refuses.
    pub fn with_profile(
        calibration_json: &str,
        threads: usize,
        profile: SlamProfile,
        overrides: &[(String, serde_json::Value)],
    ) -> Result<Self, SlamError> {
        let calibration = Calibration::<f64>::from_json_str(calibration_json).map_err(|e| SlamError::Config(e.to_string()))?;
        Self::with_configuration(calibration, profile_config(profile, overrides)?, threads)
    }

    /// Any 4-camera 640x360 calibration and configuration.
    ///
    /// # Errors
    ///
    /// [`SlamError`] when slam-rs refuses them or the rig is not four 640x360 cameras.
    pub fn with_configuration(calibration: Calibration<f64>, config: VioConfig, threads: usize) -> Result<Self, SlamError> {
        if calibration.resolution.len() != SLAM_CAMERAS.len()
            || calibration.resolution.iter().any(|r| r[0] as usize != SMALL_SIZE.width || r[1] as usize != SMALL_SIZE.height)
        {
            return Err(SlamError::Config(format!("expected four 640x360 cameras, got {:?}", calibration.resolution)));
        }
        let vio = Self::build(&calibration, &config, threads)?;
        Ok(Self { vio, calibration, config, threads, imu_unordered: 0, resets: 0 })
    }

    fn build(calibration: &Calibration<f64>, config: &VioConfig, threads: usize) -> Result<Vio<f32>, SlamError> {
        let options = FrontendOptions { threads, ..Default::default() };
        Ok(Vio::with_backend(config.clone(), calibration.clone(), options, slam_rs::Backend::Cpu)?)
    }

    /// Restart the estimator (a new world from the next frameset).
    ///
    /// # Errors
    ///
    /// [`SlamError`] if slam-rs refuses the (unchanged) configuration.
    pub fn reset(&mut self) -> Result<(), SlamError> {
        self.vio = Self::build(&self.calibration, &self.config, self.threads)?;
        self.resets += 1;
        Ok(())
    }

    /// Add an IMU0 sample (rig frame, SI). Samples that do not strictly follow the previous one (slam-rs's
    /// `Vio::last_imu_t_ns`) are dropped and counted.
    ///
    /// # Errors
    ///
    /// [`SlamError::Vio`] on a non-finite sample.
    pub fn push_imu(&mut self, sample: &ImuSample) -> Result<(), SlamError> {
        if self.vio.last_imu_t_ns().is_some_and(|last| sample.t_ns <= last) {
            self.imu_unordered += 1;
            return Ok(());
        }
        self.vio.push_imu(sample.t_ns, sample.gyro, sample.accel)?;
        Ok(())
    }

    /// Whether the IMU reaches past `t_ns` (slam-rs tracks a frameset only then).
    pub fn imu_covers(&self, t_ns: i64) -> bool {
        self.vio.last_imu_t_ns().is_some_and(|last| last > t_ns)
    }

    /// Track one frameset: `images` are the 640x360 small images of cameras `[4, 0, 1, 5]`, in that order.
    ///
    /// # Errors
    ///
    /// [`SlamError::Input`] for a wrong image size; [`SlamError::Vio`] when slam-rs fails (the caller resets).
    pub fn track(&mut self, index: u64, t_ns: i64, images: [&Image<u8, 1>; 4]) -> Result<SlamPose, SlamError> {
        if images.iter().any(|image| image.size() != SMALL_SIZE) {
            return Err(SlamError::Input("SLAM images must be 640x360".into()));
        }
        let views: [ImageView<'_>; 4] = images.map(|image| ImageView {
            width: image.width(),
            height: image.height(),
            stride: image.width(),
            data: image.as_slice(),
        });
        let started = Instant::now();
        let result = self.vio.track(t_ns, &views)?;
        let compute_ms = started.elapsed().as_secs_f64() * 1e3;
        let stats = self.vio.last_stats();
        let landmarks = stats.map_or(0, |s| s.num_landmarks);
        let tracked: usize = stats.map_or(0, |s| s.connected.iter().sum());
        let optimised = stats.is_some_and(|s| s.opt_started);
        let finite = result.pose.is_some_and(|pose| pose.world_from_rig.iter().all(|v| v.is_finite()));
        let frontend = self.vio.frontend_timings();
        let ms = |ns: u64| ns as f64 / 1e6;
        let stages = SlamStages {
            frontend_ms: ms(frontend.flow.pyramid_ns + frontend.flow.detect_ns + frontend.flow.track_ns + frontend.flow.stereo_ns + frontend.imu_ns),
            optimize_ms: stats.map_or(0.0, |s| ms(s.timings.optimize_ns)),
            marginalize_ms: stats.map_or(0.0, |s| ms(s.timings.marginalize_ns)),
            keyframe_ms: stats.map_or(0.0, |s| ms(s.timings.keyframe_ns)),
            keyframe: stats.is_some_and(|s| s.took_keyframe),
            pyramid_ms: ms(frontend.flow.pyramid_ns),
            detect_ms: ms(frontend.flow.detect_ns),
            track_ms: ms(frontend.flow.track_ns),
            stereo_ms: ms(frontend.flow.stereo_ns),
            deferred_ms: self.vio.last_deferred_keyframe().map_or(0.0, |d| ms(d.timings.measure_ns)),
            deferred_wait_ms: ms(self.vio.deferred_wait_ns()),
            keyframe_deferred: stats.is_some_and(|s| s.keyframe_deferred),
        };
        let status = match result.status {
            VioStatus::NeedMoreImu => SlamStatus::WaitingForImu,
            VioStatus::Tracking if landmarks >= 10 && tracked >= 10 && optimised && finite => SlamStatus::Tracking,
            VioStatus::Tracking | VioStatus::Buffered | VioStatus::NoVisualFeatures => SlamStatus::NoVisualFeatures,
        };
        let world_from_rig = result.pose.filter(|_| finite).map_or_else(Isometry3::identity, |pose| isometry_from_array(&pose.world_from_rig));
        Ok(SlamPose {
            index,
            t_ns,
            world_from_rig,
            ok: status == SlamStatus::Tracking,
            status,
            compute_ms,
            landmarks,
            tracked,
            optimised,
            stages,
            resets: self.resets,
        })
    }
}

/// Looks up the dump's reference poses by time (`--slam reference`), undoing a replay loop's time shift.
pub struct ReferencePoses {
    poses: Vec<(i64, [f64; 16])>,
    first_t_ns: i64,
    span_ns: Option<i64>,
}

impl ReferencePoses {
    /// `poses` from `read_reference_poses`; `first_t_ns` and `span_ns` from the replay (loop shift = k * span).
    pub fn new(mut poses: Vec<(i64, [f64; 16])>, first_t_ns: i64, span_ns: Option<i64>) -> Self {
        // Framesets without a reference pose carry NaN rows in the dump (s66-full: 7 of 1362): drop them.
        poses.retain(|(_, matrix)| matrix.iter().all(|v| v.is_finite()));
        poses.sort_by_key(|(t, _)| *t);
        Self { poses, first_t_ns, span_ns }
    }

    /// The reference pose within 2 ms of `t_ns`, if any.
    pub fn at(&self, t_ns: i64) -> Option<Isometry3<f64>> {
        let t = match self.span_ns {
            Some(span) if span > 0 && t_ns >= self.first_t_ns => self.first_t_ns + (t_ns - self.first_t_ns) % span,
            _ => t_ns,
        };
        let at = self.poses.partition_point(|(pose_t, _)| *pose_t < t);
        [at.checked_sub(1), Some(at)]
            .into_iter()
            .flatten()
            .filter_map(|i| self.poses.get(i))
            .filter(|(pose_t, _)| (pose_t - t).abs() <= 2_000_000)
            .min_by_key(|(pose_t, _)| (pose_t - t).abs())
            .and_then(|(_, matrix)| isometry_from_matrix(matrix))
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::ImageSize;
    use slam_rs::calib::{CameraModel, PinholeParams};
    use slam_rs::lie::{Se3, So3};

    use super::*;

    /// Each profile is its overlay over MSDMO_CONFIG; live30 is live with a 70 px grid; the names round-trip.
    #[test]
    fn profiles_are_their_overlays_and_live30_is_live_with_a_70_px_grid() -> Result<(), Box<dyn std::error::Error>> {
        for profile in [SlamProfile::Fast, SlamProfile::Live, SlamProfile::Live30] {
            let written: serde_json::Value = serde_json::from_str(&profile_config(profile, &[])?.to_json_string()?)?;
            let overlay: serde_json::Value = serde_json::from_str(profile.overlay())?;
            for (key, value) in overlay.as_object().ok_or("profile is not an object")? {
                let key = key.strip_prefix("config.").filter(|rest| rest.starts_with("port.")).unwrap_or(key);
                assert_eq!(&written["value0"][key], value, "{} {key}", profile.as_str());
            }
            assert_eq!(profile.as_str().parse::<SlamProfile>()?, profile);
        }
        let live = profile_config(SlamProfile::Live, &[])?;
        assert!(live.port_keyframe_solve_deferred && live.vio_max_iterations == 5);
        let mut live70 = live;
        live70.optical_flow_detection_grid_size = 70;
        assert_eq!(profile_config(SlamProfile::Live30, &[])?, live70);
        assert!("reference".parse::<SlamProfile>().is_err());
        Ok(())
    }

    #[test]
    fn overrides_parse_json_values_or_strings() -> Result<(), SlamError> {
        assert_eq!(parse_override("config.optical_flow_max_iterations=4")?, ("config.optical_flow_max_iterations".into(), serde_json::json!(4)));
        assert_eq!(parse_override("port.keyframe_solve_deferred=false")?.1, serde_json::json!(false));
        assert_eq!(parse_override("x=a=b")?, ("x".into(), serde_json::json!("a=b")), "the first = splits; a value that is not JSON is a string");
        assert!(parse_override("no_value").is_err());
        Ok(())
    }

    #[test]
    fn overrides_take_port_keys_with_or_without_a_config_prefix() -> Result<(), Box<dyn std::error::Error>> {
        let off = |key: &str| vec![(key.to_string(), serde_json::Value::Bool(false))];
        for key in ["port.keyframe_solve_deferred", "config.port.keyframe_solve_deferred"] {
            assert!(!profile_config(SlamProfile::Live, &off(key))?.port_keyframe_solve_deferred, "{key}");
        }
        let four = |key: &str| vec![(key.to_string(), serde_json::json!(4))];
        for key in ["config.optical_flow_max_iterations", "optical_flow_max_iterations"] {
            assert_eq!(profile_config(SlamProfile::Live, &four(key))?.optical_flow_max_iterations, 4, "{key}");
        }
        assert!(matches!(SlamEstimator::with_profile(CAP_A_CALIBRATION, 1, SlamProfile::Live, &off("config.no_such_key")), Err(SlamError::Config(_))));
        Ok(())
    }

    #[test]
    fn the_rate_selector_takes_every_frame_at_30_hz_and_every_other_at_15() {
        let period = 33_333_333;
        for (hz, expected) in [(30.0, 30), (15.0, 15), (10.0, 10), (0.0, 30)] {
            let mut selector = RateSelector::new(hz, 4_000_000);
            let mut taken = 0;
            for frame in 0..30 {
                let t = frame * period + if frame % 2 == 0 { 1_500_000 } else { 0 };
                if selector.due(t) {
                    selector.selected(t);
                    taken += 1;
                }
            }
            assert_eq!(taken, expected, "{hz} Hz");
        }
    }

    #[test]
    fn reference_poses_are_found_by_time_across_loops() {
        let pose = |x: f64| {
            let mut m = [0.0; 16];
            for d in 0..4 {
                m[5 * d] = 1.0;
            }
            m[3] = x;
            m
        };
        let reference = ReferencePoses::new(vec![(1_000, pose(1.0)), (34_000_000, pose(2.0)), (60_000_000, [f64::NAN; 16])], 1_000, Some(100_000_000));
        let x = |t: i64| reference.at(t).map(|p| p.translation.x);
        assert_eq!(x(1_500_000), Some(1.0));
        assert_eq!(x(34_000_000 + 100_000_000 + 1_000_000), Some(2.0));
        assert_eq!(x(17_000_000), None);
        assert_eq!(x(60_000_000), None, "a NaN reference pose is no pose");
    }

    #[test]
    fn each_cap_gets_its_own_factory_calibration() -> Result<(), Box<dyn std::error::Error>> {
        let calibration = |device: &str| crate::capture::Cap::from_device(device).map(crate::capture::Cap::slam_calibration);
        assert_eq!(calibration("cap_a"), Some(CAP_A_CALIBRATION));
        assert_eq!(calibration("cap_b"), Some(CAP_B_CALIBRATION));
        assert_eq!(calibration("cap_c"), None);
        // left_front (SLAM camera 1) at 640x360 = the factory Kalibr fx at 1920x1080 over 3: Cap A 636.4361 (its
        // camchain-imucam.yaml), Cap B 612.4247 (fe6fede545c972fa/imus_cam_lr_front_extrinsic).
        let fx = |slam: &SlamEstimator| match &slam.calibration.intrinsics[1] {
            CameraModel::Kb4(kb4) => kb4.fx,
            _ => f64::NAN,
        };
        let cap_a = SlamEstimator::with_profile(CAP_A_CALIBRATION, 1, SlamProfile::Live, &[])?;
        let cap_b = SlamEstimator::with_profile(CAP_B_CALIBRATION, 1, SlamProfile::Live, &[])?;
        assert!((fx(&cap_a) - 636.4360961914062 / 3.0).abs() < 1e-3, "Cap A fx {}", fx(&cap_a));
        assert!((fx(&cap_b) - 612.4247264335909 / 3.0).abs() < 1e-9, "Cap B fx {}", fx(&cap_b));
        assert!(SlamEstimator::with_profile("{}", 1, SlamProfile::Live, &[]).is_err());
        Ok(())
    }

    #[test]
    fn the_cap_a_estimator_builds_and_refuses_a_wrong_size() -> Result<(), Box<dyn std::error::Error>> {
        let mut slam = SlamEstimator::with_profile(CAP_A_CALIBRATION, 2, SlamProfile::Live, &[])?;
        let small = Image::<u8, 1>::from_size_val(ImageSize { width: 320, height: 180 }, 0)?;
        assert!(matches!(slam.track(0, 0, [&small, &small, &small, &small]), Err(SlamError::Input(_))));
        slam.push_imu(&ImuSample { t_ns: 10, gyro: [0.0; 3], accel: [0.0, 0.0, 9.81] })?;
        slam.push_imu(&ImuSample { t_ns: 10, gyro: [0.0; 3], accel: [0.0, 0.0, 9.81] })?;
        assert_eq!(slam.imu_unordered, 1);
        assert!(slam.imu_covers(9) && !slam.imu_covers(10));
        slam.reset()?;
        assert!(!slam.imu_covers(9), "a reset estimator has no IMU");
        slam.push_imu(&ImuSample { t_ns: 5, gyro: [0.0; 3], accel: [0.0, 0.0, 9.81] })?;
        assert_eq!(slam.imu_unordered, 1, "an earlier time is accepted after a reset");
        Ok(())
    }

    /// PR #270's `a_known_stationary_textured_rig_stays_at_its_origin`: a synthetic textured plane at 3 m seen by four pinhole
    /// cameras 8 cm apart, a stationary IMU; the estimator must stay within 2 cm of its origin.
    #[test]
    fn a_known_stationary_textured_rig_stays_at_its_origin() -> Result<(), Box<dyn std::error::Error>> {
        let mut calibration = Calibration::<f64>::from_json_str(CAP_A_CALIBRATION)?;
        for camera in 0..4 {
            calibration.t_i_c[camera] = Se3::new(So3::identity(), [camera as f64 * 0.08, 0.0, 0.0].into());
            calibration.intrinsics[camera] = CameraModel::Pinhole(PinholeParams { fx: 300.0, fy: 300.0, cx: 320.0, cy: 180.0 });
        }
        let config = VioConfig::from_json_str(MSDMO_CONFIG)?;
        let mut slam = SlamEstimator::with_configuration(calibration, config, 4)?;
        let images: Vec<Image<u8, 1>> = (0..4usize)
            .map(|camera| {
                let pixels = (0..640 * 360)
                    .map(|i| {
                        let x = i % 640 + camera * 8;
                        let y = i / 640;
                        let hash = ((x / 5) as u32).wrapping_mul(73856093) ^ ((y / 5) as u32).wrapping_mul(19349663);
                        (40 + hash % 180) as u8
                    })
                    .collect();
                Image::new(SMALL_SIZE, pixels)
            })
            .collect::<Result<_, _>>()?;
        let mut poses = Vec::new();
        let mut index = 0;
        for tick in 0..=1200_i64 {
            let t = 1_000_000_000 + tick * 5_000_000;
            slam.push_imu(&ImuSample { t_ns: t, gyro: [0.0; 3], accel: [0.0, 0.0, 9.81] })?;
            if tick % 14 == 8 {
                let frame_t = t - 6_000_000;
                assert!(slam.imu_covers(frame_t));
                let pose = slam.track(index, frame_t, [&images[0], &images[1], &images[2], &images[3]])?;
                index += 1;
                if pose.ok {
                    poses.push(pose.world_from_rig.translation.vector.norm());
                }
            }
        }
        assert!(poses.len() > 60, "insufficient visually supported poses: {}", poses.len());
        let largest = poses.iter().cloned().fold(0.0f64, f64::max);
        assert!(largest < 0.02, "stationary rig moved {largest} m");
        Ok(())
    }
}
