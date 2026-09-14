//! Bounded live input adapter for the existing four-camera CPU estimator.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use slam_rs::{
    ImageView, Vio, VioStatus, calib::Calibration, config::VioConfig,
    frontend::flow::FrontendOptions,
};
use std::{collections::VecDeque, time::Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuChannel {
    Gyro,
    Accel,
}

/// Left, left_front, right_front, right in estimator order.
pub const SLAM_CAMERAS: [u8; 4] = [4, 0, 1, 5];

#[derive(Debug, PartialEq)]
pub enum SlamInput {
    /// Camera index in estimator order: left, left-front, right-front, right.
    Frame {
        camera: usize,
        timestamp_ns: i64,
        pixels: Vec<u8>,
    },
    /// Gyro or accelerometer from IMU 0.
    Imu {
        channel: ImuChannel,
        timestamp_ns: i64,
        xyz: [f64; 3],
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlamStatus {
    WaitingForImu,
    NoVisualFeatures,
    TrackingProvisional,
    Failed(String),
}

impl std::fmt::Display for SlamStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WaitingForImu => f.write_str("waiting_for_imu"),
            Self::NoVisualFeatures => f.write_str("no_visual_features"),
            Self::TrackingProvisional => f.write_str("tracking_provisional"),
            Self::Failed(reason) => write!(f, "failed: {reason}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamReport {
    pub timestamp_ns: i64,
    pub pose: Option<[f64; 7]>,
    pub status: SlamStatus,
    pub processing_ms: f64,
    pub latency_ms: f64,
    pub landmarks: usize,
    pub tracked_observations: usize,
    pub optimization_started: bool,
    pub updates: u64,
}

struct Frame {
    timestamp_ns: i64,
    pixels: Vec<u8>,
}
type ImuSample = (i64, [f64; 3]);

#[derive(Default)]
pub struct LiveSlamOptions {
    pub profile: bool,
    pub joint: bool,
}

/// Apply the recorder's fast estimator schedule (configs/profiles/fast.json).
pub fn fast_profile(config: &mut VioConfig) {
    config.vio_max_iterations = 7;
    config.port_redetect_survivor_ratio = 0.85;
    config.port_frame_update_max_iterations = 5;
}

pub struct LiveSlam {
    vio: Vio<f32>,
    resolution: [[u32; 2]; 4],
    frames: [VecDeque<Frame>; 4],
    gyro: VecDeque<ImuSample>,
    accel: VecDeque<ImuSample>,
    last_imu: Option<i64>,
    last_frame: Option<i64>,
    updates: u64,
    profile: bool,
}

impl LiveSlam {
    /// Cap A calibration is deliberately provisional on Cap B. The recorder
    /// persists the original document and both serials in every part.
    pub fn cap_a_fast_profile(options: LiveSlamOptions) -> Result<Self> {
        let calibration = Calibration::<f64>::from_json_str(include_str!(
            "../../../configs/robocap_calib_downscale3.json"
        ))?;
        let mut config =
            VioConfig::from_json_str(include_str!("../../../configs/msdmo_config.json"))?;
        fast_profile(&mut config);
        Self::with_configuration(calibration, config, options)
    }

    /// The same live adapter can be checked against an independently calibrated
    /// four-camera rig. Device capture still supplies Cap B's fixed geometry.
    pub fn with_configuration(
        calibration: Calibration<f64>,
        mut config: VioConfig,
        options: LiveSlamOptions,
    ) -> Result<Self> {
        // Diagnostic comparison with the existing joint optimizer changes only
        // this estimator's schedule, never camera/sensor capture.
        if options.joint {
            config.port_frame_update_max_iterations = 0;
        }
        let resolution: [[u32; 2]; 4] = calibration
            .resolution
            .clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("live adapter requires four cameras"))?;
        let vio = Vio::with_backend(
            config,
            calibration,
            FrontendOptions {
                threads: 4,
                ..Default::default()
            },
            slam_rs::Backend::Cpu,
        )?;
        Ok(Self {
            vio,
            resolution,
            frames: std::array::from_fn(|_| VecDeque::new()),
            gyro: VecDeque::new(),
            accel: VecDeque::new(),
            last_imu: None,
            last_frame: None,
            updates: 0,
            profile: options.profile,
        })
    }

    pub fn push(&mut self, input: SlamInput, now_ns: i64) -> Result<Option<SlamReport>> {
        match input {
            SlamInput::Frame {
                camera,
                timestamp_ns,
                pixels,
            } => {
                ensure!(
                    camera < 4
                        && pixels.len()
                            == self.resolution[camera][0] as usize
                                * self.resolution[camera][1] as usize,
                    "invalid SLAM image"
                );
                let queue = &mut self.frames[camera];
                ensure!(
                    queue
                        .back()
                        .is_none_or(|last| timestamp_ns > last.timestamp_ns),
                    "unordered SLAM camera"
                );
                if queue.len() == 8 {
                    queue.pop_front();
                }
                queue.push_back(Frame {
                    timestamp_ns,
                    pixels,
                });
            }
            SlamInput::Imu {
                channel,
                timestamp_ns,
                xyz,
            } => {
                ensure!(xyz.iter().all(|v| v.is_finite()), "nonfinite SLAM IMU");
                let queue = match channel {
                    ImuChannel::Gyro => &mut self.gyro,
                    ImuChannel::Accel => &mut self.accel,
                };
                ensure!(
                    queue.back().is_none_or(|last| timestamp_ns > last.0),
                    "unordered SLAM IMU"
                );
                // Failure is contained in the SLAM child. Raw recording continues.
                ensure!(queue.len() < 512, "SLAM IMU backlog exceeded its bound");
                queue.push_back((timestamp_ns, xyz));
            }
        }
        loop {
            let Some(t) = self.frames.iter().try_fold(i64::MIN, |latest, queue| {
                Some(latest.max(queue.front()?.timestamp_ns))
            }) else {
                return Ok(None);
            };
            let mut dropped = false;
            for queue in &mut self.frames {
                while queue.front().is_some_and(|f| {
                    f.timestamp_ns < t - 3_000_000
                        || now_ns - f.timestamp_ns > 250_000_000
                        || self
                            .last_frame
                            .is_some_and(|last| f.timestamp_ns - last < 60_000_000)
                }) {
                    queue.pop_front();
                    dropped = true;
                }
            }
            if dropped {
                continue;
            }
            // Interpolate accelerometer samples onto each gyro timestamp. Keep
            // the bracketing sample, and only advance the estimator just past t.
            while self.last_imu.is_none_or(|last| last <= t) {
                let Some(&(gt, gyro)) = self.gyro.front() else {
                    return Ok(None);
                };
                if self.accel.front().is_some_and(|a| a.0 > gt) {
                    self.gyro.pop_front();
                    continue;
                }
                while self.accel.get(1).is_some_and(|a| a.0 <= gt) {
                    self.accel.pop_front();
                }
                let (Some(&(at, a)), Some(&(bt, b))) = (self.accel.front(), self.accel.get(1))
                else {
                    return Ok(None);
                };
                ensure!(bt - at <= 50_000_000, "SLAM accelerometer gap");
                let alpha = (gt - at) as f64 / (bt - at) as f64;
                let accel = std::array::from_fn(|i| a[i] + alpha * (b[i] - a[i]));
                self.vio.push_imu(gt, gyro, accel)?;
                self.last_imu = Some(gt);
                self.gyro.pop_front();
            }
            let frames = self
                .frames
                .iter_mut()
                .map(|q| q.pop_front().context("matched frame disappeared"))
                .collect::<Result<Vec<_>>>()?;
            let views = frames
                .iter()
                .enumerate()
                .map(|(camera, f)| ImageView {
                    width: self.resolution[camera][0] as usize,
                    height: self.resolution[camera][1] as usize,
                    stride: self.resolution[camera][0] as usize,
                    data: &f.pixels,
                })
                .collect::<Vec<_>>();
            let start = Instant::now();
            let result = self.vio.track(t, &views)?;
            let processing_ms = start.elapsed().as_secs_f64() * 1000.0;
            if self.profile {
                let frontend = self.vio.frontend_timings();
                let stages = self.vio.last_stats().map(|stats| &stats.timings);
                eprintln!(
                    "slam_profile={}",
                    serde_json::json!({
                        "timestamp_ns": t,
                        "input_age_ms": (now_ns - t) as f64 / 1e6,
                        "processing_ms": processing_ms,
                        "took_keyframe": self.vio.last_stats().map(|s| s.took_keyframe),
                        "pyramid_ns": frontend.pyramid_ns,
                        "detect_ns": frontend.detect_ns,
                        "track_ns": frontend.track_ns,
                        "stereo_ns": frontend.stereo_ns,
                        "imu_ns": frontend.imu_ns,
                        "predict_ns": stages.map(|s| s.predict_ns),
                        "keyframe_ns": stages.map(|s| s.keyframe_ns),
                        "optimize_ns": stages.map(|s| s.optimize_ns),
                        "linearize_ns": stages.map(|s| s.linearize_ns),
                        "solver_ns": stages.map(|s| s.solver_ns),
                        "back_substitution_ns": stages.map(|s| s.back_substitution_ns),
                        "error_ns": stages.map(|s| s.error_ns),
                        "marginalize_ns": stages.map(|s| s.marginalize_ns),
                        "measure_ns": stages.map(|s| s.measure_ns),
                    })
                );
            }
            let landmarks = self.vio.last_stats().map_or(0, |s| s.num_landmarks);
            let tracked_observations = self
                .vio
                .last_stats()
                .map_or(0, |s| s.connected.iter().sum());
            let optimization_started = self.vio.last_stats().is_some_and(|s| s.opt_started);
            let supported = result.status == VioStatus::Tracking
                && landmarks >= 10
                && tracked_observations >= 10
                && optimization_started
                && result.world_from_rig.iter().all(|v| v.is_finite());
            self.last_frame = Some(t);
            self.updates += 1;
            return Ok(Some(SlamReport {
                timestamp_ns: t,
                pose: supported.then_some(result.world_from_rig),
                status: if result.status == VioStatus::NeedMoreImu {
                    SlamStatus::WaitingForImu
                } else if !supported {
                    SlamStatus::NoVisualFeatures
                } else {
                    SlamStatus::TrackingProvisional
                },
                processing_ms,
                latency_ms: (now_ns - t) as f64 / 1e6 + processing_ms,
                landmarks,
                tracked_observations,
                optimization_started,
                updates: self.updates,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};

    /// The typed overlay must say exactly what the checked-in profile file says.
    #[test]
    fn fast_profile_matches_the_checked_in_profile_file() -> Result<()> {
        let file: serde_json::Value =
            serde_json::from_str(include_str!("../../../configs/profiles/fast.json"))?;
        let mut config = slam_rs::config::VioConfig::from_json_str(include_str!(
            "../../../configs/msdmo_config.json"
        ))?;
        super::fast_profile(&mut config);
        let written: serde_json::Value = serde_json::from_str(&config.to_json_string()?)?;
        for (key, value) in file.as_object().context("profile is not an object")? {
            assert_eq!(&written["value0"][key], value, "{key}");
        }
        Ok(())
    }
}
