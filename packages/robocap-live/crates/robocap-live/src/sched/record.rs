//! What the output stage hands its sinks, the sink trait, and the `--record` JSONL writer (SPEC "Runtime").

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use nalgebra::Isometry3;
use serde::Serialize;

use crate::downsample::SmallImages;
use crate::frame::matrix_from_isometry;
use crate::hands::HandFrameResult;
use crate::slam::SlamPose;
use kornia_staging_sensors::Frameset;

/// Per-frameset stage timings, milliseconds; NaN (`null` in the record) for a stage that did not run for this frameset.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FrameTimings {
    /// Small images of the frameset.
    pub downsample_ms: f64,
    /// The call that produced this frameset's own pose; see [`crate::slam::SlamStages`].
    pub slam_ms: f64,
    /// Of `slam_ms`: the frontend.
    pub slam_frontend_ms: f64,
    /// Of `slam_ms`: the LM loop.
    pub slam_optimize_ms: f64,
    /// Of `slam_ms`: marginalisation.
    pub slam_marginalize_ms: f64,
    /// Waiting for a pose (in the hands stage, or in the output stage for a frameset that skipped it).
    pub pose_wait_ms: f64,
    /// `HandTracking::step`, when it produced hands.
    pub hands_ms: f64,
    /// DetNet (letterbox + net + decode).
    pub detnet_ms: f64,
    /// Crop planning and sampling.
    pub crops_ms: f64,
    /// KeyNet.
    pub keynet_ms: f64,
    /// handfit fits.
    pub fit_ms: f64,
    /// Tracker bookkeeping.
    pub tracker_ms: f64,
    /// Source emission -> output start.
    pub pipeline_ms: f64,
}

impl FrameTimings {
    /// A frameset's timings after its small images: every other stage not run (NaN).
    pub fn downsampled(downsample_ms: f64) -> Self {
        Self {
            downsample_ms,
            slam_ms: f64::NAN,
            slam_frontend_ms: f64::NAN,
            slam_optimize_ms: f64::NAN,
            slam_marginalize_ms: f64::NAN,
            pose_wait_ms: f64::NAN,
            hands_ms: f64::NAN,
            detnet_ms: f64::NAN,
            crops_ms: f64::NAN,
            keynet_ms: f64::NAN,
            fit_ms: f64::NAN,
            tracker_ms: f64::NAN,
            pipeline_ms: f64::NAN,
        }
    }

    /// The pose lookup's share: the wait, and `Vio::track`'s times when `pose` is frameset `index`'s own and SLAM computed it.
    pub(super) fn set_pose(&mut self, pose: Option<&SlamPose>, index: u64, waited: Duration) {
        self.pose_wait_ms = waited.as_secs_f64() * 1e3;
        if let Some(pose) = pose.filter(|pose| pose.index == index && pose.compute_ms > 0.0) {
            self.slam_ms = pose.compute_ms;
            self.slam_frontend_ms = pose.stages.frontend_ms;
            self.slam_optimize_ms = pose.stages.optimize_ms;
            self.slam_marginalize_ms = pose.stages.marginalize_ms;
        }
    }

    /// The hands stage's share: the step and the tracker's own stage times.
    pub(super) fn set_hands(&mut self, hands_ms: f64, tracker: &crate::hands::HandTimings) {
        self.hands_ms = hands_ms;
        self.detnet_ms = tracker.detnet_ms;
        self.crops_ms = tracker.crops_ms;
        self.keynet_ms = tracker.keynet_ms;
        self.fit_ms = tracker.fit_ms;
        self.tracker_ms = tracker.tracker_ms;
    }
}

/// Everything the output stage knows about one frameset.
pub struct OutputRecord<'a> {
    /// The frameset (full images).
    pub frameset: &'a Frameset,
    /// The 640x360 images by camera.
    pub small: &'a SmallImages,
    /// The pose used (newest at or before the frameset), if any.
    pub pose: Option<&'a SlamPose>,
    /// The hands, when the hands stage ran.
    pub hands: Option<&'a HandFrameResult>,
    /// Stage timings.
    pub timings: &'a FrameTimings,
}

impl OutputRecord<'_> {
    /// `world_from_rig` of the record (identity without a pose).
    pub fn world_from_rig(&self) -> Isometry3<f64> {
        self.pose
            .map_or_else(Isometry3::identity, |pose| pose.world_from_rig)
    }
}

/// Errors of an output sink.
#[derive(Debug, thiserror::Error)]
#[error("{sink}: {message}")]
pub struct SinkError {
    /// Which sink.
    pub sink: String,
    /// What happened.
    pub message: String,
}

/// Something the output stage hands every frameset to (the `--record` writer, the Rerun logger).
pub trait FramesetSink: Send {
    /// Take one frameset. Must not block for long: the output queue is latest-wins in realtime runs.
    ///
    /// # Errors
    ///
    /// [`SinkError`] when the sink cannot take it; the output stage counts it and goes on.
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError>;
    /// End of the run.
    ///
    /// # Errors
    ///
    /// [`SinkError`] when the sink cannot finish (e.g. a flush fails); the run then reports it.
    fn finish(&mut self) -> Result<(), SinkError>;
}

#[derive(Serialize)]
struct RecordView {
    camera: usize,
    keypoints_px: Vec<[f32; 2]>,
    presence: f32,
    pinch: Option<f32>,
}

#[derive(Serialize)]
struct RecordHand {
    tracked: bool,
    reported: bool,
    landmarks: Option<Vec<[f64; 3]>>,
    views: Vec<RecordView>,
    detnet_camera: Option<usize>,
    detnet_circle: Option<[f32; 3]>,
}

#[derive(Serialize)]
struct RecordLine<'a> {
    index: u64,
    t_ns: i64,
    slam_index: Option<u64>,
    slam_t_ns: Option<i64>,
    world_from_rig: [f64; 16],
    slam_ok: bool,
    slam_status: &'a str,
    slam_landmarks: usize,
    slam_tracked: usize,
    slam_optimised: bool,
    hands: Vec<RecordHand>,
    detnet_camera: Option<usize>,
    scale: Option<f64>,
    timings_ms: &'a FrameTimings,
}

/// The `--record` JSONL writer (SPEC "Runtime": one line per frameset).
pub struct RecordWriter {
    out: std::io::BufWriter<std::fs::File>,
    path: String,
}

impl RecordWriter {
    /// Create (truncate) the file.
    ///
    /// # Errors
    ///
    /// [`SinkError`] when it cannot be created.
    pub fn create(path: &Path) -> Result<Self, SinkError> {
        let file = std::fs::File::create(path).map_err(|e| SinkError {
            sink: "record".into(),
            message: format!("{}: {e}", path.display()),
        })?;
        Ok(Self {
            out: std::io::BufWriter::new(file),
            path: path.display().to_string(),
        })
    }
}

impl FramesetSink for RecordWriter {
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError> {
        let hands = record
            .hands
            .map(|result| {
                result
                    .hands
                    .iter()
                    .map(|hand| RecordHand {
                        tracked: hand.tracked,
                        reported: hand.reported,
                        landmarks: hand.landmarks_world.map(|l| l.to_vec()),
                        views: hand
                            .fitted_views()
                            .map(|v| RecordView {
                                camera: v.camera,
                                keypoints_px: v.keypoints_px.to_vec(),
                                presence: v.presence,
                                pinch: v.pinch,
                            })
                            .collect(),
                        detnet_camera: hand.detnet_camera,
                        detnet_circle: hand.detnet_circle,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let line = RecordLine {
            index: record.frameset.index,
            t_ns: record.frameset.timestamp_ns,
            slam_index: record.pose.map(|pose| pose.index),
            slam_t_ns: record.pose.map(|pose| pose.t_ns),
            world_from_rig: matrix_from_isometry(&record.world_from_rig()),
            slam_ok: record.pose.is_some_and(|pose| pose.ok),
            slam_status: record.pose.map_or("none", |pose| pose.status.as_str()),
            slam_landmarks: record.pose.map_or(0, |pose| pose.landmarks),
            slam_tracked: record.pose.map_or(0, |pose| pose.tracked),
            slam_optimised: record.pose.is_some_and(|pose| pose.optimised),
            hands,
            detnet_camera: record.hands.and_then(|h| h.detnet_camera),
            scale: record.hands.map(|h| h.scale),
            timings_ms: record.timings,
        };
        let error = |e: &dyn std::fmt::Display| SinkError {
            sink: "record".into(),
            message: format!("{}: {e}", self.path),
        };
        serde_json::to_writer(&mut self.out, &line).map_err(|e| error(&e))?;
        self.out.write_all(b"\n").map_err(|e| error(&e))
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        self.out.flush().map_err(|e| SinkError {
            sink: "record".into(),
            message: format!("{}: {e}", self.path),
        })
    }
}
