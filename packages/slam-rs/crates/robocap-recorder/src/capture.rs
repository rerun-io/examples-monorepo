use std::{path::Path, sync::Arc, time::Duration};

use anyhow::{Result, ensure};
use arrow_array::{
    ArrayRef, BooleanArray, Int32Array, Int64Array, StringArray, UInt32Array, UInt64Array,
};
use rerun::{
    ComponentDescriptor, RecordingStream, RecordingStreamBuilder, SerializedComponentBatch,
    TimeCell,
};

use crate::{CAMERAS, DurableRrdSink, SENSORS};
use sha2::{Digest, Sha256};

/// Original calibration input, retained separately from the target device identity.
/// The estimator owns parsing and validating this document before using it.
#[derive(Clone)]
pub struct CalibrationSource {
    pub device_serial: String,
    pub document: String,
    pub placeholder: bool,
}

/// Stable device/session identity and the part's boot-relative start time.
#[derive(Clone)]
pub struct CaptureIdentity {
    pub device_serial: String,
    pub session: String,
    pub part: u32,
    pub start_ns: i64,
    pub calibration: Option<CalibrationSource>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionKind {
    Gyro,
    Accel,
    Mag,
}

/// An original sensor sample. No interpolation or clock correction is applied.
#[derive(Clone)]
pub struct MotionSample {
    pub device: u8,
    pub kind: MotionKind,
    pub timestamp_ns: i64,
    pub sequence: u64,
    pub raw: [i32; 3],
    pub temperature_raw: Option<i16>,
    /// Verified conversion from counts to rad/s, m/s², or tesla, respectively.
    /// None means the physical scale is unknown; only raw counts are recorded.
    pub scale: Option<f64>,
}

impl MotionSample {
    /// Index after the six cameras: three gyro/accel pairs, then magnetometer.
    pub fn stream_index(&self) -> Result<usize> {
        SENSORS
            .iter()
            .position(|channel| channel.device == self.device && channel.kind == self.kind)
            .map(|index| CAMERAS.len() + index)
            .ok_or_else(|| {
                anyhow::anyhow!(match self.kind {
                    MotionKind::Gyro | MotionKind::Accel => "unknown IMU index",
                    MotionKind::Mag => "unknown magnetometer index",
                })
            })
    }
}

/// One complete hardware-encoded H.264 access unit, with no B-frame reordering.
pub struct VideoSample<'a> {
    /// Canonical order: left_front, right_front, left_eye, right_eye, left, right.
    pub camera: u8,
    pub timestamp_ns: i64,
    pub sequence: u64,
    pub keyframe: bool,
    pub annex_b: &'a [u8],
}

/// Persists live capture samples directly, using Dataforge's entity paths.
///
/// This writer does not configure or acquire hardware, align IMU clocks, or
/// select samples for an estimator. Those belong at their respective boundaries.
pub struct DirectWriter {
    recording: RecordingStream,
    sink: DurableRrdSink,
    last_video: [Option<(i64, u64)>; 6],
    #[cfg(feature = "live-slam")]
    last_pose: Option<[f32; 3]>,
}

impl DirectWriter {
    pub fn create(
        path: &Path,
        identity: CaptureIdentity,
        display: Option<&crate::DisplayAssets>,
    ) -> Result<Self> {
        ensure!(
            !identity.device_serial.is_empty() && !identity.session.is_empty(),
            "device and session identities are required"
        );
        ensure!(
            identity.part > 0 && identity.start_ns >= 0,
            "invalid recording part or monotonic start"
        );
        if let Some(calibration) = &identity.calibration {
            ensure!(
                !calibration.device_serial.is_empty() && !calibration.document.trim().is_empty(),
                "calibration source identity and document are required"
            );
            ensure!(
                calibration.placeholder || calibration.device_serial == identity.device_serial,
                "a calibration from another device must be marked as a placeholder"
            );
        }
        let sink = DurableRrdSink::create(path)?;
        let recording = RecordingStreamBuilder::new("robocap")
            .recording_id(format!(
                "{}-{}-part-{:04}",
                identity.device_serial, identity.session, identity.part
            ))
            .set_sinks(vec![Box::new(sink.clone()) as Box<dyn rerun::sink::LogSink>])?;
        let calibration_status = match &identity.calibration {
            None => "unavailable",
            Some(source) if source.placeholder => "provisional",
            Some(_) => "provided",
        };
        if let Some(source) = &identity.calibration {
            recording.log_serialized_batches(
                "/__properties/calibration",
                true,
                [
                    field(
                        "source_device_serial",
                        Arc::new(StringArray::from(vec![source.device_serial.as_str()])),
                    ),
                    field(
                        "target_device_serial",
                        Arc::new(StringArray::from(vec![identity.device_serial.as_str()])),
                    ),
                    field(
                        "source_document",
                        Arc::new(StringArray::from(vec![source.document.as_str()])),
                    ),
                    field(
                        "source_sha256",
                        Arc::new(StringArray::from(vec![format!(
                            "{:x}",
                            Sha256::digest(source.document.as_bytes())
                        )])),
                    ),
                    field(
                        "is_placeholder",
                        Arc::new(BooleanArray::from(vec![source.placeholder])),
                    ),
                ],
            )?;
        }
        recording.log_serialized_batches(
            "/__properties/capture",
            true,
            [
                field(
                    "device_serial",
                    Arc::new(StringArray::from(vec![identity.device_serial])),
                ),
                field(
                    "session",
                    Arc::new(StringArray::from(vec![identity.session])),
                ),
                field("part", Arc::new(UInt32Array::from(vec![identity.part]))),
                field(
                    "start_ns",
                    Arc::new(Int64Array::from(vec![identity.start_ns])),
                ),
                field(
                    "clock",
                    Arc::new(StringArray::from(vec!["device_monotonic_unadjusted"])),
                ),
                field(
                    "format",
                    Arc::new(StringArray::from(vec!["robocap_direct_rrd_v1"])),
                ),
                field(
                    "calibration_status",
                    Arc::new(StringArray::from(vec![calibration_status])),
                ),
            ],
        )?;
        for (camera, spec) in CAMERAS.iter().enumerate() {
            let entity = format!("/world/rig_00/cam_{camera:02}");
            recording.log_serialized_batches(
                entity.as_str(),
                true,
                [field("name", Arc::new(StringArray::from(vec![spec.name])))],
            )?;
            recording.log_static(
                format!("{entity}/pinhole/video"),
                &rerun::VideoStream::new(rerun::components::VideoCodec::H264),
            )?;
        }
        if let Some(display) = display {
            display.send(&recording)?;
        }
        #[cfg(feature = "live-slam")]
        recording.log_static("/world", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;
        Ok(Self {
            recording,
            sink,
            last_video: [None; 6],
            #[cfg(feature = "live-slam")]
            last_pose: None,
        })
    }

    pub fn video(&mut self, sample: VideoSample<'_>) -> Result<()> {
        ensure!(sample.camera < 6, "unknown camera index");
        ensure!(
            sample.timestamp_ns >= 0 && !sample.annex_b.is_empty(),
            "invalid video sample"
        );
        let previous = self.last_video[usize::from(sample.camera)];
        if let Some((time, sequence)) = previous {
            ensure!(
                sample.timestamp_ns > time && sample.sequence > sequence,
                "video timestamp or sequence did not advance"
            );
        } else {
            ensure!(
                sample.keyframe
                    && has_nal(sample.annex_b, 7)
                    && has_nal(sample.annex_b, 8)
                    && has_nal(sample.annex_b, 5),
                "a camera must start each part with SPS, PPS, and IDR in Annex B format"
            );
        }
        let entity = format!("/world/rig_00/cam_{:02}/pinhole/video", sample.camera);
        self.recording.set_time(
            "video_time",
            TimeCell::from_duration_nanos(sample.timestamp_ns),
        );
        self.recording.log(
            entity.as_str(),
            &rerun::VideoStream::update_fields()
                .with_sample(sample.annex_b.to_vec())
                .with_is_keyframe(sample.keyframe),
        )?;
        self.recording.log_serialized_batches(
            entity.as_str(),
            false,
            [
                field(
                    "source_timestamp_ns",
                    Arc::new(Int64Array::from(vec![sample.timestamp_ns])),
                ),
                field(
                    "source_sequence",
                    Arc::new(UInt64Array::from(vec![sample.sequence])),
                ),
            ],
        )?;
        self.last_video[usize::from(sample.camera)] = Some((sample.timestamp_ns, sample.sequence));
        Ok(())
    }

    pub fn motion(&mut self, sample: MotionSample) -> Result<()> {
        ensure!(sample.timestamp_ns >= 0, "negative sensor timestamp");
        ensure!(
            sample
                .scale
                .is_none_or(|scale| scale.is_finite() && scale > 0.0),
            "invalid sensor scale"
        );
        let entity = match sample.kind {
            MotionKind::Gyro | MotionKind::Accel => {
                ensure!(sample.device < 3, "unknown IMU index");
                let suffix = if matches!(sample.kind, MotionKind::Gyro) {
                    "gyro"
                } else {
                    "accel"
                };
                format!("/world/rig_00/imu_{:02}/{suffix}", sample.device)
            }
            MotionKind::Mag => {
                ensure!(sample.device == 0, "unknown magnetometer index");
                "/world/rig_00/mag_00".to_owned()
            }
        };
        self.recording.set_time(
            "video_time",
            TimeCell::from_duration_nanos(sample.timestamp_ns),
        );
        self.recording.log_serialized_batches(
            entity.as_str(),
            false,
            [
                field(
                    "raw_counts",
                    Arc::new(Int32Array::from(sample.raw.to_vec())),
                ),
                field(
                    "source_timestamp_ns",
                    Arc::new(Int64Array::from(vec![sample.timestamp_ns])),
                ),
                field(
                    "source_sequence",
                    Arc::new(UInt64Array::from(vec![sample.sequence])),
                ),
            ],
        )?;
        self.recording.log(
            format!("{entity}/raw"),
            &rerun::Scalars::new(sample.raw.map(f64::from)),
        )?;
        if let Some(temperature) = sample.temperature_raw {
            self.recording.log_serialized_batches(
                entity.as_str(),
                false,
                [field(
                    "temperature_raw",
                    Arc::new(Int32Array::from(vec![i32::from(temperature)])),
                )],
            )?;
        }
        if let Some(scale) = sample.scale {
            self.recording.log(
                entity,
                &rerun::Scalars::new(sample.raw.map(|value| f64::from(value) * scale)),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "live-slam")]
    pub fn slam(&mut self, report: &crate::SlamReport) -> Result<()> {
        self.recording.set_time(
            "video_time",
            TimeCell::from_duration_nanos(report.timestamp_ns),
        );
        self.recording.log(
            "/derived/slam/status",
            &rerun::TextLog::new(report.status.to_string()),
        )?;
        self.recording.log(
            "/derived/slam/processing_ms",
            &rerun::Scalars::new([report.processing_ms]),
        )?;
        self.recording.log(
            "/derived/slam/latency_ms",
            &rerun::Scalars::new([report.latency_ms]),
        )?;
        self.recording.log(
            "/derived/slam/landmarks",
            &rerun::Scalars::new([report.landmarks as f64]),
        )?;
        self.recording.log(
            "/derived/slam/updates",
            &rerun::Scalars::new([report.updates as f64]),
        )?;
        if let Some(pose) = report.pose {
            ensure!(pose.iter().all(|v| v.is_finite()), "nonfinite SLAM pose");
            self.recording.log(
                "/world/rig_00",
                &rerun::Transform3D::from_translation_rotation(
                    [pose[0] as f32, pose[1] as f32, pose[2] as f32],
                    rerun::Quaternion::from_xyzw([
                        pose[3] as f32,
                        pose[4] as f32,
                        pose[5] as f32,
                        pose[6] as f32,
                    ]),
                ),
            )?;
            let position = [pose[0] as f32, pose[1] as f32, pose[2] as f32];
            if let Some(previous) = self.last_pose {
                // Persist each edge once. The blueprint accumulates the overview
                // and limits the follow trail to its cursor-relative window.
                let edge = rerun::LineStrips3D::new([vec![previous, position]])
                    .with_colors([rerun::Color::from_rgb(255, 180, 40)])
                    .with_radii([0.004]);
                self.recording
                    .log("/world/runs/slam_rs/trajectory", &edge)?;
                self.recording.log("/world/runs/slam_rs/trail", &edge)?;
            }
            self.last_pose = Some(position);
        } else {
            self.last_pose = None;
            self.recording.log("/world/rig_00", &rerun::Clear::flat())?;
        }
        Ok(())
    }

    /// Synchronize queued records without publishing a completed file.
    /// The capture coordinator should call this at least once per second.
    /// The timeout bounds SDK dispatch waiting, not an OS storage syscall.
    pub fn checkpoint(&self, timeout: Duration) -> Result<()> {
        self.recording.flush_with_timeout(timeout)?;
        Ok(())
    }

    /// Synchronize queued records and publish only a complete RRD.
    pub fn finish(self) -> Result<()> {
        self.checkpoint(Duration::from_secs(30))?;
        self.sink.finish()
    }
}

fn field(name: &'static str, array: ArrayRef) -> SerializedComponentBatch {
    SerializedComponentBatch::new(array, ComponentDescriptor::partial(name))
}

fn has_nal(bytes: &[u8], kind: u8) -> bool {
    // A four-byte start code includes the three-byte suffix. Emulation
    // prevention keeps this delimiter out of an encoded NAL payload.
    bytes
        .windows(4)
        .any(|window| window[..3] == [0, 0, 1] && window[3] & 0x1f == kind)
}
