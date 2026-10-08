//! Sensor frames and source contracts on a monotonic integer clock.
use crate::imu::CombinedImuSample;
use kornia_image::Image;
use std::sync::Arc;

/// Capture metadata independent of device-specific controls and image geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureMeta {
    /// Driver sequence number or source frame index.
    pub sequence: u64,
    /// Monotonic capture timestamp, integer nanoseconds.
    pub timestamp_ns: i64,
    /// Source camera index.
    pub camera_slot: usize,
}
/// A camera frame at its captured resolution.
#[derive(Clone)]
pub struct CameraFrame {
    /// Capture time and sequence.
    pub meta: CaptureMeta,
    /// Captured luma; resolution comes from the image itself.
    pub full: Arc<Image<u8, 1>>,
}
/// Available camera frames for one trigger instant, indexed by the rig's camera order.
#[derive(Clone)]
pub struct Frameset {
    /// Source frameset index.
    pub index: u64,
    /// Earliest present camera timestamp, monotonic nanoseconds.
    pub timestamp_ns: i64,
    /// Runtime camera slots; missing frames have `None`.
    pub cameras: Vec<Option<CameraFrame>>,
}
/// Ordered sensor event; IMU coverage up to a frameset precedes that frameset.
#[derive(Clone)]
pub enum SourceEvent {
    /// Camera frames for one trigger instant.
    Frameset(Frameset),
    /// Combined gyro and acceleration.
    Imu(CombinedImuSample),
}
