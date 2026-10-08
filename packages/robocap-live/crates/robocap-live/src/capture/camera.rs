//! RoboCap capture format and queue policy.
use crate::frame::FULL_SIZE;
use kornia_staging_io::v4l::mplane::{CaptureFormat, MplaneError, PlaneLayout};

/// Native luma bytes in one cap camera frame.
pub const LUMA_BYTES: usize = FULL_SIZE.width * FULL_SIZE.height;
/// Reserve three of eight buffers outside downstream leases. During the next DQBUF/copy,
/// at least two remain driver-owned. Completed (not yet dequeued) buffers can still consume
/// that reserve; this is not a promise about the driver's instantaneous empty queue.
/// Five leases instead of six costs no extra mapped memory; pressure falls back to luma copies.
pub const MIN_QUEUED: u32 = 3;

/// The cap's one-plane NV12 layout.
/// # Errors
/// Returns a format error if the app's geometry is invalid.
pub fn capture_format() -> Result<CaptureFormat, MplaneError> {
    CaptureFormat::new(
        FULL_SIZE,
        u32::from_le_bytes(*b"NV12"),
        vec![PlaneLayout {
            stride: FULL_SIZE.width,
            bytes: LUMA_BYTES * 3 / 2,
        }],
    )
}
