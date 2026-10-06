//! RoboCap capture format and queue policy.
use crate::frame::FULL_SIZE;
use kornia_staging_io::v4l::mplane::{CaptureFormat, MplaneError, PlaneLayout};

/// Native luma bytes in one cap camera frame.
pub const LUMA_BYTES: usize = FULL_SIZE.width * FULL_SIZE.height;
/// Driver buffers retained while readers hold zero-copy images.
pub const MIN_QUEUED: u32 = 2;

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
