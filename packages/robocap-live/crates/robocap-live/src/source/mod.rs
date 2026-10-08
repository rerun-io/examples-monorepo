//! Frame sources: the live cameras + IMU on the cap, a `robocap-live-dump/1` replay, or framesets handed over by another thread
//! ([`channel`]). Each yields [`kornia_staging_sensors::SourceEvent`]s in time order (IMU samples up to a frameset's time come before it, as slam-rs
//! needs). The pull shape follows kornia-slam's app `FrameSource`
//! (`next_frame() -> Result<Option<_>, SourceError>`), generalised to a six-camera rig with integer-nanosecond time.
#![deny(missing_docs)]

pub mod channel;
#[cfg(target_os = "linux")]
pub mod live;
pub mod replay;
pub(crate) mod sync;
pub use sync::CaptureSync;
mod health;
pub use health::{CaptureHealth, CaptureSnapshot};

use crate::frame::{FrameError, NUM_CAMERAS, Rig};
use kornia_staging_sensors::SourceEvent;

/// Errors of a frame source.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    /// Reading the shared types or the dump format failed.
    #[error(transparent)]
    Frame(#[from] FrameError),
    /// A camera, the frame trigger or an IMU device failed.
    #[error("device: {0}")]
    Device(String),
    /// The source was asked to stop or lost its input.
    #[error("stopped: {0}")]
    Stopped(String),
}

/// Application source with calibrated rig metadata and mounting orientation.
pub trait FrameSource: Send {
    /// Calibrated six-camera rig.
    fn rig(&self) -> &Rig;
    /// Cameras whose stored pixels are turned relative to calibration.
    fn turned_180(&self) -> [bool; NUM_CAMERAS] {
        [false; NUM_CAMERAS]
    }
    /// Live capture health shared with the monitor; replay never recovers a trigger.
    fn capture_health(&self) -> Option<CaptureHealth> { None }
    /// Read the next event; `None` means EOF or requested stop.
    /// # Errors
    /// Returns a capture, replay, or source shutdown error.
    fn next_event(&mut self) -> Result<Option<SourceEvent>, SourceError>;
}
