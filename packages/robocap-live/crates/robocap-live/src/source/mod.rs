//! Frame sources: the live cameras + IMU on the cap, or a `robocap-live-dump/1` replay. Both yield [`SourceEvent`]s in time order
//! (IMU samples up to a frameset's time come before it, as slam-rs needs). The pull shape follows kornia-slam's app `FrameSource`
//! (`next_frame() -> Result<Option<_>, SourceError>`), generalised to a six-camera rig with integer-nanosecond time.
#![deny(missing_docs)]

#[cfg(target_os = "linux")]
pub mod live;
pub mod replay;

use crate::frame::{FrameError, Rig};

pub use crate::frame::SourceEvent;

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

/// Anything that produces framesets and IMU samples in time order.
pub trait FrameSource: Send {
    /// The rig the frames come from.
    fn rig(&self) -> &Rig;
    /// The next event, or `None` when the source is exhausted (replay end, or a stop request).
    ///
    /// # Errors
    ///
    /// [`SourceError`] when a device or the dump fails; the source is unusable afterwards.
    fn next_event(&mut self) -> Result<Option<SourceEvent>, SourceError>;
}
