//! Framesets handed over by another thread as a [`FrameSource`]: a caller that already holds the frames (the Python binding,
//! fed from a Rerun catalog segment) runs them through [`crate::sched::run`] like the cameras or a replay. It carries no IMU, so
//! pair it with `SlamMode::Reference` (or `Off`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::time::Duration;

use super::{FrameSource, SourceError, SourceEvent};
use crate::frame::{Frameset, Rig};

/// How often a source waiting for the next frameset looks at the stop flag.
const STOP_POLL: Duration = Duration::from_millis(50);

/// The receiving end: yields each frameset sent, and ends when every sender is gone or `stop` is set.
pub struct ChannelSource {
    rig: Rig,
    framesets: Receiver<Frameset>,
    stop: Arc<AtomicBool>,
}

/// A source for `rig` and the sender that feeds it; a send blocks while `depth` framesets wait.
pub fn channel(rig: Rig, depth: usize, stop: Arc<AtomicBool>) -> (SyncSender<Frameset>, ChannelSource) {
    let (sender, framesets) = sync_channel(depth);
    (sender, ChannelSource { rig, framesets, stop })
}

impl FrameSource for ChannelSource {
    fn rig(&self) -> &Rig {
        &self.rig
    }

    fn next_event(&mut self) -> Result<Option<SourceEvent>, SourceError> {
        while !self.stop.load(Ordering::Relaxed) {
            match self.framesets.recv_timeout(STOP_POLL) {
                Ok(frameset) => return Ok(Some(SourceEvent::Frameset(frameset))),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(None),
            }
        }
        Ok(None)
    }
}
