//! The logger as one of the output stage's sinks: the live binary's `--viewer`/`--save` and the hands catalog layer
//! both log through it.

use std::sync::{Arc, Mutex, PoisonError};

use super::{FrameLog, LogStats, Logger};
use crate::sched::{FramesetSink, OutputRecord, SinkError};

/// Hands every frameset's record to a [`Logger`] and finishes it at the end of the run.
pub struct LoggerSink {
    logger: Option<Logger>,
    finished: Arc<Mutex<Option<LogStats>>>,
}

impl LoggerSink {
    /// Sink into `logger`.
    pub fn new(logger: Logger) -> Self {
        Self { logger: Some(logger), finished: Arc::default() }
    }

    /// Where [`FramesetSink::finish`] leaves the logger's final counters (the sink itself moves into the pipeline).
    pub fn final_stats(&self) -> Arc<Mutex<Option<LogStats>>> {
        self.finished.clone()
    }
}

impl FramesetSink for LoggerSink {
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError> {
        let Some(logger) = self.logger.as_mut() else { return Ok(()) };
        // The logger draws a pose only when it is a usable one.
        let frame = FrameLog {
            t_ns: record.frameset.t_ns,
            small: std::array::from_fn(|c| record.small[c].as_ref()),
            world_from_rig: record.pose.filter(|pose| pose.ok).map(|pose| &pose.world_from_rig),
            slam_status: record.pose.map_or("none", |pose| pose.status.as_str()),
            hands: record.hands,
            timings: record.timings,
        };
        logger.log_frameset(&frame).map_err(|e| SinkError { sink: "rerun".into(), message: e.to_string() })
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        let Some(logger) = self.logger.take() else { return Ok(()) };
        let (stats, encoders) = logger.finish().map_err(|e| SinkError { sink: "rerun".into(), message: e.to_string() })?;
        eprintln!("robocap-live: rerun logger {stats:?}");
        for encoder in encoders {
            eprintln!("robocap-live: encoder {encoder:?}");
        }
        *self.finished.lock().unwrap_or_else(PoisonError::into_inner) = Some(stats);
        Ok(())
    }
}
