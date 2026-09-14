use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::{CaptureIdentity, DirectWriter, MotionSample, VideoSample};

struct NextPart {
    writer: DirectWriter,
    identity: CaptureIdentity,
    switched: [bool; 13],
}

/// Rotates a full six-camera, three-IMU, one-MAG recording without resampling.
///
/// Cameras cross a boundary at their first IDR after it; sensors cross at their
/// first sample after it. At most two files are open. The old file closes once
/// all streams have crossed, so late-arriving samples need no replay or queue.
/// Estimator state is deliberately outside this owner and survives rotation.
pub struct SegmentedWriter {
    directory: PathBuf,
    identity: CaptureIdentity,
    current: DirectWriter,
    next: Option<NextPart>,
    duration_ns: i64,
    display: Option<crate::DisplayAssets>,
    last_input: [Option<i64>; 13],
    #[cfg(feature = "live-slam")]
    last_slam: Option<crate::SlamReport>,
}

impl SegmentedWriter {
    pub fn checkpoint(&self, timeout: std::time::Duration) -> Result<()> {
        self.current.checkpoint(timeout)?;
        if let Some(next) = &self.next {
            next.writer.checkpoint(timeout)?;
        }
        Ok(())
    }
    pub fn create(
        directory: &Path,
        identity: CaptureIdentity,
        duration_ns: i64,
        display: Option<crate::DisplayAssets>,
    ) -> Result<Self> {
        ensure!(duration_ns > 0, "segment duration must be positive");
        identity
            .start_ns
            .checked_add(duration_ns)
            .context("segment deadline overflow")?;
        let writer = DirectWriter::create(
            &directory.join(format!("part-{:04}.rrd", identity.part)),
            identity.clone(),
            display.as_ref(),
        )?;
        Ok(Self {
            directory: directory.to_path_buf(),
            identity,
            current: writer,
            next: None,
            duration_ns,
            display,
            last_input: [None; 13],
            #[cfg(feature = "live-slam")]
            last_slam: None,
        })
    }

    pub fn video(&mut self, sample: VideoSample<'_>) -> Result<()> {
        ensure!(sample.camera < 6, "unknown camera index");
        let stream = usize::from(sample.camera);
        self.check_order(stream, sample.timestamp_ns)?;
        let deadline = self
            .identity
            .start_ns
            .checked_add(self.duration_ns)
            .context("segment deadline overflow")?;
        let crosses = sample.keyframe && sample.timestamp_ns >= deadline;
        let rotating = self.next.is_some() || crosses;
        let duration_ns = self.duration_ns;
        let timestamp = sample.timestamp_ns;
        let writer = self.route(stream, crosses)?;
        if rotating {
            ensure!(
                timestamp - deadline < duration_ns,
                "rotation stalled waiting for a capture stream"
            );
        }
        writer.video(sample)?;
        self.last_input[stream] = Some(timestamp);
        self.complete_transition()
    }

    fn route(&mut self, stream: usize, crosses: bool) -> Result<&mut DirectWriter> {
        if self.next.is_none() && crosses {
            ensure!(
                self.last_input.iter().all(Option::is_some),
                "cannot rotate: a capture stream never started"
            );
            let mut identity = self.identity.clone();
            identity.part = identity
                .part
                .checked_add(1)
                .context("part number overflow")?;
            identity.start_ns = self
                .identity
                .start_ns
                .checked_add(self.duration_ns)
                .context("segment deadline overflow")?;
            #[allow(unused_mut)]
            let mut writer = DirectWriter::create(
                &self
                    .directory
                    .join(format!("part-{:04}.rrd", identity.part)),
                identity.clone(),
                self.display.as_ref(),
            )?;
            #[cfg(feature = "live-slam")]
            if let Some(report) = &self.last_slam {
                writer.slam(report)?;
            }
            self.next = Some(NextPart {
                writer,
                identity,
                switched: [false; 13],
            });
        }
        if let Some(next) = &mut self.next
            && (next.switched[stream] || crosses)
        {
            next.switched[stream] = true;
            return Ok(&mut next.writer);
        }
        Ok(&mut self.current)
    }

    pub fn motion(&mut self, sample: MotionSample) -> Result<()> {
        let stream = sample.stream_index()?;
        self.check_order(stream, sample.timestamp_ns)?;
        let timestamp = sample.timestamp_ns;
        let deadline = self
            .identity
            .start_ns
            .checked_add(self.duration_ns)
            .context("segment deadline overflow")?;
        self.route(stream, timestamp >= deadline)?.motion(sample)?;
        self.last_input[stream] = Some(timestamp);
        self.complete_transition()
    }

    /// Results retain their original time and update number. New parts are
    /// seeded with the previous result so they open independently. A session
    /// reader can deduplicate that boundary seed by timestamp/update number.
    #[cfg(feature = "live-slam")]
    pub fn slam(&mut self, report: &crate::SlamReport) -> Result<()> {
        if let Some(next) = &mut self.next {
            next.writer.slam(report)?;
        } else {
            self.current.slam(report)?;
        }
        self.last_slam = Some(report.clone());
        Ok(())
    }

    /// Publish every open part, even during an incomplete keyframe transition.
    pub fn finish(self) -> Result<()> {
        ensure!(
            self.last_input.iter().all(Option::is_some),
            "cannot finalize: a capture stream never started"
        );
        self.current.finish()?;
        if let Some(next) = self.next {
            next.writer.finish()?;
        }
        Ok(())
    }

    fn check_order(&self, stream: usize, timestamp: i64) -> Result<()> {
        ensure!(
            timestamp >= self.identity.start_ns || self.last_input[stream].is_some(),
            "sample precedes session start"
        );
        ensure!(
            self.last_input[stream].is_none_or(|last| timestamp > last),
            "capture stream timestamp did not advance"
        );
        Ok(())
    }

    fn complete_transition(&mut self) -> Result<()> {
        if self
            .next
            .as_ref()
            .is_some_and(|next| next.switched.iter().all(|&switched| switched))
            && let Some(next) = self.next.take()
        {
            let finished = std::mem::replace(&mut self.current, next.writer);
            self.identity = next.identity;
            finished.finish()?;
        }
        Ok(())
    }
}
