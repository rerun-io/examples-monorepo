//! A hands catalog layer: the runtime's own pipeline ([`crate::sched::run`]) on framesets handed over by the caller.
//!
//! An offline caller that already holds a segment's frames (robocap-live-py, fed from a Rerun catalog segment) pushes them into
//! a [`channel`] source, and the pipeline runs them exactly as `robocap-live --source replay <dump> --slam reference` runs a
//! dump: lossless queues, the downsample stage's 640x360 images, the reference poses through the pose store (a frameset without
//! its own pose takes the newest usable one), the hands stage's tracker step on the networks, and the output stage handing each
//! record to the logger's [`LoggerSink`] in [`Content::HandsLayer`] mode. The layer's recording id is the segment's and its
//! `video_time` is the catalog's (origin 0), so it registers beside the segment's `base` and `slam_rs` layers.
#![deny(missing_docs)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

use crate::frame::{CameraFrame, Frameset, NUM_CAMERAS, Rig};
use crate::hands::{self, HandTimings, HandsConfig, HandsError};
use crate::log::scene::{Content, HandOverlays};
use crate::log::{LogError, LogStats, Logger, LoggerConfig, LoggerSink, VideoMode};
use crate::nets::HandNets;
use crate::sched::{self, FramesetSink, HandsStage, OutputRecord, PipelineConfig, RunSummary, SchedError, SinkError};
use crate::slam::{ReferencePoses, SlamMode};
use crate::source::channel::channel;

/// Framesets a push may run ahead of the pipeline's source stage (the stages' own queues hold more).
const PUSH_QUEUE: usize = 2;

/// Errors of the layer writer.
#[derive(Debug, thiserror::Error)]
pub enum LayerError {
    /// The tracker could not be built.
    #[error("hands: {0}")]
    Hands(#[from] HandsError),
    /// The logger could not be opened.
    #[error("log: {0}")]
    Log(#[from] LogError),
    /// The pipeline failed.
    #[error("pipeline: {0}")]
    Sched(#[from] SchedError),
    /// The caller supplied invalid input.
    #[error("{0}")]
    Input(String),
    /// The pipeline could not complete the layer.
    #[error("{0}")]
    Incomplete(String),
}

/// How to write a layer.
#[derive(Clone, Debug)]
pub struct HandsLayerConfig {
    /// The `.rrd` to write (created; an existing file is truncated, so never point this at a registered layer).
    pub output: PathBuf,
    /// Recording id: the catalog segment id.
    pub recording_id: String,
    /// How much of the hand pipeline the camera panes show.
    pub hand_overlays: HandOverlays,
    /// The tracker's settings.
    pub hands: HandsConfig,
    /// Threads for the 640x360 images.
    pub downsample_threads: usize,
}

/// What the output stage saw, counted by [`Tally`].
#[derive(Clone, Debug, Default)]
pub struct LayerCounts {
    /// Framesets through the output stage.
    pub framesets: u64,
    /// Framesets tracked on their own reference pose.
    pub with_pose: u64,
    /// Framesets without one, tracked on an earlier frameset's (the pose store's newest usable pose).
    pub held_pose: u64,
    /// Per hand (left, right): framesets on which it was tracked.
    pub tracked: [u64; 2],
    /// Per hand: framesets on which it was reported (tracked and confirmed; only reported hands are drawn).
    pub reported: [u64; 2],
    /// The hand scale phi at the end.
    pub scale: f64,
    /// Whether the live scale calibration finished.
    pub scale_final: bool,
    /// The tracker's split of its steps, summed, milliseconds.
    pub hand_timings: HandTimings,
}

/// A sink beside the logger that counts what the layer holds.
struct Tally(Arc<Mutex<LayerCounts>>);

impl FramesetSink for Tally {
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError> {
        let mut counts = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        counts.framesets += 1;
        match record.pose.filter(|pose| pose.ok) {
            Some(pose) if pose.index == record.frameset.index => counts.with_pose += 1,
            Some(_) => counts.held_pose += 1,
            None => {}
        }
        if let Some(hands) = record.hands {
            for (side, hand) in hands.hands.iter().enumerate() {
                counts.tracked[side] += u64::from(hand.tracked);
                counts.reported[side] += u64::from(hand.reported);
            }
            (counts.scale, counts.scale_final) = (hands.scale, hands.scale_final);
            let total = &mut counts.hand_timings;
            total.detnet_ms += hands.timings.detnet_ms;
            total.crops_ms += hands.timings.crops_ms;
            total.keynet_ms += hands.timings.keynet_ms;
            total.fit_ms += hands.timings.fit_ms;
            total.tracker_ms += hands.timings.tracker_ms;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

/// A finished layer: what it holds, the pipeline's own summary and the logger's counters.
#[derive(Debug)]
pub struct LayerSummary {
    /// What the output stage saw.
    pub counts: LayerCounts,
    /// The scheduler's summary (per-stage timings, counters).
    pub run: RunSummary,
    /// The logger's counters.
    pub log: LogStats,
}

/// Writes one segment's hands layer: [`HandsLayerWriter::push`] every frameset in time order, then [`HandsLayerWriter::finish`].
pub struct HandsLayerWriter {
    framesets: Option<SyncSender<Frameset>>,
    run: Option<JoinHandle<Result<RunSummary, SchedError>>>,
    stop: Arc<AtomicBool>,
    counts: Arc<Mutex<LayerCounts>>,
    log_stats: Arc<Mutex<Option<LogStats>>>,
    next_index: u64,
    last_t_ns: Option<i64>,
}

impl HandsLayerWriter {
    /// Build the tracker for `rig` on `nets`, open the layer file and start the pipeline.
    ///
    /// `reference` holds the segment's SLAM poses keyed by frameset time (`ReferencePoses::at` takes the one within 2 ms).
    ///
    /// # Errors
    ///
    /// [`LayerError::Hands`] for a bad rig or tracker setting, [`LayerError::Log`] when the file cannot be opened,
    /// [`LayerError::Incomplete`] when the logger refuses to save (its filesystem is below the free-space floor) or the pipeline
    /// thread cannot start.
    pub fn new(rig: &Rig, nets: Box<dyn HandNets>, reference: ReferencePoses, config: HandsLayerConfig) -> Result<Self, LayerError> {
        let tracker = hands::new_tracker(rig, config.hands)?;
        let output = config.output.clone();
        let options = LoggerConfig {
            save: Some(config.output),
            video: VideoMode::Off,
            recording_id: Some(config.recording_id),
            time_origin_ns: Some(0),
            hand_overlays: config.hand_overlays,
            content: Content::HandsLayer,
            lossless: true,
            ..LoggerConfig::default()
        };
        let logger = Logger::new(rig, options)?;
        if logger.stats().save_stopped_low_disk {
            return Err(LayerError::Incomplete(format!("{}: too little free space to write the layer", output.display())));
        }
        let sink = LoggerSink::new(logger);
        let log_stats = sink.final_stats();
        let counts = Arc::new(Mutex::new(LayerCounts::default()));
        let sinks: Vec<Box<dyn FramesetSink>> = vec![Box::new(sink), Box::new(Tally(counts.clone()))];
        let stop = Arc::new(AtomicBool::new(false));
        let (framesets, source) = channel(rig.clone(), PUSH_QUEUE, stop.clone());
        // `robocap-live --source replay <dump> --slam reference --hands on` with the CLI's defaults.
        let pipeline = PipelineConfig {
            lossless: true,
            slam_mode: SlamMode::Reference,
            reference: Some(reference),
            hands: Some(HandsStage { tracker, nets, nets_factory: None }),
            downsample_threads: config.downsample_threads.max(1),
            ..PipelineConfig::default()
        };
        let run_stop = stop.clone();
        let run = std::thread::Builder::new()
            .name("rl-layer".into())
            .spawn(move || sched::run(Box::new(source), pipeline, sinks, run_stop))
            .map_err(|error| LayerError::Incomplete(format!("pipeline thread: {error}")))?;
        Ok(Self {
            framesets: Some(framesets),
            run: Some(run),
            stop,
            counts,
            log_stats,
            next_index: 0,
            last_t_ns: None,
        })
    }

    /// Hand one frameset to the pipeline; blocks while the pipeline is [`PUSH_QUEUE`] framesets behind.
    ///
    /// # Errors
    ///
    /// [`LayerError::Input`] for a time that does not increase, and the pipeline's error when it
    /// has stopped.
    pub fn push(&mut self, t_ns: i64, mut cameras: [Option<CameraFrame>; NUM_CAMERAS]) -> Result<u64, LayerError> {
        let index = self.next_index;
        if let Some(last) = self.last_t_ns.filter(|&last| t_ns <= last) {
            return Err(LayerError::Input(format!("frameset {index} at {t_ns} ns does not follow {last} ns")));
        }
        for camera in cameras.iter_mut().flatten() {
            camera.meta.seq = index;
        }
        let frameset = Frameset { index, t_ns, cameras };
        let Some(framesets) = &self.framesets else { return Err(LayerError::Input("the layer is finished".into())) };
        if framesets.send(frameset).is_err() {
            // The source stage has ended: the pipeline stopped on an error.
            self.framesets = None;
            return Err(self.join().err().unwrap_or_else(|| LayerError::Incomplete("the pipeline ended early".into())));
        }
        self.next_index += 1;
        self.last_t_ns = Some(t_ns);
        Ok(index)
    }

    /// Wait for the pipeline thread.
    fn join(&mut self) -> Result<RunSummary, LayerError> {
        let run = self.run.take().ok_or_else(|| LayerError::Input("the layer is finished".into()))?;
        Ok(run.join().map_err(|_| LayerError::Incomplete("the pipeline thread panicked".into()))??)
    }

    /// End the input, drain the pipeline and close the layer file.
    ///
    /// # Errors
    ///
    /// The pipeline's error, [`LayerError::Input`] when no frameset was pushed, and [`LayerError::Incomplete`] when a frameset did not
    /// reach the output or was dropped by the logger, the hands stage or a sink counted an error, or saving stopped for low disk.
    pub fn finish(mut self) -> Result<LayerSummary, LayerError> {
        drop(self.framesets.take());
        let run = self.join()?;
        let counts = self.counts.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let log = self.log_stats.lock().unwrap_or_else(PoisonError::into_inner).take().ok_or_else(|| LayerError::Incomplete("the logger did not finish".into()))?;
        let c = &run.counters;
        let pipeline_errors = c.hands_errors + c.nets_failures + c.hands_without_nets + c.hands_bypassed + c.output_late + c.output_errors;
        let dropped = run.downsample_dropped + run.hands_dropped + run.output_dropped;
        if self.next_index == 0 {
            return Err(LayerError::Input("no frameset was pushed".into()));
        }
        if counts.framesets != self.next_index || pipeline_errors + dropped > 0 || log.save_stopped_low_disk || log.framesets_dropped > 0 || log.errors > 0 {
            return Err(LayerError::Incomplete(format!(
                "the layer is incomplete: {} of {} framesets reached the output, {pipeline_errors} hands/output errors, {dropped} dropped in the \
                 pipeline, {} dropped by the logger, {} logger errors, saving stopped for low disk: {}",
                counts.framesets, self.next_index, log.framesets_dropped, log.errors, log.save_stopped_low_disk
            )));
        }
        Ok(LayerSummary { counts, run, log })
    }
}

impl Drop for HandsLayerWriter {
    /// A layer dropped before [`HandsLayerWriter::finish`] stops its pipeline and leaves an incomplete file.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        drop(self.framesets.take());
        if let Some(run) = self.run.take() {
            let _ = run.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use kornia_image::Image;

    use super::*;
    use crate::frame::{FULL_SIZE, FrameMeta};
    use crate::nets::NoNets;

    use crate::log::tests::test_rig;

    fn cameras(t_ns: i64, cameras: &[usize]) -> Result<[Option<CameraFrame>; NUM_CAMERAS], kornia_image::ImageError> {
        let image = Arc::new(Image::<u8, 1>::from_size_val(FULL_SIZE, 90)?);
        let frames: [Option<CameraFrame>; NUM_CAMERAS] = std::array::from_fn(|camera| {
            cameras.contains(&camera).then(|| CameraFrame {
                meta: FrameMeta { seq: 0, pts_ns: t_ns, source_id: camera as u32, turned_180: false },
                full: image.clone(),
            })
        });
        Ok(frames)
    }

    fn config(output: PathBuf) -> HandsLayerConfig {
        HandsLayerConfig {
            output,
            recording_id: "segment".into(),
            hand_overlays: HandOverlays::Debug,
            hands: HandsConfig { scale_wait: true, ..HandsConfig::default() },
            downsample_threads: 2,
        }
    }

    #[test]
    fn a_layer_runs_its_framesets_through_the_pipeline_in_order_and_refuses_the_rest() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-layer-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let output = dir.join("layer.rrd");
        let pose = [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0];
        // Framesets 33 ms apart, a pose for the first only (a reference pose matches within 2 ms): the second is tracked on
        // the first one's, the pose store's newest usable pose.
        let (t0, t1) = (1_000_000_000, 1_033_000_000);
        let reference = ReferencePoses::new(vec![(t0, pose), (t1, [f64::NAN; 16])], t0, None);
        let mut layer = HandsLayerWriter::new(&test_rig(), Box::new(NoNets), reference, config(output.clone()))?;
        assert_eq!(layer.push(t0, cameras(t0, &[0, 1, 2, 3, 4, 5])?)?, 0);
        assert_eq!(layer.push(t1, cameras(t1, &[0, 2])?)?, 1);
        assert!(matches!(layer.push(t1, cameras(t1, &[0])?), Err(LayerError::Input(_))), "time does not increase");
        let summary = layer.finish()?;
        let counts = &summary.counts;
        assert_eq!((counts.framesets, counts.with_pose, counts.held_pose, counts.reported), (2, 1, 1, [0, 0]));
        assert_eq!((summary.log.framesets_in, summary.run.framesets), (2, 2));
        assert!(output.metadata()?.len() > 0);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_layer_dropped_unfinished_stops_its_pipeline() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-layer-drop-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let mut layer = HandsLayerWriter::new(&test_rig(), Box::new(NoNets), ReferencePoses::new(Vec::new(), 0, None), config(dir.join("layer.rrd")))?;
        layer.push(1_000, cameras(1_000, &[0])?)?;
        drop(layer);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
