//! Rerun logging: the six camera videos, the cap's SLAM pose and trajectory, both hands in 3D and on the camera
//! panes, and per-stage timings, streamed live to a viewer and optionally saved to an `.rrd`, without slowing the pipeline.
//!
//! The runtime calls [`Logger::log_frameset`] from its logging thread. That call never blocks: it hands the frameset to the
//! logger's worker through a bounded queue and drops it (counted) when the worker is behind. Two `RecordingStream`s, as
//! PR #270's agreed design says:
//! - the **preview** stream to the gRPC viewer (`rerun+http://<host>:9876/proxy`). A bounded queue sits in front of it: when
//!   the Wi-Fi link stalls, items are dropped and counted, never waited for; a dropped video sample pauses that camera's
//!   preview until its next keyframe, so the viewer never decodes a broken reference chain. If the viewer goes away the
//!   stream reconnects and re-sends the static scene and layout.
//! - the **save** stream to an `.rrd` file, when asked; it gets every frameset the worker takes.
//!
//! Threads: the worker (encoder input, records), one reader per H.264 encoder (access units to both streams), and the
//! preview sender. Entity layout and the record contents are in [`scene`]; the encoder in [`video`]; the layout asset in
//! [`display`].

pub mod display;
pub mod scene;
pub mod video;

mod preview;
mod sink;
mod worker;

#[cfg(test)]
pub(crate) mod tests;

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nalgebra::Isometry3;
use rerun::RecordingStream;

use crate::sched::FrameTimings;
use crate::frame::{Luma, NUM_CAMERAS, Rig, SMALL_SIZE};
use crate::hands::HandFrameResult;
use display::{APPLICATION_ID, DisplayAssets, DisplayError};
use preview::{PreviewQueue, PreviewSender};
pub use sink::LoggerSink;
use scene::{DeliveredState, RecordState};
use worker::{FrameItem, SaveSlot, VideoSink, Worker};
use kornia_staging_io::video::{EncoderConfig, VideoError};
use video::{EncoderReport, VideoSample};
use worker::EncoderState;
use kornia_staging_io::video::H264Encoder;

/// Errors of the logger.
#[derive(Debug, thiserror::Error)]
pub enum LogError {
    /// Unknown application encoder preset.
    #[error("encoder {0:?}: expected mpp, x264 or openh264")]
    Encoder(String),
    /// The Rerun SDK refused a stream or a component.
    #[error("rerun: {0}")]
    Rerun(#[from] rerun::RecordingStreamError),
    /// The display asset could not be loaded or sent.
    #[error(transparent)]
    Display(#[from] DisplayError),
    /// An H.264 encoder failed.
    #[error(transparent)]
    Video(#[from] VideoError),
    /// Invalid options.
    #[error("{0}")]
    Invalid(String),
    /// The worker thread is gone (it failed earlier; see its error from [`Logger::finish`]).
    #[error("the logging worker has stopped")]
    WorkerGone,
}

/// How the camera images reach the viewer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoMode {
    /// H.264 through the encoder child processes (the cap: ~1 Mbit/s per camera).
    H264,
    /// Raw 640x360 luma images (a local viewer; ~7 MB/s per camera at 30 fps).
    Raw,
    /// No images.
    Off,
}

impl FromStr for VideoMode {
    type Err = LogError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "h264" => Ok(Self::H264),
            "raw" => Ok(Self::Raw),
            "off" => Ok(Self::Off),
            other => Err(LogError::Invalid(format!("video mode {other:?}: expected h264, raw or off"))),
        }
    }
}

/// Logger settings.
#[derive(Clone, Debug)]
pub struct LoggerConfig {
    /// The viewer's gRPC proxy URL, e.g. `rerun+http://<viewer>:9876/proxy`; `None` = no preview stream.
    pub viewer: Option<String>,
    /// Save everything to this `.rrd` too.
    pub save: Option<PathBuf>,
    /// Video mode.
    pub video: VideoMode,
    /// The H.264 encoder command (default: the cap's `mpph264enc` at 1 Mbit/s, GOP 30).
    pub encoder: EncoderConfig,
    /// The display asset (layout + cap mesh) sent at the start of each stream; `None` = the viewer's default layout.
    pub display: Option<PathBuf>,
    /// Recording id; `None` = `robocap-live-<unix seconds>`.
    pub recording_id: Option<String>,
    /// `video_time` = `t_ns - origin`; `None` = the first frameset's `t_ns` (a live session starts at 0).
    pub time_origin_ns: Option<i64>,
    /// Framesets the worker may lag behind before `log_frameset` drops (or, [`LoggerConfig::lossless`], waits).
    pub input_queue: usize,
    /// `log_frameset` waits for the worker instead of dropping a frameset: an offline catalog layer must hold every frameset,
    /// and nothing upstream of it runs in real time.
    pub lossless: bool,
    /// Items (video samples, images, frame records) the preview may lag behind before it drops.
    pub preview_queue: usize,
    /// Bytes the preview stream's SDK pipeline may buffer before it pushes back (bounds the latency after a stall).
    pub preview_max_bytes_in_flight: u64,
    /// How long the preview stream batches rows before sending: `ChunkBatcherConfig::LOW_LATENCY`'s 8 ms, as so100-hackathon's
    /// live view uses (smaller chunks, worse for a saved file, but each frameset goes out as soon as it is logged).
    pub preview_flush: Duration,
    /// The cameras that get a video pane (H.264 encoders or raw images); fewer encoders draw less power on the cap.
    pub video_cameras: Vec<usize>,
    /// Stop saving (and say so) when the save file's filesystem has less free space than this.
    pub save_min_free_bytes: u64,
    /// How much of the hand pipeline the camera panes show (more costs more of the preview link).
    pub hand_overlays: scene::HandOverlays,
    /// The whole scene, or only the hands for a catalog layer ([`scene::Content`]).
    pub content: scene::Content,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            viewer: None,
            save: None,
            video: VideoMode::H264,
            encoder: video::mpp(SMALL_SIZE, 30, 1_000_000, 30).expect("valid fixed preview geometry"),
            display: None,
            recording_id: None,
            time_origin_ns: None,
            input_queue: 4,
            lossless: false,
            preview_queue: 256,
            preview_max_bytes_in_flight: 4 * 1024 * 1024,
            preview_flush: rerun::log::ChunkBatcherConfig::LOW_LATENCY.flush_tick,
            video_cameras: (0..NUM_CAMERAS).collect(),
            save_min_free_bytes: 4 * 1024 * 1024 * 1024,
            hand_overlays: scene::HandOverlays::default(),
            content: scene::Content::Full,
        }
    }
}

/// One frameset for the logger (all borrowed; the logger keeps `Arc` clones of the images, it copies nothing big).
pub struct FrameLog<'a> {
    /// Frameset time, nanoseconds (CLOCK_MONOTONIC live, the catalog's video time in replay).
    pub t_ns: i64,
    /// The 640x360 small images, per camera.
    pub small: [Option<&'a Luma>; NUM_CAMERAS],
    /// SLAM's `world_from_rig`, when it has a pose.
    pub world_from_rig: Option<&'a Isometry3<f64>>,
    /// SLAM's status text (logged when it changes).
    pub slam_status: &'a str,
    /// The hand tracker's result, when hands ran.
    pub hands: Option<&'a HandFrameResult>,
    /// Stage timings, logged on `/timings` ([`scene::TIMING_SERIES`]).
    pub timings: &'a FrameTimings,
}

/// A snapshot of the logger's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogStats {
    /// Framesets passed to `log_frameset`.
    pub framesets_in: u64,
    /// Framesets dropped because the worker was behind.
    pub framesets_dropped: u64,
    /// Items handed to the preview stream.
    pub preview_sent: u64,
    /// Items dropped because the preview queue was full (the link stalled).
    pub preview_dropped: u64,
    /// Video samples withheld from the preview until their camera's next keyframe, after a drop or a reconnect.
    pub preview_gated: u64,
    /// Bytes of video samples and images handed to the preview stream.
    pub preview_payload_bytes: u64,
    /// Preview reconnects.
    pub preview_reconnects: u64,
    /// Encoded video samples.
    pub video_samples: u64,
    /// Encoded video bytes.
    pub video_bytes: u64,
    /// Encoder or SDK errors (each also printed once).
    pub errors: u64,
    /// Mean worker time per frameset, milliseconds.
    pub worker_ms_mean: f64,
    /// Max worker time per frameset, milliseconds.
    pub worker_ms_max: f64,
    /// Seconds since the logger started.
    pub elapsed_s: f64,
    /// Encoder CPU seconds, summed over the cameras (last sample).
    pub encoder_cpu_s: f64,
    /// The save file was closed early because its filesystem ran low on space.
    pub save_stopped_low_disk: bool,
}

/// The logger's counters as its threads update them (caller, worker, encoder readers, preview sender); [`LogStats`] is a snapshot.
#[derive(Default)]
struct LogCounters {
    framesets_in: AtomicU64,
    framesets_dropped: AtomicU64,
    preview_sent: AtomicU64,
    preview_dropped: AtomicU64,
    preview_gated: AtomicU64,
    preview_payload_bytes: AtomicU64,
    preview_reconnects: AtomicU64,
    preview_queued: AtomicU64,
    video_samples: AtomicU64,
    video_bytes: AtomicU64,
    errors: AtomicU64,
    worker_ns_total: AtomicU64,
    worker_ns_max: AtomicU64,
    worker_framesets: AtomicU64,
    encoder_cpu_ms: AtomicU64,
    save_stopped: AtomicBool,
}

impl LogCounters {
    fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    fn error(&self, what: &str, error: &dyn std::fmt::Display) {
        if self.errors.fetch_add(1, Ordering::Relaxed) < 5 {
            eprintln!("robocap-live log: {what}: {error}");
        }
    }

    fn snapshot(&self, started: Instant) -> LogStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let framesets = get(&self.worker_framesets).max(1);
        LogStats {
            framesets_in: get(&self.framesets_in),
            framesets_dropped: get(&self.framesets_dropped),
            preview_sent: get(&self.preview_sent),
            preview_dropped: get(&self.preview_dropped),
            preview_gated: get(&self.preview_gated),
            preview_payload_bytes: get(&self.preview_payload_bytes),
            preview_reconnects: get(&self.preview_reconnects),
            video_samples: get(&self.video_samples),
            video_bytes: get(&self.video_bytes),
            errors: get(&self.errors),
            worker_ms_mean: get(&self.worker_ns_total) as f64 / framesets as f64 / 1e6,
            worker_ms_max: get(&self.worker_ns_max) as f64 / 1e6,
            elapsed_s: started.elapsed().as_secs_f64(),
            encoder_cpu_s: get(&self.encoder_cpu_ms) as f64 / 1e3,
            save_stopped_low_disk: self.save_stopped.load(Ordering::Relaxed),
        }
    }
}

/// Everything a new preview stream needs before data: the static scene and the layout.
struct Prelude {
    rig: Rig,
    h264: bool,
    display: Option<DisplayAssets>,
    content: scene::Content,
}

impl Prelude {
    fn send(&self, rec: &RecordingStream) -> Result<(), LogError> {
        scene::log_static_hands(rec, &self.rig)?;
        if self.content == scene::Content::Full {
            scene::log_static_scene(rec, &self.rig, self.h264)?;
        }
        if let Some(display) = self.display.as_ref() {
            display.send(rec)?;
        }
        Ok(())
    }
}

/// Free bytes on the filesystem holding `path` (`statvfs`), if it can be read.
pub fn free_bytes(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let dir = if path.is_dir() { path } else { path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new(".")) };
    let c_path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` points to writable memory of the right size; statvfs only
    // writes into it, and we read it only when the call succeeded.
    let status = unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) };
    if status != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0, so it filled the struct.
    let stat = unsafe { stat.assume_init() };
    // Both fields are 64-bit on the 64-bit targets this runs on; `from` keeps it exact elsewhere too.
    #[allow(clippy::useless_conversion)]
    Some(u64::from(stat.f_bavail) * u64::from(stat.f_frsize))
}

/// One deadline for input drain, encoder readers, file finalization, and preview drain.
#[derive(Default)]
struct Shutdown(Mutex<Option<Instant>>);

impl Shutdown {
    fn deadline(&self) -> Instant {
        let mut deadline = self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *deadline.get_or_insert_with(|| Instant::now() + Duration::from_secs(5))
    }
}

/// The live logger.
pub struct Logger {
    input: Option<SyncSender<FrameItem>>,
    worker: Option<JoinHandle<Result<Vec<EncoderReport>, LogError>>>,
    preview: Option<JoinHandle<()>>,
    shutdown: Arc<Shutdown>,
    counters: Arc<LogCounters>,
    started: Instant,
    time_origin_ns: Option<i64>,
    recording_id: String,
    lossless: bool,
}

impl Logger {
    /// Open the streams, send the static scene and the layout, and start the encoders and threads.
    ///
    /// # Arguments
    ///
    /// * `rig` - The six cameras (poses and intrinsics, at their calibration size).
    /// * `options` - Where to stream and save, and how to send video.
    ///
    /// # Errors
    ///
    /// [`LogError`] if the display asset is invalid, a stream cannot be opened, or an encoder cannot start.
    pub fn new(rig: &Rig, options: LoggerConfig) -> Result<Self, LogError> {
        if rig.cameras.len() != NUM_CAMERAS {
            return Err(LogError::Invalid(format!("rig has {} cameras, expected {NUM_CAMERAS}", rig.cameras.len())));
        }
        let recording_id = options.recording_id.clone().unwrap_or_else(|| {
            let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            format!("robocap-live-{seconds}")
        });
        let display = options.display.as_deref().map(DisplayAssets::load).transpose()?;
        let prelude = Arc::new(Prelude { rig: rig.clone(), h264: options.video == VideoMode::H264, display, content: options.content });
        let counters = Arc::new(LogCounters::default());
        let started = Instant::now();
        let shutdown = Arc::new(Shutdown::default());

        let save = match &options.save {
            Some(path) => match free_bytes(path) {
                Some(free) if free < options.save_min_free_bytes => {
                    eprintln!(
                        "robocap-live log: not saving {}: {:.1} GB free, below the {:.1} GB floor",
                        path.display(),
                        free as f64 / 1e9,
                        options.save_min_free_bytes as f64 / 1e9
                    );
                    counters.save_stopped.store(true, Ordering::Relaxed);
                    None
                }
                _ => {
                    // A catalog layer carries no recording properties: registered beside the segment's base layer, its
                    // RecordingInfo (start time, name) would replace the segment's.
                    let rec = rerun::RecordingStreamBuilder::new(APPLICATION_ID)
                        .recording_id(recording_id.clone())
                        .send_properties(options.content == scene::Content::Full)
                        .save(path)?;
                    rec.set_log_time_enabled(false);
                    prelude.send(&rec)?;
                    Some(rec)
                }
            },
            None => None,
        };
        let save_path = options.save.clone();
        let save: SaveSlot = Arc::new(std::sync::Mutex::new(save));

        let (preview_queue, preview) = match &options.viewer {
            Some(url) => {
                let uri: rerun::external::re_uri::ProxyUri =
                    url.parse().map_err(|e| LogError::Invalid(format!("viewer URL {url:?}: {e} (expected rerun+http://<host>:9876/proxy)")))?;
                let (tx, rx) = sync_channel(options.preview_queue.max(1));
                let queue = PreviewQueue { tx, counters: counters.clone() };
                let sender = PreviewSender {
                    uri,
                    recording_id: recording_id.clone(),
                    max_bytes_in_flight: options.preview_max_bytes_in_flight,
                    flush: options.preview_flush,
                    prelude: prelude.clone(),
                    counters: counters.clone(),
                    shutdown: shutdown.clone(),
                };
                let handle = std::thread::Builder::new()
                    .name("log-preview".into())
                    .spawn(move || sender.run(rx))
                    .map_err(|e| LogError::Invalid(format!("preview thread: {e}")))?;
                (Some(queue), Some(handle))
            }
            None => (None, None),
        };

        if let Some(&camera) = options.video_cameras.iter().find(|&&c| c >= NUM_CAMERAS) {
            return Err(LogError::Invalid(format!("video camera {camera}: there are {NUM_CAMERAS} cameras")));
        }
        let mut encoders: Vec<Option<EncoderState>> = (0..NUM_CAMERAS).map(|_| None).collect();
        if options.video == VideoMode::H264 {
            for (camera, slot) in encoders.iter_mut().enumerate().filter(|(camera, _)| options.video_cameras.contains(camera)) {
                let mut sink = VideoSink { save: save.clone(), preview: preview_queue.clone(), counters: counters.clone(), seq: 0 };
                *slot = Some(EncoderState {
                    encoder: H264Encoder::spawn(&options.encoder, move |sample| sink.send(VideoSample {
                        camera, t_ns: sample.timestamp_ns, data: sample.unit.data.into(), keyframe: sample.unit.keyframe,
                    }))?,
                    cpu_seconds: 0.0,
                });
            }
        }

        let (input, rx) = sync_channel(options.input_queue.max(1));
        let worker = Worker {
            shutdown: shutdown.clone(),
            finalizer: None,
            video: options.video,
            video_cameras: options.video_cameras.clone(),
            encoders,
            save,
            save_path,
            save_min_free_bytes: options.save_min_free_bytes,
            last_disk_check: Instant::now(),
            preview: preview_queue,
            counters: counters.clone(),
            state: RecordState::new(rig, options.hand_overlays),
            delivered: DeliveredState::default(),
            content: options.content,
            notice: None,
            fps_window: Default::default(),
            last_worker_ms: 0.0,
        };
        let worker = std::thread::Builder::new()
            .name("log-worker".into())
            .spawn(move || worker.run(rx))
            .map_err(|e| LogError::Invalid(format!("worker thread: {e}")))?;
        Ok(Self {
            input: Some(input),
            worker: Some(worker),
            preview,
            shutdown,
            counters,
            started,
            time_origin_ns: options.time_origin_ns,
            recording_id,
            lossless: options.lossless,
        })
    }

    /// The recording id both streams use.
    pub fn recording_id(&self) -> &str {
        &self.recording_id
    }

    /// Hand one frameset to the logger. Never blocks: if the worker is behind, the frameset is dropped and counted. A
    /// [`LoggerConfig::lossless`] logger waits for the worker instead.
    ///
    /// # Errors
    ///
    /// [`LogError::WorkerGone`] if the worker stopped (its error comes from [`Logger::finish`]).
    pub fn log_frameset(&mut self, frame: &FrameLog<'_>) -> Result<(), LogError> {
        let origin = *self.time_origin_ns.get_or_insert(frame.t_ns);
        LogCounters::add(&self.counters.framesets_in, 1);
        let item = FrameItem {
            t_ns: frame.t_ns - origin,
            small: frame.small.map(|image| image.cloned()),
            world_from_rig: frame.world_from_rig.copied(),
            slam_status: frame.slam_status.to_string(),
            hands: frame.hands.cloned(),
            timings: *frame.timings,
            received: Instant::now(),
        };
        let Some(input) = &self.input else { return Err(LogError::WorkerGone) };
        if self.lossless {
            return input.send(item).map_err(|_| LogError::WorkerGone);
        }
        match input.try_send(item) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                LogCounters::add(&self.counters.framesets_dropped, 1);
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(LogError::WorkerGone),
        }
    }

    /// The counters so far.
    pub fn stats(&self) -> LogStats {
        self.counters.snapshot(self.started)
    }

    /// Drain the worker, flush the encoders and the save file, give the preview up to a few seconds to drain, and return
    /// the counters.
    ///
    /// # Errors
    ///
    /// The worker's first error, if it stopped on one.
    pub fn finish(mut self) -> Result<(LogStats, Vec<EncoderReport>), LogError> {
        let deadline = self.shutdown.deadline();
        drop(self.input.take());
        let encoders = if let Some(worker) = self.worker.take() {
            while !worker.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if !worker.is_finished() {
                return Err(LogError::Invalid("logging worker exceeded the shutdown deadline".into()));
            }
            worker.join().map_err(|_| LogError::Invalid("the logging worker panicked".into()))??
        } else {
            Vec::new()
        };
        if let Some(preview) = self.preview.take() {
            while !preview.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if preview.is_finished() {
                preview.join().map_err(|_| LogError::Invalid("the preview sender panicked".into()))?;
            }
        }
        Ok((self.counters.snapshot(self.started), encoders))
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        self.shutdown.deadline();
        drop(self.input.take());
    }
}

/// Encoder per-camera CPU, as a share of one core over the encoder's life (for reports).
pub fn encoder_cpu_percent(stats: &EncoderReport) -> f64 {
    if stats.stats.wall_seconds > 0.0 { 100.0 * stats.cpu_seconds / stats.stats.wall_seconds } else { 0.0 }
}
