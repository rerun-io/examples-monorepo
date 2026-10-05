//! Lossy preview delivery: the sender alone gates each camera's video on keyframes, and computes scene transitions only after
//! taking a snapshot from the queue.

use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rerun::RecordingStream;
use rerun::sink::{GrpcSinkConnectionState, LogSink};

use super::{LogCounters, LogError, Prelude, Shutdown, scene};
use super::display::APPLICATION_ID;
use super::scene::{DeliveredState, SceneSnapshot};
use super::video::EncodedSample;
use super::worker::{log_image, log_video_sample};
use crate::frame::{Luma, NUM_CAMERAS};

/// What travels to the preview sender.
pub(super) enum PreviewItem {
    /// An access unit; `seq` numbers the camera's samples from 0, so the sender sees one the queue dropped.
    Video { seq: u64, sample: EncodedSample },
    Image { camera: usize, t_ns: i64, luma: Luma },
    Frame(Arc<SceneSnapshot>),
}

/// The bounded queue in front of the preview stream, shared by the worker and the encoder readers.
#[derive(Clone)]
pub(super) struct PreviewQueue {
    pub(super) tx: SyncSender<PreviewItem>,
    pub(super) counters: Arc<LogCounters>,
}

impl PreviewQueue {
    /// Try to queue an item; `false` when the queue is full (the item is dropped and counted).
    pub(super) fn offer(&self, item: PreviewItem) -> bool {
        // Count first: the sender may take the item (and count it out) before try_send returns.
        LogCounters::add(&self.counters.preview_queued, 1);
        match self.tx.try_send(item) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.counters.preview_queued.fetch_sub(1, Ordering::Relaxed);
                LogCounters::add(&self.counters.preview_dropped, 1);
                false
            }
        }
    }
}

/// The preview's gRPC sink: the SDK's message-proxy client (what `GrpcSink` wraps) without LZ4, since nearly all of the stream is
/// H.264 that does not compress, and shared, so the preview sender can watch the connection.
struct ProxySink(Arc<rerun::external::re_grpc_client::write::Client>);

impl ProxySink {
    fn new(uri: rerun::external::re_uri::ProxyUri) -> Self {
        let options = rerun::external::re_grpc_client::write::Options {
            compression: re_log_encoding::rrd::Compression::Off,
            ..rerun::external::re_grpc_client::write::Options::default()
        };
        Self(Arc::new(rerun::external::re_grpc_client::write::Client::new(uri, options)))
    }
}

impl LogSink for ProxySink {
    fn send(&self, msg: rerun::log::LogMsg) {
        self.0.send_blocking(msg);
    }

    fn flush_blocking(&self, timeout: Duration) -> Result<(), rerun::sink::SinkFlushError> {
        self.0.flush_blocking(timeout).map_err(|e| rerun::sink::SinkFlushError::failed(e.to_string()))
    }
}

/// The preview's keyframe gate: on a fresh stream, and after a gap in a camera's sample numbers (the queue dropped one), that
/// camera's non-key samples are withheld until its next keyframe, so the viewer never decodes a broken reference chain.
pub(super) struct KeyGate {
    /// Per camera: withhold non-key samples.
    need_key: [bool; NUM_CAMERAS],
    /// Per camera: the number of the sample that follows the last one seen.
    next_seq: [u64; NUM_CAMERAS],
}

impl KeyGate {
    /// A fresh stream: every camera waits for a keyframe.
    pub(super) fn new() -> Self {
        Self { need_key: [true; NUM_CAMERAS], next_seq: [0; NUM_CAMERAS] }
    }

    /// A new stream after a reconnect: every camera waits for a keyframe again.
    pub(super) fn reconnected(&mut self) {
        self.need_key = [true; NUM_CAMERAS];
    }

    /// Whether sample `seq` of `camera` goes to the viewer.
    pub(super) fn admit(&mut self, camera: usize, seq: u64, keyframe: bool) -> bool {
        let gap = seq != self.next_seq[camera];
        self.next_seq[camera] = seq + 1;
        if (self.need_key[camera] || gap) && !keyframe {
            self.need_key[camera] = true;
            return false;
        }
        self.need_key[camera] = false;
        true
    }
}

pub(super) struct PreviewSender {
    pub(super) shutdown: Arc<Shutdown>,
    pub(super) uri: rerun::external::re_uri::ProxyUri,
    pub(super) recording_id: String,
    pub(super) max_bytes_in_flight: u64,
    pub(super) flush: Duration,
    pub(super) prelude: Arc<Prelude>,
    pub(super) counters: Arc<LogCounters>,
}

impl PreviewSender {
    fn connect(&self) -> Result<(RecordingStream, Arc<rerun::external::re_grpc_client::write::Client>), LogError> {
        let sink = ProxySink::new(self.uri.clone());
        let client = sink.0.clone();
        let config = rerun::log::ChunkBatcherConfig {
            max_bytes_in_flight: self.max_bytes_in_flight,
            flush_tick: self.flush,
            ..rerun::log::ChunkBatcherConfig::LOW_LATENCY
        };
        let rec = rerun::RecordingStreamBuilder::new(APPLICATION_ID)
            .recording_id(self.recording_id.clone())
            .batcher_config(config)
            .set_sinks(vec![Box::new(sink) as Box<dyn LogSink>])?;
        rec.set_log_time_enabled(false);
        self.prelude.send(&rec)?;
        Ok((rec, client))
    }

    pub(super) fn run(self, rx: Receiver<PreviewItem>) {
        let mut connection = match self.connect() {
            Ok(connection) => Some(connection),
            Err(error) => {
                self.counters.error("preview connect", &error);
                None
            }
        };
        let mut gate = KeyGate::new();
        let mut delivered = DeliveredState::reconnected();
        let mut latest: Option<Arc<SceneSnapshot>> = None;
        let mut last_check = Instant::now();
        loop {
            let item = match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(item) => Some(item),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            if last_check.elapsed() > Duration::from_millis(500) {
                last_check = Instant::now();
                let lost = connection.as_ref().is_none_or(|(_, client)| matches!(client.status(), GrpcSinkConnectionState::Disconnected(_)));
                if lost {
                    // The viewer went away (or never answered): open a fresh stream; the client retries until it connects.
                    if let Some((old, _)) = connection.take() {
                        // Dropping a stream flushes it without a timeout; never do that on this thread.
                        std::thread::spawn(move || drop(old));
                    }
                    match self.connect() {
                        Ok(fresh) => {
                            delivered = DeliveredState::reconnected();
                            if let Some(snapshot) = &latest
                                && let Err(error) = scene::write_record(&fresh.0, &delivered.record(snapshot), self.prelude.content)
                            {
                                self.counters.error("preview scene restore", &error);
                            }
                            connection = Some(fresh);
                            gate.reconnected();
                            LogCounters::add(&self.counters.preview_reconnects, 1);
                        }
                        Err(error) => self.counters.error("preview reconnect", &error),
                    }
                }
            }
            let Some(item) = item else { continue };
            self.counters.preview_queued.fetch_sub(1, Ordering::Relaxed);
            if let PreviewItem::Frame(snapshot) = &item {
                latest = Some(snapshot.clone());
            }
            let Some((rec, _)) = &connection else { continue };
            let result = match &item {
                PreviewItem::Video { seq, sample } => {
                    if !gate.admit(sample.camera, *seq, sample.unit.keyframe) {
                        LogCounters::add(&self.counters.preview_gated, 1);
                        continue;
                    }
                    LogCounters::add(&self.counters.preview_payload_bytes, sample.unit.data.len() as u64);
                    log_video_sample(rec, sample);
                    Ok(())
                }
                PreviewItem::Image { camera, t_ns, luma } => {
                    LogCounters::add(&self.counters.preview_payload_bytes, luma.as_slice().len() as u64);
                    log_image(rec, *camera, *t_ns, luma)
                }
                PreviewItem::Frame(record) => scene::write_record(rec, &delivered.record(record), self.prelude.content),
            };
            match result {
                Ok(()) => LogCounters::add(&self.counters.preview_sent, 1),
                Err(error) => self.counters.error("preview log", &error),
            }
        }
        if let Some((rec, _)) = connection {
            if rec.flush_with_timeout(self.shutdown.deadline().saturating_duration_since(Instant::now())).is_err() {
                self.counters.error("preview flush", &"timed out (viewer unreachable?)");
            }
            std::thread::spawn(move || drop(rec));
        }
    }
}
