//! Prepare scene snapshots, save every accepted frame, and own encoder/file shutdown.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc::Receiver};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nalgebra::Isometry3;
use rerun::RecordingStream;

use super::{LogCounters, LogError, Shutdown, VideoMode, free_bytes, scene};
use super::preview::{PreviewItem, PreviewQueue};
use super::scene::{DeliveredState, RecordState, SceneSnapshot};
use super::video::{EncodedSample, EncoderStats, H264Encoder};
use crate::sched::FrameTimings;
use crate::frame::{Luma, NUM_CAMERAS};
use crate::hands::HandFrameResult;

/// A frameset as the worker holds it.
pub(super) struct FrameItem {
    pub(super) t_ns: i64,
    pub(super) small: [Option<Luma>; NUM_CAMERAS],
    pub(super) world_from_rig: Option<Isometry3<f64>>,
    pub(super) slam_status: String,
    pub(super) hands: Option<HandFrameResult>,
    pub(super) timings: FrameTimings,
    pub(super) received: Instant,
}

/// The save stream, shared by the worker and the encoder readers, so the worker can close it early (low disk) for all of them.
pub(super) type SaveSlot = Arc<std::sync::Mutex<Option<RecordingStream>>>;

fn with_save(slot: &SaveSlot, f: impl FnOnce(&RecordingStream)) {
    if let Ok(guard) = slot.lock()
        && let Some(rec) = guard.as_ref()
    {
        f(rec);
    }
}

/// Where an encoder's access units go: the save stream and the preview queue, every one of them (the preview sender gates).
pub(super) struct VideoSink {
    pub(super) save: SaveSlot,
    pub(super) preview: Option<PreviewQueue>,
    pub(super) counters: Arc<LogCounters>,
    /// The number of this camera's next sample.
    pub(super) seq: u64,
}

impl VideoSink {
    pub(super) fn send(&mut self, sample: EncodedSample) {
        LogCounters::add(&self.counters.video_samples, 1);
        LogCounters::add(&self.counters.video_bytes, sample.unit.data.len() as u64);
        with_save(&self.save, |save| log_video_sample(save, &sample));
        let Some(queue) = &self.preview else { return };
        queue.offer(PreviewItem::Video { seq: self.seq, sample });
        self.seq += 1;
    }
}

pub(super) fn log_video_sample(rec: &RecordingStream, sample: &EncodedSample) {
    rec.set_time(scene::TIMELINE, rerun::TimeCell::from_duration_nanos(sample.t_ns));
    // A clone of the shared blob: no bytes are copied (the save stream and the preview log the same buffer).
    let video = rerun::VideoStream::update_fields().with_sample(sample.unit.data.clone()).with_is_keyframe(sample.unit.keyframe);
    // Serialisation of a byte blob cannot fail in practice; a failure is dropped with the sample.
    let _ = rec.log(scene::video_path(sample.camera), &video);
}

pub(super) fn log_image(rec: &RecordingStream, camera: usize, t_ns: i64, luma: &Luma) -> Result<(), rerun::RecordingStreamError> {
    rec.set_time(scene::TIMELINE, rerun::TimeCell::from_duration_nanos(t_ns));
    let size = luma.size();
    rec.log(scene::image_path(camera), &rerun::Image::from_l8(luma_blob(luma), [size.width as u32, size.height as u32]))
}

/// The image's pixels as a Rerun blob that borrows them: an Arrow buffer over the image's own memory, which holds a reference
/// to the image until Rerun drops the buffer (kornia's `from_borrowed` keepalive, the other way round). No bytes are copied.
pub(super) fn luma_blob(luma: &Luma) -> rerun::datatypes::Blob {
    let pixels: &[u8] = luma.as_slice();
    // Arrow asks the owner to be unwind-safe; it only keeps the image alive and is never read, so asserting it is sound.
    let owner: Arc<dyn rerun::external::arrow::alloc::Allocation> = Arc::new(std::panic::AssertUnwindSafe(luma.clone()));
    // SAFETY: `pixels` is the whole pixel slice of the image `owner` keeps alive, so it stays valid for `pixels.len()` bytes as
    // long as the buffer exists; the image sits behind an `Arc` that is now shared, so nobody can get `&mut` to it and write.
    let buffer = unsafe {
        rerun::external::arrow::buffer::Buffer::from_custom_allocation(std::ptr::NonNull::from(pixels).cast::<u8>(), pixels.len(), owner)
    };
    buffer.into()
}

pub(super) struct Worker {
    pub(super) shutdown: Arc<Shutdown>,
    pub(super) finalizer: Option<JoinHandle<Result<(), LogError>>>,
    pub(super) video: VideoMode,
    pub(super) video_cameras: Vec<usize>,
    pub(super) encoders: Vec<Option<H264Encoder>>,
    pub(super) save: SaveSlot,
    pub(super) save_path: Option<PathBuf>,
    pub(super) save_min_free_bytes: u64,
    pub(super) last_disk_check: Instant,
    pub(super) preview: Option<PreviewQueue>,
    pub(super) counters: Arc<LogCounters>,
    pub(super) state: RecordState,
    pub(super) delivered: DeliveredState,
    pub(super) content: scene::Content,
    pub(super) notice: Option<String>,
    pub(super) fps_window: VecDeque<Instant>,
    pub(super) last_worker_ms: f64,
}

const FPS_WINDOW: usize = 30;

impl Worker {
    pub(super) fn run(mut self, rx: Receiver<FrameItem>) -> Result<Vec<EncoderStats>, LogError> {
        while let Ok(item) = rx.recv() {
            let start = Instant::now();
            self.frameset(item)?;
            let ns = start.elapsed().as_nanos() as u64;
            self.last_worker_ms = ns as f64 / 1e6;
            LogCounters::add(&self.counters.worker_ns_total, ns);
            LogCounters::add(&self.counters.worker_framesets, 1);
            self.counters.worker_ns_max.fetch_max(ns, Ordering::Relaxed);
        }
        let deadline = self.shutdown.deadline();
        // Closing all inputs first lets all cameras flush concurrently, under the same deadline.
        for encoder in self.encoders.iter_mut().flatten() {
            encoder.close_input();
        }
        let mut stats = Vec::new();
        for (camera, encoder) in self.encoders.drain(..).enumerate() {
            if let Some(encoder) = encoder {
                match encoder.finish_until(deadline) {
                    Ok(s) => stats.push(s),
                    Err(error) => self.counters.error(&format!("encoder {camera} finish"), &error),
                }
            }
        }
        let cpu: f64 = stats.iter().map(|s| s.cpu_seconds).sum();
        self.counters.encoder_cpu_ms.store((cpu * 1e3) as u64, Ordering::Relaxed);
        let save = self.save.lock().ok().and_then(|mut slot| slot.take());
        if let Some(save) = save {
            self.finalizer = Some(finalize_save(save, deadline.saturating_duration_since(Instant::now()))?);
        }
        if let Some(finalizer) = self.finalizer.take() {
            while !finalizer.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if !finalizer.is_finished() {
                return Err(LogError::Invalid("save finalizer exceeded the shutdown deadline".into()));
            }
            finalizer.join().map_err(|_| LogError::Invalid("save finalizer panicked".into()))??;
        }
        Ok(stats)
    }

    pub(super) fn frameset(&mut self, item: FrameItem) -> Result<(), LogError> {
        match self.video {
            VideoMode::H264 => {
                for (camera, image) in item.small.iter().enumerate() {
                    let (Some(image), Some(encoder)) = (image, self.encoders[camera].as_mut()) else { continue };
                    if let Err(error) = encoder.push(item.t_ns, image) {
                        // A failed encoder stops its camera's video; the rest of the logging goes on.
                        self.counters.error(&format!("camera {camera} encoder"), &error);
                        self.encoders[camera] = None;
                    }
                }
            }
            VideoMode::Raw => {
                for (camera, image) in item.small.iter().enumerate() {
                    let Some(image) = image else { continue };
                    if !self.video_cameras.contains(&camera) {
                        continue;
                    }
                    with_save(&self.save, |save| {
                        if let Err(error) = log_image(save, camera, item.t_ns, image) {
                            self.counters.error("save image", &error);
                        }
                    });
                    if let Some(preview) = &self.preview {
                        preview.offer(PreviewItem::Image { camera, t_ns: item.t_ns, luma: image.clone() });
                    }
                }
            }
            VideoMode::Off => {}
        }
        self.fps_window.push_back(item.received);
        if self.fps_window.len() > FPS_WINDOW + 1 {
            self.fps_window.pop_front();
        }
        let timings = scene::timing_row(&item.timings, self.last_worker_ms);
        let fps = match (self.fps_window.front(), self.fps_window.back()) {
            (Some(first), Some(last)) if self.fps_window.len() > 1 && *last > *first => {
                Some((self.fps_window.len() - 1) as f64 / last.duration_since(*first).as_secs_f64())
            }
            _ => None,
        };
        let signals = scene::Signals { timings, fps, counters: scene::counter_row(&self.counters) };
        let prepared = self.state.prepare(item.t_ns, item.world_from_rig, &item.slam_status, item.hands.as_ref(), &signals);
        if let Some(notice) = self.check_disk()? {
            self.notice = Some(notice);
        }
        let snapshot = Arc::new(SceneSnapshot { notice: self.notice.clone(), ..prepared });
        with_save(&self.save, |save| {
            if let Err(error) = scene::write_record(save, &self.delivered.record(&snapshot), self.content) {
                self.counters.error("save record", &error);
            }
        });
        if let Some(preview) = &self.preview {
            preview.offer(PreviewItem::Frame(snapshot));
        }
        Ok(())
    }

    /// Every couple of seconds: close the save file early when its filesystem is low on space; the returned text says so.
    pub(super) fn check_disk(&mut self) -> Result<Option<String>, LogError> {
        if self.last_disk_check.elapsed() < Duration::from_secs(2) {
            return Ok(None);
        }
        self.last_disk_check = Instant::now();
        let Some(path) = self.save_path.as_ref() else { return Ok(None) };
        let Some(free) = free_bytes(path) else { return Ok(None) };
        if free >= self.save_min_free_bytes {
            return Ok(None);
        }
        let Some(save) = self.save.lock().ok().and_then(|mut slot| slot.take()) else { return Ok(None) };
        let message = format!(
            "saving stopped: {:.1} GB free on {}, below the {:.1} GB floor; the live view goes on",
            free as f64 / 1e9,
            path.display(),
            self.save_min_free_bytes as f64 / 1e9
        );
        eprintln!("robocap-live log: {message}");
        self.counters.save_stopped.store(true, Ordering::Relaxed);
        // Retain completion: Logger::finish must report this flush, including a failure after low disk stopped saving.
        self.finalizer = Some(finalize_save(save, Duration::from_secs(5))?);
        Ok(Some(message))
    }
}

fn finalize_save(save: RecordingStream, timeout: Duration) -> Result<JoinHandle<Result<(), LogError>>, LogError> {
    std::thread::Builder::new().name("log-save-finalizer".into()).spawn(move || {
        let result = save.flush_with_timeout(timeout).map_err(|e| LogError::Invalid(format!("flushing the save file: {e}")));
        // RecordingStream::drop can wait too; it remains inside the owned finalizer's deadline.
        drop(save);
        result
    }).map_err(|e| LogError::Invalid(format!("save finalizer thread: {e}")))
}
