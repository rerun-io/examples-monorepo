//! H.264 video for the viewer: one hardware encoder child process per camera, fed NV12 frames on stdin, with the Annex-B byte
//! stream read back from stdout and split into access units (one per frame) for `rerun::VideoStream`.
//!
//! The binary must not link GStreamer, so the encoder is a `gst-launch-1.0` child (on the cap: Rockchip's `mpph264enc`):
//!
//! ```text
//! filesrc location=/dev/stdin blocksize=<w*h*3/2> ! video/x-raw,format=NV12,width=640,height=360,framerate=30/1 ! queue
//!   ! mpph264enc bps=1000000 gop=30 header-mode=each-idr ! h264parse config-interval=-1
//!   ! video/x-h264,stream-format=byte-stream,alignment=au ! fdsink fd=1
//! ```
//!
//! Cap A's GStreamer has no `rawvideoparse`, so `filesrc` on `/dev/stdin` does the framing: unlike `fdsrc` it fills each
//! `blocksize` buffer completely from the pipe. The input is grey: the luma plane followed by a constant 128 chroma plane.
//!
//! Upstream candidate (UPSTREAM.md): a hardware-encoder video sink for kornia-io that needs no GStreamer link: the
//! [`AccessUnitSplitter`] and the child-process [`H264Encoder`].
#![deny(missing_docs)]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use kornia_image::{Image, ImageSize};

/// Errors of the video encoder.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// The encoder process could not be started.
    #[error("could not start the H.264 encoder `{program}`: {source}")]
    Spawn {
        /// The encoder command.
        program: String,
        /// Why it did not start.
        source: std::io::Error,
    },
    /// Writing a frame to, or reading the stream from, the encoder failed (the encoder probably exited).
    #[error("H.264 encoder for camera {camera}: {source}; encoder stderr: {stderr}")]
    Io {
        /// The camera whose encoder failed.
        camera: usize,
        /// The pipe error.
        source: std::io::Error,
        /// The encoder's stderr so far.
        stderr: String,
    },
    /// A frame did not have the encoder's size.
    #[error("camera {camera}: frame {got:?}, the encoder takes {expected:?}")]
    Size {
        /// The camera.
        camera: usize,
        /// The frame's size.
        got: ImageSize,
        /// The encoder's size.
        expected: ImageSize,
    },
    /// The encoder exited with a failure status.
    #[error("H.264 encoder for camera {camera} exited with {status}; stderr: {stderr}")]
    Exit {
        /// The camera whose encoder exited.
        camera: usize,
        /// Its exit status.
        status: String,
        /// Its stderr.
        stderr: String,
    },
    /// The encoder produced more access units than frames it was given.
    #[error("H.264 encoder for camera {camera}: an access unit without a pending frame")]
    Unmatched {
        /// The camera.
        camera: usize,
    },
    /// An encoder name that is none of [`EncoderKind`]'s.
    #[error("encoder {0:?}: expected mpp, x264 or openh264")]
    UnknownEncoder(String),
}

/// One H.264 access unit (one frame) in Annex-B form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessUnit {
    /// The access unit's bytes, start codes included, in a shared Arrow buffer: the save stream and the preview log the same
    /// allocation (cloning it copies no bytes; Rerun serialises a single blob without a copy).
    pub data: rerun::datatypes::Blob,
    /// Whether it holds an IDR slice (with `header-mode=each-idr`, also the SPS and PPS).
    pub keyframe: bool,
}

/// Splits an H.264 Annex-B byte stream into access units as bytes arrive.
///
/// An access unit ends where the next begins: at an access unit delimiter, SPS, PPS or SEI NAL, or at a slice whose
/// `first_mb_in_slice` is 0, once the current unit holds a slice (H.264 section 7.4.1.2.3). The last unit of a stream is
/// complete only at its end ([`AccessUnitSplitter::finish`]), so a live stream is one frame behind.
#[derive(Debug, Default)]
pub struct AccessUnitSplitter {
    buf: Vec<u8>,
    scan_from: usize,
    has_slice: bool,
    keyframe: bool,
}

impl AccessUnitSplitter {
    /// A splitter with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes of the stream.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The next bytes of the Annex-B stream, any length.
    /// * `out` - Receives every access unit these bytes complete, in stream order.
    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<AccessUnit>) {
        self.buf.extend_from_slice(bytes);
        let mut i = self.scan_from;
        // A start code is 00 00 01; the NAL header byte and the first slice-header byte follow it.
        while i + 4 < self.buf.len() {
            if !(self.buf[i] == 0 && self.buf[i + 1] == 0 && self.buf[i + 2] == 1) {
                i += 1;
                continue;
            }
            let nal_type = self.buf[i + 3] & 0x1f;
            let is_slice = nal_type == 1 || nal_type == 5;
            // first_mb_in_slice is ue(v); its value 0 is the single bit 1.
            let first_slice = is_slice && self.buf[i + 4] & 0x80 != 0;
            let starts_unit = matches!(nal_type, 6..=9 | 14..=18) || first_slice;
            if starts_unit && self.has_slice {
                // A four-byte start code's leading zero belongs to the new unit.
                let start = if i > 0 && self.buf[i - 1] == 0 { i - 1 } else { i };
                let data: Vec<u8> = self.buf.drain(..start).collect();
                out.push(AccessUnit { data: data.into(), keyframe: self.keyframe });
                self.has_slice = false;
                self.keyframe = false;
                i -= start;
            }
            if is_slice {
                self.has_slice = true;
                self.keyframe |= nal_type == 5;
            }
            i += 3;
        }
        self.scan_from = i;
    }

    /// End of stream: the last access unit, if the buffer holds a slice.
    pub fn finish(&mut self) -> Option<AccessUnit> {
        self.scan_from = 0;
        let has_slice = std::mem::take(&mut self.has_slice);
        let keyframe = std::mem::take(&mut self.keyframe);
        let data = std::mem::take(&mut self.buf);
        (has_slice && !data.is_empty()).then_some(AccessUnit { data: data.into(), keyframe })
    }
}

/// Whether an Annex-B buffer contains a NAL unit of `kind` (5 = IDR slice, 7 = SPS, 8 = PPS).
pub fn has_nal(bytes: &[u8], kind: u8) -> bool {
    bytes.windows(4).any(|w| w[0] == 0 && w[1] == 0 && w[2] == 1 && w[3] & 0x1f == kind)
}

/// The H.264 encoders [`EncoderConfig::for_kind`] knows, by their command-line names (`mpp`, `x264`, `openh264`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    /// The cap's hardware encoder ([`EncoderConfig::mpp`]).
    Mpp,
    /// ffmpeg's libx264 ([`EncoderConfig::x264`]).
    X264,
    /// GStreamer's openh264 ([`EncoderConfig::openh264`]).
    Openh264,
}

impl FromStr for EncoderKind {
    type Err = VideoError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "mpp" => Ok(Self::Mpp),
            "x264" => Ok(Self::X264),
            "openh264" => Ok(Self::Openh264),
            other => Err(VideoError::UnknownEncoder(other.to_string())),
        }
    }
}

/// The encoder command: a program and its arguments, which read NV12 frames on stdin and write H.264 Annex-B on stdout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderConfig {
    /// The program, e.g. `gst-launch-1.0`.
    pub program: String,
    /// Its arguments.
    pub args: Vec<String>,
    /// Frame size the encoder is told (and every frame must have).
    pub size: ImageSize,
}

/// The `gst-launch-1.0` arguments around an encoder element: stdin NV12 framing in front, Annex-B AU stream out to stdout.
fn gst_launch_args(size: ImageSize, fps: u32, encoder: &[&str]) -> Vec<String> {
    let frame_bytes = size.width * size.height * 3 / 2;
    let mut args = vec![
        "-q".to_string(),
        "filesrc".into(),
        "location=/dev/stdin".into(),
        format!("blocksize={frame_bytes}"),
        "!".into(),
        format!("video/x-raw,format=NV12,width={},height={},framerate={fps}/1", size.width, size.height),
        "!".into(),
        // filesrc reads the next frame while the encoder works on this one, so the logger's pipe write returns at once.
        "queue".into(),
        "max-size-buffers=3".into(),
        "!".into(),
    ];
    args.extend(encoder.iter().map(|s| s.to_string()));
    args.extend(
        ["!", "h264parse", "config-interval=-1", "!", "video/x-h264,stream-format=byte-stream,alignment=au", "!", "fdsink", "fd=1", "sync=false"]
            .iter()
            .map(|s| s.to_string()),
    );
    args
}

impl EncoderConfig {
    /// The command of one encoder kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - Which encoder.
    /// * `size`, `fps`, `bps`, `gop` - As for [`Self::mpp`].
    ///
    /// # Returns
    ///
    /// [`Self::mpp`], [`Self::x264`] or [`Self::openh264`] with these settings.
    pub fn for_kind(kind: EncoderKind, size: ImageSize, fps: u32, bps: u32, gop: u32) -> Self {
        match kind {
            EncoderKind::Mpp => Self::mpp(size, fps, bps, gop),
            EncoderKind::X264 => Self::x264(size, fps, bps, gop),
            EncoderKind::Openh264 => Self::openh264(size, fps, bps, gop),
        }
    }

    /// Rockchip MPP (`mpph264enc`, the RK3588's VPU), constant bitrate, SPS/PPS with every IDR so a viewer can join mid-stream.
    ///
    /// # Arguments
    ///
    /// * `size` - Frame size (640x360 on the cap).
    /// * `fps` - Nominal frame rate (rate control only; samples carry their own timestamps).
    /// * `bps` - Target bitrate, bits per second.
    /// * `gop` - Frames per group of pictures (the longest a dropped preview frame breaks a pane).
    pub fn mpp(size: ImageSize, fps: u32, bps: u32, gop: u32) -> Self {
        let bps = format!("bps={bps}");
        let gop = format!("gop={gop}");
        Self { program: "gst-launch-1.0".into(), args: gst_launch_args(size, fps, &["mpph264enc", &bps, &gop, "header-mode=each-idr", "rc-mode=cbr"]), size }
    }

    /// Cisco's software `openh264enc` (host tests and the host fallback; GStreamer's bad plugins; it takes I420, hence the
    /// `videoconvert`).
    pub fn openh264(size: ImageSize, fps: u32, bps: u32, gop: u32) -> Self {
        let bps = format!("bitrate={bps}");
        let gop = format!("gop-size={gop}");
        Self { program: "gst-launch-1.0".into(), args: gst_launch_args(size, fps, &["videoconvert", "!", "openh264enc", &bps, &gop, "complexity=low"]), size }
    }

    /// `ffmpeg` with libx264 (zerolatency, no B-frames): a host encoder where GStreamer lacks one.
    pub fn x264(size: ImageSize, fps: u32, bps: u32, gop: u32) -> Self {
        let args = [
            "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "nv12", "-s", &format!("{}x{}", size.width, size.height), "-r",
            &fps.to_string(), "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0", "-g", &gop.to_string(),
            "-b:v", &bps.to_string(), "-x264-params", "repeat-headers=1", "-flush_packets", "1", "-f", "h264", "pipe:1",
        ];
        Self { program: "ffmpeg".into(), args: args.iter().map(|s| s.to_string()).collect(), size }
    }
}

/// One encoded frame of one camera, matched to the timestamp of the frame that went in.
#[derive(Clone, Debug)]
pub struct EncodedSample {
    /// Camera index.
    pub camera: usize,
    /// The input frame's time (whatever the caller passed to [`H264Encoder::push`]).
    pub t_ns: i64,
    /// The access unit.
    pub unit: AccessUnit,
}

/// Counters of one encoder.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EncoderStats {
    /// Frames written to the encoder.
    pub frames_in: u64,
    /// Access units read back.
    pub samples_out: u64,
    /// Encoded bytes read back.
    pub bytes_out: u64,
    /// CPU seconds the encoder process used (user + system), last sampled.
    pub cpu_seconds: f64,
    /// Wall seconds since the encoder started.
    pub wall_seconds: f64,
}

type SampleSink = Box<dyn FnMut(EncodedSample) + Send>;
type ReaderTask = Box<dyn FnOnce() -> Result<(u64, u64), VideoError> + Send>;

/// A running encoder child process for one camera.
pub struct H264Encoder {
    camera: usize,
    size: ImageSize,
    child: Child,
    stdin: Option<ChildStdin>,
    pending: Arc<Mutex<VecDeque<i64>>>,
    reader: Option<JoinHandle<Result<(u64, u64), VideoError>>>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    stderr_reader: Option<JoinHandle<()>>,
    chroma: Vec<u8>,
    frames_in: u64,
    started: Instant,
    cpu_seconds: f64,
}

const STDERR_TAIL_LINES: usize = 12;
/// Pipe capacity asked for the encoder's stdin: three 640x360 NV12 frames, so a frame write returns without waiting for the
/// encoder to read it in 64 KiB pieces (each piece a context switch on the cap's A55 cores).
const PIPE_BYTES: i32 = 1 << 20;

/// Grow a pipe's kernel buffer (Linux `F_SETPIPE_SZ`; capped by `/proc/sys/fs/pipe-max-size`). Best effort: a pipe that stays at
/// 64 KiB still works, only with more context switches.
fn grow_pipe(pipe: &impl std::os::fd::AsRawFd, bytes: i32) {
    #[cfg(target_os = "linux")]
    // SAFETY: fcntl on a file descriptor we own; F_SETPIPE_SZ takes an int and touches no memory of ours.
    unsafe {
        libc::fcntl(pipe.as_raw_fd(), libc::F_SETPIPE_SZ, bytes);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (pipe.as_raw_fd(), bytes);
}

fn tail_text(tail: &Mutex<VecDeque<String>>) -> String {
    tail.lock().map(|lines| lines.iter().cloned().collect::<Vec<_>>().join(" | ")).unwrap_or_default()
}

impl H264Encoder {
    /// Start the encoder for one camera.
    ///
    /// # Arguments
    ///
    /// * `camera` - Camera index, carried into every sample.
    /// * `config` - The encoder command.
    /// * `sink` - Called on the encoder's reader thread with every access unit, in order.
    ///
    /// # Errors
    ///
    /// [`VideoError::Spawn`] if the program or a reader thread cannot be started.
    ///
    /// # Returns
    ///
    /// An encoder that owns the child process and its input and readers.
    pub fn spawn(camera: usize, config: &EncoderConfig, sink: impl FnMut(EncodedSample) + Send + 'static) -> Result<Self, VideoError> {
        Self::spawn_with_reader(camera, config, sink, |builder, task| builder.spawn(task))
    }

    fn spawn_with_reader(
        camera: usize,
        config: &EncoderConfig,
        sink: impl FnMut(EncodedSample) + Send + 'static,
        start_reader: impl FnOnce(std::thread::Builder, ReaderTask) -> std::io::Result<JoinHandle<Result<(u64, u64), VideoError>>>,
    ) -> Result<Self, VideoError> {
        let mut command = Command::new(&config.program);
        command.args(&config.args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        // Its own process group: a Ctrl-C to the runtime's group must not kill the encoder before it flushes; closing its
        // stdin ends it.
        command.process_group(0);
        let mut child = command.spawn().map_err(|source| VideoError::Spawn { program: config.program.clone(), source })?;
        let stdin = child.stdin.take();
        if let Some(stdin) = &stdin {
            grow_pipe(stdin, PIPE_BYTES);
        }
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // Own the child before either fallible thread spawn: every early return kills and reaps it through Drop.
        let mut encoder = Self {
            camera, size: config.size, child, stdin, pending: Arc::default(), reader: None,
            stderr_tail: Arc::default(), stderr_reader: None,
            chroma: vec![128u8; config.size.width * config.size.height / 2],
            frames_in: 0, started: Instant::now(), cpu_seconds: 0.0,
        };
        if let Some(stderr) = stderr {
            let tail = encoder.stderr_tail.clone();
            encoder.stderr_reader = Some(std::thread::Builder::new()
                .name(format!("h264-stderr-{camera}"))
                .spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        if let Ok(mut lines) = tail.lock() {
                            if lines.len() == STDERR_TAIL_LINES {
                                lines.pop_front();
                            }
                            lines.push_back(line);
                        }
                    }
                })
                .map_err(|source| VideoError::Spawn { program: "stderr thread".into(), source })?);
        }
        if let Some(stdout) = stdout {
            let pending = encoder.pending.clone();
            let tail = encoder.stderr_tail.clone();
            let sink: SampleSink = Box::new(sink);
            encoder.reader = Some(start_reader(
                std::thread::Builder::new().name(format!("h264-read-{camera}")),
                Box::new(move || read_units(camera, stdout, &pending, &tail, sink)),
            ).map_err(|source| VideoError::Spawn { program: "reader thread".into(), source })?);
        }
        Ok(encoder)
    }

    /// Encode one grey frame: its luma plane plus constant 128 chroma.
    ///
    /// # Arguments
    ///
    /// * `t_ns` - The frame's time; the matching [`EncodedSample`] carries it.
    /// * `luma` - The frame, the encoder's size.
    ///
    /// # Errors
    ///
    /// [`VideoError::Size`] for a frame of another size, [`VideoError::Io`] if the encoder no longer accepts input.
    pub fn push(&mut self, t_ns: i64, luma: &Image<u8, 1>) -> Result<(), VideoError> {
        if luma.size() != self.size {
            return Err(VideoError::Size { camera: self.camera, got: luma.size(), expected: self.size });
        }
        let camera = self.camera;
        let io_error = |source: std::io::Error, tail: &Mutex<VecDeque<String>>| VideoError::Io { camera, source, stderr: tail_text(tail) };
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(io_error(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "encoder input already closed"), &self.stderr_tail));
        };
        // Queue the timestamp first: the access unit may come back before this call returns.
        if let Ok(mut pending) = self.pending.lock() {
            pending.push_back(t_ns);
        }
        stdin.write_all(luma.as_slice()).map_err(|e| io_error(e, &self.stderr_tail))?;
        stdin.write_all(&self.chroma).map_err(|e| io_error(e, &self.stderr_tail))?;
        self.frames_in += 1;
        if self.frames_in % 30 == 0 {
            self.sample_cpu();
        }
        Ok(())
    }

    /// Frames written and not yet returned as access units.
    pub fn in_flight(&self) -> usize {
        self.pending.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// CPU seconds the encoder process used so far (Linux `/proc/<pid>/stat`, 100 ticks per second); 0 elsewhere.
    pub fn cpu_seconds(&mut self) -> f64 {
        self.sample_cpu();
        self.cpu_seconds
    }

    fn sample_cpu(&mut self) {
        if let Some(seconds) = process_cpu_seconds(self.child.id()) {
            self.cpu_seconds = seconds;
        }
    }

    /// Close the input and drain the child and readers under one timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Total drain budget. The child is killed and reaped on expiry; a blocked callback is detached.
    ///
    /// # Returns
    ///
    /// Counts and CPU time after the child and readers finish.
    ///
    /// # Errors
    ///
    /// [`VideoError::Exit`] if the encoder failed, a reader panicked, or the deadline expired; otherwise the reader's error.
    pub fn finish(self, timeout: Duration) -> Result<EncoderStats, VideoError> {
        self.finish_until(Instant::now() + timeout)
    }

    pub(super) fn close_input(&mut self) {
        self.sample_cpu();
        drop(self.stdin.take());
    }

    pub(super) fn finish_until(mut self, deadline: Instant) -> Result<EncoderStats, VideoError> {
        self.close_input();
        let status = loop {
            match self.child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() < deadline => {
                    self.sample_cpu();
                    std::thread::sleep(Duration::from_millis(10));
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break None;
                }
            }
        };
        while (self.reader.as_ref().is_some_and(|r| !r.is_finished()) || self.stderr_reader.as_ref().is_some_and(|r| !r.is_finished()))
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        let (samples_out, bytes_out) = match self.reader.take() {
            Some(reader) if reader.is_finished() => reader.join().map_err(|_| VideoError::Exit {
                camera: self.camera, status: "reader thread panicked".into(), stderr: tail_text(&self.stderr_tail),
            })??,
            Some(_) => return Err(VideoError::Exit {
                camera: self.camera, status: "reader exceeded the flush deadline".into(), stderr: tail_text(&self.stderr_tail),
            }),
            None => (0, 0),
        };
        if let Some(reader) = self.stderr_reader.take() {
            if !reader.is_finished() {
                return Err(VideoError::Exit {
                    camera: self.camera, status: "stderr reader exceeded the flush deadline".into(), stderr: tail_text(&self.stderr_tail),
                });
            }
            let _ = reader.join();
        }
        let stats = EncoderStats {
            frames_in: self.frames_in,
            samples_out,
            bytes_out,
            cpu_seconds: self.cpu_seconds,
            wall_seconds: self.started.elapsed().as_secs_f64(),
        };
        match status {
            Some(status) if status.success() => Ok(stats),
            Some(status) => Err(VideoError::Exit { camera: self.camera, status: status.to_string(), stderr: tail_text(&self.stderr_tail) }),
            None => Err(VideoError::Exit { camera: self.camera, status: "killed after the flush timeout".into(), stderr: tail_text(&self.stderr_tail) }),
        }
    }
}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn read_units(
    camera: usize,
    mut stdout: impl Read,
    pending: &Mutex<VecDeque<i64>>,
    tail: &Mutex<VecDeque<String>>,
    mut sink: SampleSink,
) -> Result<(u64, u64), VideoError> {
    let mut splitter = AccessUnitSplitter::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut units = Vec::new();
    let (mut samples, mut bytes) = (0u64, 0u64);
    let mut emit = |unit: AccessUnit, samples: &mut u64, bytes: &mut u64| -> Result<(), VideoError> {
        let t_ns = pending.lock().ok().and_then(|mut p| p.pop_front()).ok_or(VideoError::Unmatched { camera })?;
        *samples += 1;
        *bytes += unit.data.len() as u64;
        sink(EncodedSample { camera, t_ns, unit });
        Ok(())
    };
    loop {
        let n = match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(source) => return Err(VideoError::Io { camera, source, stderr: tail_text(tail) }),
        };
        splitter.push(&buf[..n], &mut units);
        for unit in units.drain(..) {
            emit(unit, &mut samples, &mut bytes)?;
        }
    }
    if let Some(unit) = splitter.finish() {
        emit(unit, &mut samples, &mut bytes)?;
    }
    Ok((samples, bytes))
}

/// User + system CPU seconds of a process, from `/proc/<pid>/stat` (fields 14 and 15, in 100 Hz ticks).
///
/// # Arguments
///
/// * `pid` - The process (`std::process::id()` for this one).
///
/// # Returns
///
/// The seconds, or `None` when the file cannot be read or parsed (the process is gone, or not Linux).
pub fn process_cpu_seconds(pid: u32) -> Option<f64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name (field 2) may contain spaces; the fields after its closing parenthesis are space separated.
    let rest = &stat[stat.rfind(')')? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nal(kind: u8, first_slice: bool, payload: &[u8], four_byte: bool) -> Vec<u8> {
        let mut out = if four_byte { vec![0, 0, 0, 1] } else { vec![0, 0, 1] };
        out.push(0x60 | kind);
        if kind == 1 || kind == 5 {
            out.push(if first_slice { 0x88 } else { 0x08 });
        }
        out.extend_from_slice(payload);
        out
    }

    fn stream() -> (Vec<u8>, Vec<AccessUnit>) {
        let idr = [nal(9, false, &[0x10], true), nal(7, false, &[1, 2, 3], true), nal(8, false, &[4, 5], true), nal(5, true, &[9; 40], true)].concat();
        let p1 = [nal(1, true, &[7; 20], true), nal(1, false, &[7; 10], false)].concat();
        let p2 = [nal(1, true, &[8; 25], true), nal(12, false, &[0xff; 5], false)].concat();
        let all = [idr.clone(), p1.clone(), p2.clone()].concat();
        let expected = vec![
            AccessUnit { data: idr.into(), keyframe: true },
            AccessUnit { data: p1.into(), keyframe: false },
            AccessUnit { data: p2.into(), keyframe: false },
        ];
        (all, expected)
    }

    #[test]
    fn the_splitter_cuts_access_units_at_aud_sps_and_first_slices_for_any_chunking() {
        let (all, expected) = stream();
        for chunk in [1, 2, 3, 5, 7, 64, all.len()] {
            let mut splitter = AccessUnitSplitter::new();
            let mut units = Vec::new();
            for piece in all.chunks(chunk) {
                splitter.push(piece, &mut units);
            }
            units.extend(splitter.finish());
            assert_eq!(units, expected, "chunk size {chunk}");
        }
    }

    #[test]
    fn a_second_slice_of_the_same_picture_and_filler_stay_in_their_unit() {
        let (all, expected) = stream();
        let mut splitter = AccessUnitSplitter::new();
        let mut units = Vec::new();
        splitter.push(&all, &mut units);
        // The last unit is held until the stream ends.
        assert_eq!(units.len(), 2);
        assert_eq!(splitter.finish(), Some(expected[2].clone()));
        assert!(has_nal(&expected[0].data, 7) && has_nal(&expected[0].data, 8) && has_nal(&expected[0].data, 5));
        assert!(!has_nal(&expected[1].data, 5));
    }

    #[test]
    fn an_empty_or_slice_less_stream_yields_nothing() {
        let mut splitter = AccessUnitSplitter::new();
        let mut units = Vec::new();
        splitter.push(&nal(7, false, &[1, 2, 3], true), &mut units);
        assert!(units.is_empty());
        assert_eq!(splitter.finish(), None);
    }

    #[test]
    fn gst_arguments_frame_nv12_from_stdin_and_write_annex_b_to_stdout() {
        let config = EncoderConfig::mpp(ImageSize { width: 640, height: 360 }, 30, 1_000_000, 30);
        let line = config.args.join(" ");
        assert!(line.starts_with(
            "-q filesrc location=/dev/stdin blocksize=345600 ! video/x-raw,format=NV12,width=640,height=360,framerate=30/1 ! queue max-size-buffers=3 ! mpph264enc bps=1000000 gop=30"
        ));
        assert!(line.ends_with("! video/x-h264,stream-format=byte-stream,alignment=au ! fdsink fd=1 sync=false"));
    }

    #[test]
    fn encoder_kinds_parse_by_their_command_line_names() -> Result<(), VideoError> {
        let size = ImageSize { width: 640, height: 360 };
        assert_eq!(EncoderConfig::for_kind("mpp".parse()?, size, 30, 1_000_000, 30), EncoderConfig::mpp(size, 30, 1_000_000, 30));
        assert_eq!(EncoderConfig::for_kind("x264".parse()?, size, 30, 1_000_000, 30), EncoderConfig::x264(size, 30, 1_000_000, 30));
        assert_eq!(EncoderConfig::for_kind("openh264".parse()?, size, 30, 1_000_000, 30), EncoderConfig::openh264(size, 30, 1_000_000, 30));
        assert!(matches!("h265".parse::<EncoderKind>(), Err(VideoError::UnknownEncoder(_))));
        Ok(())
    }

    #[test]
    fn cpu_seconds_of_this_process_are_readable() {
        assert!(process_cpu_seconds(std::process::id()).is_some_and(|s| s >= 0.0));
    }

    /// End to end through a real software encoder, when the host has one (GStreamer's openh264enc, or ffmpeg's libx264).
    #[test]
    fn a_host_encoder_returns_one_access_unit_per_frame_with_the_input_timestamps() -> Result<(), VideoError> {
        let size = ImageSize { width: 640, height: 360 };
        let has = |program: &str, args: &[&str]| Command::new(program).args(args).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success());
        let config = if has("gst-inspect-1.0", &["openh264enc"]) {
            EncoderConfig::openh264(size, 30, 1_000_000, 10)
        } else if has("ffmpeg", &["-hide_banner", "-h", "encoder=libx264"]) {
            EncoderConfig::x264(size, 30, 1_000_000, 10)
        } else {
            eprintln!("skipped: no host H.264 encoder (openh264enc or ffmpeg libx264)");
            return Ok(());
        };
        let samples = Arc::new(Mutex::new(Vec::new()));
        let collected = samples.clone();
        let mut encoder = H264Encoder::spawn(3, &config, move |sample| {
            if let Ok(mut s) = collected.lock() {
                s.push(sample);
            }
        })?;
        for frame in 0..25u8 {
            let data: Vec<u8> = (0..size.width * size.height).map(|i| ((i % size.width) as u8).wrapping_add(frame * 4)).collect();
            let image = Image::<u8, 1>::new(size, data).map_err(|e| VideoError::Io { camera: 3, source: std::io::Error::other(e.to_string()), stderr: String::new() })?;
            encoder.push(1_000 + i64::from(frame) * 33_333_333, &image)?;
        }
        let stats = encoder.finish(Duration::from_secs(10))?;
        let samples = samples.lock().map(|s| s.clone()).unwrap_or_default();
        assert_eq!((stats.frames_in, stats.samples_out), (25, 25));
        assert_eq!(samples.len(), 25);
        assert!(samples.iter().all(|s| s.camera == 3));
        assert!(samples[0].unit.keyframe && has_nal(&samples[0].unit.data, 7));
        assert!(samples.iter().enumerate().all(|(i, s)| s.t_ns == 1_000 + i as i64 * 33_333_333));
        assert!(samples.iter().filter(|s| s.unit.keyframe).count() >= 2, "gop 10 over 25 frames");
        Ok(())
    }
}

#[cfg(test)]
mod shutdown_tests {
    use super::*;

    #[test]
    fn a_reader_spawn_failure_kills_and_reaps_the_started_encoder() {
        let path = std::env::temp_dir().join(format!("log-encoder-pid-{}", std::process::id()));
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec!["-c".into(), "echo $$ > \"$1\"; exec cat >/dev/null".into(), "encoder".into(), path.to_string_lossy().into_owned()],
            size: ImageSize { width: 2, height: 2 },
        };
        let result = H264Encoder::spawn_with_reader(0, &config, |_| {}, |_, _| {
            let deadline = Instant::now() + Duration::from_secs(2);
            while std::fs::read_to_string(&path).is_err() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(std::io::Error::other("injected reader spawn failure"))
        });
        assert!(matches!(result, Err(VideoError::Spawn { .. })));
        let pid: i32 = std::fs::read_to_string(&path).unwrap().trim().parse().unwrap();
        let mut status = 0;
        // SAFETY: waitpid with WNOHANG only queries this test's child; status is writable.
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) }, -1);
        assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ECHILD), "child must already be reaped");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_blocked_sample_callback_is_bounded_by_the_encoder_deadline() -> Result<(), VideoError> {
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec!["-c".into(), "cat >/dev/null; printf '\\000\\000\\001\\145\\200'".into()],
            size: ImageSize { width: 2, height: 2 },
        };
        let (release, wait) = std::sync::mpsc::channel();
        let mut encoder = H264Encoder::spawn(0, &config, move |_| { let _ = wait.recv(); })?;
        encoder.push(123, &Image::from_size_val(config.size, 0).unwrap())?;
        // Release eventually even with the old unbounded join, so a regression fails instead of hanging the suite.
        let unblock = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            let _ = release.send(());
        });
        let start = Instant::now();
        let result = encoder.finish(Duration::from_millis(100));
        let elapsed = start.elapsed();
        unblock.join().unwrap();
        eprintln!("blocked callback shutdown: {elapsed:?}");
        assert!(result.is_err(), "a callback that misses the deadline must be reported");
        assert!(elapsed < Duration::from_millis(500), "reader join took {elapsed:?}");
        Ok(())
    }
}
