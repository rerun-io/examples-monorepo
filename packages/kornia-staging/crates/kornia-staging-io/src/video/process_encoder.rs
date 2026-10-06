#![allow(unsafe_code)] // FFI calls uphold the documented buffer and ownership contracts.
//! Subprocess H.264 encoding: NV12 on stdin, Annex-B access units on stdout.
//! Input timestamps are matched in FIFO order; commands must disable frame reordering.
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{AccessUnit, AccessUnitSplitter};
use kornia_image::{Image, ImageSize};

/// Errors of the video encoder.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// No encoder element was supplied to GStreamer.
    #[error("the encoder element must not be empty")]
    MissingEncoder,
    /// NV12 requires nonzero even dimensions whose storage fits in memory.
    #[error("invalid NV12 geometry {width}x{height}")]
    Geometry {
        /// Requested width.
        width: usize,
        /// Requested height.
        height: usize,
    },
    /// An encoder requires a positive frame rate.
    #[error("invalid encoder rate {fps}")]
    Rate {
        /// Requested frames per second.
        fps: u32,
    },
    /// The encoder process could not be started.
    #[error("could not start the H.264 encoder `{program}`: {source}")]
    Spawn {
        /// The encoder command.
        program: String,
        /// Why it did not start.
        source: std::io::Error,
    },
    /// Writing a frame to, or reading the stream from, the encoder failed (the encoder probably exited).
    #[error("H.264 encoder for {source}; encoder stderr: {stderr}")]
    Io {
        /// The pipe error.
        source: std::io::Error,
        /// The encoder's stderr so far.
        stderr: String,
    },
    /// A frame did not have the encoder's size.
    #[error("frame {got:?}, the encoder takes {expected:?}")]
    Size {
        /// The frame's size.
        got: ImageSize,
        /// The encoder's size.
        expected: ImageSize,
    },
    /// The encoder exited with a failure status.
    #[error("H.264 encoder exited with {status}; stderr: {stderr}")]
    Exit {
        /// Its exit status.
        status: String,
        /// Its stderr.
        stderr: String,
    },
    /// The encoder produced more access units than frames it was given.
    #[error("H.264 encoder: an access unit without a pending frame")]
    Unmatched,
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

fn validate_size(size: ImageSize) -> Result<(), VideoError> {
    if size.width == 0
        || size.height == 0
        || !size.width.is_multiple_of(2)
        || !size.height.is_multiple_of(2)
        || size
            .width
            .checked_mul(size.height)
            .and_then(|n| n.checked_mul(3))
            .is_none()
    {
        return Err(VideoError::Geometry {
            width: size.width,
            height: size.height,
        });
    }
    Ok(())
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
        format!(
            "video/x-raw,format=NV12,width={},height={},framerate={fps}/1",
            size.width, size.height
        ),
        "!".into(),
        // filesrc reads the next frame while the encoder works on this one, so the logger's pipe write returns at once.
        "queue".into(),
        "max-size-buffers=3".into(),
        "!".into(),
    ];
    args.extend(encoder.iter().map(|s| s.to_string()));
    args.extend(
        [
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    args
}

impl EncoderConfig {
    /// Configure an arbitrary NV12-to-Annex-B command.
    /// The command must not leave descendants holding its stdin, stdout or stderr pipes open.
    /// Only the direct child is terminated and reaped; see [`H264Encoder::finish`].
    /// # Arguments
    /// * `program` - executable to start.
    /// * `args` - command arguments.
    /// * `size` - nonzero even dimensions of every input frame.
    /// # Errors
    /// Rejects invalid or overflowing NV12 geometry.
    pub fn new(program: String, args: Vec<String>, size: ImageSize) -> Result<Self, VideoError> {
        validate_size(size)?;
        Ok(Self {
            program,
            args,
            size,
        })
    }

    /// Build a gst-launch pipeline around any caller-selected H.264 encoder element.
    /// The encoder must emit one access unit per frame, in input order: disable B-frame
    /// reordering and frame dropping. Timestamp matching uses an input FIFO.
    /// ```
    /// use kornia_image::ImageSize;
    /// use kornia_staging_io::video::EncoderConfig;
    /// let config = EncoderConfig::gst(ImageSize { width: 640, height: 360 }, 30,
    ///     &["x264enc", "tune=zerolatency", "bframes=0"])?;
    /// assert_eq!(config.program, "gst-launch-1.0");
    /// # Ok::<(), kornia_staging_io::video::VideoError>(())
    /// ```
    /// # Arguments
    /// * `size` - even NV12 frame dimensions.
    /// * `fps` - positive nominal frame rate.
    /// * `encoder` - encoder element and properties, with optional conversion elements.
    /// # Errors
    /// Rejects zero/odd/overflowing geometry, zero rate, or an empty encoder.
    pub fn gst(size: ImageSize, fps: u32, encoder: &[&str]) -> Result<Self, VideoError> {
        validate_size(size)?;
        if fps == 0 {
            return Err(VideoError::Rate { fps });
        }
        if encoder.first().is_none_or(|name| name.trim().is_empty()) {
            return Err(VideoError::MissingEncoder);
        }
        Ok(Self {
            program: "gst-launch-1.0".into(),
            args: gst_launch_args(size, fps, encoder),
            size,
        })
    }
}

/// One encoded frame of one camera, matched to the timestamp of the frame that went in.
#[derive(Clone, Debug)]
pub struct EncodedSample {
    /// The input frame's time (whatever the caller passed to [`H264Encoder::push`]).
    pub timestamp_ns: i64,
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
    /// Wall seconds since the encoder started.
    pub wall_seconds: f64,
}

type SampleSink = Box<dyn FnMut(EncodedSample) + Send>;
type ReaderTask = Box<dyn FnOnce() -> Result<(u64, u64), VideoError> + Send>;

/// A running encoder child process for one camera.
pub struct H264Encoder {
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
    tail.lock()
        .map(|lines| lines.iter().cloned().collect::<Vec<_>>().join(" | "))
        .unwrap_or_default()
}

impl H264Encoder {
    /// Start the encoder subprocess.
    ///
    /// # Arguments
    ///
    /// * `config` - The encoder command.
    /// * `sink` - Called on the encoder's reader thread with every access unit, in order.
    ///   If it blocks past the finish deadline, its thread is detached and callbacks may
    ///   continue after [`Self::finish`] returns; a timeout does not establish callback completion.
    ///
    /// # Errors
    ///
    /// [`VideoError::Spawn`] if the program or a reader thread cannot be started.
    ///
    /// # Returns
    ///
    /// An encoder that owns the child process and its input and readers.
    /// Custom commands must emit one access unit per input frame in input order.
    /// Reordered pictures cannot be matched by the timestamp FIFO.
    /// Termination and reaping cover only the direct child, not its process group or descendants.
    /// Commands must not leave descendants holding the encoder's stdin, stdout or stderr pipes open.
    pub fn spawn(
        config: &EncoderConfig,
        sink: impl FnMut(EncodedSample) + Send + 'static,
    ) -> Result<Self, VideoError> {
        Self::spawn_with_reader(config, sink, |builder, task| builder.spawn(task))
    }

    fn spawn_with_reader(
        config: &EncoderConfig,
        sink: impl FnMut(EncodedSample) + Send + 'static,
        start_reader: impl FnOnce(
            std::thread::Builder,
            ReaderTask,
        )
            -> std::io::Result<JoinHandle<Result<(u64, u64), VideoError>>>,
    ) -> Result<Self, VideoError> {
        validate_size(config.size)?;
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Its own process group: a Ctrl-C to the runtime's group must not kill the encoder before it flushes; closing its
        // stdin ends it.
        command.process_group(0);
        let mut child = command.spawn().map_err(|source| VideoError::Spawn {
            program: config.program.clone(),
            source,
        })?;
        let stdin = child.stdin.take();
        if let Some(stdin) = &stdin {
            grow_pipe(stdin, PIPE_BYTES);
        }
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // Own the child before either fallible thread spawn: every early return kills and reaps it through Drop.
        let mut encoder = Self {
            size: config.size,
            child,
            stdin,
            pending: Arc::default(),
            reader: None,
            stderr_tail: Arc::default(),
            stderr_reader: None,
            chroma: vec![128u8; config.size.width * config.size.height / 2],
            frames_in: 0,
            started: Instant::now(),
        };
        if let Some(stderr) = stderr {
            let tail = encoder.stderr_tail.clone();
            encoder.stderr_reader = Some(
                std::thread::Builder::new()
                    .name("h264-stderr".into())
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
                    .map_err(|source| VideoError::Spawn {
                        program: "stderr thread".into(),
                        source,
                    })?,
            );
        }
        if let Some(stdout) = stdout {
            let pending = encoder.pending.clone();
            let tail = encoder.stderr_tail.clone();
            let sink: SampleSink = Box::new(sink);
            encoder.reader = Some(
                start_reader(
                    std::thread::Builder::new().name("h264-read".into()),
                    Box::new(move || read_units(stdout, &pending, &tail, sink)),
                )
                .map_err(|source| VideoError::Spawn {
                    program: "reader thread".into(),
                    source,
                })?,
            );
        }
        Ok(encoder)
    }

    /// Encode one grey frame: its luma plane plus constant 128 chroma.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ns` - The frame's time; the matching [`EncodedSample`] carries it.
    /// * `luma` - A host-accessible frame of the encoder's size; no implicit download is performed.
    ///
    /// # Panics
    /// Panics for device-only storage before admitting the frame's timestamp.
    ///
    /// # Errors
    ///
    /// [`VideoError::Size`] for a frame of another size, [`VideoError::Io`] if the encoder no longer accepts input.
    pub fn push(&mut self, timestamp_ns: i64, luma: &Image<u8, 1>) -> Result<(), VideoError> {
        if luma.size() != self.size {
            return Err(VideoError::Size {
                got: luma.size(),
                expected: self.size,
            });
        }
        assert!(
            luma.storage.domain().is_host_accessible(),
            "encoder luma must be host-accessible"
        );
        let pixels = luma.as_slice();
        let io_error = |source: std::io::Error, tail: &Mutex<VecDeque<String>>| VideoError::Io {
            source,
            stderr: tail_text(tail),
        };
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(io_error(
                std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "encoder input already closed",
                ),
                &self.stderr_tail,
            ));
        };
        // Queue the timestamp first: the access unit may come back before this call returns.
        if let Ok(mut pending) = self.pending.lock() {
            pending.push_back(timestamp_ns);
        }
        if let Err(error) = stdin
            .write_all(pixels)
            .and_then(|()| stdin.write_all(&self.chroma))
        {
            drop(self.stdin.take());
            return Err(io_error(error, &self.stderr_tail));
        }
        self.frames_in += 1;

        Ok(())
    }

    /// Number of input frames accepted by this encoder.
    pub fn frames_in(&self) -> u64 {
        self.frames_in
    }

    /// Encoder child process id for caller-owned telemetry.
    pub fn process_id(&self) -> u32 {
        self.child.id()
    }

    /// Close stdin so the child can flush.
    pub fn close_input(&mut self) {
        drop(self.stdin.take());
    }

    /// Finish while calling an observer with the child PID before each wait.
    ///
    /// On timeout, only the direct child is killed and reaped. Descendants are not terminated
    /// and must not retain the encoder's stdin, stdout or stderr pipes.
    /// Reader threads that miss the deadline are detached. A blocked sink can resume and
    /// callbacks can still run after this method returns; timeout does not prove completion.
    /// # Arguments
    /// * `deadline` - absolute deadline for waiting on the child and readers, after which
    ///   child termination and reaping are attempted; this does not bound callback lifetime.
    /// * `observe` - caller-owned telemetry hook; it must return promptly.
    /// # Errors
    /// Returns child, reader or timeout failures.
    pub fn finish(
        mut self,
        deadline: Instant,
        mut observe: impl FnMut(u32),
    ) -> Result<EncoderStats, VideoError> {
        let tail = self.stderr_tail.clone();
        let exit = |status: String| VideoError::Exit {
            status,
            stderr: tail_text(&tail),
        };
        observe(self.child.id());
        self.close_input();
        let status = loop {
            observe(self.child.id());
            match self.child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break None;
                }
            }
        };
        while (self.reader.as_ref().is_some_and(|r| !r.is_finished())
            || self
                .stderr_reader
                .as_ref()
                .is_some_and(|r| !r.is_finished()))
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        let (samples_out, bytes_out) = match self.reader.take() {
            Some(reader) if reader.is_finished() => reader
                .join()
                .map_err(|_| exit("reader thread panicked".into()))??,
            Some(_) => return Err(exit("reader exceeded the flush deadline".into())),
            None => (0, 0),
        };
        if let Some(reader) = self.stderr_reader.take() {
            if !reader.is_finished() {
                return Err(exit("stderr reader exceeded the flush deadline".into()));
            }
            let _ = reader.join();
        }
        let stats = EncoderStats {
            frames_in: self.frames_in,
            samples_out,
            bytes_out,
            wall_seconds: self.started.elapsed().as_secs_f64(),
        };
        match status {
            Some(status) if status.success() => {
                if samples_out != self.frames_in {
                    return Err(exit(format!(
                        "encoded {samples_out} of {} input frames",
                        self.frames_in
                    )));
                }
                Ok(stats)
            }
            Some(status) => Err(exit(status.to_string())),
            None => Err(exit("killed after the flush timeout".into())),
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
    mut stdout: impl Read,
    pending: &Mutex<VecDeque<i64>>,
    tail: &Mutex<VecDeque<String>>,
    mut sink: SampleSink,
) -> Result<(u64, u64), VideoError> {
    let mut splitter = AccessUnitSplitter::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut units = Vec::new();
    let (mut samples, mut bytes) = (0u64, 0u64);
    let mut emit =
        |unit: AccessUnit, samples: &mut u64, bytes: &mut u64| -> Result<(), VideoError> {
            let timestamp_ns = pending
                .lock()
                .ok()
                .and_then(|mut p| p.pop_front())
                .ok_or(VideoError::Unmatched)?;
            *samples += 1;
            *bytes += unit.data.len() as u64;
            sink(EncodedSample { timestamp_ns, unit });
            Ok(())
        };
    loop {
        let n = match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(source) => {
                return Err(VideoError::Io {
                    source,
                    stderr: tail_text(tail),
                })
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(unsafe_code)] // A live host allocation is marked device-only; no device pointer is read.
    fn device_luma_is_refused_before_timestamp_admission() {
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec![
                "-c".into(),
                "cat >/dev/null; printf '\\000\\000\\001\\145\\200'".into(),
            ],
            size: ImageSize {
                width: 2,
                height: 2,
            },
        };
        let samples = Arc::new(Mutex::new(Vec::new()));
        let received = samples.clone();
        let mut encoder = H264Encoder::spawn(&config, move |sample| {
            received.lock().unwrap().push(sample.timestamp_ns);
        })
        .unwrap();
        let backing = Arc::new(vec![0u8; 4]);
        let device = unsafe {
            Image::from_borrowed(
                config.size,
                backing.as_ptr(),
                kornia_tensor::MemoryDomain::Device { id: 0 },
                backing,
            )
            .unwrap()
        };
        let rejected =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encoder.push(11, &device)));
        assert!(rejected.is_err());
        encoder
            .push(22, &Image::from_size_val(config.size, 0).unwrap())
            .unwrap();
        encoder
            .finish(Instant::now() + Duration::from_secs(2), |_| {})
            .unwrap();
        assert_eq!(*samples.lock().unwrap(), vec![22]);
    }

    #[test]
    fn arbitrary_encoder_configuration_rejects_invalid_input() {
        let size = ImageSize {
            width: 12,
            height: 8,
        };
        assert!(
            EncoderConfig::gst(size, 30, &["customh264enc", "bitrate=1000"])
                .unwrap()
                .args
                .iter()
                .any(|s| s == "customh264enc")
        );
        assert!(EncoderConfig::gst(size, 0, &["encoder"]).is_err());
        assert!(EncoderConfig::gst(size, 30, &[]).is_err());
        assert!(EncoderConfig::gst(size, 30, &[""]).is_err());
        assert!(EncoderConfig::gst(
            ImageSize {
                width: 3,
                height: 8
            },
            30,
            &["encoder"]
        )
        .is_err());
        assert!(EncoderConfig::gst(
            ImageSize {
                width: usize::MAX - 1,
                height: 8
            },
            30,
            &["encoder"]
        )
        .is_err());
    }

    #[test]
    fn failed_frame_write_closes_input_permanently() {
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec!["-c".into(), "head -c 1 >/dev/null".into()],
            size: ImageSize {
                width: 2048,
                height: 2048,
            },
        };
        let mut encoder = H264Encoder::spawn(&config, |_| {}).unwrap();
        let frame = Image::from_size_val(config.size, 0u8).unwrap();
        assert!(encoder.push(1, &frame).is_err());
        assert!(encoder
            .push(2, &frame)
            .unwrap_err()
            .to_string()
            .contains("input already closed"));
    }

    #[test]
    fn successful_child_exit_rejects_missing_output_frames() {
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec!["-c".into(), "cat >/dev/null".into()],
            size: ImageSize {
                width: 2,
                height: 2,
            },
        };
        let mut encoder = H264Encoder::spawn(&config, |_| {}).unwrap();
        encoder
            .push(1, &Image::from_size_val(config.size, 0u8).unwrap())
            .unwrap();
        assert!(encoder
            .finish(Instant::now() + Duration::from_secs(2), |_| {})
            .is_err());
    }

    #[test]
    fn a_reader_spawn_failure_kills_and_reaps_the_started_encoder() {
        let path = std::env::temp_dir().join(format!("log-encoder-pid-{}", std::process::id()));
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec![
                "-c".into(),
                "echo $$ > \"$1\"; exec cat >/dev/null".into(),
                "encoder".into(),
                path.to_string_lossy().into_owned(),
            ],
            size: ImageSize {
                width: 2,
                height: 2,
            },
        };
        let result = H264Encoder::spawn_with_reader(
            &config,
            |_| {},
            |_, _| {
                let deadline = Instant::now() + Duration::from_secs(2);
                while std::fs::read_to_string(&path).is_err() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(std::io::Error::other("injected reader spawn failure"))
            },
        );
        assert!(matches!(result, Err(VideoError::Spawn { .. })));
        let pid: i32 = std::fs::read_to_string(&path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let mut status = 0;
        // SAFETY: waitpid with WNOHANG only queries this test's child; status is writable.
        assert_eq!(
            unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD),
            "child must already be reaped"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_blocked_sample_callback_is_bounded_by_the_encoder_deadline() -> Result<(), VideoError> {
        let config = EncoderConfig {
            program: "sh".into(),
            args: vec![
                "-c".into(),
                "cat >/dev/null; printf '\\000\\000\\001\\145\\200'".into(),
            ],
            size: ImageSize {
                width: 2,
                height: 2,
            },
        };
        let (release, wait) = std::sync::mpsc::channel();
        let mut encoder = H264Encoder::spawn(&config, move |_| {
            let _ = wait.recv();
        })?;
        encoder.push(123, &Image::from_size_val(config.size, 0).unwrap())?;
        // Release eventually even with the old unbounded join, so a regression fails instead of hanging the suite.
        let unblock = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            let _ = release.send(());
        });
        let start = Instant::now();
        let result = encoder.finish(Instant::now() + Duration::from_millis(100), |_| {});
        let elapsed = start.elapsed();
        unblock.join().unwrap();
        eprintln!("blocked callback shutdown: {elapsed:?}");
        assert!(
            result.is_err(),
            "a callback that misses the deadline must be reported"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "reader join took {elapsed:?}"
        );
        Ok(())
    }
}
