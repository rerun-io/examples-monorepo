//! The live source on Cap A or Cap B: six rkisp mainpath cameras (V4L2 mplane NV12 1920x1080; luma lent out zero-copy or
//! copied out, [`LiveConfig::capture`]) triggered at 30 fps, and IMU0 (gyro + accel over IIO, accel interpolated onto the gyro
//! stamps), all on CLOCK_MONOTONIC.
//!
//! The device order is PR #270's (`robocap-recorder/src/session.rs` at 271ce643): check that this is a known cap, that the
//! vendor recorder is gone and that the `rig.json` is the cap's, stop the frame trigger, open the cameras (STREAMON), enable
//! the IIO buffers, start one capture thread per device, then start the trigger. On drop: stop flag, trigger stop, join the
//! threads (each closes its device; IIO attributes are restored in reverse order). In zero-copy mode a camera closes when its
//! thread has ended and the last frame over its buffers has dropped (after the trigger has stopped, so no new frames arrive).
//!
//! IMU timestamps are shifted by [`LiveConfig::imu_time_offset_ns`] (default: DataForge's -14.9 ms) onto the camera clock, the
//! same alignment the catalog (and so the replay dumps) use; the clock guard checks the raw kernel stamps.
//!
//! Event order: combined IMU samples go out as soon as they exist; a frameset goes out once every camera's frame for that
//! trigger instant has arrived (frames within 3 ms) and the IMU has reached its time (or after [`LiveConfig::imu_catchup`]),
//! so the IMU samples up to a frameset's time always come before it. IMU samples newer than a frameset may precede it.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{FrameSource, SourceError, SourceEvent};
use crate::capture::camera::{Camera, CaptureMode, CaptureStream, LumaFrame};
use crate::capture::device::{FrameTrigger, IioDevice, monotonic_ns, require_cap, require_vendor_recorder_stopped};
use crate::capture::iio::{IioScan, MotionKind};
use crate::capture::imu::{ClockCheck, ClockGuard, ImuCombiner};
use crate::capture::matcher::FramesetMatcher;
use crate::capture::{
    ACCEL_SCALE, CAMERA_DEVICES, CaptureError, DATAFORGE_IMU_TIME_OFFSET_NS, DEFAULT_RIG_PATH, GYRO_SCALE, IMU0_ACCEL_IIO, IMU0_GYRO_IIO, ROBOT_JSON,
    vendor_turned_180,
};
use crate::frame::{CameraFrame, FrameMeta, Frameset, NUM_CAMERAS, Rig};

/// Settings of the live source.
#[derive(Clone, Debug)]
pub struct LiveConfig {
    /// The cap's `rig.json` (the hands need it; SLAM uses the cap's embedded calibration). Its `device` must be this cap's.
    pub rig_path: PathBuf,
    /// The IMU clock guard.
    pub clock_guard: ClockGuard,
    /// Frames this close belong to one trigger instant.
    pub match_tolerance_ns: i64,
    /// A frameset still incomplete when frames this much newer exist goes out partial.
    pub match_max_wait_ns: i64,
    /// How long a complete frameset waits for the IMU to reach its time.
    pub imu_catchup: Duration,
    /// Frames waiting between the camera threads and the matcher before a camera thread drops (and counts) a frame.
    pub max_frames_in_flight: u64,
    /// Print the capture counters this often.
    pub report_every: Duration,
    /// Added to every IIO timestamp after the clock guard: puts the IMU on the camera clock the calibration expects. The default
    /// is DataForge's RoboCap alignment (`CAMERA_TO_IMU_OFFSET_NS`, Basalt's `kCameraToImuOffsetNs`, the median of Cap A's
    /// factory Kalibr `timeshift_cam_imu`): the catalog, and so the s66 dump, stores IMU at raw time - 14.9 ms.
    pub imu_time_offset_ns: i64,
    /// How frames leave the capture buffers. Zero-copy by default: on Cap A the mapping is cached memory, so readers lose nothing
    /// and the camera threads skip a 2 MB memcpy per frame.
    pub capture: CaptureMode,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            rig_path: PathBuf::from(DEFAULT_RIG_PATH),
            clock_guard: ClockGuard::default(),
            match_tolerance_ns: 3_000_000,
            match_max_wait_ns: 100_000_000,
            imu_catchup: Duration::from_millis(40),
            max_frames_in_flight: 24,
            report_every: Duration::from_secs(5),
            imu_time_offset_ns: DATAFORGE_IMU_TIME_OFFSET_NS,
            capture: CaptureMode::ZeroCopy,
        }
    }
}

enum CaptureEvent {
    Frame { camera: usize, frame: LumaFrame },
    Scans { kind: MotionKind, scale: f64, scans: Vec<IioScan> },
    Fault(String),
}

/// Counters shared with the capture threads.
#[derive(Default)]
struct Shared {
    frames: [AtomicU64; NUM_CAMERAS],
    sequence_gaps: [AtomicU64; NUM_CAMERAS],
    dropped_in_flight: [AtomicU64; NUM_CAMERAS],
    fallback_copies: [AtomicU64; NUM_CAMERAS],
    in_flight: AtomicU64,
    gyro: AtomicU64,
    accel: AtomicU64,
    future_skew: AtomicU64,
    future_dropped: AtomicU64,
    max_future_skew_ns: AtomicU64,
}

/// The six cameras and IMU0 of the cap as a [`FrameSource`].
pub struct LiveSource {
    rig: Rig,
    options: LiveConfig,
    rx: mpsc::Receiver<CaptureEvent>,
    matcher: FramesetMatcher,
    combiner: ImuCombiner,
    ready: VecDeque<SourceEvent>,
    pending: VecDeque<(Frameset, Instant)>,
    newest_imu_ns: Option<i64>,
    stop: Arc<AtomicBool>,
    local_stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    trigger: Option<FrameTrigger>,
    started: Instant,
    last_report: Instant,
    last_frames: [u64; NUM_CAMERAS],
    last_imu: (u64, u64),
    imu_late_framesets: u64,
    /// The cameras the vendor turns 180 degrees (`/userdata/robot.json`): their frames are stamped [`FrameMeta::turned_180`].
    turned_180: [bool; NUM_CAMERAS],
}

fn device_error(error: CaptureError) -> SourceError {
    SourceError::Device(error.to_string())
}

impl LiveSource {
    /// Take over the cameras, trigger and IMU0 (the vendor recorder must already be stopped by `robocap-panel handoff`) and
    /// start capturing. `stop` ends the source at the next event.
    ///
    /// # Errors
    ///
    /// [`SourceError::Device`] when this is not Cap A or Cap B, the rig is another cap's, the vendor recorder still owns capture,
    /// or a device refuses; whatever was opened is closed and restored again.
    pub fn open(options: LiveConfig, stop: Arc<AtomicBool>) -> Result<Self, SourceError> {
        let cap = require_cap().map_err(device_error)?;
        require_vendor_recorder_stopped().map_err(device_error)?;
        let rig = Rig::load(&options.rig_path)?;
        if rig.device != cap.device() {
            return Err(SourceError::Device(format!(
                "{} is the rig of device {:?}, but this is {}: pass this cap's rig.json (--rig)",
                options.rig_path.display(),
                rig.device,
                cap.device()
            )));
        }
        let robot = std::fs::read_to_string(ROBOT_JSON).map_err(|source| device_error(CaptureError::Io { what: format!("read {ROBOT_JSON}"), source }))?;
        let turned_180 = vendor_turned_180(&robot).map_err(device_error)?;
        let turned: Vec<&str> = (0..NUM_CAMERAS).filter(|&c| turned_180[c]).map(|c| crate::frame::CAMERA_NAMES[c]).collect();
        eprintln!("robocap-live: the vendor turns {turned:?} 180 degrees ({ROBOT_JSON}): their frames are read upright");
        let trigger = FrameTrigger::stopped().map_err(device_error)?;
        let mut cameras = Vec::with_capacity(NUM_CAMERAS);
        for path in CAMERA_DEVICES {
            cameras.push(Camera::open(path).map_err(device_error)?);
        }
        let gyro = IioDevice::start(IMU0_GYRO_IIO, MotionKind::Gyro).map_err(device_error)?;
        let accel = IioDevice::start(IMU0_ACCEL_IIO, MotionKind::Accel).map_err(device_error)?;
        for (device, expected) in [(&gyro, GYRO_SCALE), (&accel, ACCEL_SCALE)] {
            if (device.scale - expected).abs() > 1e-9 {
                eprintln!("robocap-live: iio:device{} scale {} differs from PR #270's {expected}", device.index, device.scale);
            }
        }
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared::default());
        let local_stop = Arc::new(AtomicBool::new(false));
        let mut source = Self {
            rig,
            matcher: FramesetMatcher::new(options.match_tolerance_ns, options.match_max_wait_ns),
            options,
            rx,
            combiner: ImuCombiner::default(),
            ready: VecDeque::new(),
            pending: VecDeque::new(),
            newest_imu_ns: None,
            stop,
            local_stop,
            shared,
            threads: Vec::new(),
            trigger: Some(trigger),
            started: Instant::now(),
            last_report: Instant::now(),
            last_frames: [0; NUM_CAMERAS],
            last_imu: (0, 0),
            imu_late_framesets: 0,
            turned_180,
        };
        for (camera, device) in cameras.into_iter().enumerate() {
            let device = CaptureStream::new(device, source.options.capture);
            let (tx, shared, stop) = (tx.clone(), source.shared.clone(), source.local_stop.clone());
            let max_in_flight = source.options.max_frames_in_flight;
            let handle = std::thread::Builder::new()
                .name(format!("rl-cam-{camera}"))
                .spawn(move || camera_thread(camera, device, &tx, &shared, &stop, max_in_flight))
                .map_err(|e| SourceError::Device(format!("spawn camera thread: {e}")))?;
            source.threads.push(handle);
        }
        for (device, kind) in [(gyro, MotionKind::Gyro), (accel, MotionKind::Accel)] {
            let (tx, shared, stop, guard) = (tx.clone(), source.shared.clone(), source.local_stop.clone(), source.options.clock_guard);
            let name = match kind {
                MotionKind::Gyro => "rl-iio-gyro",
                MotionKind::Accel => "rl-iio-accel",
            };
            let handle = std::thread::Builder::new()
                .name(name.into())
                .spawn(move || iio_thread(kind, device, &tx, &shared, &stop, guard))
                .map_err(|e| SourceError::Device(format!("spawn IIO thread: {e}")))?;
            source.threads.push(handle);
        }
        drop(tx);
        if let Some(trigger) = &source.trigger {
            trigger.start().map_err(device_error)?;
        }
        source.started = Instant::now();
        source.last_report = Instant::now();
        eprintln!("robocap-live: live capture started (6 cameras, {:?} capture, IMU0 gyro + accel, trigger 30 fps)", source.options.capture);
        Ok(source)
    }

    /// A one-line summary of the capture counters.
    pub fn report(&self) -> String {
        let frames: Vec<u64> = self.shared.frames.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let gaps: Vec<u64> = self.shared.sequence_gaps.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let dropped: Vec<u64> = self.shared.dropped_in_flight.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let copies: Vec<u64> = self.shared.fallback_copies.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let counts = self.matcher.counts;
        format!(
            "live: {:.1} s frames {frames:?} seq_gaps {gaps:?} dropped {dropped:?} fallback_copies {copies:?} | gyro {} accel {} future_skew {} dropped {} (max {:.3} ms) | \
             framesets complete {} partial {} dup {} late {} imu_late {} | combiner {:?}",
            self.started.elapsed().as_secs_f64(),
            self.shared.gyro.load(Ordering::Relaxed),
            self.shared.accel.load(Ordering::Relaxed),
            self.shared.future_skew.load(Ordering::Relaxed),
            self.shared.future_dropped.load(Ordering::Relaxed),
            self.shared.max_future_skew_ns.load(Ordering::Relaxed) as f64 / 1e6,
            counts.complete,
            counts.partial,
            counts.duplicate,
            counts.late,
            self.imu_late_framesets,
            self.combiner.counts,
        )
    }

    fn periodic_report(&mut self) -> Result<(), SourceError> {
        let elapsed = self.last_report.elapsed();
        if elapsed < self.options.report_every {
            return Ok(());
        }
        let frames: [u64; NUM_CAMERAS] = std::array::from_fn(|c| self.shared.frames[c].load(Ordering::Relaxed));
        let fps: Vec<String> = frames.iter().zip(self.last_frames.iter()).map(|(a, b)| format!("{:.1}", (a - b) as f64 / elapsed.as_secs_f64())).collect();
        let imu = (self.shared.gyro.load(Ordering::Relaxed), self.shared.accel.load(Ordering::Relaxed));
        eprintln!(
            "robocap-live: camera fps [{}] gyro {:.0} Hz accel {:.0} Hz | {}",
            fps.join(", "),
            (imu.0 - self.last_imu.0) as f64 / elapsed.as_secs_f64(),
            (imu.1 - self.last_imu.1) as f64 / elapsed.as_secs_f64(),
            self.report()
        );
        if self.started.elapsed() > Duration::from_secs(3) && (frames.contains(&0) || imu.0 == 0 || imu.1 == 0) {
            return Err(SourceError::Device(format!("a stream failed to start after 3 s: {}", self.report())));
        }
        self.last_frames = frames;
        self.last_imu = imu;
        self.last_report = Instant::now();
        Ok(())
    }
}

impl FrameSource for LiveSource {
    fn rig(&self) -> &Rig {
        &self.rig
    }

    fn next_event(&mut self) -> Result<Option<SourceEvent>, SourceError> {
        loop {
            if let Some(event) = self.ready.pop_front() {
                return Ok(Some(event));
            }
            if self.stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            if let Some((frameset, waiting)) = self.pending.front() {
                let covered = self.newest_imu_ns.is_some_and(|t| t >= frameset.t_ns);
                if covered || waiting.elapsed() >= self.options.imu_catchup {
                    if !covered {
                        self.imu_late_framesets += 1;
                    }
                    if let Some((frameset, _)) = self.pending.pop_front() {
                        self.ready.push_back(SourceEvent::Frameset(frameset));
                    }
                    continue;
                }
            }
            self.periodic_report()?;
            match self.rx.recv_timeout(Duration::from_millis(5)) {
                Ok(CaptureEvent::Frame { camera, frame }) => {
                    self.shared.in_flight.fetch_sub(1, Ordering::Relaxed);
                    let meta = FrameMeta { seq: u64::from(frame.sequence), pts_ns: frame.timestamp_ns, source_id: camera as u32, turned_180: self.turned_180[camera] };
                    let full = Arc::new(frame.luma);
                    let mut out = Vec::new();
                    self.matcher.push(camera, CameraFrame { meta, full }, &mut out);
                    let now = Instant::now();
                    self.pending.extend(out.into_iter().map(|frameset| (frameset, now)));
                }
                Ok(CaptureEvent::Scans { kind, scale, scans }) => {
                    for scan in scans {
                        let xyz = scan.raw.map(|v| f64::from(v) * scale);
                        let t_ns = scan.timestamp_ns + self.options.imu_time_offset_ns;
                        match kind {
                            MotionKind::Gyro => self.combiner.push_gyro(t_ns, xyz),
                            MotionKind::Accel => self.combiner.push_accel(t_ns, xyz),
                        }
                    }
                    while let Some(sample) = self.combiner.pop() {
                        self.newest_imu_ns = Some(sample.t_ns);
                        self.ready.push_back(SourceEvent::Imu(sample));
                    }
                }
                Ok(CaptureEvent::Fault(message)) => return Err(SourceError::Device(message)),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
            }
        }
    }
}

impl Drop for LiveSource {
    fn drop(&mut self) {
        self.local_stop.store(true, Ordering::Relaxed);
        drop(self.trigger.take());
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
        eprintln!("robocap-live: capture stopped; {}", self.report());
    }
}

/// One camera's capture loop, until `stop` (the source's own flag, set when it drops), a fault or a closed channel.
fn camera_thread(
    camera: usize,
    mut device: CaptureStream,
    tx: &mpsc::Sender<CaptureEvent>,
    shared: &Shared,
    stop: &AtomicBool,
    max_in_flight: u64,
) {
    let mut previous: Option<u32> = None;
    while !stop.load(Ordering::Relaxed) {
        match device.next_frame(200) {
            Ok(Some(frame)) => {
                if let Some(last) = previous
                    && frame.sequence != last.wrapping_add(1)
                {
                    shared.sequence_gaps[camera].fetch_add(u64::from(frame.sequence.wrapping_sub(last).wrapping_sub(1)), Ordering::Relaxed);
                }
                previous = Some(frame.sequence);
                shared.frames[camera].fetch_add(1, Ordering::Relaxed);
                shared.fallback_copies[camera].store(device.fallback_copies(), Ordering::Relaxed);
                if shared.in_flight.load(Ordering::Relaxed) >= max_in_flight {
                    shared.dropped_in_flight[camera].fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                shared.in_flight.fetch_add(1, Ordering::Relaxed);
                if tx.send(CaptureEvent::Frame { camera, frame }).is_err() {
                    break;
                }
            }
            Ok(None) => {}
            Err(error) => {
                let _ = tx.send(CaptureEvent::Fault(format!("camera {camera} ({}): {error}", CAMERA_DEVICES[camera])));
                break;
            }
        }
    }
}

/// One IIO device's read loop, until `stop` (the source's own flag, set when it drops), a fault or a closed channel. Every scan
/// passes the clock guard: future skew is counted (and kept or dropped), a wrong clock or a backlog ends the run.
fn iio_thread(
    kind: MotionKind,
    mut device: IioDevice,
    tx: &mpsc::Sender<CaptureEvent>,
    shared: &Shared,
    stop: &AtomicBool,
    guard: ClockGuard,
) {
    let (what, counter) = match kind {
        MotionKind::Gyro => ("IMU0 gyro", &shared.gyro),
        MotionKind::Accel => ("IMU0 accel", &shared.accel),
    };
    let fault = |message: String| {
        let _ = tx.send(CaptureEvent::Fault(format!("{what}: {message}")));
    };
    let mut read = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        read.clear();
        let now = match device.read_scans(100, &mut read).and_then(|()| monotonic_ns()) {
            Ok(now) => now,
            Err(error) => return fault(error.to_string()),
        };
        let mut scans = Vec::with_capacity(read.len());
        for &scan in &read {
            let ahead_ns = scan.timestamp_ns - now;
            match guard.check(scan.timestamp_ns, now) {
                ClockCheck::Ok => scans.push(scan),
                ClockCheck::FutureSkew => {
                    shared.future_skew.fetch_add(1, Ordering::Relaxed);
                    shared.max_future_skew_ns.fetch_max(u64::try_from(ahead_ns).unwrap_or(0), Ordering::Relaxed);
                    scans.push(scan);
                }
                ClockCheck::FutureDropped => {
                    shared.future_dropped.fetch_add(1, Ordering::Relaxed);
                    shared.max_future_skew_ns.fetch_max(u64::try_from(ahead_ns).unwrap_or(0), Ordering::Relaxed);
                }
                ClockCheck::WrongClock => {
                    return fault(format!("sample {:.3} ms in the future of CLOCK_MONOTONIC (wrong clock)", ahead_ns as f64 / 1e6));
                }
                ClockCheck::TooOld => return fault(format!("sample {:.1} ms old (clock mismatch or backlog)", -ahead_ns as f64 / 1e6)),
            }
        }
        if scans.is_empty() {
            continue;
        }
        counter.fetch_add(scans.len() as u64, Ordering::Relaxed);
        if tx.send(CaptureEvent::Scans { kind, scale: device.scale, scans }).is_err() {
            break;
        }
    }
}
