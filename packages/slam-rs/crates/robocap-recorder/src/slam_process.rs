//! SLAM process isolation. Capture never waits for the estimator.
use crate::{
    DeviceProfile, ImuChannel, LiveSlam, LiveSlamOptions, SLAM_PIXELS, SlamInput, SlamReport,
    SlamStatus, monotonic_ns,
};
use anyhow::{Context, Result, ensure};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone)]
pub struct SlamSender {
    frames: mpsc::SyncSender<SlamInput>,
    imu: mpsc::SyncSender<SlamInput>,
    failed: Arc<AtomicBool>,
    pub dropped_frames: Arc<AtomicU64>,
}
impl SlamSender {
    pub fn submit(&self, input: SlamInput) {
        if self.failed.load(Ordering::Relaxed) {
            return;
        }
        let tx = if matches!(input, SlamInput::Frame { .. }) {
            &self.frames
        } else {
            &self.imu
        };
        match tx.try_send(input) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(SlamInput::Frame { .. })) => {
                self.dropped_frames.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.failed.store(true, Ordering::Relaxed);
            }
        }
    }
}

pub struct SlamProcess {
    child: Child,
    sender: SlamSender,
    reports: mpsc::Receiver<Result<SlamReport>>,
    workers: Vec<thread::JoinHandle<()>>,
    last_report: Instant,
    failure_reported: bool,
}

impl SlamProcess {
    pub fn spawn(executable: &Path) -> Result<Self> {
        let mut child = Command::new(executable)
            .arg("--slam-worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let mut input = child.stdin.take().context("child stdin")?;
        let output = child.stdout.take().context("child stdout")?;
        // IMU always drains before the next image. Image backlog cannot consume
        // sensor queue capacity. At most 16 images (3.7 MiB) await the pipe.
        let (frame_tx, frame_rx) = mpsc::sync_channel(16);
        let (imu_tx, imu_rx) = mpsc::sync_channel(512);
        let (report_tx, reports) = mpsc::sync_channel(64);
        let failed = Arc::new(AtomicBool::new(false));
        let sender = SlamSender {
            frames: frame_tx,
            imu: imu_tx,
            failed: failed.clone(),
            dropped_frames: Arc::new(AtomicU64::new(0)),
        };
        let write_failed = failed.clone();
        let writer = thread::spawn(move || {
            while !write_failed.load(Ordering::Relaxed) {
                let next = match imu_rx.try_recv() {
                    Ok(message) => Ok(message),
                    Err(_) => frame_rx.recv_timeout(Duration::from_millis(2)),
                };
                match next {
                    Ok(message) => {
                        let result = message.write_to(&mut input);
                        if result.is_err() {
                            write_failed.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        });
        let reader = thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let report = line
                    .map_err(anyhow::Error::from)
                    .and_then(|line| serde_json::from_str(&line).map_err(Into::into));
                if report_tx.try_send(report).is_err() {
                    failed.store(true, Ordering::Relaxed);
                    break;
                }
            }
        });
        Ok(Self {
            child,
            sender,
            reports,
            workers: vec![writer, reader],
            last_report: Instant::now(),
            failure_reported: false,
        })
    }
    pub fn sender(&self) -> SlamSender {
        self.sender.clone()
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn poll(&mut self) -> Result<Option<SlamReport>> {
        if self.failure_reported {
            return Ok(None);
        }
        let status = self.child.try_wait()?;
        let failed = status.is_some()
            || self.sender.failed.load(Ordering::Relaxed)
            || self.last_report.elapsed() > Duration::from_secs(5);
        if failed {
            return self.report_failure(format!(
                "worker={status:?}; queue fault or no updates for 5 seconds"
            ));
        }
        match self.reports.try_recv() {
            Ok(Ok(report)) => {
                self.last_report = Instant::now();
                Ok(Some(report))
            }
            Ok(Err(error)) => self.report_failure(error.to_string()),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.sender.failed.store(true, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    fn report_failure(&mut self, cause: String) -> Result<Option<SlamReport>> {
        self.failure_reported = true;
        self.sender.failed.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        Ok(Some(SlamReport {
            timestamp_ns: monotonic_ns()?,
            pose: None,
            status: SlamStatus::Failed(cause),
            processing_ms: 0.0,
            latency_ms: 0.0,
            landmarks: 0,
            tracked_observations: 0,
            optimization_started: false,
            updates: 0,
        }))
    }
}
impl Drop for SlamProcess {
    fn drop(&mut self) {
        self.sender.failed.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

pub fn slam_worker() -> Result<()> {
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")?;
    let serial = std::fs::read_to_string("/proc/device-tree/serial-number")?;
    let profile = DeviceProfile::identify(hostname.trim(), serial.trim_end_matches('\0'))?;
    if let Some(range) = profile.slam_cpus() {
        // Only this child and its subsequently created frontend threads use the
        // four Cortex-A76 cores. Capture remains under the normal scheduler.
        unsafe {
            let mut cpus: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut cpus);
            for cpu in range {
                libc::CPU_SET(cpu, &mut cpus);
            }
            ensure!(
                libc::sched_setaffinity(0, std::mem::size_of_val(&cpus), &cpus) == 0,
                "SLAM CPU affinity failed"
            );
        }
    }
    let mut slam = LiveSlam::cap_a_fast_profile(LiveSlamOptions {
        profile: std::env::var_os("ROBOCAP_SLAM_PROFILE").is_some(),
        joint: std::env::var_os("ROBOCAP_SLAM_JOINT").is_some(),
    })?;
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    while let Some(message) = SlamInput::read_from(&mut input, SLAM_PIXELS)? {
        if let Some(report) = slam.push(message, monotonic_ns()?)? {
            serde_json::to_writer(&mut output, &report)?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    Ok(())
}

impl SlamInput {
    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Frame {
                camera,
                timestamp_ns,
                pixels,
            } => {
                if *camera >= 4 || pixels.len() != SLAM_PIXELS {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid SLAM image",
                    ));
                }
                w.write_all(&[*camera as u8])?;
                w.write_all(&timestamp_ns.to_le_bytes())?;
                w.write_all(pixels)?;
            }
            Self::Imu {
                channel,
                timestamp_ns,
                xyz,
            } => {
                let kind = match channel {
                    ImuChannel::Gyro => 4,
                    ImuChannel::Accel => 5,
                };
                w.write_all(&[kind])?;
                w.write_all(&timestamp_ns.to_le_bytes())?;
                for value in xyz {
                    w.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    fn read_from(r: &mut impl Read, pixels_len: usize) -> Result<Option<Self>> {
        let mut kind = [0];
        if r.read(&mut kind)? == 0 {
            return Ok(None);
        }
        let mut timestamp = [0; 8];
        r.read_exact(&mut timestamp)?;
        let timestamp_ns = i64::from_le_bytes(timestamp);
        let message = if kind[0] < 4 {
            let mut pixels = vec![0; pixels_len];
            r.read_exact(&mut pixels)?;
            Self::Frame {
                camera: usize::from(kind[0]),
                timestamp_ns,
                pixels,
            }
        } else {
            let channel = match kind[0] {
                4 => ImuChannel::Gyro,
                5 => ImuChannel::Accel,
                _ => anyhow::bail!("unknown worker message"),
            };
            let mut xyz = [0.0; 3];
            for value in &mut xyz {
                let mut bytes = [0; 8];
                r.read_exact(&mut bytes)?;
                *value = f64::from_le_bytes(bytes);
            }
            Self::Imu {
                channel,
                timestamp_ns,
                xyz,
            }
        };
        Ok(Some(message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_keeps_camera_and_imu_tags_and_rejects_truncated_messages() -> Result<()> {
        let pixels_len = SLAM_PIXELS;
        for tag in 0..6 {
            let message = match tag {
                0..=3 => SlamInput::Frame {
                    camera: tag,
                    timestamp_ns: 123,
                    pixels: vec![27; pixels_len],
                },
                _ => SlamInput::Imu {
                    channel: if tag == 4 {
                        ImuChannel::Gyro
                    } else {
                        ImuChannel::Accel
                    },
                    timestamp_ns: 123,
                    xyz: [1.0, -2.0, 3.0],
                },
            };
            let mut bytes = Vec::new();
            message.write_to(&mut bytes)?;
            assert_eq!(&bytes[..9], &[tag as u8, 123, 0, 0, 0, 0, 0, 0, 0]);
            if tag >= 4 {
                assert_eq!(
                    &bytes[9..],
                    &[
                        0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 8,
                        64,
                    ]
                );
            } else {
                assert_eq!(&bytes[9..], vec![27; 640 * 360]);
            }
            let mut reader = bytes.as_slice();
            assert_eq!(
                SlamInput::read_from(&mut reader, pixels_len)?,
                Some(message)
            );
            assert!(SlamInput::read_from(&mut reader, pixels_len)?.is_none());
            assert!(SlamInput::read_from(&mut &bytes[..bytes.len() - 1], pixels_len).is_err());
        }
        assert!(SlamInput::read_from(&mut &[6, 0, 0, 0, 0, 0, 0, 0, 0][..], pixels_len).is_err());
        Ok(())
    }
}
