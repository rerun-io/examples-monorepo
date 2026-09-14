use crate::{
    Camera, CaptureIdentity, CapturedBuffer, DeviceProfile, FRAME_HEIGHT, FRAME_WIDTH,
    FrameTrigger, IioDevice, MotionKind, MotionSample, SENSORS, SamplePipeline, SegmentedWriter,
    SensorChannel, VideoSample, monotonic_ns,
};
#[cfg(feature = "live-slam")]
use crate::{ImuChannel, SLAM_CAMERAS, SlamInput, SlamProcess, SlamSender, slam_luma};
use anyhow::{Context, Result, ensure};
use gstreamer as gst;
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

enum Event {
    Video {
        camera: u8,
        sequence: u32,
        frame: CapturedBuffer,
    },
    Motion(MotionSample),
    Ready,
    Fault(String),
}

fn camera_capture(
    camera: u8,
    path: &str,
    tx: mpsc::SyncSender<Event>,
    stop: &'static AtomicBool,
    #[cfg(feature = "live-slam")] slam: SlamSender,
) -> Result<()> {
    let mut device = Camera::open(path)?;
    let mut pipeline = SamplePipeline::start(
        &format!(
            "appsrc name=raw is-live=true format=time block=false max-buffers=3 caps=video/x-raw,format=NV12,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate=30/1 ! mpph264enc bps=4000000 gop=30 header-mode=1 max-pending=2 ! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=frames sync=false max-buffers=4 drop=false"
        ),
        "frames",
    )?;
    let source = pipeline.source("raw")?;
    let (identity_tx, identity_rx) = mpsc::sync_channel(16);
    thread::scope(|scope| -> Result<()> {
        let encoder_tx = tx.clone();
        let consumer = scope.spawn(move || -> Result<()> {
            let result = (|| {
                while let Some(frame) = pipeline.next(Duration::from_secs(5))? {
                    let (timestamp_ns, sequence) =
                        identity_rx.recv_timeout(Duration::from_secs(1))?;
                    ensure!(
                        frame.pts_ns == timestamp_ns,
                        "encoder altered camera timestamp"
                    );
                    encoder_tx.send(Event::Video {
                        camera,
                        sequence,
                        frame,
                    })?;
                }
                ensure!(
                    identity_rx.try_recv().is_err(),
                    "encoder dropped trailing frames"
                );
                Ok(())
            })();
            if result.is_err() {
                stop.store(true, Ordering::Relaxed);
            }
            result
        });
        let capture_result = (|| -> Result<()> {
            tx.send(Event::Ready)?;
            let mut previous = None;
            while !stop.load(Ordering::Relaxed) {
                let Some(frame) = device.read_frame()? else {
                    continue;
                };
                if let Some(last) = previous {
                    ensure!(
                        frame.sequence == last + 1,
                        "camera {camera} lost frames: {last} -> {}",
                        frame.sequence
                    );
                }
                previous = Some(frame.sequence);
                #[cfg(feature = "live-slam")]
                if let Some(index) = SLAM_CAMERAS.iter().position(|&id| id == camera) {
                    let pixels = slam_luma(&frame.nv12);
                    slam.submit(SlamInput::Frame {
                        camera: index,
                        timestamp_ns: frame.timestamp_ns,
                        pixels,
                    });
                }
                let timestamp_ns = u64::try_from(frame.timestamp_ns)?;
                let mut buffer = gst::Buffer::from_mut_slice(frame.nv12);
                let writable = buffer.get_mut().context("new camera buffer is shared")?;
                writable.set_pts(gst::ClockTime::from_nseconds(timestamp_ns));
                writable.set_duration(gst::ClockTime::from_nseconds(33_333_333));
                writable.set_offset(u64::from(frame.sequence));
                let wait = Instant::now();
                while source.current_level_buffers() >= 3 {
                    ensure!(
                        !stop.load(Ordering::Relaxed)
                            && wait.elapsed() < Duration::from_millis(500),
                        "camera encoder queue stalled"
                    );
                    thread::sleep(Duration::from_millis(1));
                }
                identity_tx.send((timestamp_ns, frame.sequence))?;
                source.push_buffer(buffer)?;
            }
            Ok(())
        })();
        source.end_of_stream()?;
        capture_result?;
        consumer
            .join()
            .map_err(|_| anyhow::anyhow!("encoder thread panicked"))??;
        Ok(())
    })
}

fn sensor_capture(
    channel: SensorChannel,
    mut device: IioDevice,
    tx: mpsc::SyncSender<Event>,
    stop: &'static AtomicBool,
    #[cfg(feature = "live-slam")] slam: SlamSender,
) -> Result<()> {
    let index = channel.iio_index;
    let scale = if matches!(channel.kind, MotionKind::Mag) {
        None
    } else {
        let prefix = channel.prefix;
        Some(
            fs::read_to_string(format!(
                "/sys/bus/iio/devices/iio:device{index}/{prefix}_scale"
            ))?
            .trim()
            .parse::<f64>()?,
        )
    };
    let mut sequence = 0;
    tx.send(Event::Ready)?;
    while !stop.load(Ordering::Relaxed) {
        for sample in device.read_scans()? {
            let age = monotonic_ns()? - sample.timestamp_ns;
            ensure!(
                (0..1_000_000_000).contains(&age),
                "IIO {index} clock mismatch or backlog: {age}"
            );
            #[cfg(feature = "live-slam")]
            if channel.device == 0 && !matches!(channel.kind, MotionKind::Mag) {
                slam.submit(SlamInput::Imu {
                    channel: if matches!(channel.kind, MotionKind::Gyro) {
                        ImuChannel::Gyro
                    } else {
                        ImuChannel::Accel
                    },
                    timestamp_ns: sample.timestamp_ns,
                    xyz: sample.raw.map(|v| f64::from(v) * scale.unwrap_or(1.0)),
                });
            }
            tx.send(Event::Motion(MotionSample {
                device: channel.device,
                kind: channel.kind,
                timestamp_ns: sample.timestamp_ns,
                sequence,
                raw: sample.raw,
                scale,
                temperature_raw: sample.temperature_raw,
            }))?;
            sequence += 1;
        }
    }
    Ok(())
}

pub fn run(directory: PathBuf, seconds: u64, stop: &'static AtomicBool) -> Result<()> {
    ensure!(
        (1..=3600).contains(&seconds),
        "trial duration must be 1..3600 seconds"
    );
    let hostname = fs::read_to_string("/proc/sys/kernel/hostname")?;
    let serial = fs::read_to_string("/proc/device-tree/serial-number")?;
    let profile = DeviceProfile::identify(hostname.trim(), serial.trim_end_matches('\0'))?;
    // Refuse device access while the vendor application can own capture. A
    // stopped launcher can retain its dead child as a zombie until resumed.
    for process in fs::read_dir("/proc")? {
        let path = process?.path();
        let Ok(name) = fs::read_to_string(path.join("comm")) else {
            continue;
        };
        if name.trim() == "omni-specs.bin" {
            let status = fs::read_to_string(path.join("status"))?;
            ensure!(
                status
                    .lines()
                    .any(|line| line.starts_with("State:") && line.contains("Z (zombie)")),
                "vendor recorder still owns capture; use a supervised idle handoff"
            );
        }
    }
    fs::create_dir(&directory).context("create exclusive session directory")?;
    let trigger = FrameTrigger::stopped()?;
    let identity = CaptureIdentity {
        device_serial: profile.serial().into(),
        session: directory
            .file_name()
            .context("session directory name")?
            .to_string_lossy()
            .into_owned(),
        part: 1,
        start_ns: monotonic_ns()?,
        calibration: Some(profile.calibration()),
    };
    let display = std::env::var_os("ROBOCAP_DISPLAY_RRD")
        .map(|path| {
            crate::DisplayAssets::load(
                std::path::Path::new(&path),
                &profile.calibration().device_serial,
            )
        })
        .transpose()?;
    let mut writer = SegmentedWriter::create(&directory, identity, 600_000_000_000, display)?;
    let devices = SENSORS
        .iter()
        .map(|channel| IioDevice::start(channel.iio_index))
        .collect::<Result<Vec<_>>>()?;
    #[cfg(feature = "live-slam")]
    let mut slam = SlamProcess::spawn(&std::env::current_exe()?)?;
    #[cfg(feature = "live-slam")]
    eprintln!("slam_worker_pid={}", slam.pid());
    let (tx, rx) = mpsc::sync_channel(2048);
    let mut workers = Vec::new();
    for (camera, path) in profile.camera_paths().into_iter().enumerate() {
        let tx = tx.clone();
        #[cfg(feature = "live-slam")]
        let slam_sender = slam.sender();
        workers.push(thread::spawn(move || {
            if let Err(error) = camera_capture(
                camera as u8,
                path,
                tx.clone(),
                stop,
                #[cfg(feature = "live-slam")]
                slam_sender,
            ) {
                stop.store(true, Ordering::Relaxed);
                let _ = tx.send(Event::Fault(format!("camera {camera}: {error:#}")));
            }
        }));
    }
    for (channel, device) in SENSORS.into_iter().zip(devices) {
        let index = channel.iio_index;
        let tx = tx.clone();
        #[cfg(feature = "live-slam")]
        let slam_sender = slam.sender();
        workers.push(thread::spawn(move || {
            if let Err(error) = sensor_capture(
                channel,
                device,
                tx.clone(),
                stop,
                #[cfg(feature = "live-slam")]
                slam_sender,
            ) {
                stop.store(true, Ordering::Relaxed);
                let _ = tx.send(Event::Fault(format!("IIO {index}: {error:#}")));
            }
        }));
    }
    drop(tx);
    let start = Instant::now();
    let mut ready = 0;
    let mut started = None;
    let mut checkpoint = Instant::now();
    let mut counts = [0_u64; 13];
    let result = (|| -> Result<()> {
        loop {
            #[cfg(feature = "live-slam")]
            while let Some(mut report) = slam.poll()? {
                report.latency_ms = (monotonic_ns()? - report.timestamp_ns) as f64 / 1e6;
                writer.slam(&report)?;
                eprintln!("slam_report={}", serde_json::to_string(&report)?);
            }
            if started.is_some_and(|at: Instant| at.elapsed() >= Duration::from_secs(seconds)) {
                stop.store(true, Ordering::Relaxed);
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Event::Ready) => {
                    ready += 1;
                    if ready == 13 {
                        trigger.start()?;
                        started = Some(Instant::now());
                        eprintln!(
                            "capture started; calibration=provisional; live_slam={}",
                            cfg!(feature = "live-slam")
                        );
                    }
                }
                Ok(Event::Video {
                    camera,
                    sequence,
                    frame,
                }) => {
                    writer.video(VideoSample {
                        camera,
                        timestamp_ns: i64::try_from(frame.pts_ns)?,
                        sequence: u64::from(sequence),
                        keyframe: frame.keyframe,
                        annex_b: &frame.bytes,
                    })?;
                    counts[usize::from(camera)] += 1;
                }
                Ok(Event::Motion(sample)) => {
                    let stream = sample.stream_index()?;
                    writer.motion(sample)?;
                    counts[stream] += 1;
                }
                Ok(Event::Fault(error)) => anyhow::bail!(error),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            ensure!(
                started.is_some() || start.elapsed() < Duration::from_secs(20),
                "capture startup timed out"
            );
            if started.is_some_and(|at| at.elapsed() > Duration::from_secs(3)) {
                ensure!(
                    counts.iter().all(|&count| count > 0),
                    "capture stream failed to start: counts={counts:?}"
                );
            }
            if checkpoint.elapsed() >= Duration::from_secs(1) {
                let checkpoint_started = Instant::now();
                writer.checkpoint(Duration::from_secs(3))?;
                if std::env::var_os("ROBOCAP_SLAM_PROFILE").is_some() {
                    eprintln!(
                        "capture_checkpoint_profile={}",
                        serde_json::json!({
                            "timestamp_ns": monotonic_ns()?,
                            "duration_ms": checkpoint_started.elapsed().as_secs_f64() * 1000.0,
                        })
                    );
                }
                #[cfg(feature = "live-slam")]
                let slam_dropped_frames = slam.sender().dropped_frames.load(Ordering::Relaxed);
                #[cfg(not(feature = "live-slam"))]
                let slam_dropped_frames = 0;
                eprintln!(
                    "capture_counts={counts:?} elapsed={:.1} slam_dropped_frames={slam_dropped_frames}",
                    start.elapsed().as_secs_f64()
                );
                checkpoint = Instant::now();
            }
        }
        Ok(())
    })();
    stop.store(true, Ordering::Relaxed);
    drop(rx);
    drop(trigger);
    let mut join_result = Ok(());
    for worker in workers {
        if worker.join().is_err() && join_result.is_ok() {
            join_result = Err(anyhow::anyhow!("capture worker panicked"));
        }
    }
    join_result?;
    result?;
    writer.finish()?;
    eprintln!(
        "capture complete counts={counts:?} elapsed={:.3}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
