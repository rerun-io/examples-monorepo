mod common;

use std::fs::File;
use std::io::BufReader;
use std::time::Duration;

use re_log_encoding::Decoder;
use re_log_types::LogMsg;
use rerun::sink::LogSink;
use robocap_recorder::{CaptureIdentity, DirectWriter, DurableRrdSink, MotionKind, MotionSample};

#[test]
fn checkpoint_exposes_complete_records_without_publishing_a_finished_file() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("checkpoint.rrd");
    let mut writer = DirectWriter::create(
        &path,
        CaptureIdentity {
            device_serial: "test".into(),
            session: "checkpoint".into(),
            part: 1,
            start_ns: 100,
            calibration: None,
        },
        None,
    )?;
    writer.motion(MotionSample {
        temperature_raw: None,
        device: 0,
        kind: MotionKind::Gyro,
        timestamp_ns: 123,
        sequence: 1,
        raw: [7, -8, 9],
        scale: None,
    })?;
    writer.checkpoint(Duration::from_secs(5))?;
    assert!(!path.exists());
    let raw_records: usize = common::rows(&path.with_extension("rrd.partial"))?
        .values()
        .filter_map(|fields| fields.get("raw_counts"))
        .map(Vec::len)
        .sum();
    assert_eq!(raw_records, 1);
    writer.finish()?;
    Ok(())
}

#[test]
fn live_messages_are_readable_unchanged_after_durable_completion() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("segment-0001.rrd");
    let (recording, memory) = rerun::RecordingStreamBuilder::new("capture-contract").memory()?;
    recording.set_time(
        "video_time",
        rerun::TimeCell::from_duration_nanos(625_255_220_000),
    );
    recording.log(
        "/world/rig_00/imu_00/gyro",
        &rerun::Scalars::new([0.125, -0.25, 0.5]),
    )?;
    recording.flush_with_timeout(Duration::from_secs(5))?;
    let expected = memory.take();
    assert!(!expected.is_empty());

    let sink = DurableRrdSink::create(&path)?;
    sink.send_all(expected.clone());
    sink.flush_blocking(Duration::from_secs(5))?;
    assert!(!path.exists(), "a checkpoint is not a completed recording");
    sink.finish()?;

    let decoded = Decoder::<LogMsg>::decode_eager(BufReader::new(File::open(&path)?))?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(decoded, expected);
    assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
    Ok(())
}

#[test]
fn finishing_never_replaces_a_recording_created_by_another_writer() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("segment-0001.rrd");
    let sink = DurableRrdSink::create(&path)?;
    std::fs::write(&path, b"existing recording must remain unchanged")?;
    assert!(sink.finish().is_err());
    assert_eq!(
        std::fs::read(&path)?,
        b"existing recording must remain unchanged"
    );
    assert!(path.with_extension("rrd.partial").exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn storage_failure_is_reported_and_never_published_as_complete() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", "write_limit_child", "--ignored", "--nocapture"])
        .env("ROBOCAP_WRITE_LIMIT_DIRECTORY", directory.path())
        .status()?;
    assert!(
        status.success(),
        "isolated storage-failure check failed: {status}"
    );
    assert!(!directory.path().join("segment-0001.rrd").exists());
    assert!(directory.path().join("segment-0001.rrd.partial").exists());
    Ok(())
}

#[cfg(unix)]
#[test]
#[ignore = "run by storage_failure_is_reported_and_never_published_as_complete in an isolated process"]
fn write_limit_child() -> anyhow::Result<()> {
    let directory = std::env::var_os("ROBOCAP_WRITE_LIMIT_DIRECTORY")
        .ok_or_else(|| anyhow::anyhow!("requires isolated test directory"))?;
    let path = std::path::PathBuf::from(directory).join("segment-0001.rrd");
    let (recording, memory) = rerun::RecordingStreamBuilder::new("storage-failure").memory()?;
    let mut state = 0x74a9_0813_u32;
    let payload: String = (0..131_072)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            char::from(32 + (state % 95) as u8)
        })
        .collect();
    recording.log("/capture_payload", &rerun::TextDocument::new(payload))?;
    recording.flush_with_timeout(Duration::from_secs(5))?;
    let messages = memory.take();
    let sink = DurableRrdSink::create(&path)?;

    // SAFETY: this is an isolated child test process. Ignore the signal so the
    // regular-file write returns EFBIG and exercises the real sink error path.
    let result = unsafe {
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        libc::setrlimit(
            libc::RLIMIT_FSIZE,
            &libc::rlimit {
                rlim_cur: 4096,
                rlim_max: 4096,
            },
        )
    };
    anyhow::ensure!(
        result == 0,
        "setrlimit failed: {}",
        std::io::Error::last_os_error()
    );
    sink.send_all(messages);
    assert!(sink.flush_blocking(Duration::from_secs(5)).is_err());
    assert!(sink.finish().is_err());
    Ok(())
}
