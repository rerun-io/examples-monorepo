#![cfg(all(feature = "live-slam", feature = "gstreamer-capture"))]
use anyhow::Result;
use robocap_recorder::{ImuChannel, SlamInput, SlamProcess, SlamStatus};
use std::{
    os::unix::fs::PermissionsExt,
    path::Path,
    time::{Duration, Instant},
};

#[test]
fn malformed_report_preserves_the_parse_error_and_is_reported_once() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let executable = directory.path().join("malformed-worker");
    std::fs::write(
        &executable,
        "#!/bin/sh\nprintf '%s\\n' '{\"timestamp_ns\":\"broken\"}'\nexec cat >/dev/null\n",
    )?;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))?;
    let mut process = SlamProcess::spawn(&executable)?;
    let start = Instant::now();
    let report = loop {
        if let Some(report) = process.poll()? {
            break report;
        }
        anyhow::ensure!(
            start.elapsed() < Duration::from_secs(3),
            "parse failure was not reported"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(
        matches!(report.status, SlamStatus::Failed(_)),
        "{}",
        report.status
    );
    let SlamStatus::Failed(reason) = &report.status else {
        anyhow::bail!("expected failure, got {}", report.status);
    };
    assert!(
        reason.contains("invalid type: string \"broken\", expected i64"),
        "{reason}"
    );
    assert!(report.pose.is_none());
    assert_eq!(report.updates, 0);
    assert!(process.poll()?.is_none());
    Ok(())
}

#[test]
fn estimator_process_death_is_reported_and_capture_submission_stays_bounded() -> Result<()> {
    let mut process = SlamProcess::spawn(Path::new(env!("CARGO_BIN_EXE_robocap-direct")))?;
    // The process is our own child, never a device or shared service.
    assert_eq!(
        unsafe { libc::kill(process.pid() as i32, libc::SIGKILL) },
        0
    );
    let start = Instant::now();
    loop {
        if process
            .poll()?
            .is_some_and(|report| matches!(report.status, SlamStatus::Failed(_)))
        {
            break;
        }
        anyhow::ensure!(
            start.elapsed() < Duration::from_secs(3),
            "child failure was not reported"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let start = Instant::now();
    for tick in 0..1000 {
        process.sender().submit(SlamInput::Imu {
            channel: ImuChannel::Gyro,
            timestamp_ns: tick,
            xyz: [0.0; 3],
        });
    }
    assert!(start.elapsed() < Duration::from_millis(100));
    Ok(())
}

#[test]
fn invalid_frame_length_fails_the_sender_without_writing_a_partial_message() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let executable = directory.path().join("input-worker");
    let received = directory.path().join("received");
    std::fs::write(
        &executable,
        format!("#!/bin/sh\nexec cat > '{}'\n", received.display()),
    )?;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))?;
    let mut process = SlamProcess::spawn(&executable)?;
    process.sender().submit(SlamInput::Frame {
        camera: 0,
        timestamp_ns: 123,
        pixels: vec![0; 1],
    });
    let start = Instant::now();
    loop {
        if let Some(report) = process.poll()? {
            assert!(matches!(report.status, SlamStatus::Failed(_)));
            break;
        }
        anyhow::ensure!(
            start.elapsed() < Duration::from_secs(3),
            "invalid frame was not rejected"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(process);
    assert!(std::fs::read(received).unwrap_or_default().is_empty());
    Ok(())
}
