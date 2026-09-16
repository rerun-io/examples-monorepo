//! Offline writer validation only: feed existing Cap B samples through the
//! same public writer used by live capture. This is not a recording mode.
use std::{path::PathBuf, process::Command, time::Duration};

use anyhow::{Context, Result, ensure};
use robocap_recorder::{
    CAMERAS, CaptureIdentity, DirectWriter, MotionKind, MotionSample, SamplePipeline, VideoSample,
};
use rusqlite::{Connection, OpenFlags};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: validate_cap_fixture SESSION_DIRECTORY OUTPUT.rrd DURATION_SECONDS"
    );
    let directory = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let duration_ns = args[3]
        .parse::<i64>()?
        .checked_mul(1_000_000_000)
        .context("duration overflow")?;
    ensure!(duration_ns > 0, "duration must be positive");
    let files = std::fs::read_dir(&directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut videos = Vec::new();
    for camera in CAMERAS {
        let suffix = camera.name.replace('_', "-");
        let matches: Vec<_> = files
            .iter()
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().ends_with(&format!("_{suffix}.mp4")))
            })
            .collect();
        ensure!(
            matches.len() == 1,
            "fixture requires exactly one segment for camera {suffix}"
        );
        let path = matches[0];
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format_tags=comment",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(path)
            .output()?;
        ensure!(
            probe.status.success(),
            "cannot read video epoch: {}",
            path.display()
        );
        let epoch = std::str::from_utf8(&probe.stdout)?
            .trim()
            .parse::<i64>()?
            .checked_mul(1000)
            .context("epoch overflow")?;
        videos.push((path, epoch));
    }
    let start_ns = videos
        .iter()
        .map(|(_, epoch)| *epoch)
        .min()
        .context("no videos")?;
    let end_ns = start_ns
        .checked_add(duration_ns)
        .context("end time overflow")?;
    let mut writer = DirectWriter::create(
        &output,
        CaptureIdentity {
            device_serial: "fe6fede545c972fa".into(),
            session: "offline-writer-validation".into(),
            part: 1,
            start_ns,
            calibration: None,
        },
        None,
    )?;
    for (camera, (path, epoch)) in videos.into_iter().enumerate() {
        let location = path
            .to_str()
            .context("non-UTF8 fixture path")?
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        let mut pipeline = SamplePipeline::start(
            &format!(
                "filesrc location=\"{location}\" ! qtdemux ! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=frames sync=false max-buffers=4 drop=false"
            ),
            "frames",
        )?;
        let mut sequence = 0;
        while let Some(frame) = pipeline.next(Duration::from_secs(10))? {
            let timestamp_ns = epoch
                .checked_add(i64::try_from(frame.pts_ns)?)
                .context("video timestamp overflow")?;
            if timestamp_ns >= end_ns {
                break;
            }
            writer.video(VideoSample {
                camera: u8::try_from(camera)?,
                timestamp_ns,
                sequence,
                keyframe: frame.keyframe,
                annex_b: &frame.bytes,
            })?;
            sequence += 1;
        }
        println!("camera={camera} samples={sequence} epoch_ns={epoch}");
    }
    for device in 0..3 {
        let prefix = format!("IMUWriter_dev{device}_");
        let path = files
            .iter()
            .find(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(&prefix))
            })
            .context("IMU fixture missing")?;
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        for (table, kind) in [
            ("gyro_data", MotionKind::Gyro),
            ("acc_data", MotionKind::Accel),
        ] {
            let mut statement = connection.prepare(&format!("SELECT id, timestamp, x, y, z FROM {table} WHERE timestamp >= ?1 AND timestamp < ?2 ORDER BY timestamp"))?;
            let mut rows = statement.query([start_ns, end_ns])?;
            let mut count = 0;
            while let Some(row) = rows.next()? {
                // No scale is inferred from the device's current idle range.
                // Original fixture counts and their independent times are kept.
                writer.motion(MotionSample {
                    temperature_raw: None,
                    device,
                    kind,
                    timestamp_ns: row.get(1)?,
                    sequence: u64::try_from(row.get::<_, i64>(0)?)?,
                    raw: [row.get(2)?, row.get(3)?, row.get(4)?],
                    scale: None,
                })?;
                count += 1;
            }
            println!("imu={device} kind={kind:?} samples={count}");
        }
    }
    let path = files
        .iter()
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("MAGWriter_"))
        })
        .context("MAG fixture missing")?;
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut statement = connection.prepare("SELECT id, timestamp, mag_x, mag_y, mag_z FROM mag_data WHERE timestamp >= ?1 AND timestamp < ?2 ORDER BY timestamp")?;
    let mut rows = statement.query([start_ns, end_ns])?;
    let mut count = 0;
    while let Some(row) = rows.next()? {
        writer.motion(MotionSample {
            temperature_raw: None,
            device: 0,
            kind: MotionKind::Mag,
            timestamp_ns: row.get(1)?,
            sequence: u64::try_from(row.get::<_, i64>(0)?)?,
            raw: [row.get(2)?, row.get(3)?, row.get(4)?],
            scale: None,
        })?;
        count += 1;
    }
    println!("mag samples={count} start_ns={start_ns} end_ns={end_ns}");
    writer.finish()?;
    Ok(())
}
