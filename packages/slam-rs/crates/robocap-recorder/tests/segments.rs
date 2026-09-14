mod common;

use arrow_array::{BooleanArray, Int64Array, StringArray};
use common::{DELTA, IDR, rows, values};
use robocap_recorder::{
    CalibrationSource, CaptureIdentity, MotionKind, MotionSample, SegmentedWriter, VideoSample,
};
use std::collections::BTreeMap;

#[test]
fn ten_minute_rotation_retains_every_stream_without_resetting_the_session_clock()
-> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let mut writer = SegmentedWriter::create(
        directory.path(),
        CaptureIdentity {
            device_serial: "fe6fede545c972fa".into(),
            session: "boundaries".into(),
            part: 1,
            start_ns: 10_000_000_000,
            calibration: Some(CalibrationSource {
                device_serial: "f408193e6447b3b0".into(),
                document: "{\"value0\":{}}".into(),
                placeholder: true,
            }),
        },
        600_000_000_000,
        None,
    )?;
    let times = [
        10_000_000_000,
        609_999_999_000,
        610_000_000_000,
        610_033_333_333,
        1_210_000_000_000,
        1_210_033_333_333,
    ];
    for (sequence, timestamp_ns) in times.into_iter().enumerate() {
        let keyframe = sequence % 2 == 0;
        for camera in 0..6 {
            writer.video(VideoSample {
                camera,
                timestamp_ns,
                sequence: sequence as u64,
                keyframe,
                annex_b: if keyframe { &IDR } else { &DELTA },
            })?;
        }
        for device in 0..3 {
            for kind in [MotionKind::Gyro, MotionKind::Accel] {
                writer.motion(MotionSample {
                    temperature_raw: None,
                    device,
                    kind,
                    timestamp_ns,
                    sequence: sequence as u64,
                    raw: [1, 2, 3],
                    scale: None,
                })?;
            }
        }
        writer.motion(MotionSample {
            temperature_raw: None,
            device: 0,
            kind: MotionKind::Mag,
            timestamp_ns,
            sequence: sequence as u64,
            raw: [4, 5, 6],
            scale: None,
        })?;
        #[cfg(feature = "live-slam")]
        writer.slam(&robocap_recorder::SlamReport {
            timestamp_ns,
            pose: None,
            status: robocap_recorder::SlamStatus::NoVisualFeatures,
            processing_ms: 2.0,
            latency_ms: 10.0,
            landmarks: 0,
            tracked_observations: 0,
            optimization_started: false,
            updates: sequence as u64 + 1,
        })?;
    }
    writer.finish()?;
    let mut files: Vec<_> = std::fs::read_dir(directory.path())?.collect::<Result<_, _>>()?;
    files.sort_by_key(|entry| entry.file_name());
    assert_eq!(files.len(), 3);
    let mut all = BTreeMap::<String, Vec<i64>>::new();
    for (index, file) in files.into_iter().enumerate() {
        assert_eq!(
            file.file_name().to_string_lossy(),
            format!("part-{:04}.rrd", index + 1)
        );
        let entities = rows(&file.path())?;
        let streams = stream_timestamps(&file.path())?;
        for (entity, timestamps) in &streams {
            all.entry(entity.clone()).or_default().extend(timestamps);
        }
        let fields = &entities["/__properties/calibration"];
        let mut calibration = BTreeMap::new();
        for name in [
            "source_device_serial",
            "target_device_serial",
            "source_document",
            "source_sha256",
        ] {
            calibration.insert(
                name,
                values::<StringArray>(&fields[name][0])?.value(0).to_owned(),
            );
        }
        let placeholder = Some(values::<BooleanArray>(&fields["is_placeholder"][0])?.value(0));
        #[cfg(feature = "live-slam")]
        assert!(
            entities.contains_key("/derived/slam/status"),
            "each independently readable part must carry SLAM status"
        );
        assert_eq!(calibration["source_device_serial"], "f408193e6447b3b0");
        assert_eq!(calibration["target_device_serial"], "fe6fede545c972fa");
        assert_eq!(calibration["source_document"], "{\"value0\":{}}");
        assert_eq!(
            calibration["source_sha256"],
            "ca0d60f57e888b64cd49953a5a33ca57f088f9cd529ce810a9540ab7fb8db874"
        );
        assert_eq!(
            placeholder,
            Some(true),
            "each part must identify the borrowed calibration"
        );
        assert_eq!(
            streams.len(),
            13,
            "six cameras, six motion axes groups, and MAG in each part"
        );
        for timestamps in streams.values() {
            assert_eq!(timestamps, &times[index * 2..index * 2 + 2]);
        }
    }
    assert_eq!(all.len(), 13);
    for timestamps in all.values() {
        assert_eq!(timestamps, &times);
    }
    Ok(())
}

fn writer_at(directory: &std::path::Path, duration_ns: i64) -> anyhow::Result<SegmentedWriter> {
    SegmentedWriter::create(
        directory,
        CaptureIdentity {
            device_serial: "test".into(),
            session: "boundary".into(),
            part: 1,
            start_ns: 0,
            calibration: None,
        },
        duration_ns,
        None,
    )
}

fn all_cameras(
    writer: &mut SegmentedWriter,
    timestamp_ns: i64,
    sequence: u64,
) -> anyhow::Result<()> {
    for camera in 0..6 {
        writer.video(VideoSample {
            camera,
            timestamp_ns,
            sequence,
            keyframe: true,
            annex_b: &IDR,
        })?;
    }
    Ok(())
}

fn all_sensors(
    writer: &mut SegmentedWriter,
    timestamp_ns: i64,
    sequence: u64,
) -> anyhow::Result<()> {
    for device in 0..3 {
        for kind in [MotionKind::Gyro, MotionKind::Accel] {
            writer.motion(MotionSample {
                temperature_raw: None,
                device,
                kind,
                timestamp_ns,
                sequence,
                raw: [1, 2, 3],
                scale: None,
            })?;
        }
    }
    writer.motion(MotionSample {
        temperature_raw: None,
        device: 0,
        kind: MotionKind::Mag,
        timestamp_ns,
        sequence,
        raw: [4, 5, 6],
        scale: None,
    })
}

/// Source timestamps per entity path in one finished part.
fn stream_timestamps(path: &std::path::Path) -> anyhow::Result<BTreeMap<String, Vec<i64>>> {
    let mut streams = BTreeMap::<String, Vec<i64>>::new();
    for (entity, fields) in rows(path)? {
        if let Some(records) = fields.get("source_timestamp_ns") {
            for row in records {
                streams
                    .entry(entity.clone())
                    .or_default()
                    .push(values::<Int64Array>(row)?.value(0));
            }
        }
    }
    Ok(streams)
}

#[test]
fn finishing_during_a_keyframe_transition_publishes_both_parts() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let deadline = 1_000_000_000;
    let mut writer = writer_at(directory.path(), deadline)?;
    all_cameras(&mut writer, 0, 0)?;
    all_sensors(&mut writer, 0, 0)?;
    writer.video(VideoSample {
        camera: 0,
        timestamp_ns: deadline + 10_000,
        sequence: 1,
        keyframe: true,
        annex_b: &IDR,
    })?;
    writer.finish()?;

    let mut files: Vec<_> = std::fs::read_dir(directory.path())?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<Result<_, _>>()?;
    files.sort();
    assert_eq!(files, ["part-0001.rrd", "part-0002.rrd"]);
    let first = stream_timestamps(&directory.path().join("part-0001.rrd"))?;
    assert_eq!(first.len(), 13);
    for (entity, times) in &first {
        assert_eq!(times, &[0], "{entity} in part 1");
    }
    let second = stream_timestamps(&directory.path().join("part-0002.rrd"))?;
    assert_eq!(
        second,
        BTreeMap::from([(
            "/world/rig_00/cam_00/pinhole/video".to_owned(),
            vec![deadline + 10_000]
        )])
    );
    Ok(())
}

#[test]
fn sensors_past_the_deadline_enter_the_next_part_even_when_they_arrive_before_the_keyframe()
-> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let deadline = 1_000_000_000;
    let mut writer = writer_at(directory.path(), deadline)?;
    all_cameras(&mut writer, 0, 0)?;
    all_sensors(&mut writer, 0, 0)?;
    // Encoded video lags the IMU by a few frames, so sensor samples beyond the
    // boundary reach the writer before any camera's keyframe does.
    all_sensors(&mut writer, deadline + 5_000, 1)?;
    all_cameras(&mut writer, deadline + 10_000, 1)?;
    all_sensors(&mut writer, deadline + 20_000, 2)?;
    writer.finish()?;

    let first = stream_timestamps(&directory.path().join("part-0001.rrd"))?;
    let second = stream_timestamps(&directory.path().join("part-0002.rrd"))?;
    assert_eq!(first.len(), 13);
    assert_eq!(second.len(), 13);
    for (entity, times) in &first {
        assert_eq!(times, &[0], "{entity} in part 1 must stop at the boundary");
    }
    for (entity, times) in &second {
        let expected: &[i64] = if entity.contains("/cam_") {
            &[deadline + 10_000]
        } else {
            &[deadline + 5_000, deadline + 20_000]
        };
        assert_eq!(times, expected, "{entity} in part 2");
    }
    Ok(())
}
