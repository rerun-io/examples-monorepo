mod common;

use arrow_array::{Float64Array, Int32Array, Int64Array, ListArray, UInt8Array};
use common::{DELTA, IDR, rows, values};
use robocap_recorder::{CaptureIdentity, DirectWriter, MotionKind, MotionSample, VideoSample};
use std::collections::BTreeMap;

#[test]
fn unknown_sensor_scale_has_a_raw_plot_without_claiming_physical_units() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("raw-sensor.rrd");
    let mut writer = DirectWriter::create(
        &path,
        CaptureIdentity {
            device_serial: "test".into(),
            session: "unknown-scale".into(),
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
    writer.finish()?;
    let mut plots = BTreeMap::new();
    for (entity, fields) in rows(&path)? {
        if let Some(scalars) = fields.get("Scalars:scalars") {
            plots.insert(
                entity,
                values::<Float64Array>(&scalars[0])?.values().to_vec(),
            );
        }
    }
    assert_eq!(
        plots,
        BTreeMap::from([("/world/rig_00/imu_00/gyro/raw".into(), vec![7.0, -8.0, 9.0])])
    );
    Ok(())
}

#[test]
fn six_encoded_streams_keep_access_units_and_source_identity() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("part-0001.rrd");
    let mut writer = DirectWriter::create(
        &path,
        CaptureIdentity {
            device_serial: "fe6fede545c972fa".into(),
            session: "video-contract".into(),
            part: 1,
            start_ns: 625_255_220_000,
            calibration: None,
        },
        None,
    )?;
    // A transport fixture: SPS, PPS, and IDR NAL units. Pixel decoding is
    // checked separately against real captured frames, not these short bytes.
    let access_unit = [
        0, 0, 0, 1, 0x67, 0x64, 0, 0x28, 0, 0, 1, 0x68, 0xef, 0, 0, 1, 0x65, 0x88,
    ];
    for camera in 0..6 {
        writer.video(VideoSample {
            camera,
            timestamp_ns: 625_255_220_000 + i64::from(camera),
            sequence: 0,
            keyframe: true,
            annex_b: &access_unit,
        })?;
    }
    writer.finish()?;
    let mut samples = BTreeMap::new();
    for (entity, fields) in rows(&path)? {
        if let Some(samples_rows) = fields.get("VideoStream:sample") {
            let blobs = values::<ListArray>(&samples_rows[0])?;
            samples.insert(
                entity,
                values::<UInt8Array>(&blobs.value(0))?.values().to_vec(),
            );
        }
    }
    assert_eq!(samples.len(), 6);
    for camera in 0..6 {
        assert_eq!(
            samples[&format!("/world/rig_00/cam_{camera:02}/pinhole/video")],
            access_unit
        );
    }
    assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
    Ok(())
}

#[test]
fn a_part_requires_a_decodable_first_frame_and_ordered_video() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let mut writer = DirectWriter::create(
        &directory.path().join("part.rrd"),
        CaptureIdentity {
            device_serial: "test".into(),
            session: "ordered".into(),
            part: 1,
            start_ns: 100,
            calibration: None,
        },
        None,
    )?;
    let missing_headers = [0, 0, 1, 0x65, 0x88];
    assert!(
        writer
            .video(VideoSample {
                camera: 0,
                timestamp_ns: 100,
                sequence: 0,
                keyframe: false,
                annex_b: &DELTA
            })
            .is_err()
    );
    assert!(
        writer
            .video(VideoSample {
                camera: 0,
                timestamp_ns: 100,
                sequence: 0,
                keyframe: true,
                annex_b: &missing_headers
            })
            .is_err()
    );
    writer.video(VideoSample {
        camera: 0,
        timestamp_ns: 100,
        sequence: 0,
        keyframe: true,
        annex_b: &IDR,
    })?;
    assert!(
        writer
            .video(VideoSample {
                camera: 0,
                timestamp_ns: 100,
                sequence: 1,
                keyframe: false,
                annex_b: &DELTA
            })
            .is_err()
    );
    assert!(
        writer
            .video(VideoSample {
                camera: 0,
                timestamp_ns: 101,
                sequence: 0,
                keyframe: false,
                annex_b: &DELTA
            })
            .is_err()
    );
    writer.video(VideoSample {
        camera: 0,
        timestamp_ns: 101,
        sequence: 1,
        keyframe: false,
        annex_b: &DELTA,
    })?;
    writer.finish()?;
    Ok(())
}

#[test]
fn all_motion_streams_keep_original_counts_and_independent_timestamps() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("part-0001.rrd");
    let identity = CaptureIdentity {
        device_serial: "fe6fede545c972fa".into(),
        session: "contract-test".into(),
        part: 1,
        start_ns: 625_255_220_000,
        calibration: None,
    };
    let mut writer = DirectWriter::create(&path, identity, None)?;
    let mut expected = BTreeMap::new();
    for device in 0..3 {
        for (kind, suffix, offset) in [
            (MotionKind::Gyro, "gyro", 0),
            (MotionKind::Accel, "accel", 731),
        ] {
            let timestamp_ns = 625_255_220_100 + 1_000 * i64::from(device) + offset;
            let raw = [i32::MIN, i32::from(device), i32::MAX];
            writer.motion(MotionSample {
                temperature_raw: None,
                device,
                kind,
                timestamp_ns,
                sequence: 17,
                raw,
                scale: Some(0.125),
            })?;
            expected.insert(
                format!("/world/rig_00/imu_{device:02}/{suffix}"),
                (timestamp_ns, raw),
            );
        }
    }
    writer.motion(MotionSample {
        temperature_raw: None,
        device: 0,
        kind: MotionKind::Mag,
        timestamp_ns: 625_255_220_777,
        sequence: 2,
        raw: [20, -31, 42],
        scale: None,
    })?;
    expected.insert(
        "/world/rig_00/mag_00".to_owned(),
        (625_255_220_777, [20, -31, 42]),
    );
    writer.finish()?;

    let mut actual = BTreeMap::new();
    for (entity, fields) in rows(&path)? {
        if let Some(counts) = fields.get("raw_counts") {
            let counts = values::<Int32Array>(&counts[0])?;
            let time = values::<Int64Array>(&fields["source_timestamp_ns"][0])?;
            actual.insert(
                entity,
                (
                    time.value(0),
                    [counts.value(0), counts.value(1), counts.value(2)],
                ),
            );
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(
        std::fs::read_dir(directory.path())?.count(),
        1,
        "direct mode must not produce SQLite or MP4 intermediates"
    );
    Ok(())
}

#[test]
fn buffered_sensor_temperature_survives_rrd_persistence() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("temperature.rrd");
    let mut writer = DirectWriter::create(
        &path,
        CaptureIdentity {
            device_serial: "test".into(),
            session: "temperature".into(),
            part: 1,
            start_ns: 0,
            calibration: None,
        },
        None,
    )?;
    writer.motion(MotionSample {
        device: 0,
        kind: MotionKind::Gyro,
        timestamp_ns: 100,
        sequence: 0,
        raw: [1, 2, 3],
        scale: None,
        temperature_raw: Some(-111),
    })?;
    writer.finish()?;
    let mut temperatures = Vec::new();
    for fields in rows(&path)?.values() {
        if let Some(records) = fields.get("temperature_raw") {
            for row in records {
                temperatures.extend_from_slice(values::<Int32Array>(row)?.values());
            }
        }
    }
    assert_eq!(temperatures, [-111]);
    Ok(())
}
