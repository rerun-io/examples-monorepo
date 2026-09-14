#![cfg(feature = "live-slam")]
mod common;
use robocap_recorder::{CaptureIdentity, DirectWriter, SlamReport};
use std::collections::BTreeMap;

#[test]
fn live_poses_move_the_dataforge_rig_and_connect_the_path_without_bridging_tracking_gaps()
-> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("scene.rrd");
    let mut writer = DirectWriter::create(
        &path,
        CaptureIdentity {
            device_serial: "test".into(),
            session: "scene".into(),
            part: 1,
            start_ns: 0,
            calibration: None,
        },
        None,
    )?;
    for (index, pose) in [
        Some([0., 0., 0., 0., 0., 0., 1.]),
        Some([1., 0., 0., 0., 0., 0., 1.]),
        None,
        Some([3., 0., 0., 0., 0., 0., 1.]),
    ]
    .into_iter()
    .enumerate()
    {
        writer.slam(&SlamReport {
            timestamp_ns: index as i64 * 100_000_000,
            pose,
            status: robocap_recorder::SlamStatus::TrackingProvisional,
            processing_ms: 1.,
            latency_ms: 2.,
            landmarks: 20,
            tracked_observations: 20,
            optimization_started: true,
            updates: index as u64 + 1,
        })?;
    }
    writer.finish()?;
    let mut rows = BTreeMap::<(String, String), usize>::new();
    for (entity, fields) in common::rows(&path)? {
        for (name, records) in fields {
            rows.insert((entity.clone(), name), records.len());
        }
    }
    assert_eq!(
        rows.get(&("/world/rig_00".into(), "Transform3D:translation".into())),
        Some(&3)
    );
    assert_eq!(
        rows.get(&(
            "/world/runs/slam_rs/trajectory".into(),
            "LineStrips3D:strips".into()
        )),
        Some(&1)
    );
    assert_eq!(
        rows.get(&(
            "/world/runs/slam_rs/trail".into(),
            "LineStrips3D:strips".into()
        )),
        Some(&1)
    );
    assert!(!rows.keys().any(|(entity, _)| entity == "/derived/slam/rig"));
    Ok(())
}
