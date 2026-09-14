use re_log_encoding::Decoder;
use re_log_types::LogMsg;
use robocap_recorder::{CaptureIdentity, DirectWriter, DisplayAssets};
use std::{fs::File, io::BufReader, time::Duration};

#[test]
fn saved_scene_and_blueprint_follow_the_new_recording_identity() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let asset = dir.path().join("display.rrd");
    {
        let rec = rerun::RecordingStreamBuilder::new("robocap")
            .recording_id("test-display")
            .save(&asset)?;
        rec.log_static(
            "/world/rig_00/cam_00",
            &rerun::Transform3D::from_translation([0.1, 0.0, 0.0]),
        )?;
        rerun::blueprint::Blueprint::auto()
            .send(&rec, rerun::blueprint::BlueprintActivation::default())?;
        rec.flush_with_timeout(Duration::from_secs(5))?;
    }
    let display = DisplayAssets::load(&asset, "test")?;
    assert!(DisplayAssets::load(&asset, "other-device").is_err());
    let output = dir.path().join("capture.rrd");
    DirectWriter::create(
        &output,
        CaptureIdentity {
            device_serial: "test".into(),
            session: "live".into(),
            part: 1,
            start_ns: 0,
            calibration: None,
        },
        Some(&display),
    )?
    .finish()?;
    let mut camera_seen = false;
    let mut blueprint_seen = false;
    for msg in Decoder::<LogMsg>::decode_eager(BufReader::new(File::open(output)?))? {
        match msg? {
            LogMsg::ArrowMsg(id, msg)
                if msg.batch.schema().metadata()["rerun:entity_path"] == "/world/rig_00/cam_00"
                    && msg
                        .batch
                        .column_by_name("Transform3D:translation")
                        .is_some() =>
            {
                assert_eq!(id.recording_id().as_str(), "test-live-part-0001");
                camera_seen = true;
            }
            LogMsg::BlueprintActivationCommand(_) => blueprint_seen = true,
            _ => {}
        }
    }
    assert!(camera_seen && blueprint_seen);
    Ok(())
}
