use anyhow::{Context, Result, ensure};
use re_log_encoding::Decoder;
use re_log_types::{LogMsg, StoreKind};
use rerun::{RecordingStream, log::Chunk};
use std::{fs, path::Path};

/// DataForge's prepared static rig and blueprint; reused for every live file part.
pub struct DisplayAssets {
    messages: Vec<LogMsg>,
}

impl DisplayAssets {
    pub fn load(path: &Path, calibration_device: &str) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() <= 32 * 1024 * 1024,
            "display asset exceeds 32 MiB"
        );
        let data = fs::read(path)?;
        let mut messages = Vec::new();
        let mut scene_seen = false;
        let mut blueprint_seen = false;
        for message in Decoder::<LogMsg>::decode_eager(data.as_slice())? {
            let message = message?;
            ensure!(
                message.store_id().application_id().as_str() == "robocap",
                "display belongs to another application"
            );
            if message.store_id().kind() == StoreKind::Recording {
                ensure!(
                    message.store_id().recording_id().as_str()
                        == format!("{calibration_device}-display"),
                    "display calibration source does not match"
                );
                let LogMsg::ArrowMsg(_, arrow) = &message else {
                    continue;
                };
                let chunk = Chunk::from_arrow_msg(arrow)?;
                if chunk.entity_path().to_string().starts_with("/__properties") {
                    continue;
                }
                ensure!(
                    chunk.is_static(),
                    "display must not contain recorded motion or sensor data"
                );
                let entity = chunk.entity_path().to_string();
                ensure!(
                    entity == "/world/rig_00" || entity.starts_with("/world/rig_00/"),
                    "display contains an unexpected entity"
                );
                ensure!(
                    !(entity == "/world/rig_00"
                        && arrow
                            .batch
                            .schema()
                            .fields()
                            .iter()
                            .any(|f| f.name().starts_with("Transform3D:"))),
                    "a static rig pose would hide live motion"
                );
                scene_seen = true;
            }
            if matches!(message, LogMsg::BlueprintActivationCommand(_)) {
                blueprint_seen = true;
            }
            messages.push(message);
        }
        ensure!(
            scene_seen && blueprint_seen,
            "display needs both static geometry and a blueprint"
        );
        Ok(Self { messages })
    }

    pub(crate) fn send(&self, recording: &RecordingStream) -> Result<()> {
        let id = recording
            .store_info()
            .context("recording disabled")?
            .store_id;
        for message in &self.messages {
            let mut message = message.clone();
            if message.store_id().kind() == StoreKind::Recording {
                message.set_store_id(id.clone());
            }
            recording.record_msg(message);
        }
        Ok(())
    }
}
