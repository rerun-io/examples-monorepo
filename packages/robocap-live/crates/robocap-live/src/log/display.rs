//! The display asset: a small `.rrd` holding the viewer layout (blueprint) and the static cap mesh, prepared once in Python
//! from DataForge's RoboCap blueprint pieces (`handtrack.apis.robocap_live_display`) and sent at the start of every stream.
//!
//! Adapted from PR #270's `robocap-recorder/src/display.rs`: the asset's recording messages are re-addressed to the live
//! recording, its blueprint keeps its own store, and the activation command is forced to make the layout active, so a
//! fresh viewer and a viewer that cached an older `robocap` layout both show it.

use std::io::BufReader;
use std::path::Path;

use rerun::RecordingStream;
use rerun::external::re_log_types::{LogMsg, StoreKind};
use rerun::log::Chunk;

use super::scene::RIG_PATH;

/// The application id the asset, the live recording and the blueprint share (the blueprint applies per application).
pub const APPLICATION_ID: &str = "robocap";
const MAX_ASSET_BYTES: u64 = 32 * 1024 * 1024;

/// Errors of loading a display asset.
#[derive(Debug, thiserror::Error)]
pub enum DisplayError {
    /// The file could not be read.
    #[error("{path}: {source}")]
    Io { path: String, source: std::io::Error },
    /// The file is not a readable rrd.
    #[error("{path}: not a readable rrd: {message}")]
    Decode { path: String, message: String },
    /// The file breaks the asset's contract.
    #[error("{path}: {message}")]
    Invalid { path: String, message: String },
}

/// The decoded asset, ready to send.
#[derive(Clone, Debug)]
pub struct DisplayAssets {
    messages: Vec<LogMsg>,
    /// Number of static scene chunks (the mesh).
    pub scene_chunks: usize,
    /// Number of blueprint messages.
    pub blueprint_messages: usize,
}

impl DisplayAssets {
    /// Load and check an asset.
    ///
    /// The asset must belong to the `robocap` application, hold a blueprint with its activation command, and carry only
    /// static scene data that cannot hide live data: no `Transform3D` on `/world/rig_00` (a static pose would shadow the
    /// SLAM pose) and nothing under the camera pinholes (the binary logs the cameras from its rig).
    ///
    /// # Errors
    ///
    /// [`DisplayError`] if the file is missing, too large, not an rrd, or breaks the contract above.
    pub fn load(path: &Path) -> Result<Self, DisplayError> {
        let name = path.display().to_string();
        let io = |source| DisplayError::Io { path: name.clone(), source };
        let invalid = |message: String| DisplayError::Invalid { path: name.clone(), message };
        if std::fs::metadata(path).map_err(io)?.len() > MAX_ASSET_BYTES {
            return Err(invalid(format!("larger than {MAX_ASSET_BYTES} bytes")));
        }
        let file = std::fs::File::open(path).map_err(io)?;
        let decode = |e: &dyn std::fmt::Display| DisplayError::Decode { path: name.clone(), message: e.to_string() };
        let decoder = re_log_encoding::DecoderApp::decode_eager(BufReader::new(file)).map_err(|e| decode(&e))?;
        let (mut messages, mut scene_chunks, mut blueprint_messages, mut activation) = (Vec::new(), 0, 0, false);
        for message in decoder {
            let mut message = message.map_err(|e| decode(&e))?;
            if message.store_id().application_id().as_str() != APPLICATION_ID {
                return Err(invalid(format!("belongs to application {}, not {APPLICATION_ID}", message.store_id().application_id())));
            }
            let kind = message.store_id().kind();
            match (&mut message, kind) {
                (LogMsg::BlueprintActivationCommand(command), _) => {
                    command.make_active = true;
                    command.make_default = true;
                    activation = true;
                }
                (_, StoreKind::Blueprint) => blueprint_messages += 1,
                // The live stream announces its own store; the asset's would overwrite its info.
                (LogMsg::SetStoreInfo(_), StoreKind::Recording) => continue,
                (LogMsg::ArrowMsg(_, arrow), StoreKind::Recording) => {
                    let chunk = Chunk::from_arrow_msg(arrow).map_err(|e| decode(&e))?;
                    let entity = chunk.entity_path().to_string();
                    if entity.starts_with("/__properties") || entity.starts_with("/__warnings") {
                        continue;
                    }
                    if !chunk.is_static() {
                        return Err(invalid(format!("{entity}: temporal data; the asset may hold static scene data only")));
                    }
                    if entity.contains("/pinhole") {
                        return Err(invalid(format!("{entity}: camera data; the binary logs the cameras from its rig")));
                    }
                    let has_transform = chunk.components().keys().any(|descriptor| descriptor.to_string().contains("Transform3D"));
                    if entity == RIG_PATH && has_transform {
                        return Err(invalid(format!("a static Transform3D on {RIG_PATH} would hide the live SLAM pose")));
                    }
                    scene_chunks += 1;
                }
            }
            messages.push(message);
        }
        if !activation || blueprint_messages == 0 {
            return Err(invalid("no blueprint (with its activation command)".into()));
        }
        Ok(Self { messages, scene_chunks, blueprint_messages })
    }

    /// Send the asset into a recording: scene messages re-addressed to it, the blueprint as is.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Invalid`] if the stream is disabled (it has no store).
    pub fn send(&self, recording: &RecordingStream) -> Result<(), DisplayError> {
        let store_id = recording
            .store_info()
            .ok_or_else(|| DisplayError::Invalid { path: "<recording>".into(), message: "the recording stream is disabled".into() })?
            .store_id;
        for message in &self.messages {
            let mut message = message.clone();
            if message.store_id().kind() == StoreKind::Recording {
                message.set_store_id(store_id.clone());
            }
            recording.record_msg(message);
        }
        Ok(())
    }
}
