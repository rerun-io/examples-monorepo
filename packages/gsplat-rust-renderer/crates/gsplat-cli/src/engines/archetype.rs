//! PLY -> native Arrow archetype -> the viewer's shared core conversion.
use crate::Result;
use anyhow::Context as _;
use re_sdk_types::archetypes::GaussianSplats3D;
use re_types_core::Archetype as _;
use std::path::Path;

pub fn archetype_splats(path: &Path) -> Result<gsplat_core::Splats> {
    let native = if path.extension().is_some_and(|e| e == "rrd") {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut found = None;
        for message in re_log_encoding::rrd::DecoderApp::decode_lazy(reader) {
            let message = message?;
            let re_log_types::LogMsg::ArrowMsg(store, message) = message else {
                continue;
            };
            if store.kind() != re_log_types::StoreKind::Recording {
                continue;
            }
            let chunk = re_chunk::Chunk::from_arrow_msg(&message)?;
            if chunk
                .raw_component_array(GaussianSplats3D::descriptor_centers().component)
                .is_none()
            {
                continue;
            }
            if found.is_some() {
                anyhow::bail!("archetype parity RRD requires exactly one complete splat row");
            }
            let row = chunk.into_unit().ok_or_else(|| {
                anyhow::anyhow!("archetype parity RRD must contain one complete row")
            })?;
            found = Some(GaussianSplats3D::from_arrow_components(
                GaussianSplats3D::all_components()
                    .iter()
                    .filter_map(|descriptor| {
                        row.component_batch_raw(descriptor.component)
                            .map(|array| (descriptor.clone(), array))
                    }),
            )?);
        }
        found.context("RRD has no GaussianSplats3D row")?
    } else {
        GaussianSplats3D::from_ply_file_path(path)?
    };
    Ok(super::decode(&native)?.native().to_core())
}
