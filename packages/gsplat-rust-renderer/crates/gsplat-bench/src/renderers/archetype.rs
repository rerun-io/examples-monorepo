//! PLY -> native Arrow archetype -> the viewer's shared core conversion.
use crate::{Error, Result};
use re_sdk_types::{archetypes::GaussianSplats3D, components};
use re_types_core::{Archetype as _, FromArrow as _};
use std::path::Path;

pub fn archetype_splats(path: &Path) -> Result<gsplat_core::Splats> {
    let native = if path.extension().is_some_and(|e| e == "rrd") {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut found = None;
        for message in re_log_encoding::rrd::DecoderApp::decode_lazy(reader) {
            let message = message.map_err(|e| Error::Invalid(e.to_string()))?;
            let re_log_types::LogMsg::ArrowMsg(store, message) = message else {
                continue;
            };
            if store.kind() != re_log_types::StoreKind::Recording {
                continue;
            }
            let chunk = re_chunk::Chunk::from_arrow_msg(&message)
                .map_err(|e| Error::Invalid(e.to_string()))?;
            if chunk
                .raw_component_array(GaussianSplats3D::descriptor_centers().component)
                .is_none()
            {
                continue;
            }
            if found.is_some() {
                return Err(Error::Invalid(
                    "archetype parity RRD requires exactly one complete splat row".into(),
                ));
            }
            let row = chunk.into_unit().ok_or_else(|| {
                Error::Invalid("archetype parity RRD must contain one complete row".into())
            })?;
            found = Some(
                GaussianSplats3D::from_arrow_components(
                    GaussianSplats3D::all_components()
                        .iter()
                        .filter_map(|descriptor| {
                            row.component_batch_raw(descriptor.component)
                                .map(|array| (descriptor.clone(), array))
                        }),
                )
                .map_err(|e| Error::Invalid(e.to_string()))?,
            );
        }
        found.ok_or_else(|| Error::Invalid("RRD has no GaussianSplats3D row".into()))?
    } else {
        GaussianSplats3D::from_ply_file_path(path)?
    };
    let centers = components::Position3D::from_arrow(
        &native
            .centers
            .as_ref()
            .ok_or_else(|| Error::Invalid("PLY has no centers".into()))?
            .array,
    )
    .map_err(|e| Error::Invalid(e.to_string()))?
    .into_iter()
    .map(|p| p.0.0)
    .collect::<Vec<_>>();
    let scales = native
        .scales
        .as_ref()
        .map(|batch| components::Scale3D::from_arrow(&batch.array))
        .transpose()
        .map_err(|e| Error::Invalid(e.to_string()))?
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.0.0)
        .collect::<Vec<_>>();
    let quaternions = native
        .quaternions
        .as_ref()
        .map(|batch| components::RotationQuat::from_arrow(&batch.array))
        .transpose()
        .map_err(|e| Error::Invalid(e.to_string()))?
        .unwrap_or_default()
        .into_iter()
        .map(|q| q.0.0)
        .collect::<Vec<_>>();
    let colors = native
        .colors
        .as_ref()
        .map(|batch| components::Color::from_arrow(&batch.array))
        .transpose()
        .map_err(|e| Error::Invalid(e.to_string()))?
        .unwrap_or_default()
        .into_iter()
        .map(|c| u32::from_be_bytes(c.to_array()))
        .collect::<Vec<_>>();
    let sh = native
        .sh_coefficients
        .as_ref()
        .map(|batch| components::SphericalHarmonics3Rgb::from_arrow(&batch.array))
        .transpose()
        .map_err(|e| Error::Invalid(e.to_string()))?
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.0.0)
        .collect::<Vec<_>>();
    let degree = native
        .spherical_harmonics_degree
        .as_ref()
        .map(|batch| components::SphericalHarmonicsDegree::from_arrow(&batch.array))
        .transpose()
        .map_err(|e| Error::Invalid(e.to_string()))?
        .and_then(|v| v.first().map(|d| d.0.0))
        .unwrap_or(3);
    gsplat_core::native::NativeSplats {
        centers: &centers,
        scales: &scales,
        quaternions: &quaternions,
        colors: &colors,
        sh: &sh,
        degree,
    }
    .to_core()
    .map_err(|e| Error::Invalid(e.to_string()))
}
