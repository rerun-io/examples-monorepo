//! PLY -> native Arrow archetype -> the viewer's shared core conversion.
use anyhow::Context as _;
use anyhow::Result;
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
    Ok(decode(&native)?.native().to_core())
}

pub(super) struct Decoded {
    pub(super) centers: Vec<[f32; 3]>,
    pub(super) scales: Vec<[f32; 3]>,
    pub(super) quaternions: Vec<[f32; 4]>,
    pub(super) colors: Vec<u32>,
    pub(super) sh: Vec<[[half::f16; 3]; 15]>,
    pub(super) degree: u32,
}

impl Decoded {
    pub(super) fn native(&self) -> gsplat_core::native::NativeSplats<'_> {
        gsplat_core::native::NativeSplats {
            centers: &self.centers,
            scales: &self.scales,
            quaternions: &self.quaternions,
            colors: &self.colors,
            sh: &self.sh,
            degree: self.degree,
        }
    }
}

fn column<C: re_types_core::FromArrow>(
    batch: Option<&re_sdk_types::SerializedComponentBatch>,
) -> Result<Vec<C>> {
    batch
        .map(|b| C::from_arrow(&b.array))
        .transpose()
        .map(|v| v.unwrap_or_default())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub(super) fn decode(native: &re_sdk_types::archetypes::GaussianSplats3D) -> Result<Decoded> {
    use re_sdk_types::components as c;
    if native.centers.is_none() {
        anyhow::bail!("native splats have no centers");
    }
    Ok(Decoded {
        centers: column::<c::Position3D>(native.centers.as_ref())?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        scales: column::<c::Scale3D>(native.scales.as_ref())?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        quaternions: column::<c::RotationQuat>(native.quaternions.as_ref())?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        colors: column::<c::Color>(native.colors.as_ref())?
            .into_iter()
            .map(|v| u32::from_be_bytes(v.to_array()))
            .collect(),
        sh: column::<c::SphericalHarmonics3Rgb>(native.sh_coefficients.as_ref())?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        degree: column::<c::SphericalHarmonicsDegree>(native.spherical_harmonics_degree.as_ref())?
            .first()
            .map_or(3, |v| v.0.0),
    })
}
