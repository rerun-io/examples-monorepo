//! Immutable GPU parameters shared by views without another upload.
use crate::{Error, Splats};

pub struct Scene {
    pub(crate) transforms: wgpu::Buffer,
    pub(crate) opacity: wgpu::Buffer,
    pub(crate) sh: wgpu::Buffer,
    pub(crate) min_scale: wgpu::Buffer,
    pub(crate) has_min_scale: bool,
    pub(crate) n: u32,
    pub(crate) degree: u32,
}
impl Scene {
    pub(crate) fn upload(
        device: &wgpu::Device,
        splats: &Splats,
        limit: u64,
    ) -> Result<Self, Error> {
        let n =
            u32::try_from(splats.transforms.len()).map_err(|_| Error::Input("too many splats"))?;
        if n >= 0x8000_0000 {
            return Err(Error::Input(
                "visible count reserves its high bit for arithmetic overflow",
            ));
        }
        if splats.sh_degree > 4
            || splats.raw_opacities.len() != n as usize
            || splats.sh_coefficients.len() != n as usize * (splats.sh_degree as usize + 1).pow(2)
            || splats
                .min_scale
                .as_ref()
                .is_some_and(|s| s.len() != n as usize)
        {
            return Err(Error::Input(
                "inconsistent transform, opacity, SH, or scale-floor lengths",
            ));
        }
        let upload = |label, bytes: &[u8]| -> Result<wgpu::Buffer, Error> {
            if bytes.len() as u64 > limit {
                return Err(Error::Capacity {
                    required: bytes.len() as u64,
                    limit,
                });
            }
            Ok(crate::gpu::upload(
                device,
                label,
                bytes,
                wgpu::BufferUsages::STORAGE,
            ))
        };
        Ok(Self {
            transforms: upload("raw transforms", bytemuck::cast_slice(&splats.transforms))?,
            opacity: upload("raw opacity", bytemuck::cast_slice(&splats.raw_opacities))?,
            sh: upload(
                "SH RGB coefficients",
                bytemuck::cast_slice(&splats.sh_coefficients),
            )?,
            min_scale: upload(
                "3D scale floor",
                bytemuck::cast_slice(splats.min_scale.as_deref().unwrap_or(&[0.0])),
            )?,
            has_min_scale: splats.min_scale.is_some(),
            n,
            degree: splats.sh_degree,
        })
    }
}
