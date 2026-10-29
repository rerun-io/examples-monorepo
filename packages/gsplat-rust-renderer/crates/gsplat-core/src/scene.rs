//! Immutable GPU parameters shared by views without another upload.
use crate::{Error, HexelVariant, Hexels, Splats};

pub(crate) enum Inputs {
    Splats {
        transforms: wgpu::Buffer,
        opacity: wgpu::Buffer,
        sh: wgpu::Buffer,
        min_scale: wgpu::Buffer,
        has_min_scale: bool,
        degree: u32,
    },
    Hexels {
        params: wgpu::Buffer,
        variant: HexelVariant,
    },
}

pub struct Scene {
    pub(crate) inputs: Inputs,
    pub(crate) n: u32,
}

impl Scene {
    /// Number of primitives in this upload.
    pub fn len(&self) -> u32 {
        self.n
    }

    /// Whether this upload contains no primitives.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes held by this scene's GPU buffers.
    pub fn gpu_byte_size(&self) -> u64 {
        match &self.inputs {
            Inputs::Splats {
                transforms,
                opacity,
                sh,
                min_scale,
                ..
            } => [transforms, opacity, sh, min_scale]
                .into_iter()
                .map(wgpu::Buffer::size)
                .sum(),
            Inputs::Hexels { params, .. } => params.size(),
        }
    }

    pub(crate) fn hexels(&self) -> bool {
        matches!(self.inputs, Inputs::Hexels { .. })
    }

    /// Bytes of one projected primitive, matching `Splat`, `Hexel` and `ImageHexel` in the shaders.
    pub(crate) fn projected_bytes(&self) -> u64 {
        match self.inputs {
            Inputs::Splats { .. } => 40,
            Inputs::Hexels {
                variant: HexelVariant::Image,
                ..
            } => 64,
            Inputs::Hexels { .. } => 84,
        }
    }

    /// Scene buffers read by the forward (`visible == false`) or visible projection.
    pub(crate) fn bindings(&self, visible: bool) -> Vec<(u32, wgpu::BindingResource<'_>)> {
        match &self.inputs {
            Inputs::Splats {
                transforms,
                opacity,
                sh,
                min_scale,
                ..
            } => {
                let mut bindings = vec![
                    (1, transforms.as_entire_binding()),
                    (2, opacity.as_entire_binding()),
                    (3, min_scale.as_entire_binding()),
                ];
                if visible {
                    bindings.push((9, sh.as_entire_binding()));
                }
                bindings
            }
            Inputs::Hexels { params, .. } => vec![(1, params.as_entire_binding())],
        }
    }

    fn count(n: usize) -> Result<u32, Error> {
        let n = u32::try_from(n)?;
        if n >= 0x8000_0000 {
            return Err(Error::Input(
                "visible count reserves its high bit for arithmetic overflow",
            ));
        }
        Ok(n)
    }

    pub(crate) fn upload(
        device: &wgpu::Device,
        splats: &Splats,
        limit: u64,
    ) -> Result<Self, Error> {
        let n = Self::count(splats.transforms.len())?;
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
        let upload = |label, bytes: &[u8]| upload(device, label, bytes, limit);
        Ok(Self {
            inputs: Inputs::Splats {
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
                degree: splats.sh_degree,
            },
            n,
        })
    }

    pub(crate) fn upload_hexels(
        device: &wgpu::Device,
        hexels: &Hexels,
        limit: u64,
    ) -> Result<Self, Error> {
        if !hexels.params.len().is_multiple_of(hexels.variant.stride()) {
            return Err(Error::Input(
                "hexel parameters are not a multiple of the variant stride",
            ));
        }
        Ok(Self {
            n: Self::count(hexels.len())?,
            inputs: Inputs::Hexels {
                params: upload(
                    device,
                    "hexel parameters",
                    bytemuck::cast_slice(&hexels.params),
                    limit,
                )?,
                variant: hexels.variant,
            },
        })
    }
}

fn upload(
    device: &wgpu::Device,
    label: &str,
    bytes: &[u8],
    limit: u64,
) -> Result<wgpu::Buffer, Error> {
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
}
