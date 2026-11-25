//! Encode one GPU-driven forward path against shared scenes and independent views.
use crate::kernels::Kernels;
use crate::{Camera, Error, HexelVariant, Hexels, RenderOptions, Scene, Splats, Target, ViewState};
use std::sync::Arc;

/// Device-lifetime pipelines. Scene uploads and per-view scratch have independent lifetimes.
pub struct Renderer {
    device: wgpu::Device,
    limit: u64,
    kernels: Kernels,
}

impl Renderer {
    pub fn new(device: &wgpu::Device) -> Result<Self, Error> {
        let limits = device.limits();
        crate::check_adapter(&limits)?;
        let kernels = Kernels::new(device);
        Ok(Self {
            device: device.clone(),
            limit: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            kernels,
        })
    }

    /// The renderer with its splat rasterizers built from `source`, a patched `raster.wgsl`.
    #[cfg(test)]
    pub(crate) fn with_splat_raster(mut self, source: &str) -> Self {
        self.kernels.set_splat_raster(&self.device, source);
        self
    }

    /// The renderer with its splat projection and tile mapping built from `project` and `map`, a patched
    /// `project.wgsl` and `map.wgsl`.
    #[cfg(test)]
    pub(crate) fn with_splat_tiles(mut self, project: &str, map: &str) -> Self {
        self.kernels.set_splat_tiles(&self.device, project, map);
        self
    }

    /// Whether the scan and sort kernels use subgroup operations (the device enabled `SUBGROUP`)
    /// or their workgroup-memory variant.
    pub fn uses_subgroups(&self) -> bool {
        self.device.features().contains(wgpu::Features::SUBGROUP)
    }

    /// Reject oversized wire dimensions before converting the CPU data.
    pub fn check_capacity(&self, count: usize, sh_degree: u32) -> Result<(), Error> {
        self.check_bytes(Splats::gpu_bytes(count, sh_degree))
    }

    /// Hexel parameters and each view's projected hexels are one storage binding each.
    pub fn check_hexel_capacity(&self, count: usize, variant: HexelVariant) -> Result<(), Error> {
        self.check_bytes(Hexels::gpu_bytes(count, variant))
    }

    fn check_bytes(&self, required: u64) -> Result<(), Error> {
        if required > self.limit {
            return Err(Error::Capacity {
                required,
                limit: self.limit,
            });
        }
        Ok(())
    }

    pub fn upload(&self, splats: &Splats) -> Result<Arc<Scene>, Error> {
        Ok(Arc::new(Scene::upload(&self.device, splats, self.limit)?))
    }

    /// De-quantise SOG texel planes on the GPU into a splat scene (one compute pass, submitted on `queue`).
    /// The tests' CPU oracle is `upload(&sog::dequantize(q, planes)?)`; both give the same scene within f32 quaternion rounding.
    pub fn upload_sog(
        &self,
        queue: &wgpu::Queue,
        q: &crate::sog::Quantization,
        planes: &crate::sog::Planes<'_>,
    ) -> Result<Arc<Scene>, Error> {
        Ok(Arc::new(crate::sog_gpu::upload_sog(
            &self.device,
            queue,
            &self.kernels.sog_dequant,
            q,
            planes,
            self.limit,
        )?))
    }

    /// A Streamed SOG set (all loaded levels in one scene, see [`crate::lod`]) and its host-side index.
    /// WebP decode on the CPU, one GPU de-quantisation pass per file (submitted on `queue`).
    /// The finest level >= `requested` at which a Streamed SOG set (per file: level, rows, SH degree bound) fits
    /// this device; [`Self::upload_lod`] applies the same rule. `None` when not even the coarsest level fits.
    pub fn fit_lod_min_level(&self, files: &[(u32, usize, u32)], requested: u32) -> Option<u32> {
        crate::lod::fit_min_level(files, requested, |rows, degree| {
            self.check_capacity(rows, degree).is_ok()
        })
    }

    pub fn upload_lod(
        &self,
        queue: &wgpu::Queue,
        set: &crate::lod::LodSet<'_>,
        options: &crate::lod::LoadOptions,
    ) -> Result<(Arc<Scene>, Arc<crate::lod::LodIndex>), Error> {
        let (scene, index) = crate::lod::upload(
            &self.device,
            queue,
            &self.kernels.sog_dequant,
            set,
            options,
            self.limit,
        )?;
        Ok((Arc::new(scene), Arc::new(index)))
    }

    /// An entropy-decoded `.asog` on the GPU: static part through `sog_dequant.wgsl`, motion through
    /// `asog_motion.wgsl` (both submitted on `queue`), evaluated per view at `RenderOptions::time` (one dispatch per
    /// time change, no re-upload). The tests' CPU oracle is `AsogParts::motion` + `AsogMotion::evaluate`.
    pub fn upload_asog(
        &self,
        queue: &wgpu::Queue,
        parts: &crate::asog::AsogParts,
    ) -> Result<Arc<Scene>, Error> {
        let mut scene = crate::sog_gpu::upload_sog(
            &self.device,
            queue,
            &self.kernels.sog_dequant,
            &parts.quantization,
            &parts.planes(),
            self.limit,
        )?;
        let canonical = scene
            .transforms()
            .ok_or(Error::Input("an asog scene holds splats"))?;
        let motion = crate::motion_gpu::dequantize_asog(
            &self.device,
            queue,
            &self.kernels.asog_motion,
            parts,
            canonical,
            self.limit,
        )?;
        scene.attach_motion(motion)?;
        Ok(Arc::new(scene))
    }

    /// Hexel scenes share views, targets and feedback with splats; they need a pinhole camera.
    pub fn upload_hexels(&self, hexels: &Hexels) -> Result<Arc<Scene>, Error> {
        Ok(Arc::new(Scene::upload_hexels(
            &self.device,
            hexels,
            self.limit,
        )?))
    }

    pub fn create_view(
        &self,
        scene: &Arc<Scene>,
        initial_capacity: u32,
    ) -> Result<ViewState, Error> {
        ViewState::new(
            &self.device,
            &self.kernels,
            scene,
            initial_capacity,
            self.limit,
        )
    }

    /// Allocate view scratch before queuing draw data so capacity failures can select a fallback.
    pub fn prepare_view(
        &self,
        queue: &wgpu::Queue,
        view: &mut ViewState,
        camera: &Camera,
    ) -> Result<(), Error> {
        camera.validate()?;
        view.prepare(queue, &self.kernels, camera.size).map(|_| ())
    }

    /// Encode without submission or CPU waits. Submit, then poll this view's feedback.
    /// Overflow leaves the target intact; rerender after feedback requests more capacity.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &mut ViewState,
        camera: &Camera,
        options: &RenderOptions,
        target: Target,
    ) -> Result<(), Error> {
        camera.validate()?;
        options.validate()?;
        view.encode(encoder, queue, &self.kernels, camera, options, target)
    }
}
