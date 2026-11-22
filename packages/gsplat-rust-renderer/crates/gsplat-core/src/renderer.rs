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

    /// Canonical rows + per-row motion: every view evaluates the motion on the GPU at `RenderOptions::time`
    /// (one dispatch per time change, no re-upload).
    pub fn upload_animated(
        &self,
        splats: &Splats,
        motion: crate::Motion<'_>,
    ) -> Result<Arc<Scene>, Error> {
        let mut scene = Scene::upload(&self.device, splats, self.limit)?;
        scene.animate(&self.device, motion, self.limit)?;
        Ok(Arc::new(scene))
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

    /// An entropy-decoded `.asog` on the GPU: static part through `sog_dequant.wgsl`, motion through
    /// `asog_motion.wgsl` (both submitted on `queue`), evaluated per view at `RenderOptions::time`. The tests' CPU oracle is
    /// `upload_animated(&sog::dequantize(..)?, Motion::Asog(&parts.motion()))`.
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
