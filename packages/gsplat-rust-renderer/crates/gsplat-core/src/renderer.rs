//! Encode one GPU-driven forward path against shared scenes and independent views.
use crate::kernels::Kernels;
use crate::{Camera, Error, RenderOptions, Scene, Splats, Target, ViewState};
use std::sync::Arc;

pub const STAGE_NAMES: [&str; 8] = [
    "project_forward",
    "depth_sort",
    "gather_scan",
    "project_visible",
    "map_intersections",
    "tile_sort",
    "tile_offsets",
    "rasterize",
];
/// Contiguous GPU intervals; projection includes indirect-dispatch preparation.
pub fn stage_queries(stage: usize) -> (usize, usize) {
    (if stage == 0 { 0 } else { stage + 1 }, stage + 2)
}

/// Device-lifetime pipelines. Scene uploads and per-view scratch have independent lifetimes.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limit: u64,
    kernels: Kernels,
}
impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, Error> {
        let limits = device.limits();
        crate::check_adapter(device.features(), &limits)?;
        let kernels = Kernels::new(device);
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            limit: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            kernels,
        })
    }
    pub fn upload(&self, splats: &Splats) -> Result<Arc<Scene>, Error> {
        Ok(Arc::new(Scene::upload(&self.device, splats, self.limit)?))
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
    /// Encode without submission or CPU waits. Submit, then poll this view's feedback.
    /// Overflow leaves the target intact; rerender after feedback requests more capacity.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &mut ViewState,
        camera: &Camera,
        options: &RenderOptions,
        target: Target,
    ) -> Result<(), Error> {
        camera.validate()?;
        options.validate()?;
        view.encode(encoder, &self.queue, &self.kernels, camera, options, target)
    }
}
