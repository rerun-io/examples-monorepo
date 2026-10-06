//! One device-lifetime pipeline set, shared uploads, and independent asynchronous views.
use gsplat_core::{Camera, RenderMode, RenderOptions, Splats, Target};
use std::sync::Arc;

pub(crate) struct CoreRenderer(gsplat_core::Renderer);
#[derive(Clone)]
pub(crate) struct UploadedScene(Arc<gsplat_core::Scene>);
pub(crate) struct CoreView {
    state: gsplat_core::ViewState,
    last: Option<(Camera, RenderOptions)>,
}
impl CoreView {
    pub fn has_complete_image(&self) -> bool {
        self.last.is_some() && !self.state.has_pending_frames()
    }
    pub fn ready(&mut self) -> Result<bool, gsplat_core::Error> {
        match self.state.poll_feedback() {
            Ok(Some(stats)) if stats.needs_rerender => self.last = None,
            Err(error) => {
                self.last = None;
                return Err(error);
            }
            _ => {}
        }
        Ok(!self.state.has_pending_frames())
    }
}
impl CoreRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, gsplat_core::Error> {
        Ok(Self(gsplat_core::Renderer::new(device, queue)?))
    }
    pub fn upload(&self, cloud: &Splats) -> Result<UploadedScene, gsplat_core::Error> {
        let scene = UploadedScene(self.0.upload(cloud, RenderMode::Default)?);
        if std::env::var_os("GSPLAT_UPLOAD_PROBE").is_some() {
            eprintln!("GSPLAT_UPLOAD count={}", cloud.transforms.len());
        }
        Ok(scene)
    }
    pub fn create_view(
        &self,
        scene: &UploadedScene,
        count: usize,
    ) -> Result<CoreView, gsplat_core::Error> {
        Ok(CoreView {
            state: self
                .0
                .create_view(&scene.0, (count as u32).clamp(1, 1_048_576))?,
            last: None,
        })
    }
    /// Never block the UI for counts. Keep the last complete image until feedback
    /// allows another encode; a static view stops repainting once its output is valid.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        view: &mut CoreView,
        camera: &Camera,
        options: RenderOptions,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> Result<(bool, Option<Camera>), gsplat_core::Error> {
        if !view.ready()? {
            return Ok((true, None));
        }
        if view.last == Some((*camera, options)) {
            return Ok((false, None));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        self.0.render(
            &mut encoder,
            &mut view.state,
            camera,
            &options,
            Target::TextureDepth {
                color: target,
                depth,
            },
        )?;
        queue.submit([encoder.finish()]);
        view.last = Some((*camera, options));
        Ok((true, Some(*camera)))
    }
}
