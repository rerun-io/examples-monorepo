//! Rerun 0.38.1 native renderer path: public PLY conversion, per-frame builder,
//! per-frame CPU back-to-front sort with a cached seed order, and per-frame uploads.
use super::*;
use crate::wait;
use glam::Vec2;
use gsplat_render::camera::{CameraModel, opengl_to_opencv};
use re_renderer::view_builder::{Projection, TargetConfiguration, ViewBuilder};
use re_renderer::{
    GaussianShCoefficient, GaussianSplatBuilder, RenderConfig, RenderContext, Rgba, Rgba32Unmul,
    ScreenshotProcessor, SortOrderCache, device_caps,
};
use re_sdk_types::archetypes::GaussianSplats3D;
use std::path::Path;

pub struct Native {
    ctx: RenderContext,
    adapter: Adapter,
    centers: Vec<Vec3>,
    scales: Vec<Vec3>,
    rotations: Vec<glam::Quat>,
    colors: Vec<Rgba32Unmul>,
    sh_coefficients: Vec<[GaussianShCoefficient; 15]>,
    sort: SortOrderCache,
    bounds: macaw::BoundingBox,
    sh_count: usize,
    picking_ids: Vec<re_renderer::PickingLayerInstanceId>,
    screenshot_pending: bool,
    capture_next: bool,
}
impl Native {
    pub async fn new(ply_path: &Path, expected_count: usize) -> Result<Self> {
        let gaussians = GaussianSplats3D::from_ply_file_path(ply_path)?;

        let decoded = super::decode(&gaussians)?;
        let centers: Vec<_> = decoded.centers.into_iter().map(Vec3::from_array).collect();
        let scales = decoded.scales.into_iter().map(Vec3::from_array).collect();
        let rotations = decoded
            .quaternions
            .into_iter()
            .map(glam::Quat::from_array)
            .collect();
        let colors = decoded
            .colors
            .into_iter()
            .map(|c| Rgba32Unmul::from_rgba_unmul_array(c.to_be_bytes()))
            .collect();
        let sh_coefficients: Vec<_> = decoded
            .sh
            .into_iter()
            .map(|sh| sh.map(GaussianShCoefficient::from_rgb))
            .collect();
        let sh_count = if sh_coefficients.is_empty() {
            0
        } else {
            ((decoded.degree.min(3) + 1).pow(2) - 1) as usize
        };
        let bounds = macaw::BoundingBox::from_points(centers.iter().copied());
        let picking_ids = (0..centers.len())
            .map(|i| re_renderer::PickingLayerInstanceId(i as u64))
            .collect();
        if centers.len() != expected_count {
            return Err(Error::Invalid(
                "Brush and Rerun PLY loaders disagree on splat count".into(),
            ));
        }
        let mut descriptor = device_caps::instance_descriptor(None);
        descriptor.flags = wgpu::InstanceFlags::empty();
        let instance = wgpu::Instance::new(descriptor);
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        let adapter = device_caps::select_adapter(&adapters, wgpu::Backends::all(), None)
            .map_err(Error::Gpu)?;
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return Err(Error::Gpu(
                "native benchmark requires a hardware GPU, not a CPU fallback".into(),
            ));
        }
        let info = adapter.get_info().into();
        let caps = device_caps::DeviceCaps::from_adapter(&adapter).map_err(gpu)?;
        let (device, queue) = adapter
            .request_device(&caps.device_descriptor())
            .await
            .map_err(gpu)?;
        let ctx = RenderContext::new(
            &adapter,
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            |_| RenderConfig::testing(),
        )
        .map_err(gpu)?;
        Ok(Self {
            ctx,
            adapter: info,
            centers,
            scales,
            rotations,
            colors,
            sh_coefficients,
            sort: SortOrderCache::default(),
            bounds,
            sh_count,
            picking_ids,
            screenshot_pending: false,
            capture_next: false,
        })
    }
}
impl RenderEngine for Native {
    fn capture_next_frame(&mut self) {
        self.capture_next = true;
    }
    async fn render(&mut self, c: &CameraSpec, parity: bool) -> Result<Counts> {
        if !matches!(c.model, CameraModel::Pinhole) {
            return Err(Error::Unsupported("native lens model".into()));
        }
        let view = macaw::IsoTransform::from_mat4(&opengl_to_opencv(c.pose()).inverse())
            .ok_or_else(|| Error::Invalid("native rigid pose".into()))?;
        self.ctx.begin_frame();
        let mut builder = ViewBuilder::new(
            &self.ctx,
            TargetConfiguration {
                name: "gsplat-bench".into(),
                blend_with_background: if parity {
                    re_renderer::view_builder::BlendWithBackground::Premultiplied
                } else {
                    re_renderer::view_builder::BlendWithBackground::No
                },
                resolution_in_pixel: [c.width, c.height],
                view_from_world: view,
                projection_from_view: Projection::Perspective {
                    vertical_fov: 2.0 * (c.height as f32 / (2.0 * c.fy)).atan(),
                    near_plane_distance: 0.01,
                    aspect_ratio: c.width as f32 * c.fy / (c.height as f32 * c.fx),
                },
                viewport_transformation: re_renderer::RectTransform {
                    region: re_renderer::RectF32::UNIT,
                    region_of_interest: re_renderer::RectF32 {
                        min: Vec2::new(0.5 - c.cx / c.width as f32, 0.5 - c.cy / c.height as f32),
                        extent: Vec2::ONE,
                    },
                },
                ..Default::default()
            },
            re_renderer::ViewBuilderId::new(0),
        )
        .map_err(gpu)?;
        let mut splats = GaussianSplatBuilder::new(&self.ctx);
        splats
            .batch("gaussians")
            .sort_order(self.sort.clone())
            .object_space_bounding_box(self.bounds)
            .add_gaussians(
                &self.centers,
                &self.scales,
                &self.rotations,
                &self.colors,
                &self.sh_coefficients,
                self.sh_count,
                &self.picking_ids,
            );
        builder
            .queue_draw(&self.ctx, splats.into_draw_data().map_err(gpu)?)
            .map_err(gpu)?;
        let capture = std::mem::take(&mut self.capture_next) || parity;
        if capture {
            builder
                .schedule_screenshot(&self.ctx, 42, ())
                .map_err(gpu)?;
        }
        let command = builder
            .draw(
                &self.ctx,
                if parity {
                    Rgba::TRANSPARENT
                } else {
                    Rgba::BLACK
                },
            )
            .map_err(gpu)?;
        self.ctx.before_submit();
        self.ctx.queue.submit([command]);
        self.screenshot_pending = capture;
        Ok(Counts::default())
    }
    fn finish(&self) -> Result<()> {
        wait(&self.ctx.device)
    }
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        if !self.screenshot_pending {
            return Err(Error::Invalid(
                "request capture before rendering for readback".into(),
            ));
        }
        for _ in 0..20 {
            self.ctx.begin_frame();
            let mut pixels = None;
            ScreenshotProcessor::next_readback_result::<()>(&self.ctx, 42, |data, _, ()| {
                pixels = Some(data.to_vec());
            });
            self.ctx.before_submit();
            if let Some(pixels) = pixels {
                self.screenshot_pending = false;
                return Ok(pixels.into_iter().map(|v| v as f32 / 255.0).collect());
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        Err(Error::Gpu("native screenshot readback timed out".into()))
    }
    fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }
}
