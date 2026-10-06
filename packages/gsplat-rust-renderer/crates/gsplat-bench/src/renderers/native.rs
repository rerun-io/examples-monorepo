//! Rerun 0.38.1 native viewer path: public PLY conversion, per-frame builder,
//! cached CPU back-to-front sort, and per-frame uploads.
use super::*;
use re_renderer::view_builder::{Projection, TargetConfiguration, ViewBuilder};
use re_renderer::{
    GaussianShCoefficient, GaussianSplatBuilder, RenderConfig, RenderContext, Rgba, Rgba32Unmul,
    ScreenshotProcessor, SortOrderCache, device_caps,
};
use re_sdk_types::archetypes::GaussianSplats3D;
use re_types_core::FromArrow as _;
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
}
impl Native {
    pub async fn new(ply_path: &Path, expected_count: usize) -> Result<Self> {
        let gaussians = GaussianSplats3D::from_ply_file_path(ply_path)?;

        let field = |name: &str, opt: &Option<re_sdk_types::SerializedComponentBatch>| {
            opt.as_ref()
                .map(|col| col.array.clone())
                .ok_or_else(|| Error::Invalid(format!("PLY has no {name}")))
        };

        let centers: Vec<glam::Vec3> = re_sdk_types::components::Position3D::from_arrow(&field(
            "centers",
            &gaussians.centers,
        )?)
        .map_err(|e| Error::Invalid(e.to_string()))?
        .into_iter()
        .map(|p| glam::Vec3::from_array(p.0.0))
        .collect();
        let scales: Vec<glam::Vec3> =
            re_sdk_types::components::Scale3D::from_arrow(&field("scales", &gaussians.scales)?)
                .map_err(|e| Error::Invalid(e.to_string()))?
                .into_iter()
                .map(|s| glam::Vec3::from_array(s.0.0))
                .collect();
        let rotations: Vec<glam::Quat> = re_sdk_types::components::RotationQuat::from_arrow(
            &field("quaternions", &gaussians.quaternions)?,
        )
        .map_err(|e| Error::Invalid(e.to_string()))?
        .into_iter()
        .map(|q| glam::Quat::from_array(q.0.0))
        .collect();
        let colors: Vec<Rgba32Unmul> =
            re_sdk_types::components::Color::from_arrow(&field("colors", &gaussians.colors)?)
                .map_err(|e| Error::Invalid(e.to_string()))?
                .into_iter()
                .map(|c| Rgba32Unmul::from_rgba_unmul_array(c.to_array()))
                .collect();
        let sh_coefficients: Vec<[GaussianShCoefficient; 15]> = gaussians
            .sh_coefficients
            .as_ref()
            .map(|sh| {
                re_sdk_types::components::SphericalHarmonics3Rgb::from_arrow(&sh.array).map(|v| {
                    v.into_iter()
                        .map(|sh| {
                            std::array::from_fn(|i| GaussianShCoefficient::from_rgb(sh.0.0[i]))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .transpose()
            .map_err(|e| Error::Invalid(e.to_string()))?
            .unwrap_or_default();

        let bounds = macaw::BoundingBox::from_points(centers.iter().copied());
        let sh_count = gaussians
            .spherical_harmonics_degree
            .as_ref()
            .map(|batch| {
                re_sdk_types::components::SphericalHarmonicsDegree::from_arrow(&batch.array)
            })
            .transpose()
            .map_err(|e| Error::Invalid(e.to_string()))?
            .and_then(|degrees| degrees.first().copied())
            .map(|degree| degree.num_coefficients())
            .unwrap_or(if sh_coefficients.is_empty() { 0 } else { 15 });
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
        })
    }
}
impl RenderEngine for Native {
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
                    vertical_fov: c.vertical_fov(),
                    near_plane_distance: 0.01,
                    aspect_ratio: c.aspect(),
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
        if parity {
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
        self.screenshot_pending = parity;
        Ok(Counts::default())
    }
    fn finish(&self) -> Result<()> {
        wait(&self.ctx.device)
    }
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        if !self.screenshot_pending {
            return Err(Error::Invalid(
                "render with parity=true before readback".into(),
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
