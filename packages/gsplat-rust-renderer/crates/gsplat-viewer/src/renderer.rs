//! Compute draw data and transparent depth-tested composition for `re_renderer`.
use gsplat_core::{Camera, Error, RenderOptions};
use parking_lot::Mutex;
use re_renderer::external::smallvec::smallvec;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};
use re_renderer::{BindGroupDesc, BindGroupEntry, GpuBindGroup, GpuTexture};
use re_viewer::external::{parking_lot, re_log, re_renderer};
use std::sync::{Arc, OnceLock};

pub(crate) struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    composite_pipeline: re_renderer::GpuRenderPipelineHandle,
    pub(crate) core: Option<gsplat_core::Renderer>,
}
struct TargetImage {
    size: glam::UVec2,
    color: GpuTexture,
    depth: GpuTexture,
    composite: GpuBindGroup,
}
// Only Pending can carry an in-flight eye; Dirty retains the previous image on overflow.
enum Image {
    // Keep the fallback for this scene generation; moving the eye must not retry
    // an impossible allocation on every other frame. A relog resets this state.
    Capacity {
        required: u64,
        limit: u64,
    },
    Dirty,
    Pending {
        camera: Camera,
        options: RenderOptions,
    },
    Complete {
        camera: Camera,
        options: RenderOptions,
    },
}
// Bound the first allocation; feedback grows scratch for denser views.
const MAX_INITIAL_INTERSECTIONS: u32 = 1_048_576;
#[derive(Clone, Copy)]
pub(crate) enum Prepared {
    Pending,
    Current,
    Encode,
}
pub(crate) struct RenderView {
    core: gsplat_core::ViewState,
    target: TargetImage,
    image: Image,
    last_good: Option<GpuBindGroup>,
}
impl RenderView {
    pub(crate) fn gpu_byte_size(&self) -> u64 {
        self.core.gpu_byte_size()
            + u64::from(self.target.size.x) * u64::from(self.target.size.y) * 8
    }

    pub(crate) fn replace_scene(
        &mut self,
        core: &gsplat_core::Renderer,
        scene: &Arc<gsplat_core::Scene>,
    ) -> Result<(), Error> {
        let view = core_view(core, scene)?;
        self.last_good = self.completed();
        self.image = Image::Dirty;
        self.core = view;
        Ok(())
    }
    fn completed(&self) -> Option<GpuBindGroup> {
        match &self.image {
            Image::Dirty | Image::Pending { .. } | Image::Capacity { .. } => self.last_good.clone(),
            Image::Complete { .. } => Some(self.target.composite.clone()),
        }
    }
    fn is_current(&self, camera: &Camera, options: &RenderOptions) -> bool {
        matches!(&self.image, Image::Complete { camera: old_camera, options: old_options } if old_camera == camera && old_options == options)
    }
    pub(crate) fn prepare(
        &mut self,
        renderer: &GaussianRenderer,
        ctx: &re_renderer::RenderContext,
        camera: &Camera,
        options: RenderOptions,
    ) -> Result<Prepared, Error> {
        if let Image::Capacity {
            required, limit, ..
        } = &self.image
        {
            return Err(Error::Capacity {
                required: *required,
                limit: *limit,
            });
        }
        let feedback = match self.core.poll_feedback() {
            Ok(feedback) => feedback,
            Err(Error::Capacity { required, limit }) => {
                self.last_good = self.completed();
                self.image = Image::Capacity { required, limit };
                return Err(Error::Capacity { required, limit });
            }
            Err(error) => return Err(error),
        };
        if let Some(stats) = feedback
            && let Image::Pending { camera, options } = self.image
        {
            self.image = if stats.needs_rerender {
                Image::Dirty
            } else {
                self.last_good = None;
                Image::Complete { camera, options }
            };
        }
        if self.core.has_pending_frames() {
            return Ok(Prepared::Pending);
        }
        if self.is_current(camera, &options) {
            return Ok(Prepared::Current);
        }
        renderer
            .core()?
            .prepare_view(&ctx.queue, &mut self.core, camera)?;
        if self.target.size != camera.size {
            self.last_good = self.completed();
            self.image = Image::Dirty;
            self.target = renderer.target(ctx, camera.size);
        }
        Ok(Prepared::Encode)
    }
    fn encode(
        &mut self,
        renderer: &GaussianRenderer,
        ctx: &re_renderer::RenderContext,
        camera: &Camera,
        options: RenderOptions,
    ) -> Result<(), Error> {
        re_tracing::profile_function!();
        self.last_good = self.completed();
        let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
        renderer.core()?.render(
            &ctx.queue,
            encoder.get(),
            &mut self.core,
            camera,
            &options,
            gsplat_core::Target::TextureDepth {
                color: self.target.color.default_view.clone(),
                depth: self.target.depth.default_view.clone(),
            },
        )?;
        self.image = Image::Pending {
            camera: *camera,
            options,
        };
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct GaussianDrawData {
    pub(crate) view: Arc<Mutex<RenderView>>,
    pub(crate) camera: Camera,
    pub(crate) options: RenderOptions,
    pub(crate) center: glam::Vec3A,
    pub(crate) prepared: Prepared,
}
impl DrawData for GaussianDrawData {
    type Renderer = GaussianRenderer;
    fn collect_drawables(
        &self,
        view_info: &re_renderer::renderer::DrawableCollectionViewInfo,
        collector: &mut re_renderer::DrawableCollector<'_>,
    ) {
        let ctx = collector.render_ctx();
        let Ok(renderer) = ctx.renderer::<GaussianRenderer>() else {
            return;
        };
        // Rerun 0.38.1 exposes only the camera position here; the supplied camera is last_eye.
        // Upstream view_info needs the current full pose, projection, and pixel resolution.
        if matches!(self.prepared, Prepared::Encode)
            && let Err(error) = self
                .view
                .lock()
                .encode(renderer, ctx, &self.camera, self.options)
        {
            re_log::error_once!("Failed to encode Gaussian splats: {error:#}");
        }
        collector.add_drawable(
            re_renderer::DrawPhase::Transparent,
            DrawDataDrawable::from_world_position(view_info, self.center, 0),
        );
    }
}
fn core_view(
    core: &gsplat_core::Renderer,
    scene: &Arc<gsplat_core::Scene>,
) -> Result<gsplat_core::ViewState, Error> {
    core.create_view(scene, scene.len().clamp(1, MAX_INITIAL_INTERSECTIONS))
}
impl GaussianRenderer {
    pub(crate) fn core(&self) -> Result<&gsplat_core::Renderer, Error> {
        self.core.as_ref().ok_or(Error::Capabilities)
    }

    pub(crate) fn create_view(
        &self,
        ctx: &re_renderer::RenderContext,
        scene: &Arc<gsplat_core::Scene>,
        size: glam::UVec2,
    ) -> Result<RenderView, Error> {
        let core = self.core()?;
        Ok(RenderView {
            core: core_view(core, scene)?,
            target: self.target(ctx, size),
            image: Image::Dirty,
            last_good: None,
        })
    }
    fn target(&self, ctx: &re_renderer::RenderContext, size: glam::UVec2) -> TargetImage {
        // TextureDesc is not exported in 0.38.1; reuse the public 2D descriptor.
        let template = ctx.texture_manager_2d.white_texture_unorm();
        let [view, depth] = [
            ("gsplat composite", wgpu::TextureFormat::Rgba8Unorm),
            ("gsplat expected depth", wgpu::TextureFormat::R32Float),
        ]
        .map(|(label, format)| {
            let mut desc = template.creation_desc.clone();
            desc.label = label.into();
            desc.size = wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };
            desc.format = format;
            desc.usage =
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
            ctx.gpu_resources.textures.alloc(&ctx.device, &desc)
        });
        let group = ctx.gpu_resources.bind_groups.alloc(
            &ctx.device,
            &ctx.gpu_resources,
            &BindGroupDesc {
                label: "gsplat composite".into(),
                layout: self.composite_bind_group_layout,
                entries: smallvec![
                    BindGroupEntry::DefaultTextureView(view.handle),
                    BindGroupEntry::DefaultTextureView(depth.handle)
                ],
            },
        );
        TargetImage {
            size,
            color: view,
            depth,
            composite: group,
        }
    }
}
impl Renderer for GaussianRenderer {
    type RendererDrawData = GaussianDrawData;

    fn create_renderer(ctx: &re_renderer::RenderContext) -> Self {
        re_tracing::profile_function!();
        register_embedded_shaders();

        let composite_shader_module = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &re_renderer::ShaderModuleDesc {
                label: "gaussian_composite".into(),
                source: "shader/gaussian_composite.wgsl".into(),
                extra_workaround_replacements: Vec::new(),
            },
        );
        let composite_bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &re_renderer::BindGroupLayoutDesc {
                label: "GaussianRenderer::composite_bind_group_layout".into(),
                entries: (0..2)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    })
                    .collect(),
            },
        );
        let composite_pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &re_renderer::PipelineLayoutDesc {
                label: "GaussianRenderer::composite".into(),
                entries: vec![ctx.global_bindings.layout, composite_bind_group_layout],
            },
        );

        let depth_state = re_renderer::ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE_NO_WRITE;

        let composite_pipeline_desc = re_renderer::RenderPipelineDesc {
            label: "GaussianRenderer::composite".into(),
            pipeline_layout: composite_pipeline_layout,
            vertex_entrypoint: "main".into(),
            vertex_handle: re_renderer::renderer::screen_triangle_vertex_shader(ctx),
            fragment_entrypoint: "fs_main".into(),
            fragment_handle: composite_shader_module,
            vertex_buffers: smallvec![],
            render_targets: smallvec![Some(wgpu::ColorTargetState {
                format: re_renderer::ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state),
            multisample: re_renderer::ViewBuilder::main_target_default_msaa_state(
                ctx.render_config(),
                false,
            ),
        };
        let composite_pipeline = ctx
            .gpu_resources
            .render_pipelines
            .get_or_create(ctx, &composite_pipeline_desc);

        Self {
            composite_bind_group_layout,
            composite_pipeline,
            core: gsplat_core::Renderer::new(&ctx.device).ok(),
        }
    }

    fn draw(
        &self,
        render_pipelines: &re_renderer::GpuRenderPipelinePoolAccessor<'_>,
        phase: re_renderer::DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_instructions: &[DrawInstruction<'_, GaussianDrawData>],
    ) -> Result<(), DrawError> {
        re_log::debug_assert_eq!(phase, re_renderer::DrawPhase::Transparent);
        let composite_pipeline = render_pipelines.get(self.composite_pipeline)?;
        pass.set_pipeline(composite_pipeline);
        for instruction in draw_instructions {
            let entry = instruction.draw_data.view.lock();
            let Some(image) = entry.completed() else {
                continue;
            };
            pass.set_bind_group(1, &image, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

fn register_embedded_shaders() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        use re_renderer::FileSystem as _;
        re_renderer::get_filesystem()
            .create_file(
                "shader/gaussian_composite.wgsl",
                include_str!("../shader/gaussian_composite.wgsl").into(),
            )
            .expect("register composite");
    });
}
