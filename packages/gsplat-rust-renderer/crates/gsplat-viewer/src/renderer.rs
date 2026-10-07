//! Compute draw data and transparent depth-tested composition for re_renderer.
use gsplat_core::{Camera, RenderOptions};
use re_renderer::external::smallvec::smallvec;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};
use std::sync::{Arc, Mutex, OnceLock};

pub struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    render_pipeline_tile: re_renderer::GpuRenderPipelineHandle,
    pub(crate) core: Option<gsplat_core::Renderer>,
}
struct TargetImage {
    size: glam::UVec2,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    composite: Arc<wgpu::BindGroup>,
}
// Only Pending can carry an in-flight eye; Dirty retains the previous image on overflow.
enum Image {
    // Keep the fallback for this scene generation; moving the eye must not retry
    // an impossible allocation on every other frame. A relog resets this state.
    Capacity {
        required: u64,
        limit: u64,
        previous: Option<Arc<wgpu::BindGroup>>,
    },
    Dirty(Option<Arc<wgpu::BindGroup>>),
    Pending {
        camera: Camera,
        options: RenderOptions,
        previous: Option<Arc<wgpu::BindGroup>>,
    },
    Complete {
        camera: Camera,
        options: RenderOptions,
    },
}
pub(crate) struct RenderView {
    core: gsplat_core::ViewState,
    target: TargetImage,
    image: Image,
}
impl RenderView {
    pub(crate) fn replace_scene(
        &mut self,
        core: &gsplat_core::Renderer,
        scene: &Arc<gsplat_core::Scene>,
        count: usize,
    ) -> Result<(), gsplat_core::Error> {
        let view = core.create_view(scene, (count as u32).clamp(1, 1_048_576))?;
        self.image = Image::Dirty(self.completed());
        self.core = view;
        Ok(())
    }
    fn completed(&self) -> Option<Arc<wgpu::BindGroup>> {
        match &self.image {
            Image::Dirty(previous)
            | Image::Pending { previous, .. }
            | Image::Capacity { previous, .. } => previous.clone(),
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
    ) -> Result<bool, gsplat_core::Error> {
        if let Image::Capacity {
            required, limit, ..
        } = &self.image
        {
            return Err(gsplat_core::Error::Capacity {
                required: *required,
                limit: *limit,
            });
        }
        let feedback = match self.core.poll_feedback() {
            Ok(feedback) => feedback,
            Err(gsplat_core::Error::Capacity { required, limit }) => {
                self.image = Image::Capacity {
                    required,
                    limit,
                    previous: self.completed(),
                };
                return Err(gsplat_core::Error::Capacity { required, limit });
            }
            Err(error) => return Err(error),
        };
        if let Some(stats) = feedback {
            let image = std::mem::replace(&mut self.image, Image::Dirty(None));
            self.image = match image {
                Image::Pending {
                    camera, options, ..
                } if !stats.needs_rerender => Image::Complete { camera, options },
                Image::Pending { previous, .. } => Image::Dirty(previous),
                other => other,
            };
        }
        if self.core.has_pending_frames() {
            return Ok(true);
        }
        if self.is_current(camera, &options) {
            return Ok(false);
        }
        renderer
            .core
            .as_ref()
            .ok_or(gsplat_core::Error::Capabilities)?
            .prepare_view(&ctx.queue, &mut self.core, camera)?;
        if self.target.size != camera.size {
            self.image = Image::Dirty(self.completed());
            self.target = renderer.target(ctx, camera.size);
        }
        Ok(true)
    }
    fn encode(
        &mut self,
        renderer: &GaussianRenderer,
        ctx: &re_renderer::RenderContext,
        camera: &Camera,
        options: RenderOptions,
    ) -> Result<(), gsplat_core::Error> {
        if self.core.has_pending_frames() || self.is_current(camera, &options) {
            return Ok(());
        }
        let previous = self.completed();
        let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
        renderer
            .core
            .as_ref()
            .ok_or(gsplat_core::Error::Capabilities)?
            .render(
                &ctx.queue,
                encoder.get(),
                &mut self.core,
                camera,
                &options,
                gsplat_core::Target::TextureDepth {
                    color: self.target.color.clone(),
                    depth: self.target.depth.clone(),
                },
            )?;
        self.image = Image::Pending {
            camera: *camera,
            options,
            previous,
        };
        Ok(())
    }
}

#[derive(Clone)]
pub struct GaussianDrawData {
    pub(crate) view: Arc<Mutex<RenderView>>,
    pub(crate) camera: Camera,
    pub(crate) options: RenderOptions,
    pub(crate) center: glam::Vec3A,
}
impl DrawData for GaussianDrawData {
    type Renderer = GaussianRenderer;
    fn collect_drawables(
        &self,
        view_info: &re_renderer::renderer::DrawableCollectionViewInfo,
        collector: &mut re_renderer::DrawableCollector<'_>,
    ) {
        let ctx = collector.render_ctx();
        let renderer = ctx
            .renderer::<GaussianRenderer>()
            .expect("registered renderer");
        // Rerun 0.38.1 exposes only the camera position here; the supplied camera is last_eye.
        // Upstream view_info needs the current full pose, projection, and pixel resolution.
        if let Err(error) =
            self.view
                .lock()
                .expect("render view")
                .encode(renderer, ctx, &self.camera, self.options)
        {
            re_log::error_once!("Failed to encode Gaussian splats: {error}");
        }
        collector.add_drawable(
            re_renderer::DrawPhase::Transparent,
            DrawDataDrawable::from_world_position(view_info, self.center, 0),
        );
    }
}
impl GaussianRenderer {
    pub(crate) fn create_view(
        &self,
        ctx: &re_renderer::RenderContext,
        scene: &Arc<gsplat_core::Scene>,
        count: usize,
        size: glam::UVec2,
    ) -> Result<RenderView, gsplat_core::Error> {
        let core = self.core.as_ref().ok_or(gsplat_core::Error::Capabilities)?;
        Ok(RenderView {
            core: core.create_view(scene, (count as u32).clamp(1, 1_048_576))?,
            target: self.target(ctx, size),
            image: Image::Dirty(None),
        })
    }
    fn target(&self, ctx: &re_renderer::RenderContext, size: glam::UVec2) -> TargetImage {
        let [view, depth] = [
            ("gsplat composite", wgpu::TextureFormat::Rgba8Unorm),
            ("gsplat expected depth", wgpu::TextureFormat::R32Float),
        ]
        .map(|(label, format)| {
            ctx.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let layouts = ctx.gpu_resources.bind_group_layouts.resources();
        let layout = layouts
            .get(self.composite_bind_group_layout)
            .expect("composite layout");
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gsplat composite"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth),
                },
            ],
        });
        TargetImage {
            size,
            color: view,
            depth,
            composite: Arc::new(group),
        }
    }
}
impl Renderer for GaussianRenderer {
    type RendererDrawData = GaussianDrawData;

    fn create_renderer(ctx: &re_renderer::RenderContext) -> Self {
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
        let tile_pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &re_renderer::PipelineLayoutDesc {
                label: "GaussianRenderer::tile_draw".into(),
                entries: vec![ctx.global_bindings.layout, composite_bind_group_layout],
            },
        );

        let depth_state = re_renderer::ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE_NO_WRITE;

        let tile_pipeline_desc = re_renderer::RenderPipelineDesc {
            label: "GaussianRenderer::tile_draw".into(),
            pipeline_layout: tile_pipeline_layout,
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
        let render_pipeline_tile = ctx
            .gpu_resources
            .render_pipelines
            .get_or_create(ctx, &tile_pipeline_desc);

        Self {
            composite_bind_group_layout,
            render_pipeline_tile,
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
        if phase != re_renderer::DrawPhase::Transparent {
            return Ok(());
        }
        let tile_pipeline = render_pipelines.get(self.render_pipeline_tile)?;
        for instruction in draw_instructions {
            let entry = instruction.draw_data.view.lock().expect("render view");
            let image = entry
                .completed()
                .unwrap_or_else(|| entry.target.composite.clone());
            pass.set_pipeline(tile_pipeline);
            pass.set_bind_group(1, image.as_ref(), &[]);
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
                include_str!("composite.wgsl").into(),
            )
            .expect("register composite");
    });
}
