//! Compute draw data and transparent depth-tested composition for re_renderer.
use gsplat_core::{Camera, Error, RenderOptions};
use re_renderer::external::smallvec::smallvec;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};
use re_renderer::{BindGroupDesc, BindGroupEntry, GpuBindGroup, GpuTexture};
use std::sync::{Arc, Mutex, OnceLock};

pub struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    render_pipeline_tile: re_renderer::GpuRenderPipelineHandle,
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
        previous: Option<GpuBindGroup>,
    },
    Dirty(Option<GpuBindGroup>),
    Pending {
        camera: Camera,
        options: RenderOptions,
        previous: Option<GpuBindGroup>,
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
    ) -> Result<(), Error> {
        let view = core_view(core, scene, count)?;
        self.image = Image::Dirty(self.completed());
        self.core = view;
        Ok(())
    }
    fn completed(&self) -> Option<GpuBindGroup> {
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
    ) -> Result<bool, Error> {
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
                self.image = Image::Capacity {
                    required,
                    limit,
                    previous: self.completed(),
                };
                return Err(Error::Capacity { required, limit });
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
            .core()?
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
    ) -> Result<(), Error> {
        if self.core.has_pending_frames() || self.is_current(camera, &options) {
            return Ok(());
        }
        let previous = self.completed();
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
fn core_view(
    core: &gsplat_core::Renderer,
    scene: &Arc<gsplat_core::Scene>,
    count: usize,
) -> Result<gsplat_core::ViewState, Error> {
    core.create_view(scene, (count as u32).clamp(1, 1_048_576))
}
impl GaussianRenderer {
    pub(crate) fn core(&self) -> Result<&gsplat_core::Renderer, Error> {
        self.core.as_ref().ok_or(Error::Capabilities)
    }

    pub(crate) fn create_view(
        &self,
        ctx: &re_renderer::RenderContext,
        scene: &Arc<gsplat_core::Scene>,
        count: usize,
        size: glam::UVec2,
    ) -> Result<RenderView, Error> {
        let core = self.core()?;
        Ok(RenderView {
            core: core_view(core, scene, count)?,
            target: self.target(ctx, size),
            image: Image::Dirty(None),
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
            core: gsplat_core::Renderer::new(&ctx.device)
                .inspect_err(|error| re_log::warn_once!("Gaussian compute unavailable: {error}"))
                .ok(),
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
            let Some(image) = entry.completed() else {
                continue;
            };
            pass.set_pipeline(tile_pipeline);
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
                include_str!("composite.wgsl").into(),
            )
            .expect("register composite");
    });
}
