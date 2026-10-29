//! Compute draw data: transparent depth-tested composition, picking ids and outline masks for `re_renderer`.
use gsplat_core::{Camera, Error, RenderOptions};
use parking_lot::Mutex;
use re_renderer::external::bytemuck;
use re_renderer::external::smallvec::smallvec;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};
use re_renderer::{BindGroupDesc, BindGroupEntry, GpuBindGroup, GpuTexture};
use re_viewer::external::{parking_lot, re_log, re_renderer, re_viewer_context};
use std::sync::{Arc, OnceLock};

pub(crate) struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    draw_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    composite_pipeline: re_renderer::GpuRenderPipelineHandle,
    picking_pipeline: re_renderer::GpuRenderPipelineHandle,
    outline_pipeline: re_renderer::GpuRenderPipelineHandle,
    pub(crate) core: Option<gsplat_core::Renderer>,
}

struct TargetImage {
    size: glam::UVec2,
    color: GpuTexture,
    depth: GpuTexture,
    id: GpuTexture,
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
            + u64::from(self.target.size.x) * u64::from(self.target.size.y) * 12
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
                id: self.target.id.default_view.clone(),
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
    /// `DrawUniform`: background, picking object and outline masks of this entity.
    pub(crate) uniform: GpuBindGroup,
    /// Whether the entity or any of its instances is outlined.
    pub(crate) outlined: bool,
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
        use re_renderer::DrawPhase;
        let mut phases = DrawPhase::Transparent | DrawPhase::PickingLayer;
        if self.outlined {
            phases.insert(DrawPhase::OutlineMask);
        }
        collector.add_drawable(
            phases,
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

    /// Per-frame draw uniform (see `DrawUniform`).
    pub(crate) fn draw_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        uniform: DrawUniform,
    ) -> GpuBindGroup {
        ctx.gpu_resources.bind_groups.alloc(
            &ctx.device,
            &ctx.gpu_resources,
            &BindGroupDesc {
                label: "gsplat draw".into(),
                layout: self.draw_bind_group_layout,
                entries: smallvec![re_renderer::create_and_fill_uniform_buffer(
                    ctx,
                    "gsplat draw".into(),
                    uniform,
                )],
            },
        )
    }

    fn target(&self, ctx: &re_renderer::RenderContext, size: glam::UVec2) -> TargetImage {
        // TextureDesc is not exported in 0.38.1; reuse the public 2D descriptor.
        let template = ctx.texture_manager_2d.white_texture_unorm();
        let [view, depth, id] = [
            ("gsplat composite", wgpu::TextureFormat::Rgba8Unorm),
            ("gsplat expected depth", wgpu::TextureFormat::R32Float),
            ("gsplat median contributor id", wgpu::TextureFormat::R32Uint),
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
                    BindGroupEntry::DefaultTextureView(depth.handle),
                    BindGroupEntry::DefaultTextureView(id.handle)
                ],
            },
        );
        TargetImage {
            size,
            color: view,
            depth,
            id,
            composite: group,
        }
    }
}

impl Renderer for GaussianRenderer {
    type RendererDrawData = GaussianDrawData;

    fn create_renderer(ctx: &re_renderer::RenderContext) -> Self {
        re_tracing::profile_function!();
        register_embedded_shaders();

        let module = |label: &str, source: &str| {
            ctx.gpu_resources.shader_modules.get_or_create(
                ctx,
                &re_renderer::ShaderModuleDesc {
                    label: label.into(),
                    source: source.into(),
                    extra_workaround_replacements: Vec::new(),
                },
            )
        };
        let composite_shader_module =
            module("gaussian_composite", "shader/gaussian_composite.wgsl");
        let picking_shader_module = module("gaussian_picking", "shader/gaussian_picking.wgsl");
        let composite_bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &re_renderer::BindGroupLayoutDesc {
                label: "GaussianRenderer::composite_bind_group_layout".into(),
                entries: [
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureSampleType::Uint,
                ]
                .into_iter()
                .zip(0..)
                .map(|(sample_type, binding)| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                })
                .collect(),
            },
        );
        let draw_bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &re_renderer::BindGroupLayoutDesc {
                label: "GaussianRenderer::draw_bind_group_layout".into(),
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(256),
                    },
                    count: None,
                }],
            },
        );
        let composite_pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &re_renderer::PipelineLayoutDesc {
                label: "GaussianRenderer::composite".into(),
                entries: vec![
                    ctx.global_bindings.layout,
                    composite_bind_group_layout,
                    draw_bind_group_layout,
                ],
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
        let pipelines = &ctx.gpu_resources.render_pipelines;
        let composite_pipeline = pipelines.get_or_create(ctx, &composite_pipeline_desc);
        // Picking ids and outline masks: a full-view quad that reads the id texture.
        let id_pipeline = |label: &str,
                           entrypoint: &str,
                           format: wgpu::TextureFormat,
                           depth_stencil,
                           multisample| {
            pipelines.get_or_create(
                ctx,
                &re_renderer::RenderPipelineDesc {
                    label: label.into(),
                    vertex_entrypoint: "vs_main".into(),
                    vertex_handle: picking_shader_module,
                    fragment_entrypoint: entrypoint.into(),
                    fragment_handle: picking_shader_module,
                    render_targets: smallvec![Some(format.into())],
                    depth_stencil,
                    multisample,
                    ..composite_pipeline_desc.clone()
                },
            )
        };
        let picking_pipeline = id_pipeline(
            "GaussianRenderer::picking",
            "fs_picking",
            re_renderer::PickingLayerProcessor::PICKING_LAYER_FORMAT,
            re_renderer::PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
            re_renderer::PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
        );
        let outline_pipeline = id_pipeline(
            "GaussianRenderer::outline_mask",
            "fs_outline",
            re_renderer::OutlineMaskProcessor::MASK_FORMAT,
            re_renderer::OutlineMaskProcessor::MASK_DEPTH_STATE,
            re_renderer::OutlineMaskProcessor::mask_default_msaa_state(ctx.device_caps().tier),
        );

        Self {
            composite_bind_group_layout,
            draw_bind_group_layout,
            composite_pipeline,
            picking_pipeline,
            outline_pipeline,
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
        use re_renderer::DrawPhase;
        // The composite draws a screen triangle; ids and masks draw a full-view quad.
        let (pipeline, vertices) = match phase {
            DrawPhase::Transparent => (self.composite_pipeline, 0..3),
            DrawPhase::PickingLayer => (self.picking_pipeline, 0..6),
            DrawPhase::OutlineMask => (self.outline_pipeline, 0..6),
            _ => {
                re_log::debug_panic!("unexpected draw phase {phase:?}");
                return Ok(());
            }
        };
        pass.set_pipeline(render_pipelines.get(pipeline)?);
        for instruction in draw_instructions {
            let entry = instruction.draw_data.view.lock();
            let Some(image) = entry.completed() else {
                continue;
            };
            pass.set_bind_group(1, &image, &[]);
            pass.set_bind_group(2, &instruction.draw_data.uniform, &[]);
            pass.draw(vertices.clone(), 0..1);
        }

        Ok(())
    }
}

/// Highlighted rows one draw can outline individually: the 256-byte uniform less its 32-byte header,
/// at 8 bytes per row. The lowest rows are kept; higher rows draw with the entity-wide mask only,
/// which is empty unless the whole entity is highlighted, so they lose their outline.
const MAX_HIGHLIGHTS: usize = 28;

/// The draw uniform of `shader/gaussian_picking.wgsl`; its prefix is the composite's `Background`.
/// A mask word holds channel A in bits 0-7, B in bits 8-15 and bit 16 when set.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[bytemuck(crate = "re_renderer::external::bytemuck")]
pub(crate) struct DrawUniform {
    /// Solid view background in display space (w = 1); zero for gradients.
    background: [f32; 4],
    /// Picking object id (low, high word).
    picking_object: [u32; 2],
    /// Entity-wide outline mask word.
    outline: u32,
    /// Number of valid `highlights`.
    highlight_count: u32,
    /// (row, mask word) pairs sorted by row; each row mask already falls back to the entity-wide
    /// one per channel.
    highlights: [[u32; 2]; MAX_HIGHLIGHTS],
}

const _: () = assert!(size_of::<DrawUniform>() == 256);

impl DrawUniform {
    /// `rows` is the primitive count: highlighted rows at or beyond it are ignored.
    pub(crate) fn new(
        background: Option<glam::Vec3>,
        object: re_renderer::PickingLayerObjectId,
        masks: &re_viewer_context::ViewOutlineMasks,
        rows: usize,
    ) -> Self {
        let word = |mask: re_renderer::OutlineMaskPreference| {
            mask.0
                .map_or(0, |[a, b]| u32::from(a) | (u32::from(b) << 8) | (1 << 16))
        };
        let mut highlighted: Vec<_> = outline_rows(masks, rows).collect();
        highlighted.sort_unstable_by_key(|&(row, _)| row);
        highlighted.truncate(MAX_HIGHLIGHTS);
        let mut highlights = [[0; 2]; MAX_HIGHLIGHTS];
        for (slot, &(row, mask)) in highlights.iter_mut().zip(&highlighted) {
            *slot = [row, word(mask.with_fallback_to(masks.overall))];
        }
        Self {
            background: background.map_or([0.0; 4], |color| color.extend(1.0).to_array()),
            picking_object: [(object.0 & 0xffff_ffff) as u32, (object.0 >> 32) as u32],
            outline: word(masks.overall),
            highlight_count: highlighted.len() as u32,
            highlights,
        }
    }

    /// Whether the entity or any of its rows is outlined.
    pub(crate) fn outlined(&self) -> bool {
        self.outline != 0 || self.highlight_count != 0
    }
}

/// Highlighted rows below `rows` with their own outline masks, in hash order. Each covers one row,
/// so batches can push them as non-overlapping ranges in any order.
pub(crate) fn outline_rows(
    masks: &re_viewer_context::ViewOutlineMasks,
    rows: usize,
) -> impl Iterator<Item = (u32, re_renderer::OutlineMaskPreference)> + '_ {
    masks.instances.iter().filter_map(move |(instance, &mask)| {
        let row = u32::try_from(instance.get()).ok()?;
        ((row as usize) < rows).then_some((row, mask))
    })
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
        re_renderer::get_filesystem()
            .create_file(
                "shader/gaussian_picking.wgsl",
                include_str!("../shader/gaussian_picking.wgsl").into(),
            )
            .expect("register picking");
    });
}

#[cfg(test)]
mod tests {
    use super::DrawUniform;
    use super::re_renderer::{OutlineMaskPreference, PickingLayerObjectId};
    use re_viewer::external::{re_log_types::Instance, re_viewer_context::ViewOutlineMasks};

    #[test]
    fn draw_uniform_packs_background_picking_object_and_outline_masks() {
        let mut masks = ViewOutlineMasks {
            overall: OutlineMaskPreference::some(1, 0),
            ..Default::default()
        };
        for (instance, a, b) in [
            (7, 0, 2),
            (3, 4, 5),
            (10, 9, 9),
            (u64::from(u32::MAX), 9, 9),
        ] {
            masks
                .instances
                .insert(Instance::from(instance), OutlineMaskPreference::some(a, b));
        }
        let object = PickingLayerObjectId(0x1122_3344_5566_7788);
        let uniform = DrawUniform::new(Some(glam::Vec3::new(0.25, 0.5, 1.0)), object, &masks, 10);
        assert_eq!(uniform.background, [0.25, 0.5, 1.0, 1.0]);
        assert_eq!(uniform.picking_object, [0x5566_7788, 0x1122_3344]);
        // Mask words: bit 16 = set, channel A in bits 0-7, channel B in bits 8-15.
        assert_eq!(uniform.outline, 0x1_00_01);
        // Rows sorted by id; a zero channel falls back to the entity-wide mask; rows at or beyond
        // the primitive count are no rows.
        assert_eq!(uniform.highlight_count, 2);
        assert_eq!(uniform.highlights[..2], [[3, 0x1_05_04], [7, 0x1_02_01]]);
        assert!(uniform.highlights[2..].iter().all(|&pair| pair == [0, 0]));
        assert!(uniform.outlined());

        let plain = DrawUniform::new(None, object, &ViewOutlineMasks::default(), 10);
        assert_eq!(plain.background, [0.0; 4]);
        assert_eq!([plain.outline, plain.highlight_count], [0, 0]);
        assert!(!plain.outlined());
    }

    #[test]
    fn draw_uniform_keeps_the_lowest_rows_that_fit() {
        let mut masks = ViewOutlineMasks::default();
        for instance in (0..40).rev() {
            masks
                .instances
                .insert(Instance::from(instance), OutlineMaskPreference::some(0, 1));
        }
        let uniform = DrawUniform::new(None, PickingLayerObjectId(1), &masks, 40);
        assert_eq!(uniform.outline, 0);
        assert_eq!(uniform.highlight_count, 28);
        let rows: Vec<u32> = uniform.highlights.iter().map(|pair| pair[0]).collect();
        assert_eq!(rows, (0..28).collect::<Vec<u32>>());
        assert!(uniform.highlights.iter().all(|pair| pair[1] == 0x1_01_00));
        assert!(uniform.outlined());
    }
}
