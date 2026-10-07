//! gsplat-core compute output composited into Rerun's transparent pass.
use gsplat_core::{Camera, RenderOptions, native::NativeSplats};
use re_renderer::external::smallvec::smallvec;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    render_pipeline_tile: re_renderer::GpuRenderPipelineHandle,
    core: gsplat_core::Renderer,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct BatchKey {
    pub view_id: re_viewer_context::ViewId,
    pub entity: re_log_types::EntityPath,
    pub instruction: re_sdk_types::blueprint::components::VisualizerInstructionId,
    pub instance: usize,
    pub row: (re_log_types::TimeInt, re_sdk_types::RowId),
}
pub(crate) struct Batch<'a> {
    pub key: BatchKey,
    pub cloud: &'a NativeSplats<'a>,
    pub generation: u64,
    pub camera: &'a Camera,
    pub options: RenderOptions,
}
struct TargetImage {
    size: glam::UVec2,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    composite: Arc<wgpu::BindGroup>,
}
// Only Pending can carry an in-flight eye; Dirty retains the previous image on overflow.
enum Image {
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
struct CachedEntity {
    last_frame: u64,
    generation: u64,
    core: gsplat_core::ViewState,
    target: TargetImage,
    image: Image,
}
impl CachedEntity {
    fn completed(&self) -> Option<Arc<wgpu::BindGroup>> {
        match &self.image {
            Image::Dirty(previous) | Image::Pending { previous, .. } => previous.clone(),
            Image::Complete { .. } => Some(self.target.composite.clone()),
        }
    }
    fn render(
        &mut self,
        renderer: &GaussianRenderer,
        ctx: &re_renderer::RenderContext,
        camera: &Camera,
        options: RenderOptions,
    ) -> Result<bool, gsplat_core::Error> {
        if let Some(stats) = self.core.poll_feedback()? {
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
        if matches!(&self.image, Image::Complete { camera: old_camera, options: old_options } if old_camera == camera && *old_options == options)
        {
            return Ok(false);
        }
        let previous = self.completed();
        if self.target.size != camera.size {
            self.target = renderer.target(ctx, camera.size);
        }
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        renderer.core.render(
            &mut encoder,
            &mut self.core,
            camera,
            &options,
            gsplat_core::Target::TextureDepth {
                color: self.target.color.clone(),
                depth: self.target.depth.clone(),
            },
        )?;
        ctx.queue.submit([encoder.finish()]);
        self.image = Image::Pending {
            camera: *camera,
            options,
            previous,
        };
        Ok(true)
    }
}
struct CachedScene {
    last_frame: u64,
    scene: Arc<gsplat_core::Scene>,
    count: usize,
    bounds: [glam::Vec3; 2],
}
impl CachedScene {
    fn upload(
        core: &gsplat_core::Renderer,
        cloud: &NativeSplats<'_>,
        frame: u64,
    ) -> Result<Self, gsplat_core::Error> {
        let mut bounds = [
            glam::Vec3::splat(f32::INFINITY),
            glam::Vec3::splat(f32::NEG_INFINITY),
        ];
        for (i, center) in cloud.centers.iter().enumerate() {
            let center = glam::Vec3::from_array(*center);
            let radius = glam::Vec3::from_array(
                cloud
                    .scales
                    .get(i)
                    .or_else(|| cloud.scales.last())
                    .copied()
                    .unwrap_or([0.01; 3]),
            )
            .abs()
            .max_element()
                * 3.0;
            bounds[0] = bounds[0].min(center - radius);
            bounds[1] = bounds[1].max(center + radius);
        }
        let scene = core.upload(&cloud.to_core())?;
        re_log::debug!("Uploaded {} Gaussian splats", cloud.centers.len());
        Ok(Self {
            last_frame: frame,
            scene,
            count: cloud.centers.len(),
            bounds,
        })
    }
}
#[derive(Default)]
struct Batches {
    frame: u64,
    scenes: HashMap<u64, CachedScene>,
    views: HashMap<BatchKey, CachedEntity>,
    bounds: HashMap<re_viewer_context::ViewId, ([glam::Vec3; 2], u64)>,
}
/// Store-owned cache: begin_frame also runs when the entity/view disappears.
#[derive(Default)]
pub(crate) struct GpuCache(Mutex<Batches>);
impl GpuCache {
    pub(crate) fn view_bounds(&self, id: re_viewer_context::ViewId) -> Option<[glam::Vec3; 2]> {
        self.0
            .lock()
            .expect("GPU cache")
            .bounds
            .get(&id)
            .map(|(bounds, _)| *bounds)
    }
}
impl re_viewer_context::Cache for GpuCache {
    fn name(&self) -> &'static str {
        "ComputeGaussianSplats3D GPU"
    }
    fn begin_frame(&mut self) {
        let cache = self.0.get_mut().expect("GPU cache");
        cache
            .views
            .retain(|_, entry| entry.last_frame == cache.frame);
        // Blueprint activation can omit all compute instructions for one frame.
        // Keep shared uploads across that transition, then evict genuinely unused data.
        cache
            .scenes
            .retain(|_, entry| entry.last_frame + 1 >= cache.frame);
        cache.bounds.retain(|_, (_, frame)| *frame == cache.frame);
        cache.frame += 1;
    }
    fn purge_memory(&mut self) {
        *self.0.get_mut().expect("GPU cache") = Batches::default();
    }
}
impl re_byte_size::MemUsageTreeCapture for GpuCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        let cache = self.0.lock().expect("GPU cache");
        // CPU handles only; GPU allocations are accounted by wgpu.
        re_byte_size::MemUsageTree::Bytes(
            (cache.scenes.len() * size_of::<CachedScene>()
                + cache.views.len() * size_of::<CachedEntity>()) as u64,
        )
    }
}
#[derive(Clone, Default)]
pub struct GaussianDrawData {
    batches: Vec<(Arc<wgpu::BindGroup>, glam::Vec3A)>,
}
impl DrawData for GaussianDrawData {
    type Renderer = GaussianRenderer;
    fn collect_drawables(
        &self,
        view_info: &re_renderer::renderer::DrawableCollectionViewInfo,
        collector: &mut re_renderer::DrawableCollector<'_>,
    ) {
        for index in 0..self.batches.len() {
            collector.add_drawable(
                re_renderer::DrawPhase::Transparent,
                DrawDataDrawable::from_world_position(
                    view_info,
                    self.batches[index].1,
                    index as u32,
                ),
            );
        }
    }
}
impl GaussianDrawData {
    /// Each view/instance owns its target so later views cannot overwrite earlier composites.
    pub(crate) fn add_batch(
        &mut self,
        ctx: &re_renderer::RenderContext,
        cache: &mut GpuCache,
        batch: Batch<'_>,
    ) -> Result<bool, gsplat_core::Error> {
        let Batch {
            key,
            cloud,
            generation,
            camera,
            options,
        } = batch;
        let view_id = key.view_id;
        let renderer = ctx
            .renderer::<GaussianRenderer>()
            .expect("renderer registered at startup");
        let cache = cache.0.get_mut().expect("GPU cache");
        let frame = cache.frame;
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.scenes.entry(generation) {
            entry.insert(CachedScene::upload(&renderer.core, cloud, frame)?);
        }
        let shared = cache.scenes.get_mut(&generation).expect("uploaded scene");
        shared.last_frame = frame;
        let bounds = crate::bounds::transformed(shared.bounds, options.world_from_local);
        cache
            .bounds
            .entry(view_id)
            .and_modify(|(previous, seen)| {
                *previous = if *seen == frame {
                    [previous[0].min(bounds[0]), previous[1].max(bounds[1])]
                } else {
                    bounds
                };
                *seen = frame;
            })
            .or_insert((bounds, frame));
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.views.entry(key.clone()) {
            entry.insert(CachedEntity {
                last_frame: frame,
                generation,
                core: renderer
                    .core
                    .create_view(&shared.scene, (shared.count as u32).clamp(1, 1_048_576))?,
                target: renderer.target(ctx, camera.size),
                image: Image::Dirty(None),
            });
        }
        let entry = cache.views.get_mut(&key).expect("inserted view");
        entry.last_frame = frame;
        if entry.generation != generation {
            entry.core = renderer
                .core
                .create_view(&shared.scene, (shared.count as u32).clamp(1, 1_048_576))?;
            entry.generation = generation;
            entry.image = Image::Dirty(entry.completed());
        }
        let retry = entry.render(renderer, ctx, camera, options)?;
        self.batches.push((
            entry
                .completed()
                .unwrap_or_else(|| entry.target.composite.clone()),
            ((bounds[0] + bounds[1]) * 0.5).into(),
        ));
        Ok(retry)
    }
}
impl GaussianRenderer {
    fn target(&self, ctx: &re_renderer::RenderContext, size: glam::UVec2) -> TargetImage {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gsplat composite"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let depth = ctx
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("gsplat expected depth"),
                size: texture.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
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
            core: gsplat_core::Renderer::new(&ctx.device, &ctx.queue)
                .expect("validated gsplat device"),
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
            for drawable in instruction.drawables {
                let batch_index = drawable.draw_data_payload as usize;
                let Some(batch) = instruction.draw_data.batches.get(batch_index) else {
                    continue;
                };

                pass.set_pipeline(tile_pipeline);
                pass.set_bind_group(1, batch.0.as_ref(), &[]);
                pass.draw(0..3, 0..1);
            }
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
