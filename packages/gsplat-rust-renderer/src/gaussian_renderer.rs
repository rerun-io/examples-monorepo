//! GPU renderer for Gaussian splats — the Rerun viewer layer.
//!
//! # Overview
//!
//! This module integrates the GPU compute pipeline with Rerun's rendering
//! system.  The visualizer ([`crate::gaussian_visualizer`]) hands it a
//! pre-sorted batch of visible splats each frame; the renderer uploads them
//! and dispatches the 7-stage compute pipeline.
//!
//! GPU types, bind group layouts, compute pipelines, and helper functions
//! are imported from [`crate::gsplat_core::gpu_types`] — the single source
//! of truth shared with the standalone renderer.
//!
//! # Compute Pipeline (GPU-only, Brush-aligned)
//!
//! | Stage | Shader | Description |
//! |-------|--------|-------------|
//! | 1. Project | `gaussian_project.wgsl` | 3D→2D Gaussian projection + SH evaluation |
//! | 2. Compact | `gaussian_project.wgsl` (scan) | Prefix sum to remove invisible splats |
//! | 3. Map | `gaussian_map_intersections.wgsl` | Scatter (splat, tile) pairs |
//! | 4. Sort | `gaussian_dynamic_sort.wgsl` | Radix sort by tile ID |
//! | 5. Offsets | `gaussian_tile_offsets.wgsl` | Per-tile start/end range |
//! | 6. Raster | `gaussian_raster_tiles.wgsl` | Per-pixel alpha blending |
//! | 7. Composite | `gaussian_composite.wgsl` | Blit to Rerun viewport |
//!
//! # Buffer Management
//!
//! GPU buffers are cached per-entity and grow as needed (never shrink).  This
//! avoids re-creating buffers every frame for static scenes.  The intersection
//! count is read back from the GPU (with a 2-frame delay) to right-size the
//! tile intersection buffers for the next frame.
//!
//! # Per-frame flow
//!
//! 1. Reuse or grow persistent GPU buffers for the entity
//! 2. Upload splat data (means, quats, scales, opacities, colors, SH) to GPU
//! 3. Dispatch 7-stage compute pipeline
//! 4. Composite raster texture into Rerun's viewport via fullscreen blit

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use re_renderer::external::smallvec::smallvec;
use re_renderer::external::wgpu;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};

use self::gpu_types as gpu_data;
use crate::gsplat_core::constants::{
    MAX_SPLATS_RENDERED, MIN_RADIUS_PX, OPACITY_SCALE, SIGMA_COVERAGE,
};
use crate::gsplat_core::gpu_types::{
    PROJECT_WORKGROUP_SIZE, RASTER_TEXTURE_FORMAT, SORT_BIN_COUNT, SORT_BLOCK_SIZE,
    SORT_WORKGROUP_SIZE, TILE_OFFSET_CHECKS_PER_ITER, TILE_OFFSET_WORKGROUP_SIZE,
    calc_raster_extent, calc_tile_bounds, compaction_block_count, create_compute_pipeline,
    create_filled_buffer, create_sized_buffer, dispatch_grid_1d, dispatch_grid_for_workgroups,
    intersection_capacity_for_instances, next_block_capacity, next_capacity, pack_quats, pack_rgb,
    pack_scales_opacity, pack_sh_coefficients, pack_vec3s, storage_buffer_entry,
    storage_layout_entry, tile_count, uniform_layout_entry,
};
use crate::gsplat_core::{CameraApproximation, RenderGaussianCloud, SortedSplatIndex};

const INTERSECTION_READBACK_SLOT_COUNT: usize = 2;

#[cfg(test)]
mod tests {
    #[test]
    fn dispatch_grid_tiles_large_1d_workloads() {
        assert_eq!(
            super::dispatch_grid_1d(1, super::PROJECT_WORKGROUP_SIZE),
            (1, 1)
        );
        assert_eq!(
            super::dispatch_grid_1d(
                65_535 * super::PROJECT_WORKGROUP_SIZE,
                super::PROJECT_WORKGROUP_SIZE
            ),
            (65_535, 1)
        );

        let (x, y) = super::dispatch_grid_1d(
            65_536 * super::PROJECT_WORKGROUP_SIZE,
            super::PROJECT_WORKGROUP_SIZE,
        );
        assert!(x <= 65_535);
        assert!(y > 1);
        assert!(x * y >= 65_536);
    }

    #[test]
    fn compaction_limit_covers_default_render_cap() {
        let limit = std::hint::black_box(super::intersection_capacity_for_instances(
            super::MAX_SPLATS_RENDERED,
        ));
        assert!(limit >= super::MAX_SPLATS_RENDERED);
    }

    #[test]
    fn compaction_block_count_rounds_up() {
        use crate::gsplat_core::gpu_types::COMPACTION_BLOCK_SIZE;
        assert_eq!(super::compaction_block_count(1), 1);
        assert_eq!(
            super::compaction_block_count(COMPACTION_BLOCK_SIZE as usize),
            1
        );
        assert_eq!(
            super::compaction_block_count(COMPACTION_BLOCK_SIZE as usize + 1),
            2
        );
    }

    #[test]
    fn descending_depth_sort_key_orders_farther_first() {
        let far = super::encode_descending_depth_key(10.0);
        let near = super::encode_descending_depth_key(2.0);
        assert!(far < near);
    }

    #[test]
    fn intersection_capacity_scales_with_instance_capacity() {
        assert_eq!(super::intersection_capacity_for_instances(0), 32);
        assert_eq!(super::intersection_capacity_for_instances(1), 32);
        assert_eq!(super::intersection_capacity_for_instances(32), 1_024);
        assert_eq!(super::intersection_capacity_for_instances(513), 32_768);
    }
}

// Some fields are retained as lifetime anchors for GPU resources even when not
// directly referenced in every code path.
#[allow(dead_code)]
pub struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    render_pipeline_tile: re_renderer::GpuRenderPipelineHandle,
    compute_pipelines: ComputePipelines,
    batch_cache: Mutex<HashMap<String, CachedBatchResources>>,
}

// These handles are retained so the GPU resources and bind group layouts stay alive for the
// lifetime of cached compute batches, even when not every field is read after construction.
#[allow(dead_code)]
struct ComputePipelines {
    project_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    project_pipeline: wgpu::ComputePipeline,
    scan_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scan_block_sums_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    scan_block_sums_pipeline: wgpu::ComputePipeline,
    map_intersections_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    map_intersections_pipeline: wgpu::ComputePipeline,
    clamp_intersection_count_pipeline: wgpu::ComputePipeline,
    dynamic_sort_count_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    dynamic_sort_count_pipeline: wgpu::ComputePipeline,
    dynamic_sort_reduce_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    dynamic_sort_reduce_pipeline: wgpu::ComputePipeline,
    dynamic_sort_scan_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    dynamic_sort_scan_pipeline: wgpu::ComputePipeline,
    dynamic_sort_scan_compose_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    dynamic_sort_scan_compose_pipeline: wgpu::ComputePipeline,
    dynamic_sort_scan_add_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    dynamic_sort_scan_add_pipeline: wgpu::ComputePipeline,
    dynamic_sort_scatter_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    dynamic_sort_scatter_pipeline: wgpu::ComputePipeline,
    tile_offsets_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    tile_offsets_pipeline: wgpu::ComputePipeline,
    rasterize_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    rasterize_pipeline: wgpu::ComputePipeline,
}

#[derive(Clone)]
pub struct GaussianDrawData {
    // One logical batch per logged splat entity.
    batches: Vec<GaussianBatch>,
}

/// Extra counters returned while preparing one renderer batch.
pub struct BatchSubmission {
    pub extra_draw_data: Option<re_renderer::QueueableDrawData>,
}

struct PreparedBatch {
    // Internal return value from `prepare_*_batch`: the final batch plus a few counters that are
    // useful while keeping the public surface tiny.
    batch: Option<GaussianBatch>,
    upload_ms: f32,
    extra_draw_data: Option<re_renderer::QueueableDrawData>,
    tile_intersections: usize,
    intersection_capacity: usize,
    intersection_capacity_saturated: bool,
}

#[derive(Clone)]
struct GaussianBatch {
    payload: GaussianBatchPayload,
}

#[derive(Clone)]
struct GaussianBatchPayload {
    /// The compute tile path rasterizes into an intermediate target, then blits it back.
    composite_bind_group: Arc<wgpu::BindGroup>,
}

// Some fields are retained as lifetime anchors for GPU resources.
#[allow(dead_code)]
struct CachedBatchResources {
    instance_buffer: Arc<wgpu::Buffer>,
    instance_capacity: usize,
    compute: CachedComputeResources,
}

// Like `ComputePipelines`, many of these resources are lifetime anchors for cached bind groups and
// intermediate buffers. Keeping them together makes the active Brush-style compute path easier to
// audit, even though not every field is read on every frame.
#[allow(dead_code)]
struct CachedComputeResources {
    project_uniform_buffer: Arc<wgpu::Buffer>,
    scan_uniform_buffer: Arc<wgpu::Buffer>,
    sort_uniform_buffer: Arc<wgpu::Buffer>,
    map_uniform_buffer: Arc<wgpu::Buffer>,
    project_bind_group: Arc<wgpu::BindGroup>,
    tile_count_scan_bind_group: Arc<wgpu::BindGroup>,
    tile_count_scan_block_sums_bind_group: Arc<wgpu::BindGroup>,
    map_intersections_bind_group: Arc<wgpu::BindGroup>,
    dynamic_sort_count_bind_group_primary: Arc<wgpu::BindGroup>,
    dynamic_sort_count_bind_group_alt: Arc<wgpu::BindGroup>,
    dynamic_sort_reduce_bind_group: Arc<wgpu::BindGroup>,
    dynamic_sort_scan_bind_group: Arc<wgpu::BindGroup>,
    dynamic_sort_scan_add_bind_group: Arc<wgpu::BindGroup>,
    dynamic_sort_scatter_bind_group_primary: Arc<wgpu::BindGroup>,
    dynamic_sort_scatter_bind_group_alt: Arc<wgpu::BindGroup>,
    tile_offsets_bind_group: Arc<wgpu::BindGroup>,
    rasterize_bind_group: Arc<wgpu::BindGroup>,
    composite_bind_group: Arc<wgpu::BindGroup>,
    means_buffer: Arc<wgpu::Buffer>,
    quats_buffer: Arc<wgpu::Buffer>,
    scales_opacity_buffer: Arc<wgpu::Buffer>,
    colors_buffer: Arc<wgpu::Buffer>,
    sh_coeffs_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
    visibility_flags_buffer: Arc<wgpu::Buffer>,
    projected_tile_splats_buffer: Arc<wgpu::Buffer>,
    tile_hit_counts_buffer: Arc<wgpu::Buffer>,
    tile_hit_offsets_buffer: Arc<wgpu::Buffer>,
    tile_hit_block_offsets_buffer: Arc<wgpu::Buffer>,
    tile_intersection_count_buffer: Arc<wgpu::Buffer>,
    intersection_count_readback_slots: Vec<IntersectionCountReadbackSlot>,
    num_intersections_buffer: Arc<wgpu::Buffer>,
    tile_id_from_isect_buffer: Arc<wgpu::Buffer>,
    compact_gid_from_isect_buffer: Arc<wgpu::Buffer>,
    sort_keys_buffer: Arc<wgpu::Buffer>,
    sorted_indices_alt_buffer: Arc<wgpu::Buffer>,
    sort_counts_buffer: Arc<wgpu::Buffer>,
    sort_reduced_buffer: Arc<wgpu::Buffer>,
    sort_scan_offsets_buffer: Arc<wgpu::Buffer>,
    sort_scan_block_offsets_buffer: Arc<wgpu::Buffer>,
    sort_scan_totals_buffer: Arc<wgpu::Buffer>,
    tile_offsets_buffer: Arc<wgpu::Buffer>,
    raster_uniform_buffer: Arc<wgpu::Buffer>,
    raster_texture: Arc<wgpu::Texture>,
    raster_texture_view: Arc<wgpu::TextureView>,
    raster_extent: glam::UVec2,
    index_capacity: usize,
    block_capacity: usize,
    intersection_capacity: usize,
    tile_offset_capacity: usize,
    latest_intersection_count: usize,
    intersection_capacity_saturated: bool,
    sort_workgroup_count: u32,
    sort_reduce_workgroup_count: u32,
    sort_scan_block_capacity: usize,
    cpu_indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntersectionCountReadbackState {
    Idle,
    CopySubmitted,
    Mapping,
}

struct IntersectionCountReadbackSlot {
    buffer: Arc<wgpu::Buffer>,
    result: Arc<Mutex<Option<bool>>>,
    state: IntersectionCountReadbackState,
}

impl DrawData for GaussianDrawData {
    type Renderer = GaussianRenderer;

    fn collect_drawables(
        &self,
        _view_info: &re_renderer::renderer::DrawableCollectionViewInfo,
        collector: &mut re_renderer::DrawableCollector<'_>,
    ) {
        for (index, _batch) in self.batches.iter().enumerate() {
            collector.add_drawable(
                re_renderer::DrawPhase::Transparent,
                DrawDataDrawable {
                    distance_sort_key: 0.0,
                    draw_data_payload: index as u32,
                },
            );
        }
    }
}

impl GaussianDrawData {
    pub fn new(ctx: &re_renderer::RenderContext) -> Self {
        let _ = ctx.renderer::<GaussianRenderer>();
        Self {
            batches: Vec::new(),
        }
    }

    pub fn add_batch(
        &mut self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        cloud: &Arc<RenderGaussianCloud>,
        camera: &CameraApproximation,
        visible_indices: &[SortedSplatIndex],
        farthest_depth: f32,
    ) -> BatchSubmission {
        let renderer = ctx.renderer::<GaussianRenderer>();
        if visible_indices.is_empty() {
            return BatchSubmission {
                extra_draw_data: None,
            };
        }

        let submission = renderer.prepare_compute_batch(
            ctx,
            label,
            cloud,
            camera,
            visible_indices,
            farthest_depth,
        );
        let PreparedBatch {
            batch,
            upload_ms,
            extra_draw_data,
            tile_intersections,
            intersection_capacity,
            intersection_capacity_saturated,
        } = submission;
        if let Some(batch) = batch {
            self.batches.push(batch);
        }
        let _ = (
            upload_ms,
            tile_intersections,
            intersection_capacity,
            intersection_capacity_saturated,
        );
        BatchSubmission { extra_draw_data }
    }
}

impl GaussianRenderer {
    fn create_batch_resources(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        cloud: &Arc<RenderGaussianCloud>,
        initial_capacity: usize,
    ) -> CachedBatchResources {
        // These buffers live per entity so camera movement reuses GPU allocations instead of
        // rebuilding everything every frame.
        let instance_capacity = next_capacity(initial_capacity);
        let instance_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::instances"),
            instance_capacity * std::mem::size_of::<gpu_data::InstanceData>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));

        let compute =
            self.create_compute_resources(ctx, label, cloud, &instance_buffer, instance_capacity);

        CachedBatchResources {
            instance_buffer,
            instance_capacity,
            compute,
        }
    }

    fn create_compute_resources(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        cloud: &Arc<RenderGaussianCloud>,
        output_instance_buffer: &Arc<wgpu::Buffer>,
        initial_capacity: usize,
    ) -> CachedComputeResources {
        // The compute path keeps most per-cloud data resident on the GPU:
        // canonical Gaussian attributes, temporary compaction/sort buffers, and the tile raster
        // target used by the final composite pass.
        let project_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::project_uniform")),
            size: std::mem::size_of::<gpu_data::ProjectUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let scan_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::scan_uniform")),
            size: std::mem::size_of::<gpu_data::ScanUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let sort_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::sort_uniform")),
            size: std::mem::size_of::<gpu_data::SortUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let map_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::map_uniform")),
            size: std::mem::size_of::<gpu_data::MapUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let means_buffer = Arc::new(create_filled_buffer(
            &ctx.device,
            &format!("{label}::means"),
            wgpu::BufferUsages::STORAGE,
            &pack_vec3s(cloud.means_world.iter().copied()),
        ));
        let quats_buffer = Arc::new(create_filled_buffer(
            &ctx.device,
            &format!("{label}::quats"),
            wgpu::BufferUsages::STORAGE,
            &pack_quats(cloud.quats.iter().copied()),
        ));
        let scales_opacity_buffer = Arc::new(create_filled_buffer(
            &ctx.device,
            &format!("{label}::scales_opacity"),
            wgpu::BufferUsages::STORAGE,
            &pack_scales_opacity(cloud),
        ));
        let colors_buffer = Arc::new(create_filled_buffer(
            &ctx.device,
            &format!("{label}::colors"),
            wgpu::BufferUsages::STORAGE,
            &pack_rgb(cloud.colors_dc.iter().copied()),
        ));
        let sh_coeffs_buffer = Arc::new(create_filled_buffer(
            &ctx.device,
            &format!("{label}::sh_coeffs"),
            wgpu::BufferUsages::STORAGE,
            &pack_sh_coefficients(cloud),
        ));

        let index_capacity = next_capacity(cloud.len().max(1));
        let index_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::indices"),
            index_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let visibility_flags_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::visibility_flags"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let projected_tile_splats_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::projected_tile_splats"),
            initial_capacity * std::mem::size_of::<gpu_data::TileProjectedSplat>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let tile_hit_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_counts"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let tile_hit_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_offsets"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let block_capacity = next_block_capacity(initial_capacity);
        let tile_hit_block_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_block_offsets"),
            block_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let tile_intersection_count_buffer = Arc::new(create_filled_buffer(
            &ctx.device,
            &format!("{label}::tile_intersection_count"),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            &[gpu_data::DrawIndirectArgs {
                vertex_count: 6,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }],
        ));
        let num_intersections_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::num_intersections"),
            std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let intersection_capacity = intersection_capacity_for_instances(initial_capacity);
        let tile_id_from_isect_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_id_from_isect"),
            intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        let compact_gid_from_isect_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::compact_gid_from_isect"),
            intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        let sort_keys_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_keys"),
            intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        let sorted_indices_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sorted_indices_alt"),
            intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        let sort_workgroup_count = intersection_capacity
            .div_ceil(SORT_BLOCK_SIZE as usize)
            .max(1);
        let sort_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_counts"),
            sort_workgroup_count * SORT_BIN_COUNT as usize * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_reduce_workgroup_count = (sort_workgroup_count.div_ceil(SORT_BLOCK_SIZE as usize)
            * SORT_BIN_COUNT as usize)
            .max(1);
        let sort_reduced_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_reduced"),
            sort_reduce_workgroup_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_scan_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_offsets"),
            sort_reduce_workgroup_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_scan_block_capacity = next_block_capacity(sort_reduce_workgroup_count);
        let sort_scan_block_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_block_offsets"),
            sort_scan_block_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_scan_totals_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_totals"),
            std::mem::size_of::<gpu_data::DrawIndirectArgs>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let tile_offset_capacity = 1;
        let tile_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_offsets"),
            tile_offset_capacity * 2 * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let raster_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::raster_uniform")),
            size: std::mem::size_of::<gpu_data::RasterUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let raster_extent = glam::uvec2(1, 1);
        let (raster_texture, raster_texture_view) = create_viewer_raster_texture(
            &ctx.device,
            &format!("{label}::raster_color"),
            raster_extent,
        );

        let project_bind_group = Arc::new(self.create_project_bind_group(
            ctx,
            label,
            &means_buffer,
            &quats_buffer,
            &scales_opacity_buffer,
            &colors_buffer,
            &sh_coeffs_buffer,
            &index_buffer,
            output_instance_buffer,
            &visibility_flags_buffer,
            &project_uniform_buffer,
            &projected_tile_splats_buffer,
            &tile_hit_counts_buffer,
            "project_bind_group",
        ));
        let tile_count_scan_bind_group = Arc::new(self.create_scan_bind_group(
            ctx,
            label,
            &tile_hit_counts_buffer,
            &tile_hit_offsets_buffer,
            &tile_hit_block_offsets_buffer,
            &scan_uniform_buffer,
        ));
        let tile_count_scan_block_sums_bind_group =
            Arc::new(self.create_scan_block_sums_bind_group(
                ctx,
                label,
                &tile_hit_block_offsets_buffer,
                &tile_intersection_count_buffer,
                &scan_uniform_buffer,
            ));
        let map_intersections_bind_group = Arc::new(self.create_map_intersections_bind_group(
            ctx,
            label,
            &projected_tile_splats_buffer,
            &tile_hit_offsets_buffer,
            &tile_hit_counts_buffer,
            &tile_hit_block_offsets_buffer,
            &tile_id_from_isect_buffer,
            &compact_gid_from_isect_buffer,
            &tile_intersection_count_buffer,
            &num_intersections_buffer,
            &map_uniform_buffer,
        ));
        let dynamic_sort_count_bind_group_primary =
            Arc::new(self.create_dynamic_sort_count_bind_group(
                ctx,
                label,
                &tile_id_from_isect_buffer,
                &sort_counts_buffer,
                &sort_uniform_buffer,
                &num_intersections_buffer,
                "dynamic_sort_count_bind_group_primary",
            ));
        let dynamic_sort_count_bind_group_alt =
            Arc::new(self.create_dynamic_sort_count_bind_group(
                ctx,
                label,
                &sort_keys_buffer,
                &sort_counts_buffer,
                &sort_uniform_buffer,
                &num_intersections_buffer,
                "dynamic_sort_count_bind_group_alt",
            ));
        let dynamic_sort_reduce_bind_group = Arc::new(self.create_dynamic_sort_reduce_bind_group(
            ctx,
            label,
            &sort_counts_buffer,
            &sort_reduced_buffer,
            &sort_uniform_buffer,
            &num_intersections_buffer,
        ));
        let dynamic_sort_scan_bind_group = Arc::new(self.create_dynamic_sort_scan_bind_group(
            ctx,
            label,
            &sort_reduced_buffer,
            &sort_uniform_buffer,
            &num_intersections_buffer,
        ));
        let dynamic_sort_scan_add_bind_group =
            Arc::new(self.create_dynamic_sort_scan_add_bind_group(
                ctx,
                label,
                &sort_reduced_buffer,
                &sort_counts_buffer,
                &sort_uniform_buffer,
                &num_intersections_buffer,
            ));
        let dynamic_sort_scatter_bind_group_primary =
            Arc::new(self.create_dynamic_sort_scatter_bind_group(
                ctx,
                label,
                &tile_id_from_isect_buffer,
                &compact_gid_from_isect_buffer,
                &sort_counts_buffer,
                &sort_keys_buffer,
                &sorted_indices_alt_buffer,
                &sort_uniform_buffer,
                &num_intersections_buffer,
                "dynamic_sort_scatter_bind_group_primary",
            ));
        let dynamic_sort_scatter_bind_group_alt =
            Arc::new(self.create_dynamic_sort_scatter_bind_group(
                ctx,
                label,
                &sort_keys_buffer,
                &sorted_indices_alt_buffer,
                &sort_counts_buffer,
                &tile_id_from_isect_buffer,
                &compact_gid_from_isect_buffer,
                &sort_uniform_buffer,
                &num_intersections_buffer,
                "dynamic_sort_scatter_bind_group_alt",
            ));
        let tile_offsets_bind_group = Arc::new(self.create_tile_offsets_bind_group(
            ctx,
            label,
            &tile_id_from_isect_buffer,
            &tile_offsets_buffer,
            &num_intersections_buffer,
        ));
        let rasterize_bind_group = Arc::new(self.create_rasterize_bind_group(
            ctx,
            label,
            &compact_gid_from_isect_buffer,
            &tile_offsets_buffer,
            &projected_tile_splats_buffer,
            &raster_texture_view,
            &raster_uniform_buffer,
        ));
        let composite_bind_group = Arc::new(self.create_composite_bind_group(
            ctx,
            label,
            &raster_texture_view,
            &raster_uniform_buffer,
        ));

        CachedComputeResources {
            project_uniform_buffer,
            scan_uniform_buffer,
            sort_uniform_buffer,
            map_uniform_buffer,
            project_bind_group,
            tile_count_scan_bind_group,
            tile_count_scan_block_sums_bind_group,
            map_intersections_bind_group,
            dynamic_sort_count_bind_group_primary,
            dynamic_sort_count_bind_group_alt,
            dynamic_sort_reduce_bind_group,
            dynamic_sort_scan_bind_group,
            dynamic_sort_scan_add_bind_group,
            dynamic_sort_scatter_bind_group_primary,
            dynamic_sort_scatter_bind_group_alt,
            tile_offsets_bind_group,
            rasterize_bind_group,
            composite_bind_group,
            means_buffer,
            quats_buffer,
            scales_opacity_buffer,
            colors_buffer,
            sh_coeffs_buffer,
            index_buffer,
            visibility_flags_buffer,
            projected_tile_splats_buffer,
            tile_hit_counts_buffer,
            tile_hit_offsets_buffer,
            tile_hit_block_offsets_buffer,
            tile_intersection_count_buffer,
            intersection_count_readback_slots: create_intersection_count_readback_slots(
                &ctx.device,
                label,
            ),
            num_intersections_buffer,
            tile_id_from_isect_buffer,
            compact_gid_from_isect_buffer,
            sort_keys_buffer,
            sorted_indices_alt_buffer,
            sort_counts_buffer,
            sort_reduced_buffer,
            sort_scan_offsets_buffer,
            sort_scan_block_offsets_buffer,
            sort_scan_totals_buffer,
            tile_offsets_buffer,
            raster_uniform_buffer,
            raster_texture,
            raster_texture_view,
            raster_extent,
            index_capacity,
            block_capacity,
            intersection_capacity,
            tile_offset_capacity,
            latest_intersection_count: 0,
            intersection_capacity_saturated: false,
            sort_workgroup_count: sort_workgroup_count as u32,
            sort_reduce_workgroup_count: sort_reduce_workgroup_count as u32,
            sort_scan_block_capacity,
            cpu_indices: Vec::with_capacity(index_capacity),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_project_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        means_buffer: &Arc<wgpu::Buffer>,
        quats_buffer: &Arc<wgpu::Buffer>,
        scales_opacity_buffer: &Arc<wgpu::Buffer>,
        colors_buffer: &Arc<wgpu::Buffer>,
        sh_coeffs_buffer: &Arc<wgpu::Buffer>,
        index_buffer: &Arc<wgpu::Buffer>,
        output_instance_buffer: &Arc<wgpu::Buffer>,
        visibility_flags_buffer: &Arc<wgpu::Buffer>,
        project_uniform_buffer: &Arc<wgpu::Buffer>,
        projected_tile_splats_buffer: &Arc<wgpu::Buffer>,
        tile_hit_counts_buffer: &Arc<wgpu::Buffer>,
        label_suffix: &str,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::{label_suffix}")),
            layout: self.compute_pipelines.project_bind_group_layout.as_ref(),
            entries: &[
                storage_buffer_entry(0, means_buffer),
                storage_buffer_entry(1, quats_buffer),
                storage_buffer_entry(2, scales_opacity_buffer),
                storage_buffer_entry(3, colors_buffer),
                storage_buffer_entry(4, sh_coeffs_buffer),
                storage_buffer_entry(5, index_buffer),
                storage_buffer_entry(6, output_instance_buffer),
                storage_buffer_entry(7, visibility_flags_buffer),
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: project_uniform_buffer.as_entire_binding(),
                },
                storage_buffer_entry(9, projected_tile_splats_buffer),
                storage_buffer_entry(10, tile_hit_counts_buffer),
            ],
        })
    }

    fn create_scan_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        visibility_flags_buffer: &Arc<wgpu::Buffer>,
        local_offsets_buffer: &Arc<wgpu::Buffer>,
        block_offsets_buffer: &Arc<wgpu::Buffer>,
        scan_uniform_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::scan_bind_group")),
            layout: self.compute_pipelines.scan_bind_group_layout.as_ref(),
            entries: &[
                storage_buffer_entry(16, visibility_flags_buffer),
                storage_buffer_entry(17, local_offsets_buffer),
                storage_buffer_entry(18, block_offsets_buffer),
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: scan_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_map_intersections_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        projected_tile_splats_buffer: &Arc<wgpu::Buffer>,
        tile_hit_offsets_buffer: &Arc<wgpu::Buffer>,
        tile_hit_counts_buffer: &Arc<wgpu::Buffer>,
        tile_hit_block_offsets_buffer: &Arc<wgpu::Buffer>,
        tile_id_from_isect_buffer: &Arc<wgpu::Buffer>,
        compact_gid_from_isect_buffer: &Arc<wgpu::Buffer>,
        tile_intersection_count_buffer: &Arc<wgpu::Buffer>,
        num_intersections_buffer: &Arc<wgpu::Buffer>,
        map_uniform_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::map_intersections_bind_group")),
            layout: self
                .compute_pipelines
                .map_intersections_bind_group_layout
                .as_ref(),
            entries: &[
                storage_buffer_entry(0, projected_tile_splats_buffer),
                storage_buffer_entry(1, tile_hit_offsets_buffer),
                storage_buffer_entry(2, tile_hit_counts_buffer),
                storage_buffer_entry(3, tile_hit_block_offsets_buffer),
                storage_buffer_entry(4, tile_id_from_isect_buffer),
                storage_buffer_entry(5, compact_gid_from_isect_buffer),
                storage_buffer_entry(7, tile_intersection_count_buffer),
                storage_buffer_entry(8, num_intersections_buffer),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: map_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_dynamic_sort_count_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        src_keys_buffer: &Arc<wgpu::Buffer>,
        counts_buffer: &Arc<wgpu::Buffer>,
        sort_uniform_buffer: &Arc<wgpu::Buffer>,
        total_keys_buffer: &Arc<wgpu::Buffer>,
        suffix: &str,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::{suffix}")),
            layout: self
                .compute_pipelines
                .dynamic_sort_count_bind_group_layout
                .as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_uniform_buffer.as_entire_binding(),
                },
                storage_buffer_entry(1, src_keys_buffer),
                storage_buffer_entry(2, counts_buffer),
                storage_buffer_entry(6, total_keys_buffer),
            ],
        })
    }

    fn create_dynamic_sort_reduce_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        counts_buffer: &Arc<wgpu::Buffer>,
        reduced_buffer: &Arc<wgpu::Buffer>,
        sort_uniform_buffer: &Arc<wgpu::Buffer>,
        total_keys_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::dynamic_sort_reduce_bind_group")),
            layout: self
                .compute_pipelines
                .dynamic_sort_reduce_bind_group_layout
                .as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_uniform_buffer.as_entire_binding(),
                },
                storage_buffer_entry(1, counts_buffer),
                storage_buffer_entry(2, reduced_buffer),
                storage_buffer_entry(6, total_keys_buffer),
            ],
        })
    }

    fn create_dynamic_sort_scan_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        reduced_buffer: &Arc<wgpu::Buffer>,
        sort_uniform_buffer: &Arc<wgpu::Buffer>,
        total_keys_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::dynamic_sort_scan_bind_group")),
            layout: self
                .compute_pipelines
                .dynamic_sort_scan_bind_group_layout
                .as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_uniform_buffer.as_entire_binding(),
                },
                storage_buffer_entry(1, reduced_buffer),
                storage_buffer_entry(6, total_keys_buffer),
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_dynamic_sort_scan_compose_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        offsets_buffer: &Arc<wgpu::Buffer>,
        block_offsets_buffer: &Arc<wgpu::Buffer>,
        out_buffer: &Arc<wgpu::Buffer>,
        scan_uniform_buffer: &Arc<wgpu::Buffer>,
        suffix: &str,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::{suffix}")),
            layout: self
                .compute_pipelines
                .dynamic_sort_scan_compose_bind_group_layout
                .as_ref(),
            entries: &[
                storage_buffer_entry(8, offsets_buffer),
                storage_buffer_entry(9, block_offsets_buffer),
                storage_buffer_entry(10, out_buffer),
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: scan_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_dynamic_sort_scan_add_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        reduced_buffer: &Arc<wgpu::Buffer>,
        counts_buffer: &Arc<wgpu::Buffer>,
        sort_uniform_buffer: &Arc<wgpu::Buffer>,
        total_keys_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::dynamic_sort_scan_add_bind_group")),
            layout: self
                .compute_pipelines
                .dynamic_sort_scan_add_bind_group_layout
                .as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_uniform_buffer.as_entire_binding(),
                },
                storage_buffer_entry(1, reduced_buffer),
                storage_buffer_entry(2, counts_buffer),
                storage_buffer_entry(6, total_keys_buffer),
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_dynamic_sort_scatter_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        src_keys_buffer: &Arc<wgpu::Buffer>,
        src_values_buffer: &Arc<wgpu::Buffer>,
        counts_buffer: &Arc<wgpu::Buffer>,
        out_keys_buffer: &Arc<wgpu::Buffer>,
        out_values_buffer: &Arc<wgpu::Buffer>,
        sort_uniform_buffer: &Arc<wgpu::Buffer>,
        total_keys_buffer: &Arc<wgpu::Buffer>,
        suffix: &str,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::{suffix}")),
            layout: self
                .compute_pipelines
                .dynamic_sort_scatter_bind_group_layout
                .as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_uniform_buffer.as_entire_binding(),
                },
                storage_buffer_entry(1, src_keys_buffer),
                storage_buffer_entry(2, src_values_buffer),
                storage_buffer_entry(3, counts_buffer),
                storage_buffer_entry(4, out_keys_buffer),
                storage_buffer_entry(5, out_values_buffer),
                storage_buffer_entry(6, total_keys_buffer),
            ],
        })
    }

    fn create_tile_offsets_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        sorted_tile_ids_buffer: &Arc<wgpu::Buffer>,
        tile_offsets_buffer: &Arc<wgpu::Buffer>,
        total_keys_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::tile_offsets_bind_group")),
            layout: self
                .compute_pipelines
                .tile_offsets_bind_group_layout
                .as_ref(),
            entries: &[
                storage_buffer_entry(0, sorted_tile_ids_buffer),
                storage_buffer_entry(1, tile_offsets_buffer),
                storage_buffer_entry(2, total_keys_buffer),
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_rasterize_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compact_gid_from_isect_buffer: &Arc<wgpu::Buffer>,
        tile_offsets_buffer: &Arc<wgpu::Buffer>,
        projected_tile_splats_buffer: &Arc<wgpu::Buffer>,
        raster_texture_view: &Arc<wgpu::TextureView>,
        raster_uniform_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::rasterize_bind_group")),
            layout: self.compute_pipelines.rasterize_bind_group_layout.as_ref(),
            entries: &[
                storage_buffer_entry(0, compact_gid_from_isect_buffer),
                storage_buffer_entry(1, tile_offsets_buffer),
                storage_buffer_entry(2, projected_tile_splats_buffer),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(raster_texture_view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: raster_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_composite_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        raster_texture_view: &Arc<wgpu::TextureView>,
        raster_uniform_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let layouts = ctx.gpu_resources.bind_group_layouts.resources();
        let layout = layouts
            .get(self.composite_bind_group_layout)
            .expect("gaussian composite bind-group layout should exist");
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::composite_bind_group")),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(raster_texture_view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: raster_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_scan_block_sums_bind_group(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        block_offsets_buffer: &Arc<wgpu::Buffer>,
        indirect_draw_buffer: &Arc<wgpu::Buffer>,
        scan_uniform_buffer: &Arc<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}::scan_block_sums_bind_group")),
            layout: self
                .compute_pipelines
                .scan_block_sums_bind_group_layout
                .as_ref(),
            entries: &[
                storage_buffer_entry(24, block_offsets_buffer),
                storage_buffer_entry(25, indirect_draw_buffer),
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: scan_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn refresh_compute_bind_groups(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
        output_instance_buffer: &Arc<wgpu::Buffer>,
    ) {
        compute.project_bind_group = Arc::new(self.create_project_bind_group(
            ctx,
            label,
            &compute.means_buffer,
            &compute.quats_buffer,
            &compute.scales_opacity_buffer,
            &compute.colors_buffer,
            &compute.sh_coeffs_buffer,
            &compute.index_buffer,
            output_instance_buffer,
            &compute.visibility_flags_buffer,
            &compute.project_uniform_buffer,
            &compute.projected_tile_splats_buffer,
            &compute.tile_hit_counts_buffer,
            "project_bind_group",
        ));
        compute.tile_count_scan_bind_group = Arc::new(self.create_scan_bind_group(
            ctx,
            label,
            &compute.tile_hit_counts_buffer,
            &compute.tile_hit_offsets_buffer,
            &compute.tile_hit_block_offsets_buffer,
            &compute.scan_uniform_buffer,
        ));
        compute.tile_count_scan_block_sums_bind_group =
            Arc::new(self.create_scan_block_sums_bind_group(
                ctx,
                label,
                &compute.tile_hit_block_offsets_buffer,
                &compute.tile_intersection_count_buffer,
                &compute.scan_uniform_buffer,
            ));
        compute.map_intersections_bind_group = Arc::new(self.create_map_intersections_bind_group(
            ctx,
            label,
            &compute.projected_tile_splats_buffer,
            &compute.tile_hit_offsets_buffer,
            &compute.tile_hit_counts_buffer,
            &compute.tile_hit_block_offsets_buffer,
            &compute.tile_id_from_isect_buffer,
            &compute.compact_gid_from_isect_buffer,
            &compute.tile_intersection_count_buffer,
            &compute.num_intersections_buffer,
            &compute.map_uniform_buffer,
        ));
        compute.dynamic_sort_count_bind_group_primary =
            Arc::new(self.create_dynamic_sort_count_bind_group(
                ctx,
                label,
                &compute.tile_id_from_isect_buffer,
                &compute.sort_counts_buffer,
                &compute.sort_uniform_buffer,
                &compute.num_intersections_buffer,
                "dynamic_sort_count_bind_group_primary",
            ));
        compute.dynamic_sort_count_bind_group_alt =
            Arc::new(self.create_dynamic_sort_count_bind_group(
                ctx,
                label,
                &compute.sort_keys_buffer,
                &compute.sort_counts_buffer,
                &compute.sort_uniform_buffer,
                &compute.num_intersections_buffer,
                "dynamic_sort_count_bind_group_alt",
            ));
        compute.dynamic_sort_reduce_bind_group =
            Arc::new(self.create_dynamic_sort_reduce_bind_group(
                ctx,
                label,
                &compute.sort_counts_buffer,
                &compute.sort_reduced_buffer,
                &compute.sort_uniform_buffer,
                &compute.num_intersections_buffer,
            ));
        compute.dynamic_sort_scan_bind_group = Arc::new(self.create_dynamic_sort_scan_bind_group(
            ctx,
            label,
            &compute.sort_reduced_buffer,
            &compute.sort_uniform_buffer,
            &compute.num_intersections_buffer,
        ));
        compute.dynamic_sort_scan_add_bind_group =
            Arc::new(self.create_dynamic_sort_scan_add_bind_group(
                ctx,
                label,
                &compute.sort_reduced_buffer,
                &compute.sort_counts_buffer,
                &compute.sort_uniform_buffer,
                &compute.num_intersections_buffer,
            ));
        compute.dynamic_sort_scatter_bind_group_primary =
            Arc::new(self.create_dynamic_sort_scatter_bind_group(
                ctx,
                label,
                &compute.tile_id_from_isect_buffer,
                &compute.compact_gid_from_isect_buffer,
                &compute.sort_counts_buffer,
                &compute.sort_keys_buffer,
                &compute.sorted_indices_alt_buffer,
                &compute.sort_uniform_buffer,
                &compute.num_intersections_buffer,
                "dynamic_sort_scatter_bind_group_primary",
            ));
        compute.dynamic_sort_scatter_bind_group_alt =
            Arc::new(self.create_dynamic_sort_scatter_bind_group(
                ctx,
                label,
                &compute.sort_keys_buffer,
                &compute.sorted_indices_alt_buffer,
                &compute.sort_counts_buffer,
                &compute.tile_id_from_isect_buffer,
                &compute.compact_gid_from_isect_buffer,
                &compute.sort_uniform_buffer,
                &compute.num_intersections_buffer,
                "dynamic_sort_scatter_bind_group_alt",
            ));
        compute.tile_offsets_bind_group = Arc::new(self.create_tile_offsets_bind_group(
            ctx,
            label,
            &compute.tile_id_from_isect_buffer,
            &compute.tile_offsets_buffer,
            &compute.num_intersections_buffer,
        ));
        compute.rasterize_bind_group = Arc::new(self.create_rasterize_bind_group(
            ctx,
            label,
            &compute.compact_gid_from_isect_buffer,
            &compute.tile_offsets_buffer,
            &compute.projected_tile_splats_buffer,
            &compute.raster_texture_view,
            &compute.raster_uniform_buffer,
        ));
        compute.composite_bind_group = Arc::new(self.create_composite_bind_group(
            ctx,
            label,
            &compute.raster_texture_view,
            &compute.raster_uniform_buffer,
        ));
    }

    fn ensure_intersection_capacity(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
        required_capacity: usize,
    ) -> bool {
        if required_capacity <= compute.intersection_capacity {
            return false;
        }

        compute.intersection_capacity = required_capacity;
        compute.tile_id_from_isect_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_id_from_isect"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.compact_gid_from_isect_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::compact_gid_from_isect"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.sort_keys_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_keys"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.sorted_indices_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sorted_indices_alt"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        let sort_workgroup_count = compute
            .intersection_capacity
            .div_ceil(SORT_BLOCK_SIZE as usize)
            .max(1);
        compute.sort_workgroup_count = sort_workgroup_count as u32;
        let sort_reduce_workgroup_count = (sort_workgroup_count.div_ceil(SORT_BLOCK_SIZE as usize)
            * SORT_BIN_COUNT as usize)
            .max(1);
        compute.sort_reduce_workgroup_count = sort_reduce_workgroup_count as u32;
        compute.sort_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_counts"),
            sort_workgroup_count * SORT_BIN_COUNT as usize * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.sort_reduced_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_reduced"),
            sort_reduce_workgroup_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.sort_scan_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_offsets"),
            sort_reduce_workgroup_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.sort_scan_block_capacity = next_block_capacity(sort_reduce_workgroup_count);
        compute.sort_scan_block_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_block_offsets"),
            compute.sort_scan_block_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        true
    }

    fn process_intersection_count_readbacks(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
        output_instance_buffer: &Arc<wgpu::Buffer>,
    ) {
        let _ = ctx.device.poll(wgpu::PollType::Poll);

        let mut required_capacity = None;
        for slot in &mut compute.intersection_count_readback_slots {
            match slot.state {
                IntersectionCountReadbackState::Idle => {}
                IntersectionCountReadbackState::CopySubmitted => {
                    let slice = slot.buffer.slice(..);
                    let result = slot.result.clone();
                    slice.map_async(wgpu::MapMode::Read, move |map_result| {
                        *result.lock().unwrap() = Some(map_result.is_ok());
                    });
                    slot.state = IntersectionCountReadbackState::Mapping;
                }
                IntersectionCountReadbackState::Mapping => {
                    let ready = slot.result.lock().unwrap().take();
                    let Some(mapped_ok) = ready else {
                        continue;
                    };

                    if mapped_ok {
                        let bytes = slot.buffer.slice(..).get_mapped_range();
                        let words = bytemuck::cast_slice::<u8, u32>(&bytes);
                        let total_intersections = words.get(1).copied().unwrap_or(0) as usize;
                        drop(bytes);
                        slot.buffer.unmap();

                        compute.latest_intersection_count = total_intersections;
                        compute.intersection_capacity_saturated =
                            total_intersections > compute.intersection_capacity;

                        if total_intersections > compute.intersection_capacity {
                            required_capacity = Some(
                                required_capacity.map_or(total_intersections, |current: usize| {
                                    current.max(total_intersections)
                                }),
                            );
                        }
                    }

                    slot.state = IntersectionCountReadbackState::Idle;
                }
            }
        }

        if required_capacity.is_some_and(|required| {
            self.ensure_intersection_capacity(ctx, label, compute, required)
        }) {
            self.refresh_compute_bind_groups(ctx, label, compute, output_instance_buffer);
        }
    }

    fn ensure_tile_raster_resources(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
        output_instance_buffer: &Arc<wgpu::Buffer>,
        viewport_size_px: glam::Vec2,
    ) {
        let tile_bounds = calc_tile_bounds(viewport_size_px);
        let required_tile_offset_capacity = tile_count(tile_bounds).max(1);
        let raster_extent = calc_raster_extent(viewport_size_px);

        let mut changed = false;
        if required_tile_offset_capacity > compute.tile_offset_capacity {
            compute.tile_offset_capacity = required_tile_offset_capacity;
            compute.tile_offsets_buffer = Arc::new(create_sized_buffer(
                &ctx.device,
                &format!("{label}::tile_offsets"),
                compute.tile_offset_capacity * 2 * std::mem::size_of::<u32>(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ));
            changed = true;
        }

        if compute.raster_extent != raster_extent {
            compute.raster_extent = raster_extent;
            let (raster_texture, raster_texture_view) = create_viewer_raster_texture(
                &ctx.device,
                &format!("{label}::raster_color"),
                raster_extent,
            );
            compute.raster_texture = raster_texture;
            compute.raster_texture_view = raster_texture_view;
            changed = true;
        }

        if changed {
            self.refresh_compute_bind_groups(ctx, label, compute, output_instance_buffer);
        }

        let raster_uniform = gpu_data::RasterUniformBuffer {
            tile_bounds: tile_bounds.to_array(),
            img_size: raster_extent.to_array(),
        };
        ctx.queue.write_buffer(
            &compute.raster_uniform_buffer,
            0,
            bytemuck::bytes_of(&raster_uniform),
        );
    }

    fn grow_instance_buffer(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        resources: &mut CachedBatchResources,
        required_capacity: usize,
    ) {
        resources.instance_capacity = next_capacity(required_capacity);
        resources.instance_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::instances"),
            resources.instance_capacity * std::mem::size_of::<gpu_data::InstanceData>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));
        let compute = &mut resources.compute;
        compute.visibility_flags_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::visibility_flags"),
            resources.instance_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.projected_tile_splats_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::projected_tile_splats"),
            resources.instance_capacity * std::mem::size_of::<gpu_data::TileProjectedSplat>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.tile_hit_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_counts"),
            resources.instance_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.tile_hit_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_offsets"),
            resources.instance_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let required_block_capacity = next_block_capacity(resources.instance_capacity);
        if required_block_capacity > compute.block_capacity {
            compute.block_capacity = required_block_capacity;
            compute.tile_hit_block_offsets_buffer = Arc::new(create_sized_buffer(
                &ctx.device,
                &format!("{label}::tile_hit_block_offsets"),
                compute.block_capacity * std::mem::size_of::<u32>(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ));
        }
        let required_intersection_capacity =
            intersection_capacity_for_instances(resources.instance_capacity);
        self.ensure_intersection_capacity(ctx, label, compute, required_intersection_capacity);
        self.refresh_compute_bind_groups(ctx, label, compute, &resources.instance_buffer);
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
                entries: vec![
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                                gpu_data::RasterUniformBuffer,
                            >(
                            )
                                as u64),
                        },
                        count: None,
                    },
                ],
            },
        );
        let tile_pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &re_renderer::PipelineLayoutDesc {
                label: "GaussianRenderer::tile_draw".into(),
                entries: vec![ctx.global_bindings.layout, composite_bind_group_layout],
            },
        );

        let mut depth_state = re_renderer::ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE;
        depth_state.depth_write_enabled = false;

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

        let project_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GaussianRenderer::project_compute"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shader/gaussian_project.wgsl"
                ))),
            });
        let map_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GaussianRenderer::map_intersections_compute"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shader/gaussian_map_intersections.wgsl"
                ))),
            });
        let dynamic_sort_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GaussianRenderer::dynamic_sort_compute"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shader/gaussian_dynamic_sort.wgsl"
                ))),
            });
        let tile_offsets_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GaussianRenderer::tile_offsets_compute"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shader/gaussian_tile_offsets.wgsl"
                ))),
            });
        let rasterize_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GaussianRenderer::rasterize_compute"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shader/gaussian_raster_tiles.wgsl"
                ))),
            });
        let project_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::project_bind_group_layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, true),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, true),
                    storage_layout_entry(6, false),
                    storage_layout_entry(7, false),
                    uniform_layout_entry(8, std::mem::size_of::<gpu_data::ProjectUniformBuffer>()),
                    storage_layout_entry(9, false),
                    storage_layout_entry(10, false),
                ],
            },
        ));
        let scan_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::scan_bind_group_layout"),
                entries: &[
                    storage_layout_entry(16, true),
                    storage_layout_entry(17, false),
                    storage_layout_entry(18, false),
                    uniform_layout_entry(19, std::mem::size_of::<gpu_data::ScanUniformBuffer>()),
                ],
            },
        ));
        let scan_block_sums_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::scan_block_sums_bind_group_layout"),
                entries: &[
                    storage_layout_entry(24, false),
                    storage_layout_entry(25, false),
                    uniform_layout_entry(26, std::mem::size_of::<gpu_data::ScanUniformBuffer>()),
                ],
            },
        ));
        let map_intersections_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::map_intersections_bind_group_layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, true),
                    storage_layout_entry(4, false),
                    storage_layout_entry(5, false),
                    uniform_layout_entry(6, std::mem::size_of::<gpu_data::MapUniformBuffer>()),
                    storage_layout_entry(7, true),
                    storage_layout_entry(8, false),
                ],
            },
        ));
        let dynamic_sort_count_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::dynamic_sort_count_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0, std::mem::size_of::<gpu_data::SortUniformBuffer>()),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(6, true),
                ],
            },
        ));
        let dynamic_sort_reduce_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::dynamic_sort_reduce_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0, std::mem::size_of::<gpu_data::SortUniformBuffer>()),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(6, true),
                ],
            },
        ));
        let dynamic_sort_scan_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::dynamic_sort_scan_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0, std::mem::size_of::<gpu_data::SortUniformBuffer>()),
                    storage_layout_entry(1, false),
                    storage_layout_entry(6, true),
                ],
            },
        ));
        let dynamic_sort_scan_compose_bind_group_layout = Arc::new(
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GaussianRenderer::dynamic_sort_scan_compose_bind_group_layout"),
                    entries: &[
                        storage_layout_entry(8, true),
                        storage_layout_entry(9, true),
                        storage_layout_entry(10, false),
                        uniform_layout_entry(
                            11,
                            std::mem::size_of::<gpu_data::ScanUniformBuffer>(),
                        ),
                    ],
                }),
        );
        let dynamic_sort_scan_add_bind_group_layout = Arc::new(
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GaussianRenderer::dynamic_sort_scan_add_bind_group_layout"),
                    entries: &[
                        uniform_layout_entry(0, std::mem::size_of::<gpu_data::SortUniformBuffer>()),
                        storage_layout_entry(1, true),
                        storage_layout_entry(2, false),
                        storage_layout_entry(6, true),
                    ],
                }),
        );
        let dynamic_sort_scatter_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::dynamic_sort_scatter_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0, std::mem::size_of::<gpu_data::SortUniformBuffer>()),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, true),
                    storage_layout_entry(4, false),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, true),
                ],
            },
        ));
        let tile_offsets_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::tile_offsets_bind_group_layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, false),
                    storage_layout_entry(2, true),
                ],
            },
        ));
        let rasterize_bind_group_layout = Arc::new(ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianRenderer::rasterize_bind_group_layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: RASTER_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    uniform_layout_entry(4, std::mem::size_of::<gpu_data::RasterUniformBuffer>()),
                ],
            },
        ));
        let project_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::project_compute",
            &project_shader,
            "project_main",
            &[project_bind_group_layout.as_ref()],
        );
        let scan_blocks_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::scan_blocks_compute",
            &project_shader,
            "scan_blocks_main",
            &[scan_bind_group_layout.as_ref()],
        );
        let scan_block_sums_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::scan_block_sums_compute",
            &project_shader,
            "scan_block_sums_main",
            &[scan_block_sums_bind_group_layout.as_ref()],
        );
        let map_intersections_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::map_intersections_compute",
            &map_shader,
            "map_main",
            &[map_intersections_bind_group_layout.as_ref()],
        );
        let clamp_intersection_count_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::clamp_intersection_count_compute",
            &map_shader,
            "clamp_count_main",
            &[map_intersections_bind_group_layout.as_ref()],
        );
        let dynamic_sort_count_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::dynamic_sort_count_compute",
            &dynamic_sort_shader,
            "sort_count_main",
            &[dynamic_sort_count_bind_group_layout.as_ref()],
        );
        let dynamic_sort_reduce_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::dynamic_sort_reduce_compute",
            &dynamic_sort_shader,
            "sort_reduce_main",
            &[dynamic_sort_reduce_bind_group_layout.as_ref()],
        );
        let dynamic_sort_scan_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::dynamic_sort_scan_compute",
            &dynamic_sort_shader,
            "sort_scan_main",
            &[dynamic_sort_scan_bind_group_layout.as_ref()],
        );
        let dynamic_sort_scan_compose_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::dynamic_sort_scan_compose_compute",
            &dynamic_sort_shader,
            "sort_scan_compose_main",
            &[dynamic_sort_scan_compose_bind_group_layout.as_ref()],
        );
        let dynamic_sort_scan_add_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::dynamic_sort_scan_add_compute",
            &dynamic_sort_shader,
            "sort_scan_add_main",
            &[dynamic_sort_scan_add_bind_group_layout.as_ref()],
        );
        let dynamic_sort_scatter_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::dynamic_sort_scatter_compute",
            &dynamic_sort_shader,
            "sort_scatter_main",
            &[dynamic_sort_scatter_bind_group_layout.as_ref()],
        );
        let tile_offsets_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::tile_offsets_compute",
            &tile_offsets_shader,
            "main",
            &[tile_offsets_bind_group_layout.as_ref()],
        );
        let rasterize_pipeline = create_compute_pipeline(
            &ctx.device,
            "GaussianRenderer::rasterize_compute",
            &rasterize_shader,
            "main",
            &[rasterize_bind_group_layout.as_ref()],
        );

        let compute_pipelines = ComputePipelines {
            project_bind_group_layout,
            project_pipeline,
            scan_bind_group_layout,
            scan_blocks_pipeline,
            scan_block_sums_bind_group_layout,
            scan_block_sums_pipeline,
            map_intersections_bind_group_layout,
            map_intersections_pipeline,
            clamp_intersection_count_pipeline,
            dynamic_sort_count_bind_group_layout,
            dynamic_sort_count_pipeline,
            dynamic_sort_reduce_bind_group_layout,
            dynamic_sort_reduce_pipeline,
            dynamic_sort_scan_bind_group_layout,
            dynamic_sort_scan_pipeline,
            dynamic_sort_scan_compose_bind_group_layout,
            dynamic_sort_scan_compose_pipeline,
            dynamic_sort_scan_add_bind_group_layout,
            dynamic_sort_scan_add_pipeline,
            dynamic_sort_scatter_bind_group_layout,
            dynamic_sort_scatter_pipeline,
            tile_offsets_bind_group_layout,
            tile_offsets_pipeline,
            rasterize_bind_group_layout,
            rasterize_pipeline,
        };

        Self {
            composite_bind_group_layout,
            render_pipeline_tile,
            compute_pipelines,
            batch_cache: Mutex::new(HashMap::new()),
        }
    }

    fn draw(
        &self,
        render_pipelines: &re_renderer::GpuRenderPipelinePoolAccessor<'_>,
        phase: re_renderer::DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_instructions: &[DrawInstruction<'_, GaussianDrawData>],
    ) -> Result<(), DrawError> {
        let draw_start = Instant::now();

        let tile_pipeline = render_pipelines.get(self.render_pipeline_tile)?;
        for instruction in draw_instructions {
            for drawable in instruction.drawables {
                let batch_index = drawable.draw_data_payload as usize;
                let Some(batch) = instruction.draw_data.batches.get(batch_index) else {
                    continue;
                };

                // The compute pipeline has already rasterized into an intermediate texture.
                // The draw step is just a fullscreen composite of that texture into the viewport.
                if phase != re_renderer::DrawPhase::Transparent {
                    continue;
                }
                pass.set_pipeline(tile_pipeline);
                pass.set_bind_group(1, batch.payload.composite_bind_group.as_ref(), &[]);
                pass.draw(0..3, 0..1);
            }
        }

        let _ = draw_start;

        Ok(())
    }
}

fn create_viewer_raster_texture(
    device: &wgpu::Device,
    label: &str,
    extent: glam::UVec2,
) -> (Arc<wgpu::Texture>, Arc<wgpu::TextureView>) {
    let (texture, view) = crate::gsplat_core::gpu_types::create_raster_texture(
        device,
        label,
        extent,
        wgpu::TextureUsages::empty(),
    );
    (Arc::new(texture), Arc::new(view))
}

fn create_intersection_count_readback_slots(
    device: &wgpu::Device,
    label: &str,
) -> Vec<IntersectionCountReadbackSlot> {
    (0..INTERSECTION_READBACK_SLOT_COUNT)
        .map(|slot_index| IntersectionCountReadbackSlot {
            buffer: Arc::new(create_sized_buffer(
                device,
                &format!("{label}::tile_intersection_count_readback_{slot_index}"),
                std::mem::size_of::<gpu_data::DrawIndirectArgs>(),
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )),
            result: Arc::new(Mutex::new(None)),
            state: IntersectionCountReadbackState::Idle,
        })
        .collect()
}

#[cfg(test)]
fn encode_descending_depth_key(camera_depth: f32) -> u32 {
    u32::MAX - camera_depth.to_bits()
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
            .expect("failed to register gaussian_composite.wgsl");
    });
}

mod compute {
    //! Brush-inspired compute/tile preparation path.
    //!
    //! The public facade decides when compute is available. This module owns the per-frame GPU work
    //! that dispatches the 7-stage GPU compute pipeline ending at tile raster/composite.

    use std::sync::Arc;
    use std::time::Instant;

    use super::*;
    use crate::gsplat_core::sh_degree_from_coeffs;

    impl GaussianRenderer {
        #[allow(clippy::too_many_arguments)]
        pub(super) fn prepare_compute_batch(
            &self,
            ctx: &re_renderer::RenderContext,
            label: &str,
            cloud: &Arc<RenderGaussianCloud>,
            camera: &CameraApproximation,
            visible_indices: &[SortedSplatIndex],
            _farthest_depth: f32,
        ) -> PreparedBatch {
            let upload_start = Instant::now();
            let mut cache = self.batch_cache.lock().unwrap();
            let resources = cache.entry(label.to_owned()).or_insert_with(|| {
                self.create_batch_resources(ctx, label, cloud, visible_indices.len().max(1))
            });
            let pipelines = &self.compute_pipelines;
            let selected_limit = visible_indices.len().max(1);

            // Step 1: make sure the persistent per-cloud buffers are large enough for this frame.
            if resources.instance_capacity < selected_limit {
                self.grow_instance_buffer(ctx, label, resources, selected_limit);
            }
            let compute = &mut resources.compute;
            self.process_intersection_count_readbacks(
                ctx,
                label,
                compute,
                &resources.instance_buffer,
            );
            self.ensure_tile_raster_resources(
                ctx,
                label,
                compute,
                &resources.instance_buffer,
                camera.viewport_size_px,
            );

            let coeffs_per_channel = cloud
                .sh_coeffs
                .as_ref()
                .map_or(0_u32, |sh| sh.coeffs_per_channel as u32);
            let sh_degree = sh_degree_from_coeffs(coeffs_per_channel as usize).unwrap_or(0);
            let block_count = compaction_block_count(selected_limit) as u32;
            let tile_bounds = calc_tile_bounds(camera.viewport_size_px);
            let project_uniform = gpu_data::ProjectUniformBuffer {
                view_from_world: glam::Mat4::from(camera.view_from_world).to_cols_array_2d(),
                projection_from_view: camera.projection_from_view.to_cols_array_2d(),
                camera_world_position: camera.world_position.extend(0.0).to_array(),
                viewport_and_near: [
                    camera.viewport_size_px.x,
                    camera.viewport_size_px.y,
                    camera.near_plane,
                    MIN_RADIUS_PX,
                ],
                sigma_and_counts: [
                    SIGMA_COVERAGE.to_bits(),
                    selected_limit as u32,
                    coeffs_per_channel,
                    sh_degree,
                ],
                _pad: [[
                    u32::from(cloud.sh_coeffs.is_some()),
                    OPACITY_SCALE.to_bits(),
                    MAX_SPLATS_RENDERED.min(u32::MAX as usize) as u32,
                    0,
                ]],
            };
            let scan_uniform = gpu_data::ScanUniformBuffer {
                total_selected: selected_limit as u32,
                block_count,
                _pad: [0; 2],
            };
            let map_uniform = gpu_data::MapUniformBuffer {
                total_selected: selected_limit as u32,
                intersection_capacity: compute.intersection_capacity.min(u32::MAX as usize) as u32,
                tile_bounds_x: tile_bounds.x,
                tile_bounds_y: tile_bounds.y,
            };
            ctx.queue.write_buffer(
                &compute.project_uniform_buffer,
                0,
                bytemuck::bytes_of(&project_uniform),
            );
            ctx.queue.write_buffer(
                &compute.scan_uniform_buffer,
                0,
                bytemuck::bytes_of(&scan_uniform),
            );
            ctx.queue.write_buffer(
                &compute.map_uniform_buffer,
                0,
                bytemuck::bytes_of(&map_uniform),
            );

            // Step 2: upload the pre-sorted candidate ordering from the CPU visibility pass.
            // The GPU projection/raster work
            // onto the GPU.
            compute.cpu_indices.clear();
            compute.cpu_indices.reserve(visible_indices.len());
            compute.cpu_indices.extend(
                visible_indices
                    .iter()
                    .rev()
                    .map(|visible| visible.splat_index),
            );
            ctx.queue.write_buffer(
                &compute.index_buffer,
                0,
                bytemuck::cast_slice(compute.cpu_indices.as_slice()),
            );

            let (project_x, project_y) =
                dispatch_grid_1d(selected_limit as u32, PROJECT_WORKGROUP_SIZE);
            {
                // Step 3: project, count tile intersections, and map each splat to its tile coverage.
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(label),
                            timestamp_writes: None,
                        });

                compute_pass.set_pipeline(&pipelines.project_pipeline);
                compute_pass.set_bind_group(0, compute.project_bind_group.as_ref(), &[]);
                compute_pass.dispatch_workgroups(project_x, project_y, 1);

                let (scan_x, scan_y) = dispatch_grid_1d(block_count, 1);
                compute_pass.set_pipeline(&pipelines.scan_blocks_pipeline);
                compute_pass.set_bind_group(0, compute.tile_count_scan_bind_group.as_ref(), &[]);
                compute_pass.dispatch_workgroups(scan_x, scan_y, 1);

                compute_pass.set_pipeline(&pipelines.scan_block_sums_pipeline);
                compute_pass.set_bind_group(
                    0,
                    compute.tile_count_scan_block_sums_bind_group.as_ref(),
                    &[],
                );
                compute_pass.dispatch_workgroups(1, 1, 1);

                compute_pass.set_pipeline(&pipelines.map_intersections_pipeline);
                compute_pass.set_bind_group(0, compute.map_intersections_bind_group.as_ref(), &[]);
                compute_pass.dispatch_workgroups(project_x, project_y, 1);

                compute_pass.set_pipeline(&pipelines.clamp_intersection_count_pipeline);
                compute_pass.set_bind_group(0, compute.map_intersections_bind_group.as_ref(), &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }

            if let Some(slot) = compute
                .intersection_count_readback_slots
                .iter_mut()
                .find(|slot| slot.state == IntersectionCountReadbackState::Idle)
            {
                // Read back the exact intersection demand so dense scenes can grow the staging buffers
                // safely on a later frame without stalling the normal render path.
                *slot.result.lock().unwrap() = None;
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                encoder.get().copy_buffer_to_buffer(
                    &compute.tile_intersection_count_buffer,
                    0,
                    &slot.buffer,
                    0,
                    std::mem::size_of::<gpu_data::DrawIndirectArgs>() as u64,
                );
                slot.state = IntersectionCountReadbackState::CopySubmitted;
            }

            let tile_count_bits =
                (u32::BITS - (tile_count(tile_bounds).max(1) as u32).leading_zeros()).max(1);
            let sort_passes = tile_count_bits.div_ceil(4);
            let (sort_count_x, sort_count_y) =
                dispatch_grid_for_workgroups(compute.sort_workgroup_count.max(1));
            let (sort_reduce_x, sort_reduce_y) =
                dispatch_grid_for_workgroups(compute.sort_reduce_workgroup_count.max(1));
            for pass_index in 0..sort_passes {
                // Step 4: radix-sort the tile intersections so each tile can consume a contiguous
                // intersection range during raster.
                let sort_uniform_buffer =
                    Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{label}::sort_uniform_pass_{pass_index}")),
                        size: std::mem::size_of::<gpu_data::SortUniformBuffer>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                let sort_uniform = gpu_data::SortUniformBuffer {
                    shift: pass_index * 4,
                    total_keys_unused: 0,
                    _pad: [0; 2],
                };
                ctx.queue
                    .write_buffer(&sort_uniform_buffer, 0, bytemuck::bytes_of(&sort_uniform));

                let use_primary_as_source = pass_index % 2 == 0;
                let (
                    src_keys_buffer,
                    src_values_buffer,
                    dst_keys_buffer,
                    dst_values_buffer,
                    suffix,
                ) = if use_primary_as_source {
                    (
                        &compute.tile_id_from_isect_buffer,
                        &compute.compact_gid_from_isect_buffer,
                        &compute.sort_keys_buffer,
                        &compute.sorted_indices_alt_buffer,
                        "dynamic_sort_primary_to_alt",
                    )
                } else {
                    (
                        &compute.sort_keys_buffer,
                        &compute.sorted_indices_alt_buffer,
                        &compute.tile_id_from_isect_buffer,
                        &compute.compact_gid_from_isect_buffer,
                        "dynamic_sort_alt_to_primary",
                    )
                };
                let count_bind_group = self.create_dynamic_sort_count_bind_group(
                    ctx,
                    label,
                    src_keys_buffer,
                    &compute.sort_counts_buffer,
                    &sort_uniform_buffer,
                    &compute.num_intersections_buffer,
                    suffix,
                );
                let reduce_bind_group = self.create_dynamic_sort_reduce_bind_group(
                    ctx,
                    label,
                    &compute.sort_counts_buffer,
                    &compute.sort_reduced_buffer,
                    &sort_uniform_buffer,
                    &compute.num_intersections_buffer,
                );
                let reduced_total = compute.sort_reduce_workgroup_count.max(1);
                let reduced_block_count = compaction_block_count(reduced_total as usize) as u32;
                let sort_scan_uniform_buffer =
                    Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{label}::sort_scan_uniform_pass_{pass_index}")),
                        size: std::mem::size_of::<gpu_data::ScanUniformBuffer>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                let sort_scan_uniform = gpu_data::ScanUniformBuffer {
                    total_selected: reduced_total,
                    block_count: reduced_block_count,
                    _pad: [0; 2],
                };
                ctx.queue.write_buffer(
                    &sort_scan_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&sort_scan_uniform),
                );
                let scan_blocks_bind_group = self.create_scan_bind_group(
                    ctx,
                    label,
                    &compute.sort_reduced_buffer,
                    &compute.sort_scan_offsets_buffer,
                    &compute.sort_scan_block_offsets_buffer,
                    &sort_scan_uniform_buffer,
                );
                let scan_block_sums_bind_group = self.create_scan_block_sums_bind_group(
                    ctx,
                    label,
                    &compute.sort_scan_block_offsets_buffer,
                    &compute.sort_scan_totals_buffer,
                    &sort_scan_uniform_buffer,
                );
                let scan_compose_bind_group = self.create_dynamic_sort_scan_compose_bind_group(
                    ctx,
                    label,
                    &compute.sort_scan_offsets_buffer,
                    &compute.sort_scan_block_offsets_buffer,
                    &compute.sort_reduced_buffer,
                    &sort_scan_uniform_buffer,
                    suffix,
                );
                let scan_add_bind_group = self.create_dynamic_sort_scan_add_bind_group(
                    ctx,
                    label,
                    &compute.sort_reduced_buffer,
                    &compute.sort_counts_buffer,
                    &sort_uniform_buffer,
                    &compute.num_intersections_buffer,
                );
                let scatter_bind_group = self.create_dynamic_sort_scatter_bind_group(
                    ctx,
                    label,
                    src_keys_buffer,
                    src_values_buffer,
                    &compute.sort_counts_buffer,
                    dst_keys_buffer,
                    dst_values_buffer,
                    &sort_uniform_buffer,
                    &compute.num_intersections_buffer,
                    suffix,
                );
                let (sort_scan_blocks_x, sort_scan_blocks_y) =
                    dispatch_grid_1d(reduced_block_count, 1);
                let (sort_scan_compose_x, sort_scan_compose_y) =
                    dispatch_grid_1d(reduced_total, SORT_WORKGROUP_SIZE);

                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gaussian_dynamic_sort"),
                            timestamp_writes: None,
                        });
                compute_pass.set_pipeline(&pipelines.dynamic_sort_count_pipeline);
                compute_pass.set_bind_group(0, &count_bind_group, &[]);
                compute_pass.dispatch_workgroups(sort_count_x, sort_count_y, 1);

                compute_pass.set_pipeline(&pipelines.dynamic_sort_reduce_pipeline);
                compute_pass.set_bind_group(0, &reduce_bind_group, &[]);
                compute_pass.dispatch_workgroups(sort_reduce_x, sort_reduce_y, 1);

                compute_pass.set_pipeline(&pipelines.scan_blocks_pipeline);
                compute_pass.set_bind_group(0, &scan_blocks_bind_group, &[]);
                compute_pass.dispatch_workgroups(sort_scan_blocks_x, sort_scan_blocks_y, 1);

                compute_pass.set_pipeline(&pipelines.scan_block_sums_pipeline);
                compute_pass.set_bind_group(0, &scan_block_sums_bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);

                compute_pass.set_pipeline(&pipelines.dynamic_sort_scan_compose_pipeline);
                compute_pass.set_bind_group(0, &scan_compose_bind_group, &[]);
                compute_pass.dispatch_workgroups(sort_scan_compose_x, sort_scan_compose_y, 1);

                compute_pass.set_pipeline(&pipelines.dynamic_sort_scan_add_pipeline);
                compute_pass.set_bind_group(0, &scan_add_bind_group, &[]);
                compute_pass.dispatch_workgroups(sort_reduce_x, sort_reduce_y, 1);

                compute_pass.set_pipeline(&pipelines.dynamic_sort_scatter_pipeline);
                compute_pass.set_bind_group(0, &scatter_bind_group, &[]);
                compute_pass.dispatch_workgroups(sort_count_x, sort_count_y, 1);
            }

            if sort_passes % 2 == 1 {
                let bytes = (compute.intersection_capacity * std::mem::size_of::<u32>()) as u64;
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                encoder.get().copy_buffer_to_buffer(
                    &compute.sort_keys_buffer,
                    0,
                    &compute.tile_id_from_isect_buffer,
                    0,
                    bytes,
                );
                encoder.get().copy_buffer_to_buffer(
                    &compute.sorted_indices_alt_buffer,
                    0,
                    &compute.compact_gid_from_isect_buffer,
                    0,
                    bytes,
                );
            }

            {
                // Step 5: turn sorted tile intersections into per-tile ranges, raster each tile, then
                // queue a fullscreen composite back into the normal Rerun draw graph.
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                encoder
                    .get()
                    .clear_buffer(&compute.tile_offsets_buffer, 0, None);
                let tile_offset_elements = (compute.intersection_capacity.max(1)) as u32;
                let (tile_offset_x, tile_offset_y) = dispatch_grid_1d(
                    tile_offset_elements,
                    TILE_OFFSET_WORKGROUP_SIZE * TILE_OFFSET_CHECKS_PER_ITER,
                );
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gaussian_tile_offsets"),
                            timestamp_writes: None,
                        });
                compute_pass.set_pipeline(&pipelines.tile_offsets_pipeline);
                compute_pass.set_bind_group(0, compute.tile_offsets_bind_group.as_ref(), &[]);
                compute_pass.dispatch_workgroups(tile_offset_x, tile_offset_y, 1);
            }

            {
                let tile_workgroups = tile_count(tile_bounds).max(1) as u32;
                let (raster_x, raster_y) = dispatch_grid_for_workgroups(tile_workgroups);
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gaussian_rasterize_tiles"),
                            timestamp_writes: None,
                        });
                compute_pass.set_pipeline(&pipelines.rasterize_pipeline);
                compute_pass.set_bind_group(0, compute.rasterize_bind_group.as_ref(), &[]);
                compute_pass.dispatch_workgroups(raster_x, raster_y, 1);
            }

            PreparedBatch {
                batch: Some(GaussianBatch {
                    payload: GaussianBatchPayload {
                        composite_bind_group: compute.composite_bind_group.clone(),
                    },
                }),
                upload_ms: upload_start.elapsed().as_secs_f32() * 1000.0,
                extra_draw_data: None,
                tile_intersections: compute.latest_intersection_count,
                intersection_capacity: compute.intersection_capacity,
                intersection_capacity_saturated: compute.intersection_capacity_saturated,
            }
        }
    }
}

mod gpu_types {
    //! GPU buffer layouts for the Rerun viewer splat draw path.
    //!
    //! Shared types (ProjectUniformBuffer, ScanUniformBuffer, etc.) are imported
    //! from `gsplat_core::gpu_types` — the single source of truth.  Only
    //! Rerun-specific types live here.

    use bytemuck::{Pod, Zeroable};

    // Re-export shared GPU types from gsplat_core so existing `gpu_data::*`
    // references throughout this file continue to work unchanged.
    pub use crate::gsplat_core::gpu_types::{
        DrawIndirectArgs, MapUniformBuffer, ProjectUniformBuffer, RasterUniformBuffer,
        ScanUniformBuffer, SortUniformBuffer, TileProjectedSplat,
    };

    /// Per-instance data layout -- matches the WGSL shader's output format. Used for buffer sizing.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub struct InstanceData {
        pub center_ndc: [f32; 2],
        pub ndc_depth: f32,
        pub radius_ndc: f32,
        pub inv_cov_ndc_xx_xy_yy_pad: [f32; 4],
        pub color_opacity: [f32; 4],
    }

    #[cfg(test)]
    mod tests {
        use super::TileProjectedSplat;

        #[test]
        fn tile_projected_splat_layout_matches_wgsl_storage_stride() {
            assert_eq!(std::mem::offset_of!(TileProjectedSplat, xy_px), 0);
            assert_eq!(
                std::mem::offset_of!(TileProjectedSplat, conic_xyy_opacity),
                16
            );
            assert_eq!(std::mem::offset_of!(TileProjectedSplat, color_rgba), 32);
            assert_eq!(
                std::mem::offset_of!(TileProjectedSplat, tile_bbox_min_max),
                48
            );
            assert_eq!(std::mem::size_of::<TileProjectedSplat>(), 64);
        }
    }
}
