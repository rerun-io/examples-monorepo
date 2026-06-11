//! GPU renderer for Gaussian splats — the Rerun viewer layer.
//!
//! # Overview
//!
//! This module integrates the GPU compute pipeline with Rerun's rendering
//! system.  The visualizer ([`crate::gaussian_visualizer`]) hands it the full
//! splat cloud and camera each frame; everything else — culling, depth
//! sorting, projection, rasterization — runs on the GPU.  Per-frame CPU work
//! is uniform writes and command encoding only.
//!
//! GPU types, bind group layouts, compute pipelines, and helper functions
//! are imported from [`crate::gsplat_core::gpu_types`] — the single source
//! of truth shared with the standalone renderer.
//!
//! # Compute Pipeline (GPU-only, Brush-aligned)
//!
//! | Stage | Shader | Description |
//! |-------|--------|-------------|
//! | 1. Cull | `gaussian_project.wgsl` (project_forward) | Full-cloud GPU cull, compact (gid, depth) |
//! | 2. Depth sort | `gaussian_dynamic_sort.wgsl` | Radix argsort ascending by depth bits |
//! | 3. Project | `gaussian_project.wgsl` (project_visible) | 3D→2D projection + SH evaluation |
//! | 4. Scan | `gaussian_project.wgsl` (scan) | Prefix sum over per-splat tile hit counts |
//! | 5. Map | `gaussian_map_intersections.wgsl` | Scatter (tile, splat) pairs |
//! | 6. Tile sort | `gaussian_dynamic_sort.wgsl` | Radix sort by tile ID |
//! | 7. Offsets | `gaussian_tile_offsets.wgsl` | Per-tile start/end range |
//! | 8. Raster | `gaussian_raster_tiles.wgsl` | Per-pixel alpha blending |
//! | 9. Composite | `gaussian_composite.wgsl` | Blit to Rerun viewport |
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
//! 2. Write uniforms (camera, counts)
//! 3. Dispatch the GPU-only compute pipeline
//! 4. Composite raster texture into Rerun's viewport via fullscreen blit

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use re_renderer::external::smallvec::smallvec;
use re_renderer::external::wgpu;
use re_renderer::renderer::{DrawData, DrawDataDrawable, DrawError, DrawInstruction, Renderer};

use self::gpu_types as gpu_data;
use crate::gsplat_core::gpu_types::{
    DEPTH_SORT_PASSES, GpuBindGroupLayouts, GpuComputePipelines, PROJECT_WORKGROUP_SIZE,
    PipelineBindGroups, PipelineBuffers, RadixSort, RadixSortBuffers, RadixSortScratch,
    SORT_BIN_COUNT,
    SORT_WORKGROUP_SIZE, TILE_OFFSET_CHECKS_PER_ITER, TILE_OFFSET_WORKGROUP_SIZE, TILE_WIDTH,
    build_radix_sort, calc_raster_extent, calc_tile_bounds, compaction_block_count,
    create_sort_shift_uniforms, sort_workgroup_count_for,
    create_compute_bind_group_layouts, create_compute_pipelines, create_filled_buffer,
    create_pipeline_bind_groups, create_sized_buffer, dispatch_grid_1d,
    dispatch_grid_for_workgroups, fill_map_uniform, fill_project_uniform, fill_scan_uniform,
    gid_sort_passes, intersection_capacity_for_instances, next_block_capacity, next_capacity,
    pack_quats, pack_rgb, pack_scales_opacity, pack_sh_coefficients, pack_vec3s,
    sort_reduce_workgroup_count, tile_count, tile_sort_passes,
};
use crate::gsplat_core::{CameraApproximation, RenderGaussianCloud};

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
    fn intersection_capacity_scales_with_instance_capacity() {
        assert_eq!(super::intersection_capacity_for_instances(0), 32);
        assert_eq!(super::intersection_capacity_for_instances(1), 32);
        assert_eq!(super::intersection_capacity_for_instances(32), 1_024);
        assert_eq!(super::intersection_capacity_for_instances(513), 32_768);
    }
}

pub struct GaussianRenderer {
    composite_bind_group_layout: re_renderer::GpuBindGroupLayoutHandle,
    render_pipeline_tile: re_renderer::GpuRenderPipelineHandle,
    /// Shared compute bind group layouts + pipelines from `gpu_types` — the
    /// same definitions the standalone renderer uses.
    layouts: GpuBindGroupLayouts,
    pipelines: GpuComputePipelines,
    /// Radix-sort shift uniforms (`shift = pass * 4`) — globally constant.
    shift_ubs: Vec<wgpu::Buffer>,
    batch_cache: Mutex<HashMap<String, CachedComputeResources>>,
}

#[derive(Clone)]
pub struct GaussianDrawData {
    // One logical batch per logged splat entity.
    batches: Vec<GaussianBatch>,
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

/// Per-entity cached GPU state: buffers, bind groups, and prebuilt radix
/// sorts.  Buffers grow as needed (never shrink); bind groups and sorts are
/// rebuilt only by `refresh_compute_bind_groups` when a buffer is replaced.
struct CachedComputeResources {
    buffers: ComputeBuffers,
    /// Bind groups for the non-sort pipeline stages (shared wiring with the
    /// standalone renderer via `gpu_types::create_pipeline_bind_groups`).
    bind_groups: PipelineBindGroups,
    composite_bind_group: Arc<wgpu::BindGroup>,
    /// Prebuilt radix sorts: gid canonicalization, depth argsort, tile-id
    /// sort.  Each holds per-pass bind groups — nothing is created per frame.
    gid_sort: RadixSort,
    depth_sort: RadixSort,
    tile_sort: RadixSort,
    intersection_count_readback_slots: Vec<IntersectionCountReadbackSlot>,
    raster_extent: glam::UVec2,
    /// Per-splat buffer capacity (power-of-two >= cloud.len()).
    splat_capacity: usize,
    block_capacity: usize,
    intersection_capacity: usize,
    tile_offset_capacity: usize,
    sort_workgroup_count: u32,
    depth_sort_workgroup_count: u32,
    /// Cloud build generation the splat attribute buffers were uploaded
    /// from; a mismatch in `prepare_compute_batch` triggers a re-upload.
    cloud_generation: u64,
    /// (total_splats, intersection_capacity, tile_bounds) the scan/map
    /// uniforms were last written for — they're rewritten only on change.
    last_uniform_inputs: (usize, usize, glam::UVec2),
}

/// The five per-splat attribute buffers, uploaded from one cloud build.
/// Replaced wholesale when the entity is re-logged (generation mismatch).
struct SplatAttributeBuffers {
    means: Arc<wgpu::Buffer>,
    quats: Arc<wgpu::Buffer>,
    scales_opacity: Arc<wgpu::Buffer>,
    colors: Arc<wgpu::Buffer>,
    sh_coeffs: Arc<wgpu::Buffer>,
}

fn create_splat_attribute_buffers(
    device: &wgpu::Device,
    label: &str,
    cloud: &RenderGaussianCloud,
) -> SplatAttributeBuffers {
    SplatAttributeBuffers {
        means: Arc::new(create_filled_buffer(
            device,
            &format!("{label}::means"),
            wgpu::BufferUsages::STORAGE,
            &pack_vec3s(cloud.means_world.iter().copied()),
        )),
        quats: Arc::new(create_filled_buffer(
            device,
            &format!("{label}::quats"),
            wgpu::BufferUsages::STORAGE,
            &pack_quats(cloud.quats.iter().copied()),
        )),
        scales_opacity: Arc::new(create_filled_buffer(
            device,
            &format!("{label}::scales_opacity"),
            wgpu::BufferUsages::STORAGE,
            &pack_scales_opacity(cloud),
        )),
        colors: Arc::new(create_filled_buffer(
            device,
            &format!("{label}::colors"),
            wgpu::BufferUsages::STORAGE,
            &pack_rgb(cloud.colors_dc.iter().copied()),
        )),
        sh_coeffs: Arc::new(create_filled_buffer(
            device,
            &format!("{label}::sh_coeffs"),
            wgpu::BufferUsages::STORAGE,
            &pack_sh_coefficients(cloud),
        )),
    }
}

// A few of these buffers are lifetime anchors referenced only through bind
// groups (e.g. the splat attribute buffers after upload).
#[allow(dead_code)]
struct ComputeBuffers {
    project_uniform_buffer: Arc<wgpu::Buffer>,
    scan_uniform_buffer: Arc<wgpu::Buffer>,
    scan_l2_uniform_buffer: Arc<wgpu::Buffer>,
    map_uniform_buffer: Arc<wgpu::Buffer>,
    splat: SplatAttributeBuffers,
    // GPU cull + depth-sort buffers (project_forward outputs / sort ping-pong).
    num_visible_buffer: Arc<wgpu::Buffer>,
    global_from_compact_buffer: Arc<wgpu::Buffer>,
    depth_keys_buffer: Arc<wgpu::Buffer>,
    global_from_compact_alt_buffer: Arc<wgpu::Buffer>,
    depth_keys_alt_buffer: Arc<wgpu::Buffer>,
    projected_tile_splats_buffer: Arc<wgpu::Buffer>,
    tile_hit_counts_buffer: Arc<wgpu::Buffer>,
    tile_hit_offsets_buffer: Arc<wgpu::Buffer>,
    tile_hit_block_offsets_buffer: Arc<wgpu::Buffer>,
    block_local_offsets_buffer: Arc<wgpu::Buffer>,
    block2_offsets_buffer: Arc<wgpu::Buffer>,
    tile_intersection_count_buffer: Arc<wgpu::Buffer>,
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
}

impl ComputeBuffers {
    /// Borrow every buffer the non-sort pipeline stages bind — the single
    /// wiring list handed to `gpu_types::create_pipeline_bind_groups`.
    fn pipeline_buffers(&self) -> PipelineBuffers<'_> {
        PipelineBuffers {
            means: &self.splat.means,
            quats: &self.splat.quats,
            scales_opacity: &self.splat.scales_opacity,
            colors: &self.splat.colors,
            sh_coeffs: &self.splat.sh_coeffs,
            project_ub: &self.project_uniform_buffer,
            scan_ub: &self.scan_uniform_buffer,
            scan_l2_ub: &self.scan_l2_uniform_buffer,
            map_ub: &self.map_uniform_buffer,
            raster_ub: &self.raster_uniform_buffer,
            num_visible: &self.num_visible_buffer,
            global_from_compact: &self.global_from_compact_buffer,
            depth_keys: &self.depth_keys_buffer,
            projected: &self.projected_tile_splats_buffer,
            tile_hit_counts: &self.tile_hit_counts_buffer,
            tile_hit_offsets: &self.tile_hit_offsets_buffer,
            tile_hit_block_offsets: &self.tile_hit_block_offsets_buffer,
            block_local_offsets: &self.block_local_offsets_buffer,
            block2_offsets: &self.block2_offsets_buffer,
            tile_isect_count: &self.tile_intersection_count_buffer,
            num_isect: &self.num_intersections_buffer,
            tile_id_from_isect: &self.tile_id_from_isect_buffer,
            compact_gid_from_isect: &self.compact_gid_from_isect_buffer,
            tile_offsets: &self.tile_offsets_buffer,
            raster_view: &self.raster_texture_view,
        }
    }
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
                    secondary_sort_key: 0.0,
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

    /// Prepare and queue the GPU work for one splat entity.
    ///
    /// `cloud_generation` identifies the cloud build (the visualizer bumps it
    /// whenever it rebuilds the cloud from re-logged data); the renderer
    /// re-uploads the splat attributes when it changes.
    pub fn add_batch(
        &mut self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        cloud: &Arc<RenderGaussianCloud>,
        cloud_generation: u64,
        camera: &CameraApproximation,
    ) {
        let renderer = ctx.renderer::<GaussianRenderer>();
        if cloud.is_empty() {
            return;
        }
        let batch = renderer.prepare_compute_batch(ctx, label, cloud, cloud_generation, camera);
        self.batches.push(batch);
    }
}

impl GaussianRenderer {
    /// (Re)build the three prebuilt radix sorts from the current buffers.
    ///
    /// The gid sort covers all bits of `splat_capacity` (>= any gid this
    /// entity can produce — extra stable passes over all-zero high bits are
    /// exact no-ops).  The tile sort's pass count derives from the current
    /// raster extent, which changes exactly when the tile grid does.
    fn build_radix_sorts(
        &self,
        ctx: &re_renderer::RenderContext,
        buffers: &ComputeBuffers,
        splat_capacity: usize,
        sort_workgroup_count: u32,
        depth_sort_workgroup_count: u32,
        raster_extent: glam::UVec2,
    ) -> (RadixSort, RadixSort, RadixSort) {
        let scratch = RadixSortScratch {
            counts: &buffers.sort_counts_buffer,
            reduced: &buffers.sort_reduced_buffer,
            scan_offsets: &buffers.sort_scan_offsets_buffer,
            scan_block_offsets: &buffers.sort_scan_block_offsets_buffer,
            scan_totals: &buffers.sort_scan_totals_buffer,
        };
        let gid_sort = build_radix_sort(
            &ctx.device,
            &self.layouts,
            &self.shift_ubs,
            &RadixSortBuffers {
                keys_primary: &buffers.global_from_compact_buffer,
                vals_primary: &buffers.depth_keys_buffer,
                keys_alt: &buffers.global_from_compact_alt_buffer,
                vals_alt: &buffers.depth_keys_alt_buffer,
                num_keys: &buffers.num_visible_buffer,
            },
            &scratch,
            depth_sort_workgroup_count,
            gid_sort_passes(splat_capacity),
        );
        let depth_sort = build_radix_sort(
            &ctx.device,
            &self.layouts,
            &self.shift_ubs,
            &RadixSortBuffers {
                keys_primary: &buffers.depth_keys_buffer,
                vals_primary: &buffers.global_from_compact_buffer,
                keys_alt: &buffers.depth_keys_alt_buffer,
                vals_alt: &buffers.global_from_compact_alt_buffer,
                num_keys: &buffers.num_visible_buffer,
            },
            &scratch,
            depth_sort_workgroup_count,
            DEPTH_SORT_PASSES,
        );
        let n_tiles = tile_count(raster_extent / TILE_WIDTH);
        let tile_sort = build_radix_sort(
            &ctx.device,
            &self.layouts,
            &self.shift_ubs,
            &RadixSortBuffers {
                keys_primary: &buffers.tile_id_from_isect_buffer,
                vals_primary: &buffers.compact_gid_from_isect_buffer,
                keys_alt: &buffers.sort_keys_buffer,
                vals_alt: &buffers.sorted_indices_alt_buffer,
                num_keys: &buffers.num_intersections_buffer,
            },
            &scratch,
            sort_workgroup_count,
            tile_sort_passes(n_tiles),
        );
        (gid_sort, depth_sort, tile_sort)
    }

    fn create_batch_resources(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        cloud: &Arc<RenderGaussianCloud>,
        cloud_generation: u64,
    ) -> CachedComputeResources {
        // These buffers live per entity so camera movement reuses GPU allocations instead of
        // rebuilding everything every frame.
        let initial_capacity = next_capacity(cloud.len().max(1));
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
        let scan_l2_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::scan_l2_uniform")),
            size: std::mem::size_of::<gpu_data::ScanUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let map_uniform_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::map_uniform")),
            size: std::mem::size_of::<gpu_data::MapUniformBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let splat = create_splat_attribute_buffers(&ctx.device, label, cloud);

        // GPU cull + depth-sort buffers.  project_forward appends (gid, depth
        // bits) pairs; the radix argsort ping-pongs between primary and alt.
        let num_visible_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::num_visible"),
            std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let global_from_compact_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::global_from_compact"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        let depth_keys_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::depth_keys"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        let global_from_compact_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::global_from_compact_alt"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        let depth_keys_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::depth_keys_alt"),
            initial_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
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
        // Level-2 scan buffers: scan the level-1 block sums so clouds larger
        // than 512 blocks * 512 elements = 262,144 splats scan correctly.
        let block_local_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::block_local_offsets"),
            block_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        let block2_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::block2_offsets"),
            next_block_capacity(block_capacity) * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
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
        let sort_workgroup_count = sort_workgroup_count_for(intersection_capacity) as usize;
        let sort_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_counts"),
            sort_workgroup_count * SORT_BIN_COUNT as usize * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_reduce_wg_count = sort_reduce_workgroup_count(sort_workgroup_count as u32) as usize;
        let sort_reduced_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_reduced"),
            sort_reduce_wg_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_scan_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_offsets"),
            sort_reduce_wg_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_scan_block_capacity = next_block_capacity(sort_reduce_wg_count);
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

        let depth_sort_workgroup_count = sort_workgroup_count_for(initial_capacity) as usize;

        let buffers = ComputeBuffers {
            project_uniform_buffer,
            scan_uniform_buffer,
            scan_l2_uniform_buffer,
            map_uniform_buffer,
            splat,
            num_visible_buffer,
            global_from_compact_buffer,
            depth_keys_buffer,
            global_from_compact_alt_buffer,
            depth_keys_alt_buffer,
            projected_tile_splats_buffer,
            tile_hit_counts_buffer,
            tile_hit_offsets_buffer,
            tile_hit_block_offsets_buffer,
            block_local_offsets_buffer,
            block2_offsets_buffer,
            tile_intersection_count_buffer,
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
        };

        let bind_groups =
            create_pipeline_bind_groups(&ctx.device, &self.layouts, &buffers.pipeline_buffers());
        let composite_bind_group = Arc::new(self.create_composite_bind_group(
            ctx,
            label,
            &buffers.raster_texture_view,
            &buffers.raster_uniform_buffer,
        ));
        let (gid_sort, depth_sort, tile_sort) = self.build_radix_sorts(
            ctx,
            &buffers,
            initial_capacity,
            sort_workgroup_count as u32,
            depth_sort_workgroup_count as u32,
            raster_extent,
        );

        CachedComputeResources {
            buffers,
            bind_groups,
            composite_bind_group,
            gid_sort,
            depth_sort,
            tile_sort,
            intersection_count_readback_slots: create_intersection_count_readback_slots(
                &ctx.device,
                label,
            ),
            raster_extent,
            splat_capacity: initial_capacity,
            block_capacity,
            intersection_capacity,
            tile_offset_capacity,
            sort_workgroup_count: sort_workgroup_count as u32,
            depth_sort_workgroup_count: depth_sort_workgroup_count as u32,
            cloud_generation,
            // Sentinel: forces the first prepare_compute_batch to write the
            // scan/map uniforms.
            last_uniform_inputs: (usize::MAX, 0, glam::UVec2::ZERO),
        }
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

    /// Rebuild every bind group and prebuilt radix sort after one or more
    /// buffers were replaced (splat growth, intersection growth, raster
    /// resize).  Nothing here runs on steady-state frames.
    fn refresh_compute_bind_groups(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
    ) {
        compute.bind_groups = create_pipeline_bind_groups(
            &ctx.device,
            &self.layouts,
            &compute.buffers.pipeline_buffers(),
        );
        compute.composite_bind_group = Arc::new(self.create_composite_bind_group(
            ctx,
            label,
            &compute.buffers.raster_texture_view,
            &compute.buffers.raster_uniform_buffer,
        ));
        let (gid_sort, depth_sort, tile_sort) = self.build_radix_sorts(
            ctx,
            &compute.buffers,
            compute.splat_capacity,
            compute.sort_workgroup_count,
            compute.depth_sort_workgroup_count,
            compute.raster_extent,
        );
        compute.gid_sort = gid_sort;
        compute.depth_sort = depth_sort;
        compute.tile_sort = tile_sort;
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
        compute.buffers.tile_id_from_isect_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_id_from_isect"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.buffers.compact_gid_from_isect_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::compact_gid_from_isect"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.buffers.sort_keys_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_keys"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.buffers.sorted_indices_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sorted_indices_alt"),
            compute.intersection_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        compute.sort_workgroup_count = sort_workgroup_count_for(compute.intersection_capacity);
        let sort_workgroup_count = compute.sort_workgroup_count as usize;
        let sort_reduce_wg_count = sort_reduce_workgroup_count(compute.sort_workgroup_count) as usize;
        compute.buffers.sort_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_counts"),
            sort_workgroup_count * SORT_BIN_COUNT as usize * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.buffers.sort_reduced_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_reduced"),
            sort_reduce_wg_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.buffers.sort_scan_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_offsets"),
            sort_reduce_wg_count * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let sort_scan_block_capacity = next_block_capacity(sort_reduce_wg_count);
        compute.buffers.sort_scan_block_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::sort_scan_block_offsets"),
            sort_scan_block_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        true
    }

    fn process_intersection_count_readbacks(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
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
                        // The totals buffer keeps a legacy DrawIndirect-shaped
                        // layout: word[0] is an unused legacy quad vertex count
                        // (6), word[1] holds the total written by
                        // scan_block_sums_main.
                        let total_intersections = words.get(1).copied().unwrap_or(0) as usize;
                        drop(bytes);
                        slot.buffer.unmap();

                        if std::env::var_os("GSPLAT_FPS_PROBE").is_some() {
                            eprintln!(
                                "[fps-probe] intersections {} / capacity {} (splat_capacity {})",
                                total_intersections,
                                compute.intersection_capacity,
                                compute.splat_capacity
                            );
                        }
                        if total_intersections > compute.intersection_capacity {
                            // Transiently rendered with a truncated intersection
                            // list; the growth below fixes the next frame.
                            re_log::debug!(
                                "tile intersection demand {total_intersections} exceeded \
                                 capacity {} — growing",
                                compute.intersection_capacity
                            );
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
            self.refresh_compute_bind_groups(ctx, label, compute);
        }
    }

    fn ensure_tile_raster_resources(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
        viewport_size_px: glam::Vec2,
    ) {
        let tile_bounds = calc_tile_bounds(viewport_size_px);
        let required_tile_offset_capacity = tile_count(tile_bounds).max(1);
        let raster_extent = calc_raster_extent(viewport_size_px);

        let mut changed = false;
        if required_tile_offset_capacity > compute.tile_offset_capacity {
            compute.tile_offset_capacity = required_tile_offset_capacity;
            compute.buffers.tile_offsets_buffer = Arc::new(create_sized_buffer(
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
            compute.buffers.raster_texture = raster_texture;
            compute.buffers.raster_texture_view = raster_texture_view;
            changed = true;
        }

        if changed {
            self.refresh_compute_bind_groups(ctx, label, compute);
            // The raster uniform only depends on the tile grid, so it only
            // needs rewriting when that changed.
            let raster_uniform = gpu_data::RasterUniformBuffer {
                tile_bounds: tile_bounds.to_array(),
                img_size: raster_extent.to_array(),
            };
            ctx.queue.write_buffer(
                &compute.buffers.raster_uniform_buffer,
                0,
                bytemuck::bytes_of(&raster_uniform),
            );
        }
    }

    /// Grow all per-splat buffers (used when an entity is re-logged with a
    /// larger cloud).
    fn grow_splat_capacity(
        &self,
        ctx: &re_renderer::RenderContext,
        label: &str,
        compute: &mut CachedComputeResources,
        required_capacity: usize,
    ) {
        compute.splat_capacity = next_capacity(required_capacity);
        let splat_capacity = compute.splat_capacity;
        compute.buffers.global_from_compact_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::global_from_compact"),
            splat_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        compute.buffers.depth_keys_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::depth_keys"),
            splat_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        compute.buffers.global_from_compact_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::global_from_compact_alt"),
            splat_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        compute.buffers.depth_keys_alt_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::depth_keys_alt"),
            splat_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        ));
        compute.depth_sort_workgroup_count = sort_workgroup_count_for(splat_capacity);
        compute.buffers.projected_tile_splats_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::projected_tile_splats"),
            splat_capacity * std::mem::size_of::<gpu_data::TileProjectedSplat>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.buffers.tile_hit_counts_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_counts"),
            splat_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        compute.buffers.tile_hit_offsets_buffer = Arc::new(create_sized_buffer(
            &ctx.device,
            &format!("{label}::tile_hit_offsets"),
            splat_capacity * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        let required_block_capacity = next_block_capacity(splat_capacity);
        if required_block_capacity > compute.block_capacity {
            compute.block_capacity = required_block_capacity;
            compute.buffers.tile_hit_block_offsets_buffer = Arc::new(create_sized_buffer(
                &ctx.device,
                &format!("{label}::tile_hit_block_offsets"),
                compute.block_capacity * std::mem::size_of::<u32>(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ));
            compute.buffers.block_local_offsets_buffer = Arc::new(create_sized_buffer(
                &ctx.device,
                &format!("{label}::block_local_offsets"),
                compute.block_capacity * std::mem::size_of::<u32>(),
                wgpu::BufferUsages::STORAGE,
            ));
            compute.buffers.block2_offsets_buffer = Arc::new(create_sized_buffer(
                &ctx.device,
                &format!("{label}::block2_offsets"),
                next_block_capacity(compute.block_capacity) * std::mem::size_of::<u32>(),
                wgpu::BufferUsages::STORAGE,
            ));
        }
        let required_intersection_capacity = intersection_capacity_for_instances(splat_capacity);
        self.ensure_intersection_capacity(ctx, label, compute, required_intersection_capacity);
        self.refresh_compute_bind_groups(ctx, label, compute);
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

        let layouts = create_compute_bind_group_layouts(&ctx.device);
        let pipelines = create_compute_pipelines(&ctx.device, &layouts);
        let shift_ubs = create_sort_shift_uniforms(&ctx.device);

        Self {
            composite_bind_group_layout,
            render_pipeline_tile,
            layouts,
            pipelines,
            shift_ubs,
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
    //! Brush-style GPU-only compute/tile preparation path.
    //!
    //! This module owns the per-frame GPU work: cull the full cloud
    //! (project_forward), depth-argsort the survivors, project them
    //! (project_visible), and run the map/sort/raster stages ending at the
    //! tile raster/composite.  Per-frame CPU work here is uniform writes and
    //! command encoding only.

    use std::sync::Arc;

    use super::*;

    impl GaussianRenderer {
        pub(super) fn prepare_compute_batch(
            &self,
            ctx: &re_renderer::RenderContext,
            label: &str,
            cloud: &Arc<RenderGaussianCloud>,
            cloud_generation: u64,
            camera: &CameraApproximation,
        ) -> GaussianBatch {
            // [fps-probe] period between consecutive prepares == the viewer's
            // effective frame time while the camera moves.  Enabled with
            // GSPLAT_FPS_PROBE=1; mirrors the probe used to measure Brush.
            if std::env::var_os("GSPLAT_FPS_PROBE").is_some() {
                use std::time::Instant;
                static PROBE: Mutex<Option<(Instant, f32, u32)>> = Mutex::new(None);
                let mut probe = PROBE.lock().unwrap();
                let now = Instant::now();
                if let Some((last, ema_ms, n)) = probe.as_mut() {
                    let dt_ms = now.duration_since(*last).as_secs_f32() * 1000.0;
                    *last = now;
                    if dt_ms < 2000.0 {
                        *ema_ms = if *n == 0 { dt_ms } else { *ema_ms * 0.9 + dt_ms * 0.1 };
                        *n += 1;
                        if *n % 30 == 0 {
                            eprintln!(
                                "[fps-probe] prepare period EMA {:.2} ms ({:.1} FPS), n={} viewport={:?}",
                                *ema_ms,
                                1000.0 / *ema_ms,
                                *n,
                                camera.viewport_size_px
                            );
                        }
                    }
                } else {
                    *probe = Some((now, 0.0, 0));
                }
            }
            let mut cache = self.batch_cache.lock().unwrap();
            let compute = cache
                .entry(label.to_owned())
                .or_insert_with(|| self.create_batch_resources(ctx, label, cloud, cloud_generation));

            // Step 1a: re-upload the splat attributes when the entity was
            // re-logged (the visualizer bumps the generation on every cloud
            // rebuild).  The bind-group refresh happens below — either via
            // grow_splat_capacity or explicitly.
            let needs_upload = compute.cloud_generation != cloud_generation;
            if needs_upload {
                compute.cloud_generation = cloud_generation;
                compute.buffers.splat = create_splat_attribute_buffers(&ctx.device, label, cloud);
            }

            let pipelines = &self.pipelines;
            let total_splats = cloud.len().max(1);

            // Step 1b: make sure the persistent per-cloud buffers are large enough for this frame.
            if compute.splat_capacity < total_splats {
                self.grow_splat_capacity(ctx, label, compute, total_splats); // refreshes bind groups
            } else if needs_upload {
                self.refresh_compute_bind_groups(ctx, label, compute);
            }
            self.process_intersection_count_readbacks(ctx, label, compute);
            self.ensure_tile_raster_resources(ctx, label, compute, camera.viewport_size_px);

            // Step 2: write this frame's uniforms.  Only the camera uniform
            // changes on steady-state frames; the scan/map uniforms depend on
            // (total_splats, intersection_capacity, tile_bounds) and are
            // rewritten only when one of those changed.
            let block_count = compaction_block_count(total_splats) as u32;
            let block2_count = compaction_block_count(block_count as usize) as u32;
            let tile_bounds = calc_tile_bounds(camera.viewport_size_px);
            let project_uniform = fill_project_uniform(camera, total_splats, cloud);
            ctx.queue.write_buffer(
                &compute.buffers.project_uniform_buffer,
                0,
                bytemuck::bytes_of(&project_uniform),
            );
            let uniform_inputs = (total_splats, compute.intersection_capacity, tile_bounds);
            if compute.last_uniform_inputs != uniform_inputs {
                compute.last_uniform_inputs = uniform_inputs;
                let scan_uniform = fill_scan_uniform(total_splats);
                // Level-2 scan: scans the level-1 block sums (block_count entries).
                let scan_l2_uniform = fill_scan_uniform(block_count as usize);
                let map_uniform =
                    fill_map_uniform(total_splats, compute.intersection_capacity, tile_bounds);
                ctx.queue.write_buffer(
                    &compute.buffers.scan_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&scan_uniform),
                );
                ctx.queue.write_buffer(
                    &compute.buffers.scan_l2_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&scan_l2_uniform),
                );
                ctx.queue.write_buffer(
                    &compute.buffers.map_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&map_uniform),
                );
            }

            let (project_x, project_y) =
                dispatch_grid_1d(total_splats as u32, PROJECT_WORKGROUP_SIZE);
            {
                // Step 3: cull the FULL cloud on the GPU, compacting
                // (global_gid, depth bits) pairs via atomicAdd(num_visible).
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                encoder
                    .get()
                    .clear_buffer(&compute.buffers.num_visible_buffer, 0, None);
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gaussian_project_forward"),
                            timestamp_writes: None,
                        });
                compute_pass.set_pipeline(&pipelines.project_forward);
                compute_pass.set_bind_group(0, &compute.bind_groups.project_forward, &[]);
                compute_pass.dispatch_workgroups(project_x, project_y, 1);
            }

            // Step 4a: canonicalize the compact order by gid.
            // project_forward's atomicAdd compaction is racy; sorting by gid
            // first makes depth-bit ties resolve in ascending-gid order so the
            // render is deterministic.
            // Step 4b: depth argsort, ascending by f32 depth bits — compact
            // order becomes front-to-back.  Even pass count (8), so the sorted
            // pairs land back in the primary buffers.  Both sorts use prebuilt
            // bind groups and share one compute pass — nothing is created here.
            {
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gaussian_gid_depth_sort"),
                            timestamp_writes: None,
                        });
                compute.gid_sort.encode(&mut compute_pass, pipelines);
                compute.depth_sort.encode(&mut compute_pass, pipelines);
            }

            {
                // Step 5: project visible splats in depth order, prefix-scan
                // their tile hit counts (3-level scan), and map each splat to
                // its tile coverage.
                let (scan_x, scan_y) = dispatch_grid_1d(block_count, 1);
                let (scan_l2_x, scan_l2_y) = dispatch_grid_1d(block2_count, 1);
                let (compose_x, compose_y) = dispatch_grid_1d(block_count, SORT_WORKGROUP_SIZE);

                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(label),
                            timestamp_writes: None,
                        });

                compute_pass.set_pipeline(&pipelines.project_visible);
                compute_pass.set_bind_group(0, &compute.bind_groups.project_visible, &[]);
                compute_pass.dispatch_workgroups(project_x, project_y, 1);

                compute_pass.set_pipeline(&pipelines.scan_blocks);
                compute_pass.set_bind_group(0, &compute.bind_groups.scan, &[]);
                compute_pass.dispatch_workgroups(scan_x, scan_y, 1);

                compute_pass.set_pipeline(&pipelines.scan_blocks);
                compute_pass.set_bind_group(0, &compute.bind_groups.scan_l2, &[]);
                compute_pass.dispatch_workgroups(scan_l2_x, scan_l2_y, 1);

                compute_pass.set_pipeline(&pipelines.scan_block_sums);
                compute_pass.set_bind_group(
                    0,
                    &compute.bind_groups.scan_block_sums,
                    &[],
                );
                compute_pass.dispatch_workgroups(1, 1, 1);

                compute_pass.set_pipeline(&pipelines.sort_scan_compose);
                compute_pass.set_bind_group(
                    0,
                    &compute.bind_groups.scan_compose,
                    &[],
                );
                compute_pass.dispatch_workgroups(compose_x, compose_y, 1);

                compute_pass.set_pipeline(&pipelines.map_intersections);
                compute_pass.set_bind_group(0, &compute.bind_groups.map, &[]);
                compute_pass.dispatch_workgroups(project_x, project_y, 1);

                compute_pass.set_pipeline(&pipelines.clamp_intersection_count);
                compute_pass.set_bind_group(0, &compute.bind_groups.map, &[]);
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
                    &compute.buffers.tile_intersection_count_buffer,
                    0,
                    &slot.buffer,
                    0,
                    std::mem::size_of::<gpu_data::DrawIndirectArgs>() as u64,
                );
                slot.state = IntersectionCountReadbackState::CopySubmitted;
            }

            // Step 6: radix-sort the tile intersections so each tile can
            // consume a contiguous intersection range during raster.
            {
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                let mut compute_pass =
                    encoder
                        .get()
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("gaussian_tile_sort"),
                            timestamp_writes: None,
                        });
                compute.tile_sort.encode(&mut compute_pass, pipelines);
            }

            if compute.tile_sort.num_passes() % 2 == 1 {
                let bytes = (compute.intersection_capacity * std::mem::size_of::<u32>()) as u64;
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                encoder.get().copy_buffer_to_buffer(
                    &compute.buffers.sort_keys_buffer,
                    0,
                    &compute.buffers.tile_id_from_isect_buffer,
                    0,
                    bytes,
                );
                encoder.get().copy_buffer_to_buffer(
                    &compute.buffers.sorted_indices_alt_buffer,
                    0,
                    &compute.buffers.compact_gid_from_isect_buffer,
                    0,
                    bytes,
                );
            }

            {
                // Step 7: turn sorted tile intersections into per-tile ranges, raster each tile, then
                // queue a fullscreen composite back into the normal Rerun draw graph.
                let mut encoder = ctx.active_frame.before_view_builder_encoder.lock();
                encoder
                    .get()
                    .clear_buffer(&compute.buffers.tile_offsets_buffer, 0, None);
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
                compute_pass.set_pipeline(&pipelines.tile_offsets);
                compute_pass.set_bind_group(0, &compute.bind_groups.tile_offsets, &[]);
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
                compute_pass.set_pipeline(&pipelines.rasterize);
                compute_pass.set_bind_group(0, &compute.bind_groups.rasterize, &[]);
                compute_pass.dispatch_workgroups(raster_x, raster_y, 1);
            }

            GaussianBatch {
                payload: GaussianBatchPayload {
                    composite_bind_group: compute.composite_bind_group.clone(),
                },
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

    // Re-export shared GPU types from gsplat_core so existing `gpu_data::*`
    // references throughout this file continue to work unchanged.
    pub use crate::gsplat_core::gpu_types::{
        DrawIndirectArgs, MapUniformBuffer, ProjectUniformBuffer, RasterUniformBuffer,
        ScanUniformBuffer, TileProjectedSplat,
    };

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
