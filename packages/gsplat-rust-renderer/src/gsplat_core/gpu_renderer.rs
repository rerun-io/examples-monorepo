//! Standalone GPU renderer for Gaussian splats — no Rerun dependency.
//!
//! Implements the same GPU-only compute pipeline as `gaussian_renderer.rs`
//! (project_forward cull → depth argsort → project_visible → map intersections
//! → tile radix sort → tile offsets → rasterize) but uses raw `wgpu` directly
//! instead of `re_renderer`.
//!
//! The output is read back from the GPU as an RGBA8 image buffer.
//! This module follows the [Brush](https://github.com/ArthurBrussee/brush)
//! approach: pure GPU rendering — the CPU only writes uniforms and encodes
//! commands; culling and depth sorting happen on the GPU.

use std::sync::Arc;

use super::gpu_context::GpuContext;
use super::gpu_types::*;
use super::types::{CameraApproximation, RenderGaussianCloud, RenderOutput};

// ═══════════════════════════════════════════════════════════════════════════════
// Renderer (holds reusable pipelines)
// ═══════════════════════════════════════════════════════════════════════════════

/// Holds all compute pipelines and bind group layouts.
/// Created once from the shared definitions in `gpu_types`, reused across frames.
pub struct GpuRenderer {
    pub layouts: GpuBindGroupLayouts,
    pub pipelines: GpuComputePipelines,
}

impl GpuRenderer {
    /// Create all compute pipelines from the shared definitions in `gpu_types`.
    pub fn new(device: &wgpu::Device) -> Self {
        let layouts = create_compute_bind_group_layouts(device);
        let pipelines = create_compute_pipelines(device, &layouts);
        Self { layouts, pipelines }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU radix argsort (count-buffer driven, brush v0.3.0 pattern)
// ═══════════════════════════════════════════════════════════════════════════════

/// Buffers for one GPU radix argsort over `(key, value)` pairs.
///
/// The number of keys is read on the GPU from `num_keys[0]` — dispatches are
/// capacity-sized and early-exit, so no CPU readback or indirect dispatch is
/// needed.  Used twice per frame: depth argsort (keys = f32 depth bits,
/// values = global gids) and tile-id sort (keys = tile ids, values =
/// compact gids).
struct RadixSortBuffers<'a> {
    keys_primary: &'a wgpu::Buffer,
    vals_primary: &'a wgpu::Buffer,
    keys_alt: &'a wgpu::Buffer,
    vals_alt: &'a wgpu::Buffer,
    /// GPU buffer whose first u32 is the number of keys to sort.
    num_keys: &'a wgpu::Buffer,
    counts: &'a wgpu::Buffer,
    reduced: &'a wgpu::Buffer,
    scan_offsets: &'a wgpu::Buffer,
    scan_block_offsets: &'a wgpu::Buffer,
    scan_totals: &'a wgpu::Buffer,
}

/// Encode `num_passes` 4-bit radix sort passes.  With an even pass count the
/// sorted data lands back in the primary buffers; for an odd count the caller
/// must copy back from the alt buffers.
#[allow(clippy::too_many_arguments)]
fn encode_radix_sort(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    renderer: &GpuRenderer,
    buffers: &RadixSortBuffers<'_>,
    sort_wg_count: u32,
    num_passes: u32,
) {
    let sort_reduce_wg_count: u32 = sort_reduce_workgroup_count(sort_wg_count.max(1));
    let (sort_count_x, sort_count_y) = dispatch_grid_for_workgroups(sort_wg_count.max(1));
    let (sort_reduce_x, sort_reduce_y) = dispatch_grid_for_workgroups(sort_reduce_wg_count);

    let reduced_total: u32 = sort_reduce_wg_count;
    let reduced_block_count: u32 = compaction_block_count(reduced_total as usize) as u32;
    let scan_sort_uniform: ScanUniformBuffer = fill_scan_uniform(reduced_total as usize);
    let scan_sort_ub: wgpu::Buffer = create_filled_buffer(
        device,
        "scan_sort_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&scan_sort_uniform),
    );

    for pass_index in 0..num_passes {
        let sort_uniform: SortUniformBuffer = SortUniformBuffer {
            shift: pass_index * SORT_BITS_PER_PASS,
            total_keys_unused: 0,
            _pad: [0; 2],
        };
        let sort_ub: wgpu::Buffer = create_filled_buffer(
            device,
            "sort_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&sort_uniform),
        );

        let use_primary: bool = pass_index % 2 == 0;
        let (src_keys, src_vals, dst_keys, dst_vals) = if use_primary {
            (
                buffers.keys_primary,
                buffers.vals_primary,
                buffers.keys_alt,
                buffers.vals_alt,
            )
        } else {
            (
                buffers.keys_alt,
                buffers.vals_alt,
                buffers.keys_primary,
                buffers.vals_primary,
            )
        };

        let count_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_count_bg"),
            layout: &renderer.layouts.sort_count,
            entries: &[
                storage_buffer_entry(0, &sort_ub),
                storage_buffer_entry(1, src_keys),
                storage_buffer_entry(2, buffers.counts),
                storage_buffer_entry(6, buffers.num_keys),
            ],
        });
        let reduce_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_reduce_bg"),
            layout: &renderer.layouts.sort_reduce,
            entries: &[
                storage_buffer_entry(0, &sort_ub),
                storage_buffer_entry(1, buffers.counts),
                storage_buffer_entry(2, buffers.reduced),
                storage_buffer_entry(6, buffers.num_keys),
            ],
        });
        let scan_blocks_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sort_scan_blocks_bg"),
                layout: &renderer.layouts.scan,
                entries: &[
                    storage_buffer_entry(16, buffers.reduced),
                    storage_buffer_entry(17, buffers.scan_offsets),
                    storage_buffer_entry(18, buffers.scan_block_offsets),
                    storage_buffer_entry(19, &scan_sort_ub),
                ],
            });
        let scan_block_sums_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sort_scan_block_sums_bg"),
                layout: &renderer.layouts.scan_block_sums,
                entries: &[
                    storage_buffer_entry(24, buffers.scan_block_offsets),
                    storage_buffer_entry(25, buffers.scan_totals),
                    storage_buffer_entry(26, &scan_sort_ub),
                ],
            });
        let compose_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_compose_bg"),
            layout: &renderer.layouts.sort_scan_compose,
            entries: &[
                storage_buffer_entry(8, buffers.scan_offsets),
                storage_buffer_entry(9, buffers.scan_block_offsets),
                storage_buffer_entry(10, buffers.reduced),
                storage_buffer_entry(11, &scan_sort_ub),
            ],
        });
        let scan_add_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_add_bg"),
            layout: &renderer.layouts.sort_scan_add,
            entries: &[
                storage_buffer_entry(0, &sort_ub),
                storage_buffer_entry(1, buffers.reduced),
                storage_buffer_entry(2, buffers.counts),
                storage_buffer_entry(6, buffers.num_keys),
            ],
        });
        let scatter_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scatter_bg"),
            layout: &renderer.layouts.sort_scatter,
            entries: &[
                storage_buffer_entry(0, &sort_ub),
                storage_buffer_entry(1, src_keys),
                storage_buffer_entry(2, src_vals),
                storage_buffer_entry(3, buffers.counts),
                storage_buffer_entry(4, dst_keys),
                storage_buffer_entry(5, dst_vals),
                storage_buffer_entry(6, buffers.num_keys),
            ],
        });

        let (ssb_x, ssb_y) = dispatch_grid_1d(reduced_block_count, 1);
        let (ssc_x, ssc_y) = dispatch_grid_1d(reduced_total, SORT_WORKGROUP_SIZE);

        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_sort"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&renderer.pipelines.sort_count);
        pass.set_bind_group(0, &count_bg, &[]);
        pass.dispatch_workgroups(sort_count_x, sort_count_y, 1);

        pass.set_pipeline(&renderer.pipelines.sort_reduce);
        pass.set_bind_group(0, &reduce_bg, &[]);
        pass.dispatch_workgroups(sort_reduce_x, sort_reduce_y, 1);

        pass.set_pipeline(&renderer.pipelines.scan_blocks);
        pass.set_bind_group(0, &scan_blocks_bg, &[]);
        pass.dispatch_workgroups(ssb_x, ssb_y, 1);

        pass.set_pipeline(&renderer.pipelines.scan_block_sums);
        pass.set_bind_group(0, &scan_block_sums_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        pass.set_pipeline(&renderer.pipelines.sort_scan_compose);
        pass.set_bind_group(0, &compose_bg, &[]);
        pass.dispatch_workgroups(ssc_x, ssc_y, 1);

        pass.set_pipeline(&renderer.pipelines.sort_scan_add);
        pass.set_bind_group(0, &scan_add_bg, &[]);
        pass.dispatch_workgroups(sort_reduce_x, sort_reduce_y, 1);

        pass.set_pipeline(&renderer.pipelines.sort_scatter);
        pass.set_bind_group(0, &scatter_bg, &[]);
        pass.dispatch_workgroups(sort_count_x, sort_count_y, 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main render function
// ═══════════════════════════════════════════════════════════════════════════════

/// Render a Gaussian cloud from a camera viewpoint using the GPU compute pipeline.
///
/// This is the standalone equivalent of the Rerun viewer's compute tile path,
/// following Brush's approach of pure GPU rendering: the CPU only uploads
/// data, writes uniforms, and encodes commands.
///
/// # Pipeline stages
///
/// 1. **project_forward**: cull the FULL cloud on the GPU, compact
///    `(global_gid, depth)` pairs via `atomicAdd(num_visible)`
/// 2. **Depth argsort**: GPU radix sort ascending by f32 depth bits — compact
///    order becomes front-to-back
/// 3. **project_visible**: exact 3D→2D projection + SH evaluation + tile coverage
/// 4. **Prefix scan** over per-splat tile hit counts (3-level, supports >262k splats)
/// 5. **Map intersections**: scatter (tile_id, compact_gid) pairs
/// 6. **Tile radix sort**: sort intersections by tile_id
/// 7. **Tile offsets** + **rasterize**: per-pixel alpha blending in 16×16 tiles
/// 8. **Readback**: copy raster texture to CPU as RGBA8
#[allow(clippy::too_many_lines)]
pub fn gpu_render(
    ctx: &GpuContext,
    renderer: &GpuRenderer,
    cloud: &RenderGaussianCloud,
    camera: &CameraApproximation,
    _background: [f32; 3],
) -> RenderOutput {
    let device: &wgpu::Device = &ctx.device;
    let queue: &wgpu::Queue = &ctx.queue;

    let width: u32 = camera.viewport_size_px.x.max(1.0) as u32;
    let height: u32 = camera.viewport_size_px.y.max(1.0) as u32;

    if cloud.is_empty() {
        return RenderOutput {
            pixels: vec![[0.0; 4]; (width as usize) * (height as usize)],
            width,
            height,
        };
    }
    let total_splats: usize = cloud.len();

    // ── Upload splat data to GPU ─────────────────────────────────────────
    let storage_usage: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
    let means_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "means",
        storage_usage,
        &pack_vec3s(cloud.means_world.iter().copied()),
    ));
    let quats_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "quats",
        storage_usage,
        &pack_quats(cloud.quats.iter().copied()),
    ));
    let scales_opacity_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "scales_opacity",
        storage_usage,
        &pack_scales_opacity(cloud),
    ));
    let colors_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "colors",
        storage_usage,
        &pack_rgb(cloud.colors_dc.iter().copied()),
    ));
    let sh_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "sh_coeffs",
        storage_usage,
        &pack_sh_coefficients(cloud),
    ));

    // ── Sizing ───────────────────────────────────────────────────────────
    let tile_bounds: glam::UVec2 = calc_tile_bounds(camera.viewport_size_px);
    let raster_extent: glam::UVec2 = calc_raster_extent(camera.viewport_size_px);
    let n_tiles: usize = tile_count(tile_bounds);
    let instance_capacity: usize = next_capacity(total_splats);
    let block_capacity: usize = next_block_capacity(total_splats);
    let block2_capacity: usize = next_block_capacity(block_capacity);
    let isect_capacity: usize = intersection_capacity_for_instances(instance_capacity);
    let tile_sort_wg_count: u32 = (isect_capacity as u32).div_ceil(SORT_BLOCK_SIZE).max(1);
    let depth_sort_wg_count: u32 = (instance_capacity as u32).div_ceil(SORT_BLOCK_SIZE).max(1);
    let sort_reduce_wg_count: u32 = sort_reduce_workgroup_count(tile_sort_wg_count);
    let sort_scan_block_capacity: usize = (sort_reduce_wg_count as usize)
        .div_ceil(COMPACTION_BLOCK_SIZE as usize)
        .next_power_of_two()
        .max(1);

    // ── Uniform buffers ──────────────────────────────────────────────────
    let block_count: u32 = compaction_block_count(total_splats) as u32;
    let block2_count: u32 = compaction_block_count(block_count as usize) as u32;
    let project_uniform: ProjectUniformBuffer = fill_project_uniform(camera, total_splats, cloud);
    let scan_uniform: ScanUniformBuffer = fill_scan_uniform(total_splats);
    // Level-2 scan: scans the level-1 block sums (block_count entries).
    let scan_uniform_l2: ScanUniformBuffer = fill_scan_uniform(block_count as usize);
    let map_uniform: MapUniformBuffer = fill_map_uniform(total_splats, isect_capacity, tile_bounds);
    let raster_uniform: RasterUniformBuffer = RasterUniformBuffer {
        tile_bounds: [tile_bounds.x, tile_bounds.y],
        img_size: [raster_extent.x, raster_extent.y],
    };

    let project_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "project_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&project_uniform),
    ));
    let scan_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "scan_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&scan_uniform),
    ));
    let scan_l2_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "scan_l2_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&scan_uniform_l2),
    ));
    let map_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "map_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&map_uniform),
    ));
    let raster_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "raster_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&raster_uniform),
    ));

    // ── Intermediate buffers ─────────────────────────────────────────────
    let sz = |n: usize, elem: usize| n * elem;
    let su32 = std::mem::size_of::<u32>();
    let s_splat = std::mem::size_of::<TileProjectedSplat>();

    // GPU cull + depth sort buffers (project_forward outputs).
    let num_visible_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "num_visible",
        su32,
        storage_usage | wgpu::BufferUsages::COPY_DST,
    ));
    let global_from_compact_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "global_from_compact",
        sz(instance_capacity, su32),
        storage_usage,
    ));
    let depth_keys_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "depth_keys",
        sz(instance_capacity, su32),
        storage_usage,
    ));
    let global_from_compact_alt_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "global_from_compact_alt",
        sz(instance_capacity, su32),
        storage_usage,
    ));
    let depth_keys_alt_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "depth_keys_alt",
        sz(instance_capacity, su32),
        storage_usage,
    ));

    let projected_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "projected",
        sz(instance_capacity, s_splat),
        storage_usage,
    ));
    let tile_hit_counts_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "tile_hit_counts",
        sz(instance_capacity, su32),
        storage_usage | wgpu::BufferUsages::COPY_DST,
    ));
    let tile_hit_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "tile_hit_offsets",
        sz(instance_capacity, su32),
        storage_usage,
    ));
    let tile_hit_block_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "tile_hit_block_offsets",
        sz(block_capacity, su32),
        storage_usage,
    ));
    let block_local_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "block_local_offsets",
        sz(block_capacity, su32),
        storage_usage,
    ));
    let block2_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "block2_offsets",
        sz(block2_capacity, su32),
        storage_usage,
    ));
    let tile_isect_count_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "tile_isect_count",
        storage_usage | wgpu::BufferUsages::COPY_SRC,
        &[DrawIndirectArgs {
            vertex_count: 0,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }],
    ));
    let num_isect_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "num_isect",
        su32,
        storage_usage | wgpu::BufferUsages::COPY_DST,
    ));
    let tile_id_from_isect_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "tile_id_from_isect",
        sz(isect_capacity, su32),
        storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    ));
    let compact_gid_from_isect_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "compact_gid_from_isect",
        sz(isect_capacity, su32),
        storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    ));
    let sort_keys_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_keys",
        sz(isect_capacity, su32),
        storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    ));
    let sorted_indices_alt_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sorted_indices_alt",
        sz(isect_capacity, su32),
        storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    ));
    // Sort scratch buffers are sized for the (larger) tile sort and reused by
    // the depth sort — the two sorts run sequentially in the same encoder.
    let sort_counts_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_counts",
        sz(tile_sort_wg_count as usize * SORT_BIN_COUNT as usize, su32),
        storage_usage,
    ));
    let sort_reduced_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_reduced",
        sz(sort_reduce_wg_count as usize, su32),
        storage_usage,
    ));
    let sort_scan_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_scan_offsets",
        sz(sort_reduce_wg_count as usize, su32),
        storage_usage,
    ));
    let sort_scan_block_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_scan_block_offsets",
        sz(sort_scan_block_capacity, su32),
        storage_usage,
    ));
    let sort_scan_totals_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_scan_totals",
        sz(SORT_BIN_COUNT as usize, su32),
        storage_usage | wgpu::BufferUsages::INDIRECT,
    ));
    let tile_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "tile_offsets",
        sz(n_tiles.max(1) * 2, su32),
        storage_usage | wgpu::BufferUsages::COPY_DST,
    ));

    // ── Raster texture ───────────────────────────────────────────────────
    let (raster_texture, raster_view) = create_raster_texture(
        device,
        "raster_texture",
        raster_extent,
        wgpu::TextureUsages::COPY_SRC, // readback
    );

    // ── Bind groups ──────────────────────────────────────────────────────
    // Helper macro to create a uniform buffer entry (avoids closure lifetime issues).
    macro_rules! ube {
        ($binding:expr, $buf:expr) => {
            wgpu::BindGroupEntry {
                binding: $binding,
                resource: $buf.as_entire_binding(),
            }
        };
    }

    let project_forward_bg: wgpu::BindGroup =
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project_forward_bg"),
            layout: &renderer.layouts.project_forward,
            entries: &[
                storage_buffer_entry(0, &means_buf),
                storage_buffer_entry(1, &quats_buf),
                storage_buffer_entry(2, &scales_opacity_buf),
                ube!(8, &project_ub),
                storage_buffer_entry(12, &global_from_compact_buf),
                storage_buffer_entry(13, &depth_keys_buf),
                storage_buffer_entry(14, &num_visible_buf),
            ],
        });
    let project_visible_bg: wgpu::BindGroup =
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project_visible_bg"),
            layout: &renderer.layouts.project_visible,
            entries: &[
                storage_buffer_entry(0, &means_buf),
                storage_buffer_entry(1, &quats_buf),
                storage_buffer_entry(2, &scales_opacity_buf),
                storage_buffer_entry(3, &colors_buf),
                storage_buffer_entry(4, &sh_buf),
                storage_buffer_entry(5, &global_from_compact_buf),
                ube!(8, &project_ub),
                storage_buffer_entry(9, &projected_buf),
                storage_buffer_entry(10, &tile_hit_counts_buf),
                storage_buffer_entry(11, &num_visible_buf),
            ],
        });
    let scan_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scan_bg"),
        layout: &renderer.layouts.scan,
        entries: &[
            storage_buffer_entry(16, &tile_hit_counts_buf), // scan input: per-splat tile hit counts
            storage_buffer_entry(17, &tile_hit_offsets_buf), // scan output: prefix-sum offsets
            storage_buffer_entry(18, &tile_hit_block_offsets_buf),
            ube!(19, &scan_ub),
        ],
    });
    // Level 2: scan the level-1 block sums so clouds larger than
    // 512 blocks * 512 elements = 262,144 splats scan correctly.
    let scan_l2_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scan_l2_bg"),
        layout: &renderer.layouts.scan,
        entries: &[
            storage_buffer_entry(16, &tile_hit_block_offsets_buf),
            storage_buffer_entry(17, &block_local_offsets_buf),
            storage_buffer_entry(18, &block2_offsets_buf),
            ube!(19, &scan_l2_ub),
        ],
    });
    let scan_block_sums_bg: wgpu::BindGroup =
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_block_sums_bg"),
            layout: &renderer.layouts.scan_block_sums,
            entries: &[
                storage_buffer_entry(24, &block2_offsets_buf),
                storage_buffer_entry(25, &tile_isect_count_buf), // total intersection count
                ube!(26, &scan_l2_ub),
            ],
        });
    // Compose the two scan levels back into flat per-block offsets.
    let scan_compose_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scan_compose_bg"),
        layout: &renderer.layouts.sort_scan_compose,
        entries: &[
            storage_buffer_entry(8, &block_local_offsets_buf),
            storage_buffer_entry(9, &block2_offsets_buf),
            storage_buffer_entry(10, &tile_hit_block_offsets_buf),
            ube!(11, &scan_l2_ub),
        ],
    });
    let map_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("map_bg"),
        layout: &renderer.layouts.map,
        entries: &[
            storage_buffer_entry(0, &projected_buf),
            storage_buffer_entry(1, &tile_hit_offsets_buf),
            storage_buffer_entry(2, &tile_hit_counts_buf),
            storage_buffer_entry(3, &tile_hit_block_offsets_buf),
            storage_buffer_entry(4, &tile_id_from_isect_buf),
            storage_buffer_entry(5, &compact_gid_from_isect_buf),
            ube!(6, &map_ub),
            storage_buffer_entry(7, &tile_isect_count_buf),
            storage_buffer_entry(8, &num_isect_buf),
        ],
    });
    let tile_offsets_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_offsets_bg"),
        layout: &renderer.layouts.tile_offsets,
        entries: &[
            storage_buffer_entry(0, &tile_id_from_isect_buf),
            storage_buffer_entry(1, &tile_offsets_buf),
            storage_buffer_entry(2, &num_isect_buf),
        ],
    });
    let rasterize_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rasterize_bg"),
        layout: &renderer.layouts.rasterize,
        entries: &[
            storage_buffer_entry(0, &compact_gid_from_isect_buf),
            storage_buffer_entry(1, &tile_offsets_buf),
            storage_buffer_entry(2, &projected_buf),
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&raster_view),
            },
            ube!(4, &raster_ub),
        ],
    });

    // ── GPU dispatch ─────────────────────────────────────────────────────
    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gsplat_render"),
        });

    // The visible counter accumulates atomically — reset it each frame.
    encoder.clear_buffer(&num_visible_buf, 0, None);

    let (px, py) = dispatch_grid_1d(total_splats as u32, PROJECT_WORKGROUP_SIZE);

    // Stage 1: cull the full cloud, compact (gid, depth) pairs.
    {
        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("project_forward"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&renderer.pipelines.project_forward);
        pass.set_bind_group(0, &project_forward_bg, &[]);
        pass.dispatch_workgroups(px, py, 1);
    }

    // Stage 2a: canonicalize the compact order by gid.  project_forward's
    // atomicAdd compaction is racy; sorting by gid first makes depth-bit ties
    // resolve in ascending-gid order so the render is deterministic.
    encode_radix_sort(
        device,
        &mut encoder,
        renderer,
        &RadixSortBuffers {
            keys_primary: &global_from_compact_buf,
            vals_primary: &depth_keys_buf,
            keys_alt: &global_from_compact_alt_buf,
            vals_alt: &depth_keys_alt_buf,
            num_keys: &num_visible_buf,
            counts: &sort_counts_buf,
            reduced: &sort_reduced_buf,
            scan_offsets: &sort_scan_offsets_buf,
            scan_block_offsets: &sort_scan_block_offsets_buf,
            scan_totals: &sort_scan_totals_buf,
        },
        depth_sort_wg_count,
        gid_sort_passes(total_splats),
    );

    // Stage 2b: depth argsort (ascending f32 bits => front-to-back compact
    // order).  8 passes (even), so results land back in the primary buffers.
    encode_radix_sort(
        device,
        &mut encoder,
        renderer,
        &RadixSortBuffers {
            keys_primary: &depth_keys_buf,
            vals_primary: &global_from_compact_buf,
            keys_alt: &depth_keys_alt_buf,
            vals_alt: &global_from_compact_alt_buf,
            num_keys: &num_visible_buf,
            counts: &sort_counts_buf,
            reduced: &sort_reduced_buf,
            scan_offsets: &sort_scan_offsets_buf,
            scan_block_offsets: &sort_scan_block_offsets_buf,
            scan_totals: &sort_scan_totals_buf,
        },
        depth_sort_wg_count,
        DEPTH_SORT_PASSES,
    );

    // Stage 3-5: project visible splats, prefix-scan tile counts, map intersections.
    {
        let (s1x, s1y) = dispatch_grid_1d(block_count, 1);
        let (s2x, s2y) = dispatch_grid_1d(block2_count, 1);
        let (cx, cy) = dispatch_grid_1d(block_count, SORT_WORKGROUP_SIZE);

        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("project_visible_scan_map"),
                timestamp_writes: None,
            });

        pass.set_pipeline(&renderer.pipelines.project_visible);
        pass.set_bind_group(0, &project_visible_bg, &[]);
        pass.dispatch_workgroups(px, py, 1);

        pass.set_pipeline(&renderer.pipelines.scan_blocks);
        pass.set_bind_group(0, &scan_bg, &[]);
        pass.dispatch_workgroups(s1x, s1y, 1);

        pass.set_pipeline(&renderer.pipelines.scan_blocks);
        pass.set_bind_group(0, &scan_l2_bg, &[]);
        pass.dispatch_workgroups(s2x, s2y, 1);

        pass.set_pipeline(&renderer.pipelines.scan_block_sums);
        pass.set_bind_group(0, &scan_block_sums_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        pass.set_pipeline(&renderer.pipelines.sort_scan_compose);
        pass.set_bind_group(0, &scan_compose_bg, &[]);
        pass.dispatch_workgroups(cx, cy, 1);

        pass.set_pipeline(&renderer.pipelines.map_intersections);
        pass.set_bind_group(0, &map_bg, &[]);
        pass.dispatch_workgroups(px, py, 1);

        pass.set_pipeline(&renderer.pipelines.clamp_intersection_count);
        pass.set_bind_group(0, &map_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // Stage 6: radix sort intersections by tile ID.
    let num_tile_sort_passes: u32 = tile_sort_passes(n_tiles);
    encode_radix_sort(
        device,
        &mut encoder,
        renderer,
        &RadixSortBuffers {
            keys_primary: &tile_id_from_isect_buf,
            vals_primary: &compact_gid_from_isect_buf,
            keys_alt: &sort_keys_buf,
            vals_alt: &sorted_indices_alt_buf,
            num_keys: &num_isect_buf,
            counts: &sort_counts_buf,
            reduced: &sort_reduced_buf,
            scan_offsets: &sort_scan_offsets_buf,
            scan_block_offsets: &sort_scan_block_offsets_buf,
            scan_totals: &sort_scan_totals_buf,
        },
        tile_sort_wg_count,
        num_tile_sort_passes,
    );

    // Copy back to primary if odd number of passes.
    if num_tile_sort_passes % 2 == 1 {
        let bytes: u64 = (isect_capacity * std::mem::size_of::<u32>()) as u64;
        encoder.copy_buffer_to_buffer(&sort_keys_buf, 0, &tile_id_from_isect_buf, 0, bytes);
        encoder.copy_buffer_to_buffer(
            &sorted_indices_alt_buf,
            0,
            &compact_gid_from_isect_buf,
            0,
            bytes,
        );
    }

    // Stage 7: tile offsets.
    encoder.clear_buffer(&tile_offsets_buf, 0, None);
    {
        let tile_offset_elements: u32 = isect_capacity.max(1) as u32;
        let (tox, toy) = dispatch_grid_1d(
            tile_offset_elements,
            TILE_OFFSET_WORKGROUP_SIZE * TILE_OFFSET_CHECKS_PER_ITER,
        );
        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_offsets"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&renderer.pipelines.tile_offsets);
        pass.set_bind_group(0, &tile_offsets_bg, &[]);
        pass.dispatch_workgroups(tox, toy, 1);
    }

    // Stage 8: rasterize.
    {
        let tile_workgroups: u32 = n_tiles.max(1) as u32;
        let (rx, ry) = dispatch_grid_for_workgroups(tile_workgroups);
        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&renderer.pipelines.rasterize);
        pass.set_bind_group(0, &rasterize_bg, &[]);
        pass.dispatch_workgroups(rx, ry, 1);
    }

    // ── Readback raster texture ──────────────────────────────────────────
    let bytes_per_pixel: u32 = 4; // Rgba8Unorm
    let padded_bytes_per_row: u32 = (raster_extent.x * bytes_per_pixel)
        .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging_size: u64 = padded_bytes_per_row as u64 * raster_extent.y as u64;
    let staging_buffer: wgpu::Buffer = create_sized_buffer(
        device,
        "readback_staging",
        staging_size as usize,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &raster_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: raster_extent.x,
            height: raster_extent.y,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Map and read back.
    let buffer_slice: wgpu::BufferSlice<'_> = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("map_async channel closed")
        .expect("map_async failed");

    let data = buffer_slice.get_mapped_range();
    let mut pixels: Vec<[f32; 4]> = Vec::with_capacity((width as usize) * (height as usize));

    for y in 0..height {
        let row_start: usize = y as usize * padded_bytes_per_row as usize;
        for x in 0..width {
            let offset: usize = row_start + x as usize * bytes_per_pixel as usize;
            if offset + 3 < data.len() {
                pixels.push([
                    data[offset] as f32 / 255.0,
                    data[offset + 1] as f32 / 255.0,
                    data[offset + 2] as f32 / 255.0,
                    data[offset + 3] as f32 / 255.0,
                ]);
            } else {
                pixels.push([0.0; 4]);
            }
        }
    }

    drop(data);
    staging_buffer.unmap();

    RenderOutput {
        pixels,
        width,
        height,
    }
}
