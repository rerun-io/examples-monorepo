//! Standalone GPU renderer for Gaussian splats — no Rerun dependency.
//!
//! Implements the same GPU-only compute pipeline as `gaussian_renderer.rs`
//! (project_forward cull → depth argsort → project_visible → map intersections
//! → tile radix sort → tile offsets → rasterize) but uses raw `wgpu` directly
//! instead of `re_renderer`.
//!
//! All GPU resources — splat data, scratch buffers, bind groups, the radix
//! sort passes, and the readback staging buffer — live in
//! [`GpuRenderResources`], built once per (cloud, resolution) and reused
//! across frames.  Per frame the CPU only writes the camera uniform, encodes
//! commands, and reads the output back as RGBA8 — mirroring Brush, where the
//! GPU does all culling and depth sorting.

use bytemuck::Zeroable as _;

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
/// needed.  Used three times per frame: gid canonicalization + depth argsort
/// (keys = f32 depth bits, values = global gids) and tile-id sort (keys =
/// tile ids, values = compact gids).
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

/// Bind groups that depend on the radix pass index — the 4-bit shift uniform
/// and the src/dst buffer parity (even passes read primary, odd read alt).
struct RadixSortPassBindGroups {
    count_bg: wgpu::BindGroup,
    reduce_bg: wgpu::BindGroup,
    scan_add_bg: wgpu::BindGroup,
    scatter_bg: wgpu::BindGroup,
}

/// Prebuilt bind groups and dispatch grids for one radix argsort instance.
///
/// With an even pass count the sorted data lands back in the primary buffers;
/// for an odd count the caller must copy back from the alt buffers.
struct RadixSort {
    passes: Vec<RadixSortPassBindGroups>,
    scan_blocks_bg: wgpu::BindGroup,
    scan_block_sums_bg: wgpu::BindGroup,
    compose_bg: wgpu::BindGroup,
    /// Grid for the count + scatter kernels (one thread per key slot).
    count_grid: (u32, u32),
    /// Grid for the reduce + scan_add kernels.
    reduce_grid: (u32, u32),
    scan_blocks_grid: (u32, u32),
    compose_grid: (u32, u32),
}

/// Build the per-pass bind groups for `num_passes` 4-bit radix sort passes.
///
/// `shift_ubs[i]` must hold a [`SortUniformBuffer`] with `shift = i * 4`.
fn build_radix_sort(
    device: &wgpu::Device,
    layouts: &GpuBindGroupLayouts,
    shift_ubs: &[wgpu::Buffer],
    buffers: &RadixSortBuffers<'_>,
    sort_wg_count: u32,
    num_passes: u32,
) -> RadixSort {
    let sort_reduce_wg_count: u32 = sort_reduce_workgroup_count(sort_wg_count.max(1));
    let reduced_total: u32 = sort_reduce_wg_count;
    let reduced_block_count: u32 = compaction_block_count(reduced_total as usize) as u32;
    let scan_sort_uniform: ScanUniformBuffer = fill_scan_uniform(reduced_total as usize);
    let scan_sort_ub: wgpu::Buffer = create_filled_buffer(
        device,
        "scan_sort_ub",
        wgpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&scan_sort_uniform),
    );

    let passes: Vec<RadixSortPassBindGroups> = (0..num_passes)
        .map(|pass_index| {
            let sort_ub: &wgpu::Buffer = &shift_ubs[pass_index as usize];
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

            RadixSortPassBindGroups {
                count_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_count_bg"),
                    layout: &layouts.sort_count,
                    entries: &[
                        storage_buffer_entry(0, sort_ub),
                        storage_buffer_entry(1, src_keys),
                        storage_buffer_entry(2, buffers.counts),
                        storage_buffer_entry(6, buffers.num_keys),
                    ],
                }),
                reduce_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_reduce_bg"),
                    layout: &layouts.sort_reduce,
                    entries: &[
                        storage_buffer_entry(0, sort_ub),
                        storage_buffer_entry(1, buffers.counts),
                        storage_buffer_entry(2, buffers.reduced),
                        storage_buffer_entry(6, buffers.num_keys),
                    ],
                }),
                scan_add_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_scan_add_bg"),
                    layout: &layouts.sort_scan_add,
                    entries: &[
                        storage_buffer_entry(0, sort_ub),
                        storage_buffer_entry(1, buffers.reduced),
                        storage_buffer_entry(2, buffers.counts),
                        storage_buffer_entry(6, buffers.num_keys),
                    ],
                }),
                scatter_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_scatter_bg"),
                    layout: &layouts.sort_scatter,
                    entries: &[
                        storage_buffer_entry(0, sort_ub),
                        storage_buffer_entry(1, src_keys),
                        storage_buffer_entry(2, src_vals),
                        storage_buffer_entry(3, buffers.counts),
                        storage_buffer_entry(4, dst_keys),
                        storage_buffer_entry(5, dst_vals),
                        storage_buffer_entry(6, buffers.num_keys),
                    ],
                }),
            }
        })
        .collect();

    RadixSort {
        passes,
        scan_blocks_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_blocks_bg"),
            layout: &layouts.scan,
            entries: &[
                storage_buffer_entry(16, buffers.reduced),
                storage_buffer_entry(17, buffers.scan_offsets),
                storage_buffer_entry(18, buffers.scan_block_offsets),
                storage_buffer_entry(19, &scan_sort_ub),
            ],
        }),
        scan_block_sums_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_block_sums_bg"),
            layout: &layouts.scan_block_sums,
            entries: &[
                storage_buffer_entry(24, buffers.scan_block_offsets),
                storage_buffer_entry(25, buffers.scan_totals),
                storage_buffer_entry(26, &scan_sort_ub),
            ],
        }),
        compose_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_compose_bg"),
            layout: &layouts.sort_scan_compose,
            entries: &[
                storage_buffer_entry(8, buffers.scan_offsets),
                storage_buffer_entry(9, buffers.scan_block_offsets),
                storage_buffer_entry(10, buffers.reduced),
                storage_buffer_entry(11, &scan_sort_ub),
            ],
        }),
        count_grid: dispatch_grid_for_workgroups(sort_wg_count.max(1)),
        reduce_grid: dispatch_grid_for_workgroups(sort_reduce_wg_count),
        scan_blocks_grid: dispatch_grid_1d(reduced_block_count, 1),
        compose_grid: dispatch_grid_1d(reduced_total, SORT_WORKGROUP_SIZE),
    }
}

impl RadixSort {
    /// Encode all radix passes into one compute pass.  Dispatch order is
    /// identical to encoding each pass separately — WebGPU guarantees
    /// dispatch-order visibility within a compute pass.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, pipelines: &GpuComputePipelines) {
        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_sort"),
                timestamp_writes: None,
            });
        for pass_bgs in &self.passes {
            pass.set_pipeline(&pipelines.sort_count);
            pass.set_bind_group(0, &pass_bgs.count_bg, &[]);
            pass.dispatch_workgroups(self.count_grid.0, self.count_grid.1, 1);

            pass.set_pipeline(&pipelines.sort_reduce);
            pass.set_bind_group(0, &pass_bgs.reduce_bg, &[]);
            pass.dispatch_workgroups(self.reduce_grid.0, self.reduce_grid.1, 1);

            pass.set_pipeline(&pipelines.scan_blocks);
            pass.set_bind_group(0, &self.scan_blocks_bg, &[]);
            pass.dispatch_workgroups(self.scan_blocks_grid.0, self.scan_blocks_grid.1, 1);

            pass.set_pipeline(&pipelines.scan_block_sums);
            pass.set_bind_group(0, &self.scan_block_sums_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&pipelines.sort_scan_compose);
            pass.set_bind_group(0, &self.compose_bg, &[]);
            pass.dispatch_workgroups(self.compose_grid.0, self.compose_grid.1, 1);

            pass.set_pipeline(&pipelines.sort_scan_add);
            pass.set_bind_group(0, &pass_bgs.scan_add_bg, &[]);
            pass.dispatch_workgroups(self.reduce_grid.0, self.reduce_grid.1, 1);

            pass.set_pipeline(&pipelines.sort_scatter);
            pass.set_bind_group(0, &pass_bgs.scatter_bg, &[]);
            pass.dispatch_workgroups(self.count_grid.0, self.count_grid.1, 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistent per-(cloud, resolution) render resources
// ═══════════════════════════════════════════════════════════════════════════════

/// All GPU resources for rendering one cloud at one resolution.
///
/// Splat data is uploaded once at construction; scratch buffers, bind groups,
/// the radix sort passes, the raster texture, and the readback staging buffer
/// are all reused across frames.  Per frame, [`GpuRenderResources::render`]
/// only writes the ~450-byte camera uniform, encodes commands, and reads the
/// result back.
pub struct GpuRenderResources {
    /// The cloud these resources were built for (`Arc`-backed, cheap clone).
    /// Needed per frame to refill the camera-dependent project uniform.
    cloud: RenderGaussianCloud,
    total_splats: usize,
    width: u32,
    height: u32,
    raster_extent: glam::UVec2,

    /// Camera uniform — the only buffer written per frame.
    project_ub: wgpu::Buffer,

    // Dispatch grids (all capacity-sized, fixed at build time).
    project_grid: (u32, u32),
    scan_l1_grid: (u32, u32),
    scan_l2_grid: (u32, u32),
    scan_compose_grid: (u32, u32),
    tile_offsets_grid: (u32, u32),
    raster_grid: (u32, u32),

    // Pipeline bind groups.
    project_forward_bg: wgpu::BindGroup,
    project_visible_bg: wgpu::BindGroup,
    scan_bg: wgpu::BindGroup,
    scan_l2_bg: wgpu::BindGroup,
    scan_block_sums_bg: wgpu::BindGroup,
    scan_compose_bg: wgpu::BindGroup,
    map_bg: wgpu::BindGroup,
    tile_offsets_bg: wgpu::BindGroup,
    rasterize_bg: wgpu::BindGroup,

    // Radix sorts (gid canonicalization, depth argsort, tile-id sort).
    gid_sort: RadixSort,
    depth_sort: RadixSort,
    tile_sort: RadixSort,
    num_tile_sort_passes: u32,
    isect_capacity: usize,

    // Buffers touched by encoder-level ops (clears / copies) each frame.
    num_visible_buf: wgpu::Buffer,
    tile_offsets_buf: wgpu::Buffer,
    tile_id_from_isect_buf: wgpu::Buffer,
    compact_gid_from_isect_buf: wgpu::Buffer,
    sort_keys_buf: wgpu::Buffer,
    sorted_indices_alt_buf: wgpu::Buffer,

    // Output.
    raster_texture: wgpu::Texture,
    staging_buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
}

impl GpuRenderResources {
    /// Upload the cloud and build every buffer and bind group the pipeline
    /// needs at `viewport_size_px`.  An empty cloud is fine — all dispatches
    /// no-op and [`render`](Self::render) returns a transparent image.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: &wgpu::Device,
        renderer: &GpuRenderer,
        cloud: &RenderGaussianCloud,
        viewport_size_px: glam::Vec2,
    ) -> Self {
        let width: u32 = viewport_size_px.x.max(1.0) as u32;
        let height: u32 = viewport_size_px.y.max(1.0) as u32;
        let total_splats: usize = cloud.len();

        // ── Upload splat data to GPU ─────────────────────────────────────
        let storage_usage: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
        let means_buf: wgpu::Buffer = create_filled_buffer(
            device,
            "means",
            storage_usage,
            &pack_vec3s(cloud.means_world.iter().copied()),
        );
        let quats_buf: wgpu::Buffer = create_filled_buffer(
            device,
            "quats",
            storage_usage,
            &pack_quats(cloud.quats.iter().copied()),
        );
        let scales_opacity_buf: wgpu::Buffer = create_filled_buffer(
            device,
            "scales_opacity",
            storage_usage,
            &pack_scales_opacity(cloud),
        );
        let colors_buf: wgpu::Buffer = create_filled_buffer(
            device,
            "colors",
            storage_usage,
            &pack_rgb(cloud.colors_dc.iter().copied()),
        );
        let sh_buf: wgpu::Buffer = create_filled_buffer(
            device,
            "sh_coeffs",
            storage_usage,
            &pack_sh_coefficients(cloud),
        );

        // ── Sizing ───────────────────────────────────────────────────────
        let tile_bounds: glam::UVec2 = calc_tile_bounds(viewport_size_px);
        let raster_extent: glam::UVec2 = calc_raster_extent(viewport_size_px);
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

        // ── Uniform buffers ──────────────────────────────────────────────
        let block_count: u32 = compaction_block_count(total_splats) as u32;
        let block2_count: u32 = compaction_block_count(block_count as usize) as u32;
        let scan_uniform: ScanUniformBuffer = fill_scan_uniform(total_splats);
        // Level-2 scan: scans the level-1 block sums (block_count entries).
        let scan_uniform_l2: ScanUniformBuffer = fill_scan_uniform(block_count as usize);
        let map_uniform: MapUniformBuffer =
            fill_map_uniform(total_splats, isect_capacity, tile_bounds);
        let raster_uniform: RasterUniformBuffer = RasterUniformBuffer {
            tile_bounds: [tile_bounds.x, tile_bounds.y],
            img_size: [raster_extent.x, raster_extent.y],
        };

        // The project uniform is camera-dependent; written per frame in
        // `render` (`create_filled_buffer` adds COPY_DST). Zero-init here.
        let project_ub: wgpu::Buffer = create_filled_buffer(
            device,
            "project_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&ProjectUniformBuffer::zeroed()),
        );
        let scan_ub: wgpu::Buffer = create_filled_buffer(
            device,
            "scan_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&scan_uniform),
        );
        let scan_l2_ub: wgpu::Buffer = create_filled_buffer(
            device,
            "scan_l2_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&scan_uniform_l2),
        );
        let map_ub: wgpu::Buffer = create_filled_buffer(
            device,
            "map_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&map_uniform),
        );
        let raster_ub: wgpu::Buffer = create_filled_buffer(
            device,
            "raster_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&raster_uniform),
        );
        // Radix sort shift uniforms — shift = pass * 4, shared by all sorts.
        let max_sort_passes: u32 = DEPTH_SORT_PASSES
            .max(gid_sort_passes(total_splats))
            .max(tile_sort_passes(n_tiles));
        let shift_ubs: Vec<wgpu::Buffer> = (0..max_sort_passes)
            .map(|pass_index| {
                create_filled_buffer(
                    device,
                    "sort_shift_ub",
                    wgpu::BufferUsages::UNIFORM,
                    bytemuck::bytes_of(&SortUniformBuffer {
                        shift: pass_index * SORT_BITS_PER_PASS,
                        total_keys_unused: 0,
                        _pad: [0; 2],
                    }),
                )
            })
            .collect();

        // ── Intermediate buffers ─────────────────────────────────────────
        let sz = |n: usize, elem: usize| n * elem;
        let su32 = std::mem::size_of::<u32>();
        let s_splat = std::mem::size_of::<TileProjectedSplat>();

        // GPU cull + depth sort buffers (project_forward outputs).
        let num_visible_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "num_visible",
            su32,
            storage_usage | wgpu::BufferUsages::COPY_DST,
        );
        let global_from_compact_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "global_from_compact",
            sz(instance_capacity, su32),
            storage_usage,
        );
        let depth_keys_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "depth_keys",
            sz(instance_capacity, su32),
            storage_usage,
        );
        let global_from_compact_alt_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "global_from_compact_alt",
            sz(instance_capacity, su32),
            storage_usage,
        );
        let depth_keys_alt_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "depth_keys_alt",
            sz(instance_capacity, su32),
            storage_usage,
        );

        let projected_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "projected",
            sz(instance_capacity, s_splat),
            storage_usage,
        );
        let tile_hit_counts_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "tile_hit_counts",
            sz(instance_capacity, su32),
            storage_usage | wgpu::BufferUsages::COPY_DST,
        );
        let tile_hit_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "tile_hit_offsets",
            sz(instance_capacity, su32),
            storage_usage,
        );
        let tile_hit_block_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "tile_hit_block_offsets",
            sz(block_capacity, su32),
            storage_usage,
        );
        let block_local_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "block_local_offsets",
            sz(block_capacity, su32),
            storage_usage,
        );
        let block2_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "block2_offsets",
            sz(block2_capacity, su32),
            storage_usage,
        );
        let tile_isect_count_buf: wgpu::Buffer = create_filled_buffer(
            device,
            "tile_isect_count",
            storage_usage | wgpu::BufferUsages::COPY_SRC,
            &[DrawIndirectArgs {
                vertex_count: 0,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }],
        );
        let num_isect_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "num_isect",
            su32,
            storage_usage | wgpu::BufferUsages::COPY_DST,
        );
        let tile_id_from_isect_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "tile_id_from_isect",
            sz(isect_capacity, su32),
            storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let compact_gid_from_isect_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "compact_gid_from_isect",
            sz(isect_capacity, su32),
            storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let sort_keys_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sort_keys",
            sz(isect_capacity, su32),
            storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let sorted_indices_alt_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sorted_indices_alt",
            sz(isect_capacity, su32),
            storage_usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        // Sort scratch buffers are sized for the (larger) tile sort and reused by
        // the gid + depth sorts — the sorts run sequentially in the same encoder.
        let sort_counts_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sort_counts",
            sz(tile_sort_wg_count as usize * SORT_BIN_COUNT as usize, su32),
            storage_usage,
        );
        let sort_reduced_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sort_reduced",
            sz(sort_reduce_wg_count as usize, su32),
            storage_usage,
        );
        let sort_scan_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sort_scan_offsets",
            sz(sort_reduce_wg_count as usize, su32),
            storage_usage,
        );
        let sort_scan_block_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sort_scan_block_offsets",
            sz(sort_scan_block_capacity, su32),
            storage_usage,
        );
        let sort_scan_totals_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "sort_scan_totals",
            sz(SORT_BIN_COUNT as usize, su32),
            storage_usage | wgpu::BufferUsages::INDIRECT,
        );
        let tile_offsets_buf: wgpu::Buffer = create_sized_buffer(
            device,
            "tile_offsets",
            sz(n_tiles.max(1) * 2, su32),
            storage_usage | wgpu::BufferUsages::COPY_DST,
        );

        // ── Raster texture + readback staging ────────────────────────────
        let (raster_texture, raster_view) = create_raster_texture(
            device,
            "raster_texture",
            raster_extent,
            wgpu::TextureUsages::COPY_SRC, // readback
        );
        let bytes_per_pixel: u32 = 4; // Rgba8Unorm
        let padded_bytes_per_row: u32 = (raster_extent.x * bytes_per_pixel)
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging_size: usize = padded_bytes_per_row as usize * raster_extent.y as usize;
        let staging_buffer: wgpu::Buffer = create_sized_buffer(
            device,
            "readback_staging",
            staging_size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        // ── Radix sorts ──────────────────────────────────────────────────
        // Stage 2a sorts (gid, depth) pairs ascending by gid to canonicalize
        // project_forward's racy atomicAdd compaction order; stage 2b then
        // argsorts by f32 depth bits (stable, so gid order breaks ties).
        let gid_sort: RadixSort = build_radix_sort(
            device,
            &renderer.layouts,
            &shift_ubs,
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
        let depth_sort: RadixSort = build_radix_sort(
            device,
            &renderer.layouts,
            &shift_ubs,
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
        let num_tile_sort_passes: u32 = tile_sort_passes(n_tiles);
        let tile_sort: RadixSort = build_radix_sort(
            device,
            &renderer.layouts,
            &shift_ubs,
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

        // ── Pipeline bind groups ─────────────────────────────────────────
        let project_forward_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("project_forward_bg"),
                layout: &renderer.layouts.project_forward,
                entries: &[
                    storage_buffer_entry(0, &means_buf),
                    storage_buffer_entry(1, &quats_buf),
                    storage_buffer_entry(2, &scales_opacity_buf),
                    storage_buffer_entry(8, &project_ub),
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
                    storage_buffer_entry(8, &project_ub),
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
                storage_buffer_entry(19, &scan_ub),
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
                storage_buffer_entry(19, &scan_l2_ub),
            ],
        });
        let scan_block_sums_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scan_block_sums_bg"),
                layout: &renderer.layouts.scan_block_sums,
                entries: &[
                    storage_buffer_entry(24, &block2_offsets_buf),
                    storage_buffer_entry(25, &tile_isect_count_buf), // total intersection count
                    storage_buffer_entry(26, &scan_l2_ub),
                ],
            });
        // Compose the two scan levels back into flat per-block offsets.
        let scan_compose_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scan_compose_bg"),
                layout: &renderer.layouts.sort_scan_compose,
                entries: &[
                    storage_buffer_entry(8, &block_local_offsets_buf),
                    storage_buffer_entry(9, &block2_offsets_buf),
                    storage_buffer_entry(10, &tile_hit_block_offsets_buf),
                    storage_buffer_entry(11, &scan_l2_ub),
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
                storage_buffer_entry(6, &map_ub),
                storage_buffer_entry(7, &tile_isect_count_buf),
                storage_buffer_entry(8, &num_isect_buf),
            ],
        });
        let tile_offsets_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                storage_buffer_entry(4, &raster_ub),
            ],
        });

        // ── Dispatch grids ───────────────────────────────────────────────
        let project_grid: (u32, u32) =
            dispatch_grid_1d(total_splats as u32, PROJECT_WORKGROUP_SIZE);
        let scan_l1_grid: (u32, u32) = dispatch_grid_1d(block_count, 1);
        let scan_l2_grid: (u32, u32) = dispatch_grid_1d(block2_count, 1);
        let scan_compose_grid: (u32, u32) = dispatch_grid_1d(block_count, SORT_WORKGROUP_SIZE);
        let tile_offsets_grid: (u32, u32) = dispatch_grid_1d(
            isect_capacity.max(1) as u32,
            TILE_OFFSET_WORKGROUP_SIZE * TILE_OFFSET_CHECKS_PER_ITER,
        );
        let raster_grid: (u32, u32) = dispatch_grid_for_workgroups(n_tiles.max(1) as u32);

        Self {
            cloud: cloud.clone(),
            total_splats,
            width,
            height,
            raster_extent,
            project_ub,
            project_grid,
            scan_l1_grid,
            scan_l2_grid,
            scan_compose_grid,
            tile_offsets_grid,
            raster_grid,
            project_forward_bg,
            project_visible_bg,
            scan_bg,
            scan_l2_bg,
            scan_block_sums_bg,
            scan_compose_bg,
            map_bg,
            tile_offsets_bg,
            rasterize_bg,
            gid_sort,
            depth_sort,
            tile_sort,
            num_tile_sort_passes,
            isect_capacity,
            num_visible_buf,
            tile_offsets_buf,
            tile_id_from_isect_buf,
            compact_gid_from_isect_buf,
            sort_keys_buf,
            sorted_indices_alt_buf,
            raster_texture,
            staging_buffer,
            padded_bytes_per_row,
        }
    }

    /// Render one frame: write the camera uniform, encode the pipeline,
    /// submit, and read the result back as tightly-packed RGBA8.
    ///
    /// `camera.viewport_size_px` must match the resolution these resources
    /// were built for.
    pub fn render(
        &self,
        ctx: &GpuContext,
        renderer: &GpuRenderer,
        camera: &CameraApproximation,
    ) -> RenderOutput {
        debug_assert_eq!(camera.viewport_size_px.x.max(1.0) as u32, self.width);
        debug_assert_eq!(camera.viewport_size_px.y.max(1.0) as u32, self.height);

        let project_uniform: ProjectUniformBuffer =
            fill_project_uniform(camera, self.total_splats, &self.cloud);
        ctx.queue
            .write_buffer(&self.project_ub, 0, bytemuck::bytes_of(&project_uniform));

        let mut encoder: wgpu::CommandEncoder =
            ctx.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gsplat_render"),
                });

        // The visible counter accumulates atomically — reset it each frame.
        encoder.clear_buffer(&self.num_visible_buf, 0, None);

        // Stage 1: cull the full cloud, compact (gid, depth) pairs.
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("project_forward"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&renderer.pipelines.project_forward);
            pass.set_bind_group(0, &self.project_forward_bg, &[]);
            pass.dispatch_workgroups(self.project_grid.0, self.project_grid.1, 1);
        }

        // Stage 2a: canonicalize the compact order by gid (see `new`).
        self.gid_sort.encode(&mut encoder, &renderer.pipelines);
        // Stage 2b: depth argsort (ascending f32 bits => front-to-back compact
        // order).  8 passes (even), so results land back in the primary buffers.
        self.depth_sort.encode(&mut encoder, &renderer.pipelines);

        // Stage 3-5: project visible splats, prefix-scan tile counts, map intersections.
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("project_visible_scan_map"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(&renderer.pipelines.project_visible);
            pass.set_bind_group(0, &self.project_visible_bg, &[]);
            pass.dispatch_workgroups(self.project_grid.0, self.project_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.scan_blocks);
            pass.set_bind_group(0, &self.scan_bg, &[]);
            pass.dispatch_workgroups(self.scan_l1_grid.0, self.scan_l1_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.scan_blocks);
            pass.set_bind_group(0, &self.scan_l2_bg, &[]);
            pass.dispatch_workgroups(self.scan_l2_grid.0, self.scan_l2_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.scan_block_sums);
            pass.set_bind_group(0, &self.scan_block_sums_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&renderer.pipelines.sort_scan_compose);
            pass.set_bind_group(0, &self.scan_compose_bg, &[]);
            pass.dispatch_workgroups(self.scan_compose_grid.0, self.scan_compose_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.map_intersections);
            pass.set_bind_group(0, &self.map_bg, &[]);
            pass.dispatch_workgroups(self.project_grid.0, self.project_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.clamp_intersection_count);
            pass.set_bind_group(0, &self.map_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Stage 6: radix sort intersections by tile ID.
        self.tile_sort.encode(&mut encoder, &renderer.pipelines);

        // Copy back to primary if odd number of passes.
        if self.num_tile_sort_passes % 2 == 1 {
            let bytes: u64 = (self.isect_capacity * std::mem::size_of::<u32>()) as u64;
            encoder.copy_buffer_to_buffer(
                &self.sort_keys_buf,
                0,
                &self.tile_id_from_isect_buf,
                0,
                bytes,
            );
            encoder.copy_buffer_to_buffer(
                &self.sorted_indices_alt_buf,
                0,
                &self.compact_gid_from_isect_buf,
                0,
                bytes,
            );
        }

        // Stage 7: tile offsets.
        encoder.clear_buffer(&self.tile_offsets_buf, 0, None);
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tile_offsets"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&renderer.pipelines.tile_offsets);
            pass.set_bind_group(0, &self.tile_offsets_bg, &[]);
            pass.dispatch_workgroups(self.tile_offsets_grid.0, self.tile_offsets_grid.1, 1);
        }

        // Stage 8: rasterize.
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rasterize"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&renderer.pipelines.rasterize);
            pass.set_bind_group(0, &self.rasterize_bg, &[]);
            pass.dispatch_workgroups(self.raster_grid.0, self.raster_grid.1, 1);
        }

        // ── Readback raster texture ──────────────────────────────────────
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.raster_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.raster_extent.x,
                height: self.raster_extent.y,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back, stripping the row padding (and the tile-aligned
        // right/bottom overhang) without any per-pixel conversion.
        let buffer_slice: wgpu::BufferSlice<'_> = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("map_async channel closed")
            .expect("map_async failed");

        let data = buffer_slice.get_mapped_range();
        let row_bytes: usize = self.width as usize * 4;
        let mut pixels: Vec<u8> = Vec::with_capacity(row_bytes * self.height as usize);
        for y in 0..self.height as usize {
            let row_start: usize = y * self.padded_bytes_per_row as usize;
            pixels.extend_from_slice(&data[row_start..row_start + row_bytes]);
        }
        drop(data);
        self.staging_buffer.unmap();

        RenderOutput {
            pixels,
            width: self.width,
            height: self.height,
        }
    }
}
