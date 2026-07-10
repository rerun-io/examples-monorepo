//! Standalone GPU renderer for Gaussian splats — no Rerun dependency.
//!
//! Implements the same GPU-only compute pipeline as `gaussian_renderer.rs`
//! (project_forward cull → depth argsort → project_visible → map intersections
//! → tile radix sort → tile offsets → rasterize) but uses raw `wgpu` directly
//! instead of `re_renderer`.  The bind group layouts, pipelines, bind groups,
//! and radix sorts all come from [`super::gpu_types`] — the single source of
//! truth shared with the viewer path.
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

/// Holds all compute pipelines, bind group layouts, and the radix-sort shift
/// uniforms.  Created once from the shared definitions in `gpu_types`,
/// reused across frames.
pub struct GpuRenderer {
    pub layouts: GpuBindGroupLayouts,
    pub pipelines: GpuComputePipelines,
    shift_ubs: Vec<wgpu::Buffer>,
}

impl GpuRenderer {
    /// Create all compute pipelines from the shared definitions in `gpu_types`.
    pub fn new(device: &wgpu::Device) -> Self {
        let layouts = create_compute_bind_group_layouts(device);
        let pipelines = create_compute_pipelines(device, &layouts);
        let shift_ubs = create_sort_shift_uniforms(device);
        Self {
            layouts,
            pipelines,
            shift_ubs,
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

    bind_groups: PipelineBindGroups,

    // Radix sorts (gid canonicalization, depth argsort, tile-id sort).
    gid_sort: RadixSort,
    depth_sort: RadixSort,
    tile_sort: RadixSort,
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
        let tile_sort_wg_count: u32 = sort_workgroup_count_for(isect_capacity);
        let depth_sort_wg_count: u32 = sort_workgroup_count_for(instance_capacity);
        let sort_reduce_wg_count: u32 = sort_reduce_workgroup_count(tile_sort_wg_count);
        let sort_scan_block_capacity: usize = next_block_capacity(sort_reduce_wg_count as usize);

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
            4 * su32,
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
        let scratch = RadixSortScratch {
            counts: &sort_counts_buf,
            reduced: &sort_reduced_buf,
            scan_offsets: &sort_scan_offsets_buf,
            scan_block_offsets: &sort_scan_block_offsets_buf,
            scan_totals: &sort_scan_totals_buf,
        };
        let gid_sort: RadixSort = build_radix_sort(
            device,
            &renderer.layouts,
            &renderer.shift_ubs,
            &RadixSortBuffers {
                keys_primary: &global_from_compact_buf,
                vals_primary: &depth_keys_buf,
                keys_alt: &global_from_compact_alt_buf,
                vals_alt: &depth_keys_alt_buf,
                num_keys: &num_visible_buf,
                indirect_dispatch_buffer: None,
            },
            &scratch,
            depth_sort_wg_count,
            gid_sort_passes(total_splats),
        );
        let depth_sort: RadixSort = build_radix_sort(
            device,
            &renderer.layouts,
            &renderer.shift_ubs,
            &RadixSortBuffers {
                keys_primary: &depth_keys_buf,
                vals_primary: &global_from_compact_buf,
                keys_alt: &depth_keys_alt_buf,
                vals_alt: &global_from_compact_alt_buf,
                num_keys: &num_visible_buf,
                indirect_dispatch_buffer: None,
            },
            &scratch,
            depth_sort_wg_count,
            DEPTH_SORT_PASSES,
        );
        let tile_sort: RadixSort = build_radix_sort(
            device,
            &renderer.layouts,
            &renderer.shift_ubs,
            &RadixSortBuffers {
                keys_primary: &tile_id_from_isect_buf,
                vals_primary: &compact_gid_from_isect_buf,
                keys_alt: &sort_keys_buf,
                vals_alt: &sorted_indices_alt_buf,
                num_keys: &num_isect_buf,
                indirect_dispatch_buffer: None,
            },
            &scratch,
            tile_sort_wg_count,
            tile_sort_passes(n_tiles),
        );

        // ── Pipeline bind groups (shared wiring with the viewer) ─────────
        let bind_groups: PipelineBindGroups = create_pipeline_bind_groups(
            device,
            &renderer.layouts,
            &PipelineBuffers {
                means: &means_buf,
                quats: &quats_buf,
                scales_opacity: &scales_opacity_buf,
                colors: &colors_buf,
                sh_coeffs: &sh_buf,
                project_ub: &project_ub,
                scan_ub: &scan_ub,
                scan_l2_ub: &scan_l2_ub,
                map_ub: &map_ub,
                raster_ub: &raster_ub,
                num_visible: &num_visible_buf,
                global_from_compact: &global_from_compact_buf,
                depth_keys: &depth_keys_buf,
                projected: &projected_buf,
                tile_hit_counts: &tile_hit_counts_buf,
                tile_hit_offsets: &tile_hit_offsets_buf,
                tile_hit_block_offsets: &tile_hit_block_offsets_buf,
                block_local_offsets: &block_local_offsets_buf,
                block2_offsets: &block2_offsets_buf,
                tile_isect_count: &tile_isect_count_buf,
                num_isect: &num_isect_buf,
                tile_id_from_isect: &tile_id_from_isect_buf,
                compact_gid_from_isect: &compact_gid_from_isect_buf,
                tile_offsets: &tile_offsets_buf,
                raster_view: &raster_view,
            },
        );

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
            bind_groups,
            gid_sort,
            depth_sort,
            tile_sort,
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
            pass.set_bind_group(0, &self.bind_groups.project_forward, &[]);
            pass.dispatch_workgroups(self.project_grid.0, self.project_grid.1, 1);
        }

        // Stage 2a: canonicalize the compact order by gid (see `new`).
        // Stage 2b: depth argsort (ascending f32 bits => front-to-back compact
        // order).  8 passes (even), so results land back in the primary buffers.
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gid_and_depth_sort"),
                    timestamp_writes: None,
                });
            self.gid_sort.encode(&mut pass, &renderer.pipelines);
            self.depth_sort.encode(&mut pass, &renderer.pipelines);
        }

        // Stage 3-5: project visible splats, prefix-scan tile counts, map intersections.
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("project_visible_scan_map"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(&renderer.pipelines.project_visible);
            pass.set_bind_group(0, &self.bind_groups.project_visible, &[]);
            pass.dispatch_workgroups(self.project_grid.0, self.project_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.scan_blocks);
            pass.set_bind_group(0, &self.bind_groups.scan, &[]);
            pass.dispatch_workgroups(self.scan_l1_grid.0, self.scan_l1_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.scan_blocks);
            pass.set_bind_group(0, &self.bind_groups.scan_l2, &[]);
            pass.dispatch_workgroups(self.scan_l2_grid.0, self.scan_l2_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.scan_block_sums);
            pass.set_bind_group(0, &self.bind_groups.scan_block_sums, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&renderer.pipelines.sort_scan_compose);
            pass.set_bind_group(0, &self.bind_groups.scan_compose, &[]);
            pass.dispatch_workgroups(self.scan_compose_grid.0, self.scan_compose_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.map_intersections);
            pass.set_bind_group(0, &self.bind_groups.map, &[]);
            pass.dispatch_workgroups(self.project_grid.0, self.project_grid.1, 1);

            pass.set_pipeline(&renderer.pipelines.clamp_intersection_count);
            pass.set_bind_group(0, &self.bind_groups.map, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Stage 6: radix sort intersections by tile ID.
        {
            let mut pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tile_sort"),
                    timestamp_writes: None,
                });
            self.tile_sort.encode(&mut pass, &renderer.pipelines);
        }

        // Copy back to primary if odd number of passes.
        if self.tile_sort.num_passes() % 2 == 1 {
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
            pass.set_bind_group(0, &self.bind_groups.tile_offsets, &[]);
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
            pass.set_bind_group(0, &self.bind_groups.rasterize, &[]);
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
