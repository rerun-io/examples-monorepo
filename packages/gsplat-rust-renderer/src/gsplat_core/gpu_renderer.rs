//! Standalone GPU renderer for Gaussian splats — no Rerun dependency.
//!
//! Implements the same 7-stage compute pipeline as `gaussian_renderer.rs`
//! (project → compact → map intersections → radix sort → tile offsets →
//! rasterize) but uses raw `wgpu` directly instead of `re_renderer`.
//!
//! The output is read back from the GPU as an RGBA8 image buffer.
//! This module follows the [Brush](https://github.com/ArthurBrussee/brush)
//! approach: pure GPU rendering with no CPU rasterization fallback.

use std::sync::Arc;

use glam::Mat4;

use super::constants::{MAX_SPLATS_RENDERED, MIN_RADIUS_PX, OPACITY_SCALE, SIGMA_COVERAGE};
use super::culling::rebuild_visible_indices;
use super::gpu_context::GpuContext;
use super::gpu_types::*;
use super::sh::sh_degree_from_coeffs;
use super::types::{CameraApproximation, RenderGaussianCloud, RenderOutput, SortedSplatIndex};

// ═══════════════════════════════════════════════════════════════════════════════
// Renderer (holds reusable pipelines)
// ═══════════════════════════════════════════════════════════════════════════════

/// Holds all compute pipelines and bind group layouts.
/// Created once, reused across frames.
// Some fields are retained as lifetime anchors for GPU resources even when not directly
// referenced in every dispatch path.
#[allow(dead_code)]
pub struct GpuRenderer {
    project_pipeline: wgpu::ComputePipeline,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scan_block_sums_pipeline: wgpu::ComputePipeline,
    map_intersections_pipeline: wgpu::ComputePipeline,
    clamp_intersection_count_pipeline: wgpu::ComputePipeline,
    sort_count_pipeline: wgpu::ComputePipeline,
    sort_reduce_pipeline: wgpu::ComputePipeline,
    sort_scan_pipeline: wgpu::ComputePipeline,
    sort_scan_compose_pipeline: wgpu::ComputePipeline,
    sort_scan_add_pipeline: wgpu::ComputePipeline,
    sort_scatter_pipeline: wgpu::ComputePipeline,
    tile_offsets_pipeline: wgpu::ComputePipeline,
    rasterize_pipeline: wgpu::ComputePipeline,

    project_bgl: wgpu::BindGroupLayout,
    scan_bgl: wgpu::BindGroupLayout,
    scan_block_sums_bgl: wgpu::BindGroupLayout,
    map_bgl: wgpu::BindGroupLayout,
    sort_count_bgl: wgpu::BindGroupLayout,
    sort_reduce_bgl: wgpu::BindGroupLayout,
    sort_scan_bgl: wgpu::BindGroupLayout,
    sort_scan_compose_bgl: wgpu::BindGroupLayout,
    sort_scan_add_bgl: wgpu::BindGroupLayout,
    sort_scatter_bgl: wgpu::BindGroupLayout,
    tile_offsets_bgl: wgpu::BindGroupLayout,
    rasterize_bgl: wgpu::BindGroupLayout,
}

impl GpuRenderer {
    /// Create all compute pipelines from the embedded WGSL shaders.
    pub fn new(device: &wgpu::Device) -> Self {
        let project_shader = create_shader_module(
            device,
            "project",
            include_str!("../../shader/gaussian_project.wgsl"),
        );
        let map_shader = create_shader_module(
            device,
            "map_intersections",
            include_str!("../../shader/gaussian_map_intersections.wgsl"),
        );
        let sort_shader = create_shader_module(
            device,
            "dynamic_sort",
            include_str!("../../shader/gaussian_dynamic_sort.wgsl"),
        );
        let tile_offsets_shader = create_shader_module(
            device,
            "tile_offsets",
            include_str!("../../shader/gaussian_tile_offsets.wgsl"),
        );
        let rasterize_shader = create_shader_module(
            device,
            "rasterize",
            include_str!("../../shader/gaussian_raster_tiles.wgsl"),
        );

        // ── Bind group layouts ───────────────────────────────────────────
        let project_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("project_bgl"),
            entries: &[
                storage_layout_entry(0, true),  // means_world
                storage_layout_entry(1, true),  // quats_xyzw
                storage_layout_entry(2, true),  // scales_opacity
                storage_layout_entry(3, true),  // colors_dc
                storage_layout_entry(4, true),  // sh_coefficients
                storage_layout_entry(5, true),  // sorted_indices
                storage_layout_entry(6, false), // project_output_instances (instanced quad output)
                storage_layout_entry(7, false), // visibility_flags
                uniform_layout_entry(8, std::mem::size_of::<ProjectUniformBuffer>()),
                storage_layout_entry(9, false), // projected_tile_splats
                storage_layout_entry(10, false), // projected_tile_hit_counts
            ],
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan_bgl"),
            entries: &[
                storage_layout_entry(16, true),  // visibility_flags (input)
                storage_layout_entry(17, false), // local_offsets (output)
                storage_layout_entry(18, false), // block_offsets (output)
                uniform_layout_entry(19, std::mem::size_of::<ScanUniformBuffer>()),
            ],
        });
        let scan_block_sums_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scan_block_sums_bgl"),
                entries: &[
                    storage_layout_entry(24, false), // block_offsets (in/out)
                    storage_layout_entry(25, false), // indirect_draw (output)
                    uniform_layout_entry(26, std::mem::size_of::<ScanUniformBuffer>()),
                ],
            });
        let map_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("map_bgl"),
            entries: &[
                storage_layout_entry(0, true),  // projected_tile_splats
                storage_layout_entry(1, true),  // tile_hit_offsets
                storage_layout_entry(2, true),  // tile_hit_counts
                storage_layout_entry(3, true),  // tile_hit_block_offsets
                storage_layout_entry(4, false), // tile_id_from_isect
                storage_layout_entry(5, false), // compact_gid_from_isect
                uniform_layout_entry(6, std::mem::size_of::<MapUniformBuffer>()),
                storage_layout_entry(7, true), // tile_intersection_count
                storage_layout_entry(8, false), // num_intersections
            ],
        });
        let sort_count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_count_bgl"),
            entries: &[
                uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
                storage_layout_entry(1, true),  // src_keys
                storage_layout_entry(2, false), // counts
                storage_layout_entry(6, true),  // num_intersections
            ],
        });
        let sort_reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_reduce_bgl"),
            entries: &[
                uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
                storage_layout_entry(1, true),  // counts
                storage_layout_entry(2, false), // reduced
                storage_layout_entry(6, true),  // num_intersections
            ],
        });
        let sort_scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_scan_bgl"),
            entries: &[
                uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
                storage_layout_entry(1, false), // reduced (in/out)
                storage_layout_entry(6, true),  // num_intersections
            ],
        });
        let sort_scan_compose_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sort_scan_compose_bgl"),
                entries: &[
                    storage_layout_entry(8, true),   // offsets
                    storage_layout_entry(9, true),   // block_offsets
                    storage_layout_entry(10, false), // out
                    uniform_layout_entry(11, std::mem::size_of::<ScanUniformBuffer>()),
                ],
            });
        let sort_scan_add_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_scan_add_bgl"),
            entries: &[
                uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
                storage_layout_entry(1, true),  // reduced
                storage_layout_entry(2, false), // counts
                storage_layout_entry(6, true),  // num_intersections
            ],
        });
        let sort_scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_scatter_bgl"),
            entries: &[
                uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
                storage_layout_entry(1, true),  // src_keys
                storage_layout_entry(2, true),  // src_values
                storage_layout_entry(3, true),  // counts
                storage_layout_entry(4, false), // dst_keys
                storage_layout_entry(5, false), // dst_values
                storage_layout_entry(6, true),  // num_intersections
            ],
        });
        let tile_offsets_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_offsets_bgl"),
            entries: &[
                storage_layout_entry(0, true),  // sorted_tile_ids
                storage_layout_entry(1, false), // tile_offsets
                storage_layout_entry(2, true),  // num_intersections
            ],
        });
        let rasterize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rasterize_bgl"),
            entries: &[
                storage_layout_entry(0, true), // compact_gid_from_isect
                storage_layout_entry(1, true), // tile_offsets
                storage_layout_entry(2, true), // projected
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
                uniform_layout_entry(4, std::mem::size_of::<RasterUniformBuffer>()),
            ],
        });

        // ── Compute pipelines ────────────────────────────────────────────
        let project_pipeline = create_compute_pipeline(
            device,
            "project",
            &project_shader,
            "project_main",
            &[&project_bgl],
        );
        let scan_blocks_pipeline = create_compute_pipeline(
            device,
            "scan_blocks",
            &project_shader,
            "scan_blocks_main",
            &[&scan_bgl],
        );
        let scan_block_sums_pipeline = create_compute_pipeline(
            device,
            "scan_block_sums",
            &project_shader,
            "scan_block_sums_main",
            &[&scan_block_sums_bgl],
        );
        let map_intersections_pipeline = create_compute_pipeline(
            device,
            "map_intersections",
            &map_shader,
            "map_main",
            &[&map_bgl],
        );
        let clamp_intersection_count_pipeline = create_compute_pipeline(
            device,
            "clamp_intersection_count",
            &map_shader,
            "clamp_count_main",
            &[&map_bgl],
        );
        let sort_count_pipeline = create_compute_pipeline(
            device,
            "sort_count",
            &sort_shader,
            "sort_count_main",
            &[&sort_count_bgl],
        );
        let sort_reduce_pipeline = create_compute_pipeline(
            device,
            "sort_reduce",
            &sort_shader,
            "sort_reduce_main",
            &[&sort_reduce_bgl],
        );
        let sort_scan_pipeline = create_compute_pipeline(
            device,
            "sort_scan",
            &sort_shader,
            "sort_scan_main",
            &[&sort_scan_bgl],
        );
        let sort_scan_compose_pipeline = create_compute_pipeline(
            device,
            "sort_scan_compose",
            &sort_shader,
            "sort_scan_compose_main",
            &[&sort_scan_compose_bgl],
        );
        let sort_scan_add_pipeline = create_compute_pipeline(
            device,
            "sort_scan_add",
            &sort_shader,
            "sort_scan_add_main",
            &[&sort_scan_add_bgl],
        );
        let sort_scatter_pipeline = create_compute_pipeline(
            device,
            "sort_scatter",
            &sort_shader,
            "sort_scatter_main",
            &[&sort_scatter_bgl],
        );
        let tile_offsets_pipeline = create_compute_pipeline(
            device,
            "tile_offsets",
            &tile_offsets_shader,
            "main",
            &[&tile_offsets_bgl],
        );
        let rasterize_pipeline = create_compute_pipeline(
            device,
            "rasterize",
            &rasterize_shader,
            "main",
            &[&rasterize_bgl],
        );

        Self {
            project_pipeline,
            scan_blocks_pipeline,
            scan_block_sums_pipeline,
            map_intersections_pipeline,
            clamp_intersection_count_pipeline,
            sort_count_pipeline,
            sort_reduce_pipeline,
            sort_scan_pipeline,
            sort_scan_compose_pipeline,
            sort_scan_add_pipeline,
            sort_scatter_pipeline,
            tile_offsets_pipeline,
            rasterize_pipeline,
            project_bgl,
            scan_bgl,
            scan_block_sums_bgl,
            map_bgl,
            sort_count_bgl,
            sort_reduce_bgl,
            sort_scan_bgl,
            sort_scan_compose_bgl,
            sort_scan_add_bgl,
            sort_scatter_bgl,
            tile_offsets_bgl,
            rasterize_bgl,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main render function
// ═══════════════════════════════════════════════════════════════════════════════

/// Render a Gaussian cloud from a camera viewpoint using the GPU compute pipeline.
///
/// This is the standalone equivalent of the Rerun viewer's compute tile path,
/// following Brush's approach of pure GPU rendering with no CPU fallback.
///
/// # Pipeline stages
///
/// 1. **CPU pre-pass**: frustum cull + depth sort (reuses `rebuild_visible_indices`)
/// 2. **GPU project**: 3D→2D projection + SH evaluation + tile coverage
/// 3. **GPU compact**: prefix sum to remove invisible splats
/// 4. **GPU map intersections**: scatter (tile_id, splat_id) pairs
/// 5. **GPU radix sort**: sort by tile_id
/// 6. **GPU tile offsets**: per-tile intersection ranges
/// 7. **GPU rasterize**: per-pixel alpha blending in 16×16 tiles
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

    // ── Stage 0: CPU cull + sort ─────────────────────────────────────────
    let mut visible: Vec<SortedSplatIndex> = Vec::new();
    rebuild_visible_indices(&mut visible, cloud, camera);
    if visible.is_empty() {
        return RenderOutput {
            pixels: vec![[0.0; 4]; (width as usize) * (height as usize)],
            width,
            height,
        };
    }
    let selected_limit: usize = visible.len().max(1);

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

    // Upload the CPU-sorted index list (reversed to front-to-back for the GPU).
    let cpu_indices: Vec<u32> = visible.iter().rev().map(|v| v.splat_index).collect();
    let index_buf: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
        device,
        "indices",
        storage_usage,
        &cpu_indices,
    ));

    // ── Sizing ───────────────────────────────────────────────────────────
    let tile_bounds: glam::UVec2 = calc_tile_bounds(camera.viewport_size_px);
    let raster_extent: glam::UVec2 = calc_raster_extent(camera.viewport_size_px);
    let n_tiles: usize = tile_count(tile_bounds);
    let instance_capacity: usize = next_capacity(selected_limit);
    let block_capacity: usize = next_block_capacity(selected_limit);
    let isect_capacity: usize = intersection_capacity_for_instances(instance_capacity);
    let sort_wg_count: u32 = (isect_capacity as u32).div_ceil(SORT_BLOCK_SIZE).max(1);
    let sort_reduce_wg_count: u32 = ((sort_wg_count.div_ceil(SORT_BLOCK_SIZE) as usize)
        * SORT_BIN_COUNT as usize)
        .max(1) as u32;
    let sort_scan_block_capacity: usize = (sort_reduce_wg_count as usize)
        .div_ceil(COMPACTION_BLOCK_SIZE as usize)
        .next_power_of_two()
        .max(1);

    // ── Uniform buffers ──────────────────────────────────────────────────
    let coeffs_per_channel: u32 = cloud
        .sh_coeffs
        .as_ref()
        .map_or(0, |sh| sh.coeffs_per_channel as u32);
    let sh_degree: u32 = sh_degree_from_coeffs(coeffs_per_channel as usize).unwrap_or(0);
    let block_count: u32 = compaction_block_count(selected_limit) as u32;

    let project_uniform: ProjectUniformBuffer = ProjectUniformBuffer {
        view_from_world: Mat4::from(camera.view_from_world).to_cols_array_2d(),
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
    let scan_uniform: ScanUniformBuffer = ScanUniformBuffer {
        total_selected: selected_limit as u32,
        block_count,
        _pad: [0; 2],
    };
    let map_uniform: MapUniformBuffer = MapUniformBuffer {
        total_selected: selected_limit as u32,
        intersection_capacity: isect_capacity.min(u32::MAX as usize) as u32,
        tile_bounds_x: tile_bounds.x,
        tile_bounds_y: tile_bounds.y,
    };
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

    let visibility_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "visibility",
        sz(instance_capacity, su32),
        storage_usage | wgpu::BufferUsages::COPY_DST,
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
    let sort_counts_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_counts",
        sz(sort_wg_count as usize * SORT_BIN_COUNT as usize, su32),
        storage_usage,
    ));
    let sort_reduced_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_reduced",
        sz(
            sort_reduce_wg_count as usize * SORT_BIN_COUNT as usize,
            su32,
        ),
        storage_usage,
    ));
    let sort_scan_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_scan_offsets",
        sz(
            sort_reduce_wg_count as usize * SORT_BIN_COUNT as usize,
            su32,
        ),
        storage_usage,
    ));
    let sort_scan_block_offsets_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "sort_scan_block_offsets",
        sz(sort_scan_block_capacity * SORT_BIN_COUNT as usize, su32),
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
    let raster_texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raster_texture"),
        size: wgpu::Extent3d {
            width: raster_extent.x.max(1),
            height: raster_extent.y.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RASTER_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let raster_view: wgpu::TextureView =
        raster_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

    // Binding 6 is the instance output buffer (used by CPU fallback path).
    // The standalone GPU renderer doesn't need it, but the shader writes to it.
    let dummy_instance_buf: Arc<wgpu::Buffer> = Arc::new(create_sized_buffer(
        device,
        "dummy_instances",
        sz(instance_capacity, std::mem::size_of::<TileProjectedSplat>()),
        storage_usage,
    ));

    let project_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("project_bg"),
        layout: &renderer.project_bgl,
        entries: &[
            storage_buffer_entry(0, &means_buf),
            storage_buffer_entry(1, &quats_buf),
            storage_buffer_entry(2, &scales_opacity_buf),
            storage_buffer_entry(3, &colors_buf),
            storage_buffer_entry(4, &sh_buf),
            storage_buffer_entry(5, &index_buf),
            storage_buffer_entry(6, &dummy_instance_buf), // instance output (unused)
            storage_buffer_entry(7, &visibility_buf),
            ube!(8, &project_ub),
            storage_buffer_entry(9, &projected_buf),
            storage_buffer_entry(10, &tile_hit_counts_buf),
        ],
    });
    let scan_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scan_bg"),
        layout: &renderer.scan_bgl,
        entries: &[
            storage_buffer_entry(16, &tile_hit_counts_buf), // scan input: per-splat tile hit counts
            storage_buffer_entry(17, &tile_hit_offsets_buf), // scan output: prefix-sum offsets
            storage_buffer_entry(18, &tile_hit_block_offsets_buf),
            ube!(19, &scan_ub),
        ],
    });
    let scan_block_sums_bg: wgpu::BindGroup =
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_block_sums_bg"),
            layout: &renderer.scan_block_sums_bgl,
            entries: &[
                storage_buffer_entry(24, &tile_hit_block_offsets_buf),
                storage_buffer_entry(25, &tile_isect_count_buf), // total intersection count
                ube!(26, &scan_ub),
            ],
        });
    let map_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("map_bg"),
        layout: &renderer.map_bgl,
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
        layout: &renderer.tile_offsets_bgl,
        entries: &[
            storage_buffer_entry(0, &tile_id_from_isect_buf),
            storage_buffer_entry(1, &tile_offsets_buf),
            storage_buffer_entry(2, &num_isect_buf),
        ],
    });
    let rasterize_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rasterize_bg"),
        layout: &renderer.rasterize_bgl,
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

    // Clear buffers that accumulate atomically.
    encoder.clear_buffer(&visibility_buf, 0, None);
    encoder.clear_buffer(&tile_hit_counts_buf, 0, None);

    // Stage 1-3: project, compact, map intersections.
    {
        let (px, py) = dispatch_grid_1d(selected_limit as u32, PROJECT_WORKGROUP_SIZE);
        let (sx, sy) = dispatch_grid_1d(block_count, 1);

        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("project_compact_map"),
                timestamp_writes: None,
            });

        pass.set_pipeline(&renderer.project_pipeline);
        pass.set_bind_group(0, &project_bg, &[]);
        pass.dispatch_workgroups(px, py, 1);

        pass.set_pipeline(&renderer.scan_blocks_pipeline);
        pass.set_bind_group(0, &scan_bg, &[]);
        pass.dispatch_workgroups(sx, sy, 1);

        pass.set_pipeline(&renderer.scan_block_sums_pipeline);
        pass.set_bind_group(0, &scan_block_sums_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        pass.set_pipeline(&renderer.map_intersections_pipeline);
        pass.set_bind_group(0, &map_bg, &[]);
        pass.dispatch_workgroups(px, py, 1);

        pass.set_pipeline(&renderer.clamp_intersection_count_pipeline);
        pass.set_bind_group(0, &map_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // Stage 4: radix sort by tile ID.
    let tile_count_bits: u32 = (u32::BITS - (n_tiles.max(1) as u32).leading_zeros()).max(1);
    let sort_passes: u32 = tile_count_bits.div_ceil(4);
    let (sort_count_x, sort_count_y) = dispatch_grid_for_workgroups(sort_wg_count.max(1));
    let (sort_reduce_x, sort_reduce_y) = dispatch_grid_for_workgroups(sort_reduce_wg_count.max(1));

    for pass_index in 0..sort_passes {
        let sort_uniform: SortUniformBuffer = SortUniformBuffer {
            shift: pass_index * 4,
            total_keys_unused: 0,
            _pad: [0; 2],
        };
        let sort_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
            device,
            "sort_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&sort_uniform),
        ));

        let use_primary: bool = pass_index % 2 == 0;
        let (src_keys, src_vals, dst_keys, dst_vals) = if use_primary {
            (
                &tile_id_from_isect_buf,
                &compact_gid_from_isect_buf,
                &sort_keys_buf,
                &sorted_indices_alt_buf,
            )
        } else {
            (
                &sort_keys_buf,
                &sorted_indices_alt_buf,
                &tile_id_from_isect_buf,
                &compact_gid_from_isect_buf,
            )
        };

        let count_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_count_bg"),
            layout: &renderer.sort_count_bgl,
            entries: &[
                ube!(0, &sort_ub),
                storage_buffer_entry(1, src_keys),
                storage_buffer_entry(2, &sort_counts_buf),
                storage_buffer_entry(6, &num_isect_buf),
            ],
        });
        let reduce_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_reduce_bg"),
            layout: &renderer.sort_reduce_bgl,
            entries: &[
                ube!(0, &sort_ub),
                storage_buffer_entry(1, &sort_counts_buf),
                storage_buffer_entry(2, &sort_reduced_buf),
                storage_buffer_entry(6, &num_isect_buf),
            ],
        });

        let reduced_total: u32 = sort_reduce_wg_count.max(1);
        let reduced_block_count: u32 = compaction_block_count(reduced_total as usize) as u32;
        let scan_sort_uniform: ScanUniformBuffer = ScanUniformBuffer {
            total_selected: reduced_total,
            block_count: reduced_block_count,
            _pad: [0; 2],
        };
        let scan_sort_ub: Arc<wgpu::Buffer> = Arc::new(create_filled_buffer(
            device,
            "scan_sort_ub",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&scan_sort_uniform),
        ));

        let scan_blocks_bg: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sort_scan_blocks_bg"),
                layout: &renderer.scan_bgl,
                entries: &[
                    storage_buffer_entry(16, &sort_reduced_buf),
                    storage_buffer_entry(17, &sort_scan_offsets_buf),
                    storage_buffer_entry(18, &sort_scan_block_offsets_buf),
                    ube!(19, &scan_sort_ub),
                ],
            });
        let scan_block_sums_bg2: wgpu::BindGroup =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sort_scan_block_sums_bg"),
                layout: &renderer.scan_block_sums_bgl,
                entries: &[
                    storage_buffer_entry(24, &sort_scan_block_offsets_buf),
                    storage_buffer_entry(25, &sort_scan_totals_buf),
                    ube!(26, &scan_sort_ub),
                ],
            });
        let compose_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_compose_bg"),
            layout: &renderer.sort_scan_compose_bgl,
            entries: &[
                storage_buffer_entry(8, &sort_scan_offsets_buf),
                storage_buffer_entry(9, &sort_scan_block_offsets_buf),
                storage_buffer_entry(10, &sort_reduced_buf),
                ube!(11, &scan_sort_ub),
            ],
        });
        let scan_add_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_add_bg"),
            layout: &renderer.sort_scan_add_bgl,
            entries: &[
                ube!(0, &sort_ub),
                storage_buffer_entry(1, &sort_reduced_buf),
                storage_buffer_entry(2, &sort_counts_buf),
                storage_buffer_entry(6, &num_isect_buf),
            ],
        });
        let scatter_bg: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scatter_bg"),
            layout: &renderer.sort_scatter_bgl,
            entries: &[
                ube!(0, &sort_ub),
                storage_buffer_entry(1, src_keys),
                storage_buffer_entry(2, src_vals),
                storage_buffer_entry(3, &sort_counts_buf),
                storage_buffer_entry(4, dst_keys),
                storage_buffer_entry(5, dst_vals),
                storage_buffer_entry(6, &num_isect_buf),
            ],
        });

        let (ssb_x, ssb_y) = dispatch_grid_1d(reduced_block_count, 1);
        let (ssc_x, ssc_y) = dispatch_grid_1d(reduced_total, SORT_WORKGROUP_SIZE);

        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_sort"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&renderer.sort_count_pipeline);
        pass.set_bind_group(0, &count_bg, &[]);
        pass.dispatch_workgroups(sort_count_x, sort_count_y, 1);

        pass.set_pipeline(&renderer.sort_reduce_pipeline);
        pass.set_bind_group(0, &reduce_bg, &[]);
        pass.dispatch_workgroups(sort_reduce_x, sort_reduce_y, 1);

        pass.set_pipeline(&renderer.scan_blocks_pipeline);
        pass.set_bind_group(0, &scan_blocks_bg, &[]);
        pass.dispatch_workgroups(ssb_x, ssb_y, 1);

        pass.set_pipeline(&renderer.scan_block_sums_pipeline);
        pass.set_bind_group(0, &scan_block_sums_bg2, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        pass.set_pipeline(&renderer.sort_scan_compose_pipeline);
        pass.set_bind_group(0, &compose_bg, &[]);
        pass.dispatch_workgroups(ssc_x, ssc_y, 1);

        pass.set_pipeline(&renderer.sort_scan_add_pipeline);
        pass.set_bind_group(0, &scan_add_bg, &[]);
        pass.dispatch_workgroups(sort_reduce_x, sort_reduce_y, 1);

        pass.set_pipeline(&renderer.sort_scatter_pipeline);
        pass.set_bind_group(0, &scatter_bg, &[]);
        pass.dispatch_workgroups(sort_count_x, sort_count_y, 1);
    }

    // Copy back to primary if odd number of passes.
    if sort_passes % 2 == 1 {
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

    // Stage 5: tile offsets.
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
        pass.set_pipeline(&renderer.tile_offsets_pipeline);
        pass.set_bind_group(0, &tile_offsets_bg, &[]);
        pass.dispatch_workgroups(tox, toy, 1);
    }

    // Stage 6: rasterize.
    {
        let tile_workgroups: u32 = n_tiles.max(1) as u32;
        let (rx, ry) = dispatch_grid_for_workgroups(tile_workgroups);
        let mut pass: wgpu::ComputePass<'_> =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&renderer.rasterize_pipeline);
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
