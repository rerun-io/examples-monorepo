//! GPU buffer layout types and helper functions for the standalone renderer.
//!
//! These are extracted from `gaussian_renderer.rs::gpu_types` and utility
//! functions, with all Rerun (`re_renderer`) dependencies removed.  They
//! mirror the WGSL storage/uniform layouts used by the compute shaders.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::types::RenderGaussianCloud;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants (must match gaussian_renderer.rs and the WGSL shaders)
// ═══════════════════════════════════════════════════════════════════════════════

pub const PROJECT_WORKGROUP_SIZE: u32 = 128;
pub const COMPACTION_WORKGROUP_SIZE: u32 = 256;
pub const COMPACTION_BLOCK_SIZE: u32 = COMPACTION_WORKGROUP_SIZE * 2;
pub const INTERSECTION_CAPACITY_MULTIPLIER: usize = 32;
pub const SORT_WORKGROUP_SIZE: u32 = 256;
pub const SORT_ELEMENTS_PER_THREAD: u32 = 1;
pub const SORT_BLOCK_SIZE: u32 = SORT_WORKGROUP_SIZE * SORT_ELEMENTS_PER_THREAD;
pub const SORT_BITS_PER_PASS: u32 = 4;
pub const SORT_BIN_COUNT: u32 = 1 << SORT_BITS_PER_PASS;
pub const TILE_WIDTH: u32 = 16;
pub const TILE_OFFSET_WORKGROUP_SIZE: u32 = 256;
pub const TILE_OFFSET_CHECKS_PER_ITER: u32 = 8;
pub const RASTER_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// ═══════════════════════════════════════════════════════════════════════════════
// GPU uniform/storage buffer structs
// ═══════════════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProjectUniformBuffer {
    pub view_from_world: [[f32; 4]; 4],
    pub projection_from_view: [[f32; 4]; 4],
    pub camera_world_position: [f32; 4],
    pub viewport_and_near: [f32; 4],
    pub sigma_and_counts: [u32; 4],
    /// NOT free padding — carries bit-packed values: `_pad[0][0]` = has-SH
    /// flag, `_pad[0][1]` = `OPACITY_SCALE` f32 bits.  Read as
    /// `project_uniforms.pad.x` / `.pad.y` in `gaussian_project.wgsl`;
    /// written by [`fill_project_uniform`].
    pub _pad: [[u32; 4]; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ScanUniformBuffer {
    pub total_selected: u32,
    pub block_count: u32,
    /// Pads the uniform to 16 bytes (WGSL uniform-buffer alignment).
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SortUniformBuffer {
    pub shift: u32,
    pub total_keys_unused: u32,
    /// Pads the uniform to 16 bytes (WGSL uniform-buffer alignment).
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MapUniformBuffer {
    pub total_selected: u32,
    pub intersection_capacity: u32,
    pub tile_bounds_x: u32,
    pub tile_bounds_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RasterUniformBuffer {
    pub tile_bounds: [u32; 2],
    pub img_size: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileProjectedSplat {
    pub xy_px: [f32; 2],
    /// Aligns the next `vec4` member to 16 bytes (WGSL storage layout — see
    /// the layout test in `gaussian_renderer.rs`).
    pub _pad0: [f32; 2],
    pub conic_xyy_opacity: [f32; 4],
    pub color_rgba: [f32; 4],
    pub tile_bbox_min_max: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Buffer creation helpers
// ═══════════════════════════════════════════════════════════════════════════════

pub fn create_filled_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    extra_usage: wgpu::BufferUsages,
    data: &[T],
) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let size: u64 = bytes.len().max(std::mem::size_of::<T>().max(16)) as u64;
    let buffer: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: extra_usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    mapped.slice(..bytes.len()).copy_from_slice(bytes);
    drop(mapped);
    buffer.unmap();
    buffer
}

pub fn create_sized_buffer(
    device: &wgpu::Device,
    label: &str,
    size_bytes: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes.max(16) as u64,
        usage,
        mapped_at_creation: false,
    })
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let bind_group_layouts = bind_group_layouts
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    let pipeline_layout: wgpu::PipelineLayout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}::layout")),
            bind_group_layouts: &bind_group_layouts,
            immediate_size: 0,
        });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bind group layout helpers
// ═══════════════════════════════════════════════════════════════════════════════

pub fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn uniform_layout_entry(binding: u32, size_bytes: usize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size_bytes as u64),
        },
        count: None,
    }
}

pub fn storage_buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data packing helpers (splat data → GPU format)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn pack_vec3s(values: impl Iterator<Item = Vec3>) -> Vec<[f32; 4]> {
    values
        .map(|value| [value.x, value.y, value.z, 0.0])
        .collect()
}

pub fn pack_quats(values: impl Iterator<Item = glam::Quat>) -> Vec<[f32; 4]> {
    values
        .map(|quat| [quat.x, quat.y, quat.z, quat.w])
        .collect()
}

pub fn pack_scales_opacity(cloud: &RenderGaussianCloud) -> Vec<[f32; 4]> {
    cloud
        .scales
        .iter()
        .zip(cloud.opacities.iter())
        .map(|(scale, opacity)| [scale.x, scale.y, scale.z, *opacity])
        .collect()
}

pub fn pack_rgb(values: impl Iterator<Item = [f32; 3]>) -> Vec<[f32; 4]> {
    values.map(|rgb| [rgb[0], rgb[1], rgb[2], 0.0]).collect()
}

pub fn pack_sh_coefficients(cloud: &RenderGaussianCloud) -> Vec<[f32; 4]> {
    cloud
        .sh_coeffs
        .as_ref()
        .map(|sh| {
            sh.coefficients
                .chunks_exact(3)
                .map(|coeff| [coeff[0], coeff[1], coeff[2], 0.0])
                .collect()
        })
        .unwrap_or_else(|| vec![[0.0, 0.0, 0.0, 0.0]])
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sizing / dispatch helpers
// ═══════════════════════════════════════════════════════════════════════════════

pub fn next_capacity(required: usize) -> usize {
    required.max(1).next_power_of_two().max(1024)
}

pub fn intersection_capacity_for_instances(instance_capacity: usize) -> usize {
    (instance_capacity.max(1) * INTERSECTION_CAPACITY_MULTIPLIER)
        .next_power_of_two()
        .max(16)
}

pub fn compaction_block_count(required: usize) -> usize {
    required.max(1).div_ceil(COMPACTION_BLOCK_SIZE as usize)
}

/// Number of count/scatter workgroups for a radix sort over `num_elements`
/// key slots (one thread per slot).
pub fn sort_workgroup_count_for(num_elements: usize) -> u32 {
    (num_elements as u32).div_ceil(SORT_BLOCK_SIZE).max(1)
}

/// Number of `sort_reduce` workgroups for a radix sort with `sort_wg_count`
/// count/scatter workgroups: one reduction slot per bin per reduce block.
pub fn sort_reduce_workgroup_count(sort_wg_count: u32) -> u32 {
    (sort_wg_count.div_ceil(SORT_BLOCK_SIZE) * SORT_BIN_COUNT).max(1)
}

/// Number of radix-sort passes needed to cover all tile-id bits for `n_tiles`
/// tiles.  May be odd — the caller copies back from the alt buffers then.
pub fn tile_sort_passes(n_tiles: usize) -> u32 {
    ((u32::BITS - (n_tiles.max(1) as u32).leading_zeros()).max(1)).div_ceil(SORT_BITS_PER_PASS)
}

pub fn next_block_capacity(required: usize) -> usize {
    compaction_block_count(required).next_power_of_two().max(1)
}

pub fn dispatch_grid_1d(num_elements: u32, workgroup_size: u32) -> (u32, u32) {
    let total_workgroups: u32 = num_elements.div_ceil(workgroup_size).max(1);
    dispatch_grid_for_workgroups(total_workgroups)
}

pub fn dispatch_grid_for_workgroups(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= 65_535 {
        (total_workgroups, 1)
    } else {
        let wg_y: u32 = (total_workgroups as f64).sqrt().ceil() as u32;
        let wg_x: u32 = total_workgroups.div_ceil(wg_y);
        (wg_x, wg_y)
    }
}

pub fn calc_tile_bounds(viewport_size_px: glam::Vec2) -> glam::UVec2 {
    glam::uvec2(
        viewport_size_px.x.max(1.0).ceil() as u32,
        viewport_size_px.y.max(1.0).ceil() as u32,
    )
    .map(|dimension| dimension.div_ceil(TILE_WIDTH))
}

pub fn tile_count(tile_bounds: glam::UVec2) -> usize {
    tile_bounds.x as usize * tile_bounds.y as usize
}

pub fn calc_raster_extent(viewport_size_px: glam::Vec2) -> glam::UVec2 {
    let tile_bounds: glam::UVec2 = calc_tile_bounds(viewport_size_px);
    glam::uvec2(tile_bounds.x * TILE_WIDTH, tile_bounds.y * TILE_WIDTH)
}

pub fn create_shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    })
}

/// Create a raster texture for tile-based rendering output.
///
/// Returns `(texture, view)`.  The caller may pass extra usage flags
/// (e.g. `COPY_SRC` for readback in the standalone renderer).
pub fn create_raster_texture(
    device: &wgpu::Device,
    label: &str,
    extent: glam::UVec2,
    extra_usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: extent.x.max(1),
            height: extent.y.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RASTER_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra_usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared bind group layouts + compute pipelines
// ═══════════════════════════════════════════════════════════════════════════════

/// All bind group layouts for the GPU-only Gaussian splat compute pipeline.
///
/// These layouts are shared between the Rerun viewer path and the
/// standalone GPU renderer — both bind buffers with the same WGSL
/// shader bindings.
pub struct GpuBindGroupLayouts {
    pub project_forward: wgpu::BindGroupLayout,
    pub project_visible: wgpu::BindGroupLayout,
    pub scan: wgpu::BindGroupLayout,
    pub scan_block_sums: wgpu::BindGroupLayout,
    pub map: wgpu::BindGroupLayout,
    pub sort_count: wgpu::BindGroupLayout,
    pub sort_reduce: wgpu::BindGroupLayout,
    pub sort_scan_compose: wgpu::BindGroupLayout,
    pub sort_scan_add: wgpu::BindGroupLayout,
    pub sort_scatter: wgpu::BindGroupLayout,
    pub tile_offsets: wgpu::BindGroupLayout,
    pub rasterize: wgpu::BindGroupLayout,
}

/// Create all 12 bind group layouts for the compute pipeline.
pub fn create_compute_bind_group_layouts(device: &wgpu::Device) -> GpuBindGroupLayouts {
    let project_forward = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("project_forward_bgl"),
        entries: &[
            storage_layout_entry(0, true), // means_world
            storage_layout_entry(1, true), // quats_xyzw
            storage_layout_entry(2, true), // scales_opacity
            uniform_layout_entry(8, std::mem::size_of::<ProjectUniformBuffer>()),
            storage_layout_entry(12, false), // forward_global_from_compact
            storage_layout_entry(13, false), // forward_depth_keys
            storage_layout_entry(14, false), // forward_num_visible (atomic)
        ],
    });
    let project_visible = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("project_visible_bgl"),
        entries: &[
            storage_layout_entry(0, true), // means_world
            storage_layout_entry(1, true), // quats_xyzw
            storage_layout_entry(2, true), // scales_opacity
            storage_layout_entry(3, true), // colors_dc
            storage_layout_entry(4, true), // sh_coefficients
            storage_layout_entry(5, true), // global_from_compact (depth-sorted)
            uniform_layout_entry(8, std::mem::size_of::<ProjectUniformBuffer>()),
            storage_layout_entry(9, false),  // projected_tile_splats
            storage_layout_entry(10, false), // projected_tile_hit_counts
            storage_layout_entry(11, true),  // num_visible_in
        ],
    });
    let scan = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scan_bgl"),
        entries: &[
            storage_layout_entry(16, true),  // scan input
            storage_layout_entry(17, false), // local_offsets
            storage_layout_entry(18, false), // block_offsets
            uniform_layout_entry(19, std::mem::size_of::<ScanUniformBuffer>()),
        ],
    });
    let scan_block_sums = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scan_block_sums_bgl"),
        entries: &[
            storage_layout_entry(24, false), // block_offsets
            storage_layout_entry(25, false), // indirect_draw / totals
            uniform_layout_entry(26, std::mem::size_of::<ScanUniformBuffer>()),
        ],
    });
    let map = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("map_bgl"),
        entries: &[
            storage_layout_entry(0, true),  // projected_tile_splats
            storage_layout_entry(1, true),  // tile_hit_offsets
            storage_layout_entry(2, true),  // tile_hit_counts
            storage_layout_entry(3, true),  // tile_hit_block_offsets
            storage_layout_entry(4, false), // tile_id_from_isect
            storage_layout_entry(5, false), // compact_gid_from_isect
            uniform_layout_entry(6, std::mem::size_of::<MapUniformBuffer>()),
            storage_layout_entry(7, true),  // tile_intersection_count
            storage_layout_entry(8, false), // num_intersections
        ],
    });
    let sort_count = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sort_count_bgl"),
        entries: &[
            uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
            storage_layout_entry(1, true),  // src_keys
            storage_layout_entry(2, false), // counts
            storage_layout_entry(6, true),  // num_intersections
        ],
    });
    let sort_reduce = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sort_reduce_bgl"),
        entries: &[
            uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
            storage_layout_entry(1, true),  // counts
            storage_layout_entry(2, false), // reduced
            storage_layout_entry(6, true),  // num_intersections
        ],
    });
    let sort_scan_compose = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sort_scan_compose_bgl"),
        entries: &[
            storage_layout_entry(8, true),   // offsets
            storage_layout_entry(9, true),   // block_offsets
            storage_layout_entry(10, false), // out
            uniform_layout_entry(11, std::mem::size_of::<ScanUniformBuffer>()),
        ],
    });
    let sort_scan_add = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sort_scan_add_bgl"),
        entries: &[
            uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
            storage_layout_entry(1, true),  // reduced
            storage_layout_entry(2, false), // counts
            storage_layout_entry(6, true),  // num_intersections
        ],
    });
    let sort_scatter = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let tile_offsets = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tile_offsets_bgl"),
        entries: &[
            storage_layout_entry(0, true),  // sorted_tile_ids
            storage_layout_entry(1, false), // tile_offsets
            storage_layout_entry(2, true),  // num_intersections
        ],
    });
    let rasterize = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    GpuBindGroupLayouts {
        project_forward,
        project_visible,
        scan,
        scan_block_sums,
        map,
        sort_count,
        sort_reduce,
        sort_scan_compose,
        sort_scan_add,
        sort_scatter,
        tile_offsets,
        rasterize,
    }
}

/// All compute pipelines for the GPU-only Gaussian splat pipeline.
pub struct GpuComputePipelines {
    pub project_forward: wgpu::ComputePipeline,
    pub project_visible: wgpu::ComputePipeline,
    pub scan_blocks: wgpu::ComputePipeline,
    pub scan_block_sums: wgpu::ComputePipeline,
    pub map_intersections: wgpu::ComputePipeline,
    pub clamp_intersection_count: wgpu::ComputePipeline,
    pub sort_count: wgpu::ComputePipeline,
    pub sort_reduce: wgpu::ComputePipeline,
    pub sort_scan_compose: wgpu::ComputePipeline,
    pub sort_scan_add: wgpu::ComputePipeline,
    pub sort_scatter: wgpu::ComputePipeline,
    pub tile_offsets: wgpu::ComputePipeline,
    pub rasterize: wgpu::ComputePipeline,
}

/// Create all 13 compute pipelines from the 5 embedded WGSL shaders.
pub fn create_compute_pipelines(
    device: &wgpu::Device,
    layouts: &GpuBindGroupLayouts,
) -> GpuComputePipelines {
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

    GpuComputePipelines {
        project_forward: create_compute_pipeline(
            device,
            "project_forward",
            &project_shader,
            "project_forward_main",
            &[&layouts.project_forward],
        ),
        project_visible: create_compute_pipeline(
            device,
            "project_visible",
            &project_shader,
            "project_visible_main",
            &[&layouts.project_visible],
        ),
        scan_blocks: create_compute_pipeline(
            device,
            "scan_blocks",
            &project_shader,
            "scan_blocks_main",
            &[&layouts.scan],
        ),
        scan_block_sums: create_compute_pipeline(
            device,
            "scan_block_sums",
            &project_shader,
            "scan_block_sums_main",
            &[&layouts.scan_block_sums],
        ),
        map_intersections: create_compute_pipeline(
            device,
            "map_intersections",
            &map_shader,
            "map_main",
            &[&layouts.map],
        ),
        clamp_intersection_count: create_compute_pipeline(
            device,
            "clamp_intersection_count",
            &map_shader,
            "clamp_count_main",
            &[&layouts.map],
        ),
        sort_count: create_compute_pipeline(
            device,
            "sort_count",
            &sort_shader,
            "sort_count_main",
            &[&layouts.sort_count],
        ),
        sort_reduce: create_compute_pipeline(
            device,
            "sort_reduce",
            &sort_shader,
            "sort_reduce_main",
            &[&layouts.sort_reduce],
        ),
        sort_scan_compose: create_compute_pipeline(
            device,
            "sort_scan_compose",
            &sort_shader,
            "sort_scan_compose_main",
            &[&layouts.sort_scan_compose],
        ),
        sort_scan_add: create_compute_pipeline(
            device,
            "sort_scan_add",
            &sort_shader,
            "sort_scan_add_main",
            &[&layouts.sort_scan_add],
        ),
        sort_scatter: create_compute_pipeline(
            device,
            "sort_scatter",
            &sort_shader,
            "sort_scatter_main",
            &[&layouts.sort_scatter],
        ),
        tile_offsets: create_compute_pipeline(
            device,
            "tile_offsets",
            &tile_offsets_shader,
            "main",
            &[&layouts.tile_offsets],
        ),
        rasterize: create_compute_pipeline(
            device,
            "rasterize",
            &rasterize_shader,
            "main",
            &[&layouts.rasterize],
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared pipeline bind groups
// ═══════════════════════════════════════════════════════════════════════════════

/// Borrowed references to every buffer (and the raster texture view) the
/// non-sort pipeline stages bind.  Single source of truth for the WGSL
/// binding wiring, shared by the viewer and the standalone renderer.
pub struct PipelineBuffers<'a> {
    pub means: &'a wgpu::Buffer,
    pub quats: &'a wgpu::Buffer,
    pub scales_opacity: &'a wgpu::Buffer,
    pub colors: &'a wgpu::Buffer,
    pub sh_coeffs: &'a wgpu::Buffer,
    pub project_ub: &'a wgpu::Buffer,
    pub scan_ub: &'a wgpu::Buffer,
    pub scan_l2_ub: &'a wgpu::Buffer,
    pub map_ub: &'a wgpu::Buffer,
    pub raster_ub: &'a wgpu::Buffer,
    pub num_visible: &'a wgpu::Buffer,
    pub global_from_compact: &'a wgpu::Buffer,
    pub depth_keys: &'a wgpu::Buffer,
    pub projected: &'a wgpu::Buffer,
    pub tile_hit_counts: &'a wgpu::Buffer,
    pub tile_hit_offsets: &'a wgpu::Buffer,
    pub tile_hit_block_offsets: &'a wgpu::Buffer,
    pub block_local_offsets: &'a wgpu::Buffer,
    pub block2_offsets: &'a wgpu::Buffer,
    pub tile_isect_count: &'a wgpu::Buffer,
    pub num_isect: &'a wgpu::Buffer,
    pub tile_id_from_isect: &'a wgpu::Buffer,
    pub compact_gid_from_isect: &'a wgpu::Buffer,
    pub tile_offsets: &'a wgpu::Buffer,
    pub raster_view: &'a wgpu::TextureView,
}

/// Bind groups for the non-sort pipeline stages, in dispatch order.
pub struct PipelineBindGroups {
    pub project_forward: wgpu::BindGroup,
    pub project_visible: wgpu::BindGroup,
    pub scan: wgpu::BindGroup,
    pub scan_l2: wgpu::BindGroup,
    pub scan_block_sums: wgpu::BindGroup,
    pub scan_compose: wgpu::BindGroup,
    pub map: wgpu::BindGroup,
    pub tile_offsets: wgpu::BindGroup,
    pub rasterize: wgpu::BindGroup,
}

/// Create the bind groups for every non-sort pipeline stage.
pub fn create_pipeline_bind_groups(
    device: &wgpu::Device,
    layouts: &GpuBindGroupLayouts,
    b: &PipelineBuffers<'_>,
) -> PipelineBindGroups {
    PipelineBindGroups {
        project_forward: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project_forward_bg"),
            layout: &layouts.project_forward,
            entries: &[
                storage_buffer_entry(0, b.means),
                storage_buffer_entry(1, b.quats),
                storage_buffer_entry(2, b.scales_opacity),
                storage_buffer_entry(8, b.project_ub),
                storage_buffer_entry(12, b.global_from_compact),
                storage_buffer_entry(13, b.depth_keys),
                storage_buffer_entry(14, b.num_visible),
            ],
        }),
        project_visible: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project_visible_bg"),
            layout: &layouts.project_visible,
            entries: &[
                storage_buffer_entry(0, b.means),
                storage_buffer_entry(1, b.quats),
                storage_buffer_entry(2, b.scales_opacity),
                storage_buffer_entry(3, b.colors),
                storage_buffer_entry(4, b.sh_coeffs),
                storage_buffer_entry(5, b.global_from_compact),
                storage_buffer_entry(8, b.project_ub),
                storage_buffer_entry(9, b.projected),
                storage_buffer_entry(10, b.tile_hit_counts),
                storage_buffer_entry(11, b.num_visible),
            ],
        }),
        scan: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_bg"),
            layout: &layouts.scan,
            entries: &[
                storage_buffer_entry(16, b.tile_hit_counts), // scan input: per-splat tile hit counts
                storage_buffer_entry(17, b.tile_hit_offsets), // scan output: prefix-sum offsets
                storage_buffer_entry(18, b.tile_hit_block_offsets),
                storage_buffer_entry(19, b.scan_ub),
            ],
        }),
        // Level 2: scan the level-1 block sums so clouds larger than
        // 512 blocks * 512 elements = 262,144 splats scan correctly.
        scan_l2: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_l2_bg"),
            layout: &layouts.scan,
            entries: &[
                storage_buffer_entry(16, b.tile_hit_block_offsets),
                storage_buffer_entry(17, b.block_local_offsets),
                storage_buffer_entry(18, b.block2_offsets),
                storage_buffer_entry(19, b.scan_l2_ub),
            ],
        }),
        scan_block_sums: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_block_sums_bg"),
            layout: &layouts.scan_block_sums,
            entries: &[
                storage_buffer_entry(24, b.block2_offsets),
                storage_buffer_entry(25, b.tile_isect_count), // total intersection count
                storage_buffer_entry(26, b.scan_l2_ub),
            ],
        }),
        // Compose the two scan levels back into flat per-block offsets.
        scan_compose: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_compose_bg"),
            layout: &layouts.sort_scan_compose,
            entries: &[
                storage_buffer_entry(8, b.block_local_offsets),
                storage_buffer_entry(9, b.block2_offsets),
                storage_buffer_entry(10, b.tile_hit_block_offsets),
                storage_buffer_entry(11, b.scan_l2_ub),
            ],
        }),
        map: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map_bg"),
            layout: &layouts.map,
            entries: &[
                storage_buffer_entry(0, b.projected),
                storage_buffer_entry(1, b.tile_hit_offsets),
                storage_buffer_entry(2, b.tile_hit_counts),
                storage_buffer_entry(3, b.tile_hit_block_offsets),
                storage_buffer_entry(4, b.tile_id_from_isect),
                storage_buffer_entry(5, b.compact_gid_from_isect),
                storage_buffer_entry(6, b.map_ub),
                storage_buffer_entry(7, b.tile_isect_count),
                storage_buffer_entry(8, b.num_isect),
            ],
        }),
        tile_offsets: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_offsets_bg"),
            layout: &layouts.tile_offsets,
            entries: &[
                storage_buffer_entry(0, b.tile_id_from_isect),
                storage_buffer_entry(1, b.tile_offsets),
                storage_buffer_entry(2, b.num_isect),
            ],
        }),
        rasterize: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rasterize_bg"),
            layout: &layouts.rasterize,
            entries: &[
                storage_buffer_entry(0, b.compact_gid_from_isect),
                storage_buffer_entry(1, b.tile_offsets),
                storage_buffer_entry(2, b.projected),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(b.raster_view),
                },
                storage_buffer_entry(4, b.raster_ub),
            ],
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU radix argsort (count-buffer driven, brush v0.3.0 pattern)
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum number of 4-bit radix passes over 32-bit keys.
pub const MAX_SORT_PASSES: u32 = 32 / SORT_BITS_PER_PASS;

/// Create the [`MAX_SORT_PASSES`] shift uniform buffers (`shift = pass * 4`).
/// Globally constant — create once per renderer and share across all sorts.
pub fn create_sort_shift_uniforms(device: &wgpu::Device) -> Vec<wgpu::Buffer> {
    (0..MAX_SORT_PASSES)
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
        .collect()
}

/// Buffers for one GPU radix argsort over `(key, value)` pairs.
///
/// The number of keys is read on the GPU from `num_keys[0]` — dispatches are
/// capacity-sized and early-exit, so no CPU readback or indirect dispatch is
/// needed.  Used three times per frame: gid canonicalization + depth argsort
/// (keys = f32 depth bits, values = global gids) and tile-id sort (keys =
/// tile ids, values = compact gids).
pub struct RadixSortBuffers<'a> {
    pub keys_primary: &'a wgpu::Buffer,
    pub vals_primary: &'a wgpu::Buffer,
    pub keys_alt: &'a wgpu::Buffer,
    pub vals_alt: &'a wgpu::Buffer,
    /// GPU buffer whose first u32 is the number of keys to sort.
    pub num_keys: &'a wgpu::Buffer,
}

/// Scratch buffers for the radix sort.  The sorts in a frame run
/// sequentially in one encoder, so a single set (sized for the largest sort)
/// serves all of them.
pub struct RadixSortScratch<'a> {
    pub counts: &'a wgpu::Buffer,
    pub reduced: &'a wgpu::Buffer,
    pub scan_offsets: &'a wgpu::Buffer,
    pub scan_block_offsets: &'a wgpu::Buffer,
    pub scan_totals: &'a wgpu::Buffer,
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
/// for an odd count the caller must copy back from the alt buffers
/// (check [`RadixSort::num_passes`]).
pub struct RadixSort {
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
/// `shift_ubs` comes from [`create_sort_shift_uniforms`].
pub fn build_radix_sort(
    device: &wgpu::Device,
    layouts: &GpuBindGroupLayouts,
    shift_ubs: &[wgpu::Buffer],
    buffers: &RadixSortBuffers<'_>,
    scratch: &RadixSortScratch<'_>,
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
                        storage_buffer_entry(2, scratch.counts),
                        storage_buffer_entry(6, buffers.num_keys),
                    ],
                }),
                reduce_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_reduce_bg"),
                    layout: &layouts.sort_reduce,
                    entries: &[
                        storage_buffer_entry(0, sort_ub),
                        storage_buffer_entry(1, scratch.counts),
                        storage_buffer_entry(2, scratch.reduced),
                        storage_buffer_entry(6, buffers.num_keys),
                    ],
                }),
                scan_add_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort_scan_add_bg"),
                    layout: &layouts.sort_scan_add,
                    entries: &[
                        storage_buffer_entry(0, sort_ub),
                        storage_buffer_entry(1, scratch.reduced),
                        storage_buffer_entry(2, scratch.counts),
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
                        storage_buffer_entry(3, scratch.counts),
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
                storage_buffer_entry(16, scratch.reduced),
                storage_buffer_entry(17, scratch.scan_offsets),
                storage_buffer_entry(18, scratch.scan_block_offsets),
                storage_buffer_entry(19, &scan_sort_ub),
            ],
        }),
        scan_block_sums_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_block_sums_bg"),
            layout: &layouts.scan_block_sums,
            entries: &[
                storage_buffer_entry(24, scratch.scan_block_offsets),
                storage_buffer_entry(25, scratch.scan_totals),
                storage_buffer_entry(26, &scan_sort_ub),
            ],
        }),
        compose_bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort_scan_compose_bg"),
            layout: &layouts.sort_scan_compose,
            entries: &[
                storage_buffer_entry(8, scratch.scan_offsets),
                storage_buffer_entry(9, scratch.scan_block_offsets),
                storage_buffer_entry(10, scratch.reduced),
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
    /// Total radix passes; odd means the result is in the alt buffers and the
    /// caller must copy back to primary.
    pub fn num_passes(&self) -> u32 {
        self.passes.len() as u32
    }

    /// Encode all radix passes into the given compute pass.  Dispatch order is
    /// identical to encoding each pass separately — WebGPU guarantees
    /// dispatch-order visibility within a compute pass.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>, pipelines: &GpuComputePipelines) {
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
// Uniform buffer fill helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Fill the project uniform buffer from camera and cloud parameters.
///
/// `total_splats` is the full cloud size — the GPU project_forward pass culls
/// the whole cloud itself (no CPU pre-pass).
pub fn fill_project_uniform(
    camera: &super::types::CameraApproximation,
    total_splats: usize,
    cloud: &super::types::RenderGaussianCloud,
) -> ProjectUniformBuffer {
    let coeffs_per_channel: u32 = cloud
        .sh_coeffs
        .as_ref()
        .map_or(0, |sh| sh.coeffs_per_channel as u32);
    let sh_degree: u32 = super::sh::sh_degree_from_coeffs(coeffs_per_channel as usize).unwrap_or(0);
    ProjectUniformBuffer {
        view_from_world: glam::Mat4::from(camera.view_from_world).to_cols_array_2d(),
        projection_from_view: camera.projection_from_view.to_cols_array_2d(),
        camera_world_position: camera.world_position.extend(0.0).to_array(),
        viewport_and_near: [
            camera.viewport_size_px.x,
            camera.viewport_size_px.y,
            camera.near_plane,
            super::constants::MIN_RADIUS_PX,
        ],
        sigma_and_counts: [
            super::constants::SIGMA_COVERAGE.to_bits(),
            total_splats.min(u32::MAX as usize) as u32,
            coeffs_per_channel,
            sh_degree,
        ],
        _pad: [[
            u32::from(cloud.sh_coeffs.is_some()),
            super::constants::OPACITY_SCALE.to_bits(),
            0,
            0,
        ]],
    }
}

/// Fill the scan uniform buffer for a scan over `total_elements` entries.
pub fn fill_scan_uniform(total_elements: usize) -> ScanUniformBuffer {
    ScanUniformBuffer {
        total_selected: total_elements as u32,
        block_count: compaction_block_count(total_elements) as u32,
        _pad: [0; 2],
    }
}

/// Fill the map uniform buffer.
pub fn fill_map_uniform(
    total_splats: usize,
    intersection_capacity: usize,
    tile_bounds: glam::UVec2,
) -> MapUniformBuffer {
    MapUniformBuffer {
        total_selected: total_splats.min(u32::MAX as usize) as u32,
        intersection_capacity: intersection_capacity.min(u32::MAX as usize) as u32,
        tile_bounds_x: tile_bounds.x,
        tile_bounds_y: tile_bounds.y,
    }
}

/// Number of radix-sort passes for the GPU depth argsort: 32-bit keys at
/// 4 bits per pass.  Even, so sorted results land back in the primary buffers
/// without a copy-back.
pub const DEPTH_SORT_PASSES: u32 = 32 / SORT_BITS_PER_PASS;

/// Number of radix-sort passes to canonicalize the compacted gid order before
/// the depth sort.
///
/// project_forward compacts visible splats with `atomicAdd`, so the compact
/// order is nondeterministic (brush has the same race).  Distinct splats can
/// share identical f32 depth bits, and the stable depth sort preserves input
/// order for equal keys — so without canonicalization the blend order of
/// depth-tied splats (and thus the rendered image) varies run to run.
/// Sorting the (gid, depth) pairs by gid first makes depth ties resolve in
/// ascending-gid order, deterministically.  Rounded up to an even pass count
/// so results land back in the primary buffers.
pub fn gid_sort_passes(total_splats: usize) -> u32 {
    let max_gid: u32 = total_splats.saturating_sub(1).max(1).min(u32::MAX as usize) as u32;
    let bits: u32 = u32::BITS - max_gid.leading_zeros();
    let passes: u32 = bits.div_ceil(SORT_BITS_PER_PASS);
    passes + (passes % 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL shaders are standalone compilation units, so each declares
    /// its own copies of the workgroup-size constants.  This test ties them
    /// to the Rust constants the dispatch math uses — change one side and
    /// the test names the file to update.
    #[test]
    fn wgsl_constants_match_rust() {
        let cases: [(&str, &str, String); 9] = [
            (
                "gaussian_project.wgsl",
                include_str!("../../shader/gaussian_project.wgsl"),
                format!("const PROJECT_WORKGROUP_SIZE: u32 = {PROJECT_WORKGROUP_SIZE}u;"),
            ),
            (
                "gaussian_project.wgsl",
                include_str!("../../shader/gaussian_project.wgsl"),
                format!("const COMPACTION_WORKGROUP_SIZE: u32 = {COMPACTION_WORKGROUP_SIZE}u;"),
            ),
            (
                "gaussian_project.wgsl",
                include_str!("../../shader/gaussian_project.wgsl"),
                format!("const TILE_WIDTH: u32 = {TILE_WIDTH}u;"),
            ),
            (
                "gaussian_map_intersections.wgsl",
                include_str!("../../shader/gaussian_map_intersections.wgsl"),
                format!("const PROJECT_WORKGROUP_SIZE: u32 = {PROJECT_WORKGROUP_SIZE}u;"),
            ),
            (
                "gaussian_map_intersections.wgsl",
                include_str!("../../shader/gaussian_map_intersections.wgsl"),
                format!("const COMPACTION_BLOCK_SIZE: u32 = {COMPACTION_BLOCK_SIZE}u;"),
            ),
            (
                "gaussian_dynamic_sort.wgsl",
                include_str!("../../shader/gaussian_dynamic_sort.wgsl"),
                format!("const WG: u32 = {SORT_WORKGROUP_SIZE}u;"),
            ),
            (
                "gaussian_dynamic_sort.wgsl",
                include_str!("../../shader/gaussian_dynamic_sort.wgsl"),
                format!("const BITS_PER_PASS: u32 = {SORT_BITS_PER_PASS}u;"),
            ),
            (
                "gaussian_tile_offsets.wgsl",
                include_str!("../../shader/gaussian_tile_offsets.wgsl"),
                format!("const TILE_SIZE: u32 = {TILE_OFFSET_WORKGROUP_SIZE}u;"),
            ),
            (
                "gaussian_tile_offsets.wgsl",
                include_str!("../../shader/gaussian_tile_offsets.wgsl"),
                format!("const CHECKS_PER_ITER: u32 = {TILE_OFFSET_CHECKS_PER_ITER}u;"),
            ),
        ];
        for (file, source, expected) in cases {
            assert!(
                source.contains(&expected),
                "{file} does not declare `{expected}` — keep the WGSL constants in sync with gpu_types.rs"
            );
        }
        assert!(
            include_str!("../../shader/gaussian_dynamic_sort.wgsl").contains(&format!(
                "const ELEMENTS_PER_THREAD: u32 = {SORT_ELEMENTS_PER_THREAD}u;"
            )),
            "gaussian_dynamic_sort.wgsl ELEMENTS_PER_THREAD drifted from SORT_ELEMENTS_PER_THREAD"
        );
    }

    #[test]
    fn positive_f32_depth_bits_sort_ascending_as_u32() {
        // project_forward stores bitcast<u32>(camera_depth); positive floats
        // compare identically as unsigned ints, so the radix argsort yields
        // front-to-back order directly.
        let near: u32 = 2.0_f32.to_bits();
        let mid: u32 = 10.5_f32.to_bits();
        let far: u32 = 1.0e6_f32.to_bits();
        assert!(near < mid);
        assert!(mid < far);
    }

    #[test]
    fn depth_sort_pass_count_is_even() {
        assert_eq!(super::DEPTH_SORT_PASSES % 2, 0);
        assert_eq!(super::DEPTH_SORT_PASSES * 4, 32);
    }

    #[test]
    fn gid_sort_passes_cover_all_gid_bits_and_are_even() {
        for total in [1usize, 2, 17, 256, 40_000, 262_145, 2_000_000] {
            let passes = super::gid_sort_passes(total);
            assert_eq!(passes % 2, 0, "total={total}");
            let max_gid = total.saturating_sub(1) as u64;
            assert!(u64::from(passes) * 4 >= 64 - u64::from(max_gid.max(1).leading_zeros()));
        }
    }
}
