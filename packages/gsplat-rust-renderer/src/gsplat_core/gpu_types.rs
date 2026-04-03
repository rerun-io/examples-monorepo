//! GPU buffer layout types and helper functions for the standalone renderer.
//!
//! These are extracted from `gaussian_renderer.rs::gpu_types` and utility
//! functions, with all Rerun (`re_renderer`) dependencies removed.  They
//! mirror the WGSL storage/uniform layouts used by the compute shaders.

use std::borrow::Cow;
use std::sync::Arc;

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
pub const SORT_BIN_COUNT: u32 = 16;
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
    pub _pad: [[u32; 4]; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ScanUniformBuffer {
    pub total_selected: u32,
    pub block_count: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SortUniformBuffer {
    pub shift: u32,
    pub total_keys_unused: u32,
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
    mapped[..bytes.len()].copy_from_slice(bytes);
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
    let pipeline_layout: wgpu::PipelineLayout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}::layout")),
            bind_group_layouts,
            push_constant_ranges: &[],
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

pub fn storage_buffer_entry(binding: u32, buffer: &Arc<wgpu::Buffer>) -> wgpu::BindGroupEntry<'_> {
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

/// All bind group layouts for the 7-stage Gaussian splat compute pipeline.
///
/// These layouts are shared between the Rerun viewer path and the
/// standalone GPU renderer — both bind buffers with the same WGSL
/// shader bindings.
pub struct GpuBindGroupLayouts {
    pub project: wgpu::BindGroupLayout,
    pub scan: wgpu::BindGroupLayout,
    pub scan_block_sums: wgpu::BindGroupLayout,
    pub map: wgpu::BindGroupLayout,
    pub sort_count: wgpu::BindGroupLayout,
    pub sort_reduce: wgpu::BindGroupLayout,
    pub sort_scan: wgpu::BindGroupLayout,
    pub sort_scan_compose: wgpu::BindGroupLayout,
    pub sort_scan_add: wgpu::BindGroupLayout,
    pub sort_scatter: wgpu::BindGroupLayout,
    pub tile_offsets: wgpu::BindGroupLayout,
    pub rasterize: wgpu::BindGroupLayout,
}

/// Create all 12 bind group layouts for the compute pipeline.
pub fn create_compute_bind_group_layouts(device: &wgpu::Device) -> GpuBindGroupLayouts {
    let project = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("project_bgl"),
        entries: &[
            storage_layout_entry(0, true),  // means_world
            storage_layout_entry(1, true),  // quats_xyzw
            storage_layout_entry(2, true),  // scales_opacity
            storage_layout_entry(3, true),  // colors_dc
            storage_layout_entry(4, true),  // sh_coefficients
            storage_layout_entry(5, true),  // sorted_indices
            storage_layout_entry(6, false), // project_output_instances
            storage_layout_entry(7, false), // visibility_flags
            uniform_layout_entry(8, std::mem::size_of::<ProjectUniformBuffer>()),
            storage_layout_entry(9, false),  // projected_tile_splats
            storage_layout_entry(10, false), // projected_tile_hit_counts
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
    let sort_scan = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sort_scan_bgl"),
        entries: &[
            uniform_layout_entry(0, std::mem::size_of::<SortUniformBuffer>()),
            storage_layout_entry(1, false), // reduced
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
        project,
        scan,
        scan_block_sums,
        map,
        sort_count,
        sort_reduce,
        sort_scan,
        sort_scan_compose,
        sort_scan_add,
        sort_scatter,
        tile_offsets,
        rasterize,
    }
}

/// All compute pipelines for the 7-stage Gaussian splat pipeline.
// Some pipelines are retained as lifetime anchors even when not directly
// referenced in every dispatch path.
#[allow(dead_code)]
pub struct GpuComputePipelines {
    pub project: wgpu::ComputePipeline,
    pub scan_blocks: wgpu::ComputePipeline,
    pub scan_block_sums: wgpu::ComputePipeline,
    pub map_intersections: wgpu::ComputePipeline,
    pub clamp_intersection_count: wgpu::ComputePipeline,
    pub sort_count: wgpu::ComputePipeline,
    pub sort_reduce: wgpu::ComputePipeline,
    pub sort_scan: wgpu::ComputePipeline,
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
        project: create_compute_pipeline(
            device,
            "project",
            &project_shader,
            "project_main",
            &[&layouts.project],
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
        sort_scan: create_compute_pipeline(
            device,
            "sort_scan",
            &sort_shader,
            "sort_scan_main",
            &[&layouts.sort_scan],
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
// Uniform buffer fill helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Fill the project uniform buffer from camera and cloud parameters.
pub fn fill_project_uniform(
    camera: &super::types::CameraApproximation,
    selected_limit: usize,
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
            selected_limit as u32,
            coeffs_per_channel,
            sh_degree,
        ],
        _pad: [[
            u32::from(cloud.sh_coeffs.is_some()),
            super::constants::OPACITY_SCALE.to_bits(),
            super::constants::MAX_SPLATS_RENDERED.min(u32::MAX as usize) as u32,
            0,
        ]],
    }
}

/// Fill the scan uniform buffer.
pub fn fill_scan_uniform(selected_limit: usize) -> ScanUniformBuffer {
    ScanUniformBuffer {
        total_selected: selected_limit as u32,
        block_count: compaction_block_count(selected_limit) as u32,
        _pad: [0; 2],
    }
}

/// Fill the map uniform buffer.
pub fn fill_map_uniform(
    selected_limit: usize,
    intersection_capacity: usize,
    tile_bounds: glam::UVec2,
) -> MapUniformBuffer {
    MapUniformBuffer {
        total_selected: selected_limit as u32,
        intersection_capacity: intersection_capacity.min(u32::MAX as usize) as u32,
        tile_bounds_x: tile_bounds.x,
        tile_bounds_y: tile_bounds.y,
    }
}
