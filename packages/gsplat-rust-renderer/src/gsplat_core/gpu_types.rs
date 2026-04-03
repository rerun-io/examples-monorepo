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
