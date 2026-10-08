//! GPU-resident Gaussian splat rendering. Algorithms follow Brush 1388f74c.
//!
//! Native devices must enable `wgpu::Features::SUBGROUP` and WebGPU compute limits.
//! A future web build needs the WebGPU SUBGROUP mapping from at least Brush's wgpu
//! fork commit `4db81837f`, and `enable subgroups;` prepended on WebGPU only.
//! Naga 30 rejects that directive; no browser path is implemented here.
// The renderer in the next commit is the first caller outside the tests.
#![allow(dead_code, unused_imports)]

// Optional depth raster: 256 * (9 splat floats + 1 depth float), plus four shared scalars.
pub(crate) const REQUIRED_WORKGROUP_STORAGE_BYTES: u32 = 10_256;

mod gpu;
mod kernels;
mod primitives;
mod shader;

#[cfg(test)]
mod primitive_tests;
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_utils;
