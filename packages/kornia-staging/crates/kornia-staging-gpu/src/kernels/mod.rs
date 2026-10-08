//! Shared CubeCL buffer layout and sampling operations.
pub(crate) mod layout;
mod sampling;
pub(crate) use sampling::launch_probe;

mod pyramid;
pub(crate) use layout::Buffer;
pub(crate) use layout::MAX_CUBES_PER_DIM;
pub(crate) use pyramid::{launch_ingest, launch_subsample, launch_subsample_batch};
pub(crate) use sampling::launch_copy_level0;

mod cell_select;
mod fast;
mod fast_cell;
pub(crate) use cell_select::{launch_fast_cell_select, uses_cell_kernel, CellSelectGeometry};
pub(crate) use fast::{
    launch_fast_localmax, launch_fast_mask, launch_fast_score, MASK_BITS, RING_BIAS,
};
pub(crate) use fast_cell::{cell_shared_bytes, launch_fast_cell, launch_fast_cell_batch};

/// Low-level fused KLT kernel for callers that supply validated device bindings.
pub(crate) mod klt_fused;

/// IEEE float classification for device code.
pub mod finite;
mod trig;

#[cfg(all(test, feature = "wgpu"))]
mod tests;
