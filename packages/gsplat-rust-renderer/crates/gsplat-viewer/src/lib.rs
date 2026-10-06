//! Native GaussianSplats3D conversion and custom Rerun viewer.
#[cfg(feature = "probe")]
pub mod frame_probe;
pub mod gaussian_renderer;
pub mod gaussian_visualizer;

mod core_adapter;

pub mod automatic_selection;

pub mod bounds;
