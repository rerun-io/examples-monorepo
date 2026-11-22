//! Native `GaussianSplats3D` and `Hexels3D` conversion and custom Rerun viewer.
#[cfg(not(target_arch = "wasm32"))]
pub mod application;
mod asog;
mod bounds;
mod cache;
mod fallback;
mod hexels;
mod renderer;
mod selection;
pub mod setup;
mod trigs;
mod visualizer;

pub use cache::compute_gpu_bytes;
pub use visualizer::camera_from_view;
