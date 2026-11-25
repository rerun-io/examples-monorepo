//! Native `GaussianSplats3D` and `Hexels3D` conversion and custom Rerun viewer.
#[cfg(not(target_arch = "wasm32"))]
pub mod application;
mod asog;
mod bounds;
mod cache;
mod fallback;
mod hexels;
mod payload;
pub mod profile;
mod renderer;
mod selection;
pub mod setup;
mod sog;
mod visualizer;

pub use cache::compute_gpu_bytes;
pub use renderer::compute_encoded_frames;
pub use visualizer::camera_from_view;
