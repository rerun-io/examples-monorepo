//! Native `GaussianSplats3D` and `Hexels3D` conversion and custom Rerun viewer.
#[cfg(all(feature = "app", not(target_arch = "wasm32")))]
pub mod application;
mod asog;
mod bounds;
mod cache;
mod fallback;
mod generation_cache;
mod hexels;
mod payload;
pub mod profile;
mod renderer;
mod selection;
pub mod setup;
mod sog;
mod sog_lod;
mod visualizer;

pub use cache::{compute_gpu_bytes, compute_lod_selections};
pub use renderer::{LodSelectionReport, compute_encoded_frames};
pub use visualizer::camera_from_view;
