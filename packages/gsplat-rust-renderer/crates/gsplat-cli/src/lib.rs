//! Camera and image boundaries shared by rendering, benchmarks, and evaluation.
pub mod camera;
pub mod camera_files;
pub mod engines;
pub mod hexel_ply;
pub mod lod;
mod metrics;
mod renderer;
pub mod settings;
pub mod statistics;
pub use gsplat_core::provenance::Provenance;
pub use metrics::published;
pub use metrics::{
    Convention, Evaluation, Evaluator, Metrics, RenderMetrics, ViewMetrics, evaluate_directories,
    mean, pair_directories,
};
pub use renderer::{
    Output, PlyScene, Primitives, StandaloneRenderer, Surface, parity_image, raw_splats,
};

#[derive(Debug, thiserror::Error)]
#[error("unsupported: {0}")]
pub struct Unsupported(pub String);

#[cfg(test)]
mod camera_tests;
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;

/// The provenance of this workspace (packages/gsplat-rust-renderer) for metric reports.
pub fn provenance() -> Provenance {
    Provenance::capture(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_str!("../../../../../recipes/brush-src/recipe.yaml"),
    )
}
