//! Camera and image boundaries shared by rendering, benchmarks, and evaluation.
pub mod camera;
pub mod camera_files;
pub mod engines;
mod metrics;
mod provenance;
mod renderer;
pub mod settings;
pub mod statistics;
pub use metrics::published;
pub use metrics::{
    Convention, Evaluation, Evaluator, Metrics, RenderMetrics, ViewMetrics, evaluate_directories,
    mean, pair_directories,
};
pub use provenance::Provenance;
pub use renderer::{Output, PlyScene, StandaloneRenderer, parity_image, raw_splats};

#[derive(Debug, thiserror::Error)]
#[error("unsupported: {0}")]
pub struct Unsupported(pub String);

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;
