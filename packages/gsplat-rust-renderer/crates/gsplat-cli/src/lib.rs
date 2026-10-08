//! Camera and image boundaries shared by rendering, benchmarks, and evaluation.
pub mod metrics;
mod provenance;
pub use metrics::{
    Convention, Evaluation, Evaluator, Metrics, ViewMetrics, evaluate_directories, mean,
    pair_directories,
};
pub use provenance::Provenance;

pub use anyhow::{Error, Result};
