//! Shared fixtures for GPU integration binaries.

#[allow(unused_imports, reason = "integration binaries use different fixtures")]
pub use bands::band_at;
use kornia_staging_imgproc::test_fixtures as bands;

#[allow(unused_imports, reason = "integration binaries use different fixtures")]
pub use super::gpu_flow::*;
