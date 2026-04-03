//! Core Gaussian splatting algorithm — no Rerun dependencies.
//!
//! This module contains the pure math, data structures, and GPU rendering
//! pipeline for Gaussian splatting.  Everything here is **Rerun-free** and
//! depends only on `glam` (linear algebra), `rayon` (parallelism), `wgpu`
//! (GPU compute), and `bytemuck` (buffer layout).
//!
//! The Rerun viewer module (`gaussian_visualizer.rs`) and the standalone
//! render CLI (`render_cli.rs`) both import from here.
//!
//! # Module overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`] | Data structures: `RenderGaussianCloud`, `CameraApproximation`, `RenderOutput`, etc. |
//! | [`constants`] | Shared constants: `SH_C0`, `MAX_SPLATS_RENDERED`, `SIGMA_COVERAGE`, etc. |
//! | [`projection`] | 3D→2D Gaussian projection via camera Jacobian |
//! | [`sh`] | Spherical harmonics evaluation (degrees 0–4) |
//! | [`covariance`] | 2D covariance from 3D Gaussian + camera Jacobian |
//! | [`culling`] | CPU frustum culling + depth sorting (pre-GPU pass) |
//! | [`camera`] | Camera constructors (look-at, NeRF transform, fallback) |
//! | [`gpu_types`] | GPU buffer layout structs and helpers |
//! | [`gpu_context`] | Headless wgpu device/queue initialization |
//! | [`gpu_renderer`] | 7-stage GPU compute pipeline (Brush-aligned, no CPU fallback) |

pub mod camera;
pub mod constants;
pub mod covariance;
pub mod culling;
pub mod gpu_context;
pub mod gpu_renderer;
pub mod gpu_types;
pub mod projection;
pub mod sh;
pub mod types;

// ── Convenience re-exports ───────────────────────────────────────────────
// These are the most commonly used items across the codebase.

pub use camera::{camera_from_nerf_transform, fallback_camera, make_camera_approximation};
pub use constants::*;
pub use covariance::{
    brush_bbox_extent_px, brush_camera_jacobian_rows, brush_covariance_in_pixels,
    compensate_covariance_px, mat3_from_affine, pixel_covariance_to_ndc, regularize_covariance,
};
pub use culling::{build_compute_candidate, rebuild_visible_indices};
pub use gpu_context::GpuContext;
pub use gpu_renderer::{GpuRenderer, gpu_render};
pub use projection::{build_prepared_splat, normalize_quat_or_identity, project_gaussian_to_ndc};
pub use sh::{evaluate_sh_rgb, sh_degree_from_coeffs};
pub use types::{
    CameraApproximation, PreparedSplat, ProjectedGaussian, RenderGaussianCloud, RenderOutput,
    RenderShCoefficients, SortedSplatIndex, approximate_bounds_from_points,
};
