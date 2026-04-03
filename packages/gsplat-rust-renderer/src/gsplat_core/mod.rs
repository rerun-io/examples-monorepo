//! Core Gaussian splatting algorithm — no Rerun dependencies.
//!
//! This module contains the pure math and data structures for Gaussian splat
//! rendering: 3D→2D projection, spherical harmonics evaluation, 2D covariance
//! computation, visibility culling, depth sorting, and a CPU software
//! rasterizer.
//!
//! Everything in this module is **Rerun-free** and depends only on `glam`
//! (linear algebra) and `rayon` (parallelism).  The Rerun viewer module
//! (`gaussian_visualizer.rs`) and the standalone render CLI (`render_cli.rs`)
//! both import from here.
//!
//! # Module overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`] | Data structures: `RenderGaussianCloud`, `CameraApproximation`, etc. |
//! | [`constants`] | Shared constants: `SH_C0`, `MAX_SPLATS_RENDERED`, `SIGMA_COVERAGE`, etc. |
//! | [`projection`] | 3D→2D Gaussian projection via camera Jacobian |
//! | [`sh`] | Spherical harmonics evaluation (degrees 0–4) |
//! | [`covariance`] | 2D covariance from 3D Gaussian + camera Jacobian |
//! | [`culling`] | CPU frustum culling + depth sorting |
//! | [`camera`] | Camera constructors (look-at, NeRF transform, fallback) |
//! | [`software_rasterizer`] | CPU tile rasterizer for correctness testing |

pub mod camera;
pub mod constants;
pub mod covariance;
pub mod culling;
pub mod projection;
pub mod sh;
pub mod software_rasterizer;
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
pub use projection::{build_prepared_splat, normalize_quat_or_identity, project_gaussian_to_ndc};
pub use sh::{evaluate_sh_rgb, sh_degree_from_coeffs};
pub use software_rasterizer::{RenderOutput, software_render};
pub use types::{
    CameraApproximation, PreparedSplat, ProjectedGaussian, RenderGaussianCloud,
    RenderShCoefficients, SortedSplatIndex, approximate_bounds_from_points,
};
