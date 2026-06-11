//! Core Gaussian splatting algorithm — no Rerun dependencies.
//!
//! This module contains the pure math, data structures, and GPU rendering
//! pipeline for Gaussian splatting.  Everything here is **Rerun-free** and
//! depends only on `glam` (linear algebra), `wgpu` (GPU compute), and
//! `bytemuck` (buffer layout).
//!
//! The Rerun viewer module (`gaussian_visualizer.rs`) and the standalone
//! render CLI (`render_cli.rs`) both import from here.
//!
//! # Module overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`] | Data structures: `RenderGaussianCloud`, `CameraApproximation`, `RenderOutput`, etc. |
//! | [`constants`] | Shared constants: `SH_C0`, `SIGMA_COVERAGE`, etc. |
//! | [`projection`] | Quaternion helpers |
//! | [`sh`] | Spherical harmonics metadata (degree from coefficient count) |
//! | [`camera`] | Camera constructors (look-at, NeRF transform, fallback) |
//! | [`gpu_types`] | GPU buffer layout structs and helpers |
//! | [`gpu_context`] | Headless wgpu device/queue initialization |
//! | [`gpu_renderer`] | GPU-only compute pipeline (Brush-aligned: cull + depth sort on GPU) |

pub mod camera;
pub mod constants;
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
pub use gpu_context::GpuContext;
pub use gpu_renderer::{GpuRenderResources, GpuRenderer};
pub use projection::normalize_quat_or_identity;
pub use sh::sh_degree_from_coeffs;
pub use types::{
    CameraApproximation, RenderGaussianCloud, RenderOutput, RenderShCoefficients,
    approximate_bounds_from_points,
};
