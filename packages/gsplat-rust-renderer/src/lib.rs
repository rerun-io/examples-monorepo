//! Shared library for the Gaussian splat renderer.
//!
//! This crate is structured as a library with two binary entry points:
//!
//! - **`gsplat-rust-renderer`** — Rerun viewer with custom Gaussian splat
//!   visualizer (requires the `viewer` feature).
//! - **`gsplat-render`** — Standalone GPU renderer that produces PNG images
//!   from PLY files + NeRF camera JSON (no Rerun dependency).
//!
//! # Module layout
//!
//! ```text
//! gsplat_core/              ← Core algorithm (no Rerun deps)
//!   types, constants,       ← Data structures + constants
//!   projection, sh,         ← Helpers + spherical harmonics
//!   covariance, culling,    ← 2D covariance + visibility culling
//!   camera,                 ← Camera constructors
//!   gpu_types,              ← GPU buffer layouts + pipeline definitions
//!   gpu_context,            ← Headless wgpu device/queue init
//!   gpu_renderer,           ← 7-stage GPU compute pipeline
//!
//! ply_loader                ← Rust PLY parser
//! nerf_camera               ← NeRF transforms JSON parser
//!
//! gaussian_visualizer       ← Rerun viewer glue (feature-gated)
//! gaussian_renderer         ← Rerun GPU pipeline integration (feature-gated)
//! ```

pub mod gsplat_core;
pub mod nerf_camera;
pub mod ply_loader;

#[cfg(feature = "viewer")]
pub mod gaussian_renderer;
#[cfg(feature = "viewer")]
pub mod gaussian_visualizer;
