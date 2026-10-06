//! Portable CubeCL image operations, shared across Vulkan and Metal devices.
//!
//! ```no_run
//! # #[cfg(feature = "wgpu")] {
//! use kornia_staging_gpu::runtime::{gpu_client, probe_storage, probe_subgroups};
//! let client = gpu_client()?;
//! probe_storage(&client)?;
//! let subgroup_width = probe_subgroups(&client)?;
//! assert!(subgroup_width >= 16);
//! # }
//! # Ok::<(), kornia_staging_gpu::runtime::GpuError>(())
//! ```
pub mod kernels;
pub mod runtime;
pub mod transfer;
/// Portable GPU runtime, with target-specific shader compiler selection.
#[cfg(feature = "wgpu")]
pub type GpuRuntime = cubecl_wgpu::WgpuRuntime;
