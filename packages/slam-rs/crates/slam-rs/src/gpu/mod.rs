//! CubeCL frontend: persistent pyramids, FAST selection and fused KLT.
//! The frame coordinator owns current/next resources and one typed launch list.
//! Patch tracking has no frame-phase hooks. All supported GPUs use the same
//! fused kernel after storage and subgroup startup probes.
//!
//! Pyramid levels alternate between two buffers so a reduction never binds
//! one allocation for both reading and writing. Matching-geometry cameras share
//! arenas; other rigs retain independent per-camera bindings.

mod detect;
mod kernels;
mod patches;
mod pyramid;
mod submission;
pub use submission::LaunchList;
#[cfg(test)]
mod runtime;
use kornia_staging_gpu::runtime::gpu_client;
use kornia_staging_gpu::runtime::default_capabilities;
mod frontend;
pub use frontend::GpuStages;
mod track;

pub use detect::GpuCornerScan;
pub use patches::GpuPatchSources;
pub use pyramid::{GpuPyramid, GpuPyramidBuilder};
pub use track::GpuPatchTracker;

use kornia_staging_gpu::GpuRuntime;

/// Construct the frame-stage owner on the selected device.
pub fn gpu_stages<P: kornia_staging_imgproc::optical_flow::patch_se2::Pattern>(
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    cameras: usize,
) -> Result<GpuStages<P, GpuRuntime>, crate::frontend::flow::FrontendError> {
    // Validated once per process; backend selection reads the same result.
    let capabilities = default_capabilities()?;
    let client = gpu_client()?;
    kornia_staging_gpu::transfer::execute_exclusive(&client, "frontend construction", || {
        let launches = LaunchList::default();
        let tracker = GpuPatchTracker::new(
            client.clone(),
            capabilities,
            capacity,
            num_levels,
            max_iterations,
            max_recovered_dist2,
            cameras,
            launches.clone(),
        )?;
        GpuStages::new(client.clone(), tracker, launches)
    })
}
