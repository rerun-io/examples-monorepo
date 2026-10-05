//! CubeCL frontend: persistent pyramids, FAST selection and fused KLT.
//! The frame coordinator owns current/next resources and one typed launch list.
//! Patch tracking has no frame-phase hooks. All supported GPUs use the same
//! fused kernel after storage and subgroup startup probes.
//!
//! Pyramid levels alternate between two buffers so a reduction never binds
//! one allocation for both reading and writing. Matching-geometry cameras share
//! arenas; other rigs retain independent per-camera bindings.

mod detect;
mod finite;
mod kernels;
mod patches;
mod pyramid;
mod submission;
pub use submission::LaunchList;
use submission::{read_blocking, read_failed, upload_frame};
mod runtime;
pub use runtime::{BACKEND_NAME, GpuError, gpu_client, probe_storage};
#[cfg(test)]
use runtime::{BLOCKING_READ, CORNER_SCAN_READ, arm_fault_at, fire_if_armed};
use runtime::{RUNTIME_NAME, guarded};
mod frontend;
pub use frontend::GpuStages;
mod track;
mod trig;

pub use detect::GpuCornerScan;
pub use patches::GpuPatchSources;
pub use pyramid::{GpuPyramid, GpuPyramidBuilder};
pub use track::GpuPatchTracker;

/// The runtime this build's GPU lane runs on: the portable one.
#[cfg(feature = "gpu-wgpu")]
pub type GpuRuntime = cubecl_wgpu::WgpuRuntime;

/// Construct the frame-stage owner on the selected device.
pub fn gpu_stages<P: crate::frontend::patterns::Pattern>(
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    cameras: usize,
) -> Result<GpuStages<P, GpuRuntime>, crate::frontend::tracker::TrackerError> {
    guarded(
        GpuError::ClientPanicked {
            runtime: RUNTIME_NAME,
        },
        || {
            let client = gpu_client()?;
            client
                .exclusive(|| {
                    probe_storage(&client)?;
                    let launches = LaunchList::default();
                    let tracker = GpuPatchTracker::new(
                        client.clone(),
                        capacity,
                        num_levels,
                        max_iterations,
                        max_recovered_dist2,
                        cameras,
                        launches.clone(),
                    )?;
                    GpuStages::new(client.clone(), tracker, launches)
                })
                .map_err(|error| submission::read_failed("frontend construction", &error))?
        },
    )
}
