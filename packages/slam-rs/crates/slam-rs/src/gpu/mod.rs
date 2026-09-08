//! The CubeCL frontend backend: the pyramid, the patch build and the KLT
//! tracker on the GPU, behind the stage traits the CPU port already exposes.
//!
//! ## The seam
//!
//! Nothing outside this module tree names `cubecl`. [`crate::pyramid`],
//! [`crate::frontend::tracker`] and [`crate::frontend::flow`] are unchanged: a
//! GPU frontend is [`GpuPyramidBuilder`] plus [`GpuPatchTracker`] handed to
//! `FrameToFrameOpticalFlow::with_backends`, and the CPU pair stays selectable
//! at the same call (decision D21).
//!
//! ## The runtime is a type parameter
//!
//! Every type here is generic over `R: Runtime` and every kernel is one source.
//! The only NVIDIA-specific line in the crate is [`cuda_client`]; the wgpu lane
//! is [`wgpu_client`], the same code with another client, kept compiling by
//! `cargo check --features gpu-wgpu` so the Spark, the Pi 5 and the cap are a
//! client-line change rather than a port.
//!
//! ## Residency
//!
//! A [`GpuPyramid`] is two device buffers and a host-side table of level
//! geometry; a [`GpuPatches`] is one device buffer. They are allocated once and
//! live between frames, so the previous frame's pyramid is already where the
//! tracker needs it. Per frameset the host uploads each camera's level 0 and
//! the keypoint positions, and downloads one packed result array per
//! [`PatchTracker::track`] call — one synchronisation per call, never per
//! kernel (`Robocap.md`, "Kernel-design rules learned").
//!
//! ## Level parity, and why there are two pyramid buffers
//!
//! Every level of a pyramid is a flat `u16` buffer with stride equal to its
//! width, and the levels are packed into **two** allocations: level `l` lives in
//! `a` when `l` is even and in `b` when it is odd. One buffer for everything
//! would make each subsample read and write the same allocation, which CubeCL
//! declares `const __restrict__` on the read side and WGSL refuses to bind
//! twice; separate reads and writes cost one uniform branch per sample site and
//! are correct on every runtime.

mod kernels;
mod patches;
mod pyramid;
mod track;

pub use patches::GpuPatches;
pub use pyramid::{GpuPyramid, GpuPyramidBuilder};
pub use track::GpuPatchTracker;

/// What can go wrong bringing up or running a GPU backend.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
    /// The device has no room for a buffer this geometry needs.
    #[error("a {what} buffer of {elements} elements does not fit in a usize")]
    BufferShapeOverflow {
        /// Which buffer.
        what: &'static str,
        /// Elements asked for.
        elements: usize,
    },
    /// A device read came back with the wrong number of bytes.
    ///
    /// The one failure mode a CubeCL backend has that a CPU one does not: a
    /// runtime whose CUDA installation is incomplete panics on its own worker
    /// thread and hands back a buffer of zeros rather than an error, so every
    /// download is length-checked and a short one is refused here instead of
    /// being read as data (decision D32).
    #[error("reading {what} returned {actual} bytes, expected {expected}")]
    ShortRead {
        /// Which buffer.
        what: &'static str,
        /// Bytes returned.
        actual: usize,
        /// Bytes the geometry needs.
        expected: usize,
    },
}

/// The NVIDIA runtime, so nothing outside this module names `cubecl_cuda`.
#[cfg(feature = "gpu")]
pub type CudaRuntime = cubecl_cuda::CudaRuntime;

/// The pyramid builder of the NVIDIA lane.
#[cfg(feature = "gpu")]
pub type CudaPyramidBuilder = GpuPyramidBuilder<CudaRuntime>;

/// The patch tracker of the NVIDIA lane.
#[cfg(feature = "gpu")]
pub type CudaPatchTracker<P> = GpuPatchTracker<P, CudaRuntime>;

/// A [`CudaRuntime`] client on the default device.
///
/// The one NVIDIA-specific line in the crate; [`wgpu_client`] is the same code
/// with another client.
#[cfg(feature = "gpu")]
pub fn cuda_client() -> cubecl::prelude::ComputeClient<CudaRuntime> {
    use cubecl::prelude::Runtime;
    cubecl_cuda::CudaRuntime::client(&cubecl_cuda::CudaDevice::default())
}

/// The pair of stage backends `FrameToFrameOpticalFlow::with_backends` needs,
/// on one shared NVIDIA client.
///
/// One client for both stages is what keeps a pyramid and the patches it feeds
/// on the same device queue, so the frontend synchronises once per
/// [`crate::frontend::tracker::PatchTracker::track`] call rather than once per
/// stage.
///
/// # Errors
///
/// [`crate::frontend::tracker::TrackerError`] when the capacity or the level
/// count is over its ceiling, or a buffer's element count does not fit a
/// `usize` — the same refusals the CPU tracker makes.
#[cfg(feature = "gpu")]
pub fn cuda_backends<P: crate::frontend::patterns::Pattern>(
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
) -> Result<(CudaPyramidBuilder, CudaPatchTracker<P>), crate::frontend::tracker::TrackerError> {
    let client = cuda_client();
    let tracker: CudaPatchTracker<P> = GpuPatchTracker::new(
        client.clone(),
        capacity,
        num_levels,
        max_iterations,
        max_recovered_dist2,
    )?;
    Ok((GpuPyramidBuilder::new(client, P::OFFSETS), tracker))
}

/// A [`cubecl_wgpu::WgpuRuntime`] client on the default device.
///
/// The portable lane: Vulkan on the Spark and the Pi 5, Metal on macOS. Select
/// the adapter with `CUBECL_WGPU_DEFAULT_DEVICE`; `WGPU_BACKEND` and
/// `WGPU_ADAPTER_NAME` are ignored by cubecl-wgpu.
#[cfg(feature = "gpu-wgpu")]
pub fn wgpu_client() -> cubecl::prelude::ComputeClient<cubecl_wgpu::WgpuRuntime> {
    use cubecl::prelude::Runtime;
    cubecl_wgpu::WgpuRuntime::client(&cubecl_wgpu::WgpuDevice::default())
}
