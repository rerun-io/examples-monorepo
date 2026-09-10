//! The CubeCL frontend backend (decision D21), behind the `gpu-wgpu` feature: the
//! pyramid, the patch build, the corner scan and the KLT tracker on the GPU,
//! behind the stage traits the CPU port already exposes.
//!
// The `gpu-wgpu` feature and D21 are said here rather than as an outer doc line on
// `pub mod gpu;`: an outer fragment makes rustdoc resolve this whole block in
// the crate root's scope, where none of the names below are in scope, and every
// link in the header broke silently because `cargo doc` was not a gate.
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
//! Every type here is generic over `R: Runtime` and every kernel is one source;
//! [`GpuRuntime`] carries which one this build picked. A compiling portable
//! lane proves nothing — `cargo check --features gpu-wgpu` has been green
//! while every kernel returned zeros — so the claim is checked by running:
//! `slam-rs-wgpu-test` puts the same tolerance tests through Vulkan, and
//! [`probe_storage`] refuses at construction any runtime whose device copy of a
//! known pattern does not survive.
//!
//! ## Residency
//!
//! A [`GpuPyramid`] is two device buffers and a host-side table of level
//! geometry; a [`GpuPatches`] is one device buffer. They are allocated once and
//! live between frames, so the previous frame's pyramid is already where the
//! tracker needs it. Per frameset the host uploads each camera's level 0 and
//! the keypoint positions, and downloads one packed result array per
//! [`crate::frontend::tracker::PatchTracker::track`] call — one synchronisation
//! per call, never per
//! kernel (`Robocap.md`, "Kernel-design rules learned").
//!
//! What that buys is **bounded pool growth, not zero device allocations**, and
//! the difference is worth stating plainly because the design cannot deliver
//! the stronger claim. CubeCL 0.10's only host-to-device write is `create*`
//! (`create_from_slice`, `create`, the tensor forms and `empty` —
//! `cubecl-runtime`'s client has no write into an existing handle), so three
//! allocations are per-frame by construction: the frame upload per camera
//! ([`GpuPyramidBuilder`]'s `build`), the positions buffer per patch build
//! ([`GpuPatches`]) and the transform buffer per tracking call
//! ([`GpuPatchTracker`]). Routing them through a persistent buffer would not
//! remove them — the upload still allocates and a device copy is added — which
//! is why the level-0 change kept its `create_from_slice` and moved only the
//! *big* allocation off the per-frame path. So the promise is that the pool
//! they come out of plateaus and holds flat, and
//! `the_whole_gpu_path_holds_the_pool_flat` is what says so: over 200 framesets
//! of the whole path, reserved bytes and bytes in use are constant from the
//! first frameset (40.00 MiB reserved on wgpu, 18.51 MiB in use).
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

mod detect;
mod finite;
mod kernels;
mod patches;
mod pyramid;
pub mod seam;
mod selection_batch;
mod submission;
pub use submission::CHANNEL_TASKS;
use submission::{drained, empty, read_blocking, read_failed, upload_frame};
mod runtime;
pub use runtime::{BACKEND_NAME, GpuError, gpu_client, probe_storage};
#[cfg(test)]
use runtime::{BLOCKING_READ, CORNER_SCAN_READ, arm_fault_at, fire_if_armed};
use runtime::{RUNTIME_NAME, guarded};
mod track;
mod trig;

pub use detect::GpuCornerScan;
pub use patches::{GpuPatches, StoreLayout};
pub use pyramid::{GpuPyramid, GpuPyramidBuilder, Level0, Level0Table};
pub use track::GpuPatchTracker;

/// The runtime this build's GPU lane runs on: the portable one.
#[cfg(feature = "gpu-wgpu")]
pub type GpuRuntime = cubecl_wgpu::WgpuRuntime;

/// The pyramid builder of this build's lane.
#[cfg(feature = "gpu-core")]
pub type LanePyramidBuilder = GpuPyramidBuilder<GpuRuntime>;

/// The patch tracker of this build's lane.
#[cfg(feature = "gpu-core")]
pub type LanePatchTracker<P> = GpuPatchTracker<P, GpuRuntime>;

/// The three stage backends `FrameToFrameOpticalFlow::with_backends` takes.
#[cfg(feature = "gpu-core")]
pub type LaneBackends<P> = (
    LanePyramidBuilder,
    LanePatchTracker<P>,
    Box<dyn crate::frontend::detect::CornerScan>,
);

/// The three stage backends `FrameToFrameOpticalFlow::with_backends` needs, on
/// one shared client.
///
/// One client for both stages is what keeps a pyramid and the patches it feeds
/// on the same device queue, so the frontend synchronises once per tracking
/// **batch** rather than once per stage or once per camera.
///
/// `cameras` is the rig's camera count, which is how many tracking passes the
/// tracker must be able to hold in flight at once
/// ([`crate::frontend::tracker::PatchTracker::submit_prepared`]).
///
/// # Errors
///
/// [`crate::frontend::tracker::TrackerError`] when the capacity or the level
/// count is over its ceiling, or a buffer's element count does not fit a
/// `usize` — the same refusals the CPU tracker makes; whatever [`gpu_client`]
/// refuses this host for; [`GpuError::StorageRoundTrip`] from
/// [`probe_storage`]; and [`GpuError::ClientPanicked`] if anything from the
/// client to the returned backends panics instead of returning.
#[cfg(feature = "gpu-core")]
pub fn gpu_backends<P: crate::frontend::patterns::Pattern>(
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    cameras: usize,
) -> Result<LaneBackends<P>, crate::frontend::tracker::TrackerError> {
    // The guard is around the whole of it, not around the client alone: from
    // here to the returned backends every line allocates, launches or reads on
    // the device, and CubeCL panics rather than returning an error when one of
    // those meets a device that is not there (decision D32). `gpu_client`'s own
    // inner `guarded` is for its own callers — the tolerance tests — and on this
    // path it catches nothing this one would not.
    guarded(
        GpuError::ClientPanicked {
            runtime: RUNTIME_NAME,
        },
        || -> Result<LaneBackends<P>, crate::frontend::tracker::TrackerError> {
            // Before anything else: a host with no driver, no device or no
            // adapter is a typed error rather than a panic on cubecl's worker
            // thread, and a runtime that cannot store these widths produces
            // zeros rather than an error — a trajectory made of zeros being
            // worse than a refusal.
            let client = gpu_client()?;
            probe_storage(&client)?;
            let mut tracker: LanePatchTracker<P> = GpuPatchTracker::new(
                client.clone(),
                capacity,
                num_levels,
                max_iterations,
                max_recovered_dist2,
                cameras,
            )?;
            let builder: LanePyramidBuilder = GpuPyramidBuilder::new(client.clone(), P::OFFSETS);
            let mut scanner: GpuCornerScan<GpuRuntime> = GpuCornerScan::new(client)?;
            // The two stages are handed the same frame, so they read the same
            // upload: the builder publishes level 0 per camera and the scanner
            // reads it. This is the one line that makes it one upload per camera
            // per frameset instead of two (see [`Level0`]).
            scanner.share_level0(builder.level0_table());
            // And the same again for the download: the scanner's cell keys ride
            // the tracker's temporal read instead of paying for one of their own
            // The endpoints remain private and have only one producer.
            scanner.share_reads(&mut tracker);
            Ok((builder, tracker, Box::new(scanner)))
        },
    )
}
