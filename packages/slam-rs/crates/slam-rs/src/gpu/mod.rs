//! The CubeCL frontend backend (decision D21), behind the `gpu` feature: the
//! pyramid, the patch build, the corner scan and the KLT tracker on the GPU,
//! behind the stage traits the CPU port already exposes.
//!
// The `gpu` feature and D21 are said here rather than as an outer doc line on
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
//! lane proves nothing — round 1's `cargo check --features gpu-wgpu` was green
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
mod kernels;
mod patches;
mod pyramid;
mod track;

pub use detect::GpuCornerScan;
pub use patches::GpuPatches;
pub use pyramid::{GpuPyramid, GpuPyramidBuilder, Level0, Level0Table};
pub use track::GpuPatchTracker;

/// What can go wrong bringing up or running a GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
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
    /// A device read failed outright, rather than returning the wrong length.
    ///
    /// cubecl's convenience readers panic on a `ServerError` — `read_one_unchecked`
    /// is `read_sync(..).unwrap()` and `read` is `.expect("TODO")` — and a panic
    /// on the per-frame path unwinds out of the frontend while the GIL is
    /// detached, past `process_frame`'s transactional restore, so Python would
    /// see a `PanicException` instead of the documented error and a caller that
    /// continued would be tracking against a half-committed frontend. Every
    /// download therefore takes the fallible variant and lands here (decision
    /// D32). What fails this way is a lost device or a staging allocation
    /// refused under memory pressure — the cap's shared-8 GB regime, not a
    /// healthy card; the runtime's own reason is logged where it is mapped,
    /// because this enum is `Copy` and cannot carry it.
    #[error("reading {what} from the device failed")]
    DeviceReadFailed {
        /// Which buffer.
        what: &'static str,
    },
    /// The runtime cannot store an element width the kernels bind.
    ///
    /// See [`probe_storage`]: both of this backend's bring-up failures are
    /// silent, and this is what turns them into an error.
    #[error(
        "this runtime does not store {width}-bit elements: a device copy of a \
         known pattern came back with {wrong} of {count} elements wrong"
    )]
    StorageRoundTrip {
        /// Element width in bits.
        width: usize,
        /// Elements that came back changed.
        wrong: usize,
        /// Elements copied.
        count: usize,
    },
}

/// The NVIDIA runtime, so nothing outside this module names `cubecl_cuda`.
#[cfg(feature = "gpu")]
pub type CudaRuntime = cubecl_cuda::CudaRuntime;

/// A [`CudaRuntime`] client on the default device.
///
/// The one NVIDIA-specific line in the crate; `wgpu_client` — which only a
/// `gpu-wgpu` build has — is the same code with another client.
#[cfg(feature = "gpu")]
pub fn cuda_client() -> cubecl::prelude::ComputeClient<CudaRuntime> {
    use cubecl::prelude::Runtime;
    cubecl_cuda::CudaRuntime::client(&cubecl_cuda::CudaDevice::default())
}

/// The runtime this build's GPU lane runs on.
///
/// CUDA when only `gpu` is enabled and wgpu when `gpu-wgpu` is. Every type in
/// the crate outside this file names [`GpuRuntime`], never a concrete runtime,
/// so moving a host onto the portable lane is a cargo feature rather than a
/// port — and `cargo test --features gpu-wgpu` runs the same per-kernel
/// tolerance tests through Vulkan, which is what makes that claim checkable
/// rather than merely compiled.
#[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
pub type GpuRuntime = CudaRuntime;

/// The runtime this build's GPU lane runs on: the portable one.
#[cfg(feature = "gpu-wgpu")]
pub type GpuRuntime = cubecl_wgpu::WgpuRuntime;

/// A client on this build's runtime.
///
/// The whole of what selecting a backend costs. Select the adapter with
/// `CUBECL_WGPU_DEFAULT_DEVICE` on the portable lane; `WGPU_BACKEND` and
/// `WGPU_ADAPTER_NAME` are ignored by cubecl-wgpu.
#[cfg(feature = "gpu")]
pub fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRuntime> {
    #[cfg(feature = "gpu-wgpu")]
    {
        wgpu_client()
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        cuda_client()
    }
}

/// The pyramid builder of this build's lane.
#[cfg(feature = "gpu")]
pub type LanePyramidBuilder = GpuPyramidBuilder<GpuRuntime>;

/// The patch tracker of this build's lane.
#[cfg(feature = "gpu")]
pub type LanePatchTracker<P> = GpuPatchTracker<P, GpuRuntime>;

/// The three stage backends `FrameToFrameOpticalFlow::with_backends` takes.
#[cfg(feature = "gpu")]
pub type LaneBackends<P> = (
    LanePyramidBuilder,
    LanePatchTracker<P>,
    Box<dyn crate::frontend::detect::CornerScan>,
);

/// The three stage backends `FrameToFrameOpticalFlow::with_backends` needs, on
/// one shared client.
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
pub fn gpu_backends<P: crate::frontend::patterns::Pattern>(
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
) -> Result<LaneBackends<P>, crate::frontend::tracker::TrackerError> {
    let client = gpu_client();
    // Before anything else: a runtime that cannot store these widths produces
    // zeros rather than an error, and a trajectory made of zeros is worse than
    // a refusal.
    probe_storage(&client)?;
    let tracker: LanePatchTracker<P> = GpuPatchTracker::new(
        client.clone(),
        capacity,
        num_levels,
        max_iterations,
        max_recovered_dist2,
    )?;
    let builder: LanePyramidBuilder = GpuPyramidBuilder::new(client.clone(), P::OFFSETS);
    let mut scanner: GpuCornerScan<GpuRuntime> = GpuCornerScan::new(client);
    // The two stages are handed the same frame, so they read the same upload:
    // the builder publishes level 0 per camera and the scanner reads it. This
    // is the one line that makes it one upload per camera per frameset instead
    // of two (see [`Level0`]).
    scanner.share_level0(builder.level0_table());
    Ok((builder, tracker, Box::new(scanner)))
}

/// A failed device read as a typed error, with the runtime's own reason logged.
///
/// [`GpuError`] is `Copy`, so it cannot carry the `ServerError`'s reason and
/// backtrace; the warning is where they are kept, and the returned variant is
/// what the stage errors carry to the caller.
fn read_failed(what: &'static str, error: &cubecl::server::ServerError) -> GpuError {
    log::warn!("reading {what} from the device failed: {error}");
    GpuError::DeviceReadFailed { what }
}

/// Refuse a runtime that cannot store an element width the kernels bind.
///
/// Both of this backend's bring-up failures are **silent**, and neither is
/// visible in `client.properties()`, which describes the device rather than the
/// compiler that will run on it:
///
/// * A CUDA install without `cuda-nvrtc` / `cuda-cudart-dev` panics on cubecl's
///   own worker thread. The client still constructs, every launch reports
///   success, and every read comes back as zeros (round 1's report).
/// * `cubecl-wgpu`'s **WGSL** compiler panics on `u16` and `u8` —
///   "U16 is not a valid WgpuElement" — on the same worker thread, with the
///   same result. The pyramid is `u16` and the candidate image is `u8`, so on
///   the portable lane without `cubecl-wgpu/spirv` every kernel here silently
///   produces nothing. Measured on this host: the pyramid came back all zeros
///   and the detector found no corners, with no error anywhere.
///
/// So the check is the one thing that cannot lie: write a known pattern, copy
/// it **on the device**, read it back. Four widths, 256 elements each, once at
/// construction — microseconds, and it is what a fleet machine whose driver
/// silently mishandles a width will fail on instead of producing a trajectory
/// out of zeros (decisions D21, D32).
///
/// # Errors
///
/// [`GpuError::StorageRoundTrip`] naming the width that did not survive.
#[cfg(feature = "gpu")]
pub fn probe_storage<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> Result<(), GpuError> {
    use cubecl::prelude::CubeElement;

    /// One width: a pattern no zeroing or truncation reproduces.
    fn round_trip<N, R>(
        client: &cubecl::prelude::ComputeClient<R>,
        pattern: &[N],
    ) -> Result<(), GpuError>
    where
        N: cubecl::prelude::Numeric + CubeElement + PartialEq + Copy,
        R: cubecl::prelude::Runtime,
    {
        let count: usize = pattern.len();
        let width: usize = size_of::<N>() * 8;
        let expected: usize = size_of_val(pattern);
        let source: cubecl::server::Handle = client.create_from_slice(N::as_bytes(pattern));
        let target: cubecl::server::Handle = client.empty(expected);
        kernels::launch_probe::<N, R>(client, (&source, count), (&target, count), count);
        let bytes = client
            .read_one(target)
            .map_err(|error| read_failed("the storage probe", &error))?;
        if bytes.len() != expected {
            return Err(GpuError::StorageRoundTrip {
                width,
                wrong: count,
                count,
            });
        }
        let wrong: usize = N::from_bytes(&bytes)
            .iter()
            .zip(pattern.iter())
            .filter(|(got, want)| got != want)
            .count();
        if wrong == 0 {
            Ok(())
        } else {
            Err(GpuError::StorageRoundTrip {
                width,
                wrong,
                count,
            })
        }
    }

    const COUNT: usize = 256;
    // Patterns whose every byte differs from its neighbours, so a truncation, a
    // widening or a packing slip all show up rather than cancelling.
    let bytes: Vec<u8> = (0..COUNT)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(1))
        .collect();
    let shorts: Vec<u16> = (0..COUNT)
        .map(|i| (i as u16).wrapping_mul(1_237).wrapping_add(9))
        .collect();
    let words: Vec<u32> = (0..COUNT)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x5a5a)
        .collect();
    let floats: Vec<f32> = (0..COUNT).map(|i| (i as f32) * 0.5 - 3.25).collect();
    round_trip::<u8, R>(client, &bytes)?;
    round_trip::<u16, R>(client, &shorts)?;
    round_trip::<u32, R>(client, &words)?;
    round_trip::<f32, R>(client, &floats)?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A read that fails is a typed error at every boundary, never a panic.
    ///
    /// The failure this is about cannot be forced from a test: a `ServerError`
    /// on a download means a lost device or a staging allocation refused under
    /// memory pressure, and neither is reachable from a healthy card. What is
    /// testable — and what the per-frame path actually depends on — is that the
    /// mapping produces a variant the stage errors carry, so the unwinding a
    /// panicking read would do through the released GIL (decision D32) cannot
    /// happen.
    #[test]
    fn a_failed_device_read_is_a_typed_error_at_every_stage() {
        let what: &str = "the tracker result";
        let error: cubecl::server::ServerError = cubecl::server::ServerError::Generic {
            reason: "the device is gone".to_owned(),
            backtrace: cubecl::backtrace::BackTrace::default(),
        };
        assert_eq!(
            read_failed(what, &error),
            GpuError::DeviceReadFailed { what }
        );

        // The three stages a download sits in each carry it, so the error
        // reaches the Python boundary as the documented `ValueError`.
        let tracker: crate::frontend::tracker::TrackerError =
            GpuError::DeviceReadFailed { what }.into();
        let pyramid: crate::pyramid::PyramidError = GpuError::DeviceReadFailed { what }.into();
        let detect: crate::frontend::detect::DetectError =
            GpuError::DeviceReadFailed { what }.into();
        for message in [tracker.to_string(), pyramid.to_string(), detect.to_string()] {
            assert!(
                message.contains(what),
                "the stage error dropped what failed: {message}"
            );
        }
    }
}
