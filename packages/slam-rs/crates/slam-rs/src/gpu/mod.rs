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
//! first frameset (156.74 / 40.00 MiB reserved on CUDA / wgpu, 18.51 MiB in use
//! on both).
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
pub use patches::{GpuPatches, StoreLayout};
pub use pyramid::{GpuPyramid, GpuPyramidBuilder, Level0, Level0Table};
pub use track::GpuPatchTracker;

/// What can go wrong bringing up or running a GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
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
    /// A GPU stage panicked instead of returning an error.
    ///
    /// The other half of [`GpuError::DeviceReadFailed`], and it exists because
    /// mapping the reads was not enough: CubeCL 0.10 unwraps a **failed worker
    /// submission** inside its own client (`cubecl-runtime`'s `client.rs`), so
    /// on a device that is gone the upload, the launch and the read all panic
    /// on the calling thread rather than hand back a `ServerError` there is
    /// anything to map. Every stage on the per-frame path therefore runs inside
    /// the module's `guarded`, and this is what the caller gets instead of an unwind
    /// through the released GIL and a `PanicException` in Python (decision
    /// D32). The panic's own message is logged where it is caught, because this
    /// enum is `Copy` and cannot carry it.
    #[error(
        "the GPU {what} failed on the device; the log carries the runtime's own \
         message, and the CPU frontend runs without a GPU"
    )]
    DeviceLost {
        /// Which stage was running.
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
    /// A shared library the runtime loads at run time is not on the loader path.
    ///
    /// cudarc `dlopen`s `libcuda` and `libnvrtc` on first use and **panics**
    /// (`panic_no_lib_found`) when none of its name candidates resolves, so the
    /// probe opens them itself first. `libcudart` is not in the list on purpose:
    /// nothing dlopens it — what the CUDA lane needs from `cuda-cudart-dev` is
    /// the `cuda_runtime.h` NVRTC's generated code includes, and a missing
    /// header is a compile error the runtime reports rather than a loader panic.
    #[error(
        "the GPU runtime needs the {library} shared library, which is not on the \
         loader path: install it and put its directory on LD_LIBRARY_PATH, or run \
         the CPU frontend, which needs no GPU"
    )]
    MissingLibrary {
        /// The library's cudarc name — `cuda` for the driver, `nvrtc` for the compiler.
        library: &'static str,
    },
    /// The GPU driver refused to initialise.
    #[error(
        "the {runtime} driver did not initialise (error {code}): this host has no \
         usable GPU driver, so run the CPU frontend, which needs none"
    )]
    DriverUnavailable {
        /// Which runtime asked.
        runtime: &'static str,
        /// The driver's own status code, for the reason its own docs give.
        code: i32,
    },
    /// The driver initialised and reported no devices.
    ///
    /// What `CUDA_VISIBLE_DEVICES=` produces, and what a container started
    /// without `--gpus` produces.
    #[error(
        "the {runtime} driver reports {count} devices: nothing is visible to run \
         on — check CUDA_VISIBLE_DEVICES — or run the CPU frontend, which needs \
         no device"
    )]
    NoDevice {
        /// Which runtime asked.
        runtime: &'static str,
        /// Devices the driver reported.
        count: i32,
    },
    /// No wgpu adapter for the backend this build runs on.
    ///
    /// The portable lane's equivalent of [`GpuError::NoDevice`]: the Vulkan
    /// loader found no ICD, or the device it found does not present an adapter.
    #[error(
        "wgpu found no {backend} adapter on this host: install a {backend} driver, \
         or run the CPU frontend, which needs no adapter"
    )]
    NoAdapter {
        /// The graphics backend cubecl-wgpu would have used.
        backend: &'static str,
    },
    /// Constructing the CubeCL client panicked.
    ///
    /// The last line of defence, and it is needed: CubeCL 0.10 unwraps inside
    /// its own bring-up and a panic on its worker thread reaches the caller as a
    /// `RecvError` on the main thread. Anything the probes above did not
    /// anticipate lands here rather than unwinding through the frontend and out
    /// of the released GIL as a `PanicException` (decision D32). The panic's own
    /// message is logged where it is caught, because this enum is `Copy` and
    /// cannot carry it.
    #[error(
        "building the {runtime} client panicked; the log carries the runtime's own \
         message, and the CPU frontend runs without a GPU"
    )]
    ClientPanicked {
        /// Which runtime was being built.
        runtime: &'static str,
    },
}

/// The NVIDIA runtime, so nothing outside this module names `cubecl_cuda`.
///
/// This and the module's `cuda_client` are the only two items on the `gpu`
/// feature rather than on `gpu-core`: they are what `cubecl-cuda` is for, and
/// keeping them apart is what lets a `gpu-wgpu` build carry neither the crate
/// nor its codegen.
#[cfg(feature = "gpu")]
pub type CudaRuntime = cubecl_cuda::CudaRuntime;

/// A [`CudaRuntime`] client on the default device.
///
/// The one NVIDIA-specific line in the crate; `wgpu_client` — which only a
/// `gpu-wgpu` build has — is the same code with another client.
///
/// Private, and that is the fix the re-review asked for: it constructs a client
/// without probing this host and without a panic guard, so a public one was a
/// second door into exactly the failure [`gpu_client`] exists to close. The only
/// way to a client from outside this module is [`gpu_client`], which is
/// fallible.
#[cfg(feature = "gpu")]
fn cuda_client() -> cubecl::prelude::ComputeClient<CudaRuntime> {
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

/// A client on this build's runtime, or why this host has none.
///
/// The whole of what selecting a backend costs — a cargo feature, and this
/// function is the only place it is read. Select the adapter with
/// `CUBECL_WGPU_DEFAULT_DEVICE` on the portable lane; `WGPU_BACKEND` and
/// `WGPU_ADAPTER_NAME` are ignored by cubecl-wgpu.
///
/// Fallible, and in this order, because **CubeCL 0.10 unwraps its own
/// bring-up**: `CudaRuntime::client` unwraps the driver's initialisation and
/// `cubecl-wgpu` expects an adapter, and both do it on the runtime's own worker
/// thread, so the caller gets a `RecvError` panic rather than an error. On a
/// host with `CUDA_VISIBLE_DEVICES=` that reached Python as a
/// `PanicException`, against [`crate::Vio`]'s documented contract that a
/// refusal is a `ValueError` and never a Rust panic (decision D32).
///
/// 1. `probe_availability` asks the runtime's **own** fallible API — cudarc's
///    `init` and device count, wgpu's `request_adapter` — before any client
///    exists, so the common failures name what is missing.
/// 2. The construction itself runs inside the module's `guarded`, because a
///    probe can only anticipate what it knows to ask.
///
/// **A caught panic still prints its own message to stderr**, and that is the
/// deliberate half of the trade. Round 1 swapped in a quiet panic hook around
/// the construction; the hook is process-global and unsynchronised, so it also
/// silenced unrelated threads for that window and two concurrent constructors
/// could restore it out of order — and the same guard is now on the per-frame
/// path, where swapping a global hook per frameset is not a thing that can be
/// done at all. It is gone. What it was there to hide is no longer the expected
/// case either: every failure a host without a GPU produces is caught by the
/// probe above and never panics, so the stderr line only ever appears for
/// something the probe could not anticipate — where the runtime's own message
/// is the only clue there is.
///
/// # Errors
///
/// [`GpuError::MissingLibrary`], [`GpuError::DriverUnavailable`],
/// [`GpuError::NoDevice`] or [`GpuError::NoAdapter`] from the probe, and
/// [`GpuError::ClientPanicked`] from anything it did not anticipate.
#[cfg(feature = "gpu-core")]
pub fn gpu_client() -> Result<cubecl::prelude::ComputeClient<GpuRuntime>, GpuError> {
    probe_availability()?;
    guarded(
        GpuError::ClientPanicked {
            runtime: RUNTIME_NAME,
        },
        || {
            #[cfg(feature = "gpu-wgpu")]
            {
                Ok(wgpu_client())
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            {
                Ok(cuda_client())
            }
        },
    )
}

/// What this build's lane is called in an error a user reads.
#[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
const RUNTIME_NAME: &str = "CUDA";

/// What this build's lane is called in an error a user reads.
#[cfg(feature = "gpu-wgpu")]
const RUNTIME_NAME: &str = "wgpu";

/// The same runtime as the name a machine reads: the cargo feature's own word,
/// lowercase, which is what a fleet row's `lane` says and what a chart groups on
/// ([`crate::GPU_BACKEND`]). Kept beside `RUNTIME_NAME` rather than derived from
/// it, because that one is prose in a sentence a person reads. Not a link:
/// `RUNTIME_NAME` is private, and a public item linking to a private one renders
/// as plain text, which the doc gate denies.
#[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
pub const BACKEND_NAME: &str = "cuda";

/// The same runtime as the name a machine reads: the portable lane's.
#[cfg(feature = "gpu-wgpu")]
pub const BACKEND_NAME: &str = "wgpu";

/// Refuse a host that cannot run this lane, before any CubeCL client exists.
///
/// The NVIDIA arm. Three questions in the order that makes each answerable:
/// cudarc `dlopen`s `libcuda` and `libnvrtc` and **panics** when it cannot find
/// them, so they are opened here first; then `cuInit`, which is what
/// `CudaRuntime::client` unwraps; then the device count, which is what
/// `CUDA_VISIBLE_DEVICES=` empties while leaving the driver perfectly healthy.
///
/// # Errors
///
/// [`GpuError::MissingLibrary`], [`GpuError::DriverUnavailable`] or
/// [`GpuError::NoDevice`], each naming what a user can act on.
#[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
fn probe_availability() -> Result<(), GpuError> {
    if let Some(missing) = first_missing_library(&CUDA_LIBRARIES) {
        return Err(missing);
    }
    // `CUDA_ERROR_NO_DEVICE` is what `CUDA_VISIBLE_DEVICES=` produces here: the
    // driver is loaded and healthy and `cuInit` itself refuses, so it is the
    // device error and not the driver one. The count below covers the other
    // shape, where initialisation succeeds and nothing is visible.
    cudarc::driver::result::init().map_err(|error| {
        log::warn!("cuInit failed: {error}");
        if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NO_DEVICE {
            GpuError::NoDevice {
                runtime: RUNTIME_NAME,
                count: 0,
            }
        } else {
            GpuError::DriverUnavailable {
                runtime: RUNTIME_NAME,
                code: error.0 as i32,
            }
        }
    })?;
    let count: i32 = cudarc::driver::result::device::get_count().map_err(|error| {
        log::warn!("cuDeviceGetCount failed: {error}");
        GpuError::DriverUnavailable {
            runtime: RUNTIME_NAME,
            code: error.0 as i32,
        }
    })?;
    if count > 0 {
        Ok(())
    } else {
        Err(GpuError::NoDevice {
            runtime: RUNTIME_NAME,
            count,
        })
    }
}

/// The libraries cudarc loads at run time, in the order the probe asks for them.
///
/// `libcuda` is the driver `cuInit` lives in and `libnvrtc` is the compiler
/// every kernel here goes through; cudarc `dlopen`s both on first use and
/// `panic_no_lib_found`s when none of its name candidates resolves.
#[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
const CUDA_LIBRARIES: [&str; 2] = ["cuda", "nvrtc"];

/// The first of `libraries` that cudarc's own name candidates cannot open.
///
/// The candidates come from `cudarc::get_lib_name_candidates`, which is the
/// list `panic_no_lib_found` prints, so the probe and the loader can only ever
/// disagree about *when* the answer is taken, never about what it is.
///
/// A parameter rather than the constant so a unit test can ask for a name no
/// host has. The real absence is made in a child process instead
/// (`a_cuda_host_that_cannot_load_nvrtc_is_a_typed_error`): emptying
/// `LD_LIBRARY_PATH` is not enough here, because pixi links every binary in the
/// environment with an absolute `RUNPATH` into the environment's own `lib`,
/// which the loader searches after `LD_LIBRARY_PATH`; the child is started
/// through `ld.so --inhibit-rpath ''` so that route is closed too.
#[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
fn first_missing_library(libraries: &[&'static str]) -> Option<GpuError> {
    libraries.iter().find_map(|library| {
        let loadable: bool = cudarc::get_lib_name_candidates(library)
            .iter()
            .any(|candidate| unsafe { libloading::Library::new(candidate) }.is_ok());
        (!loadable).then_some(GpuError::MissingLibrary { library })
    })
}

/// Refuse a host that cannot run this lane, before any CubeCL client exists.
///
/// The portable arm. `cubecl-wgpu` reaches its adapter through
/// `request_adapter` and `.expect`s the result on its own worker thread; the
/// same request is made here first, on the same backend
/// (`AutoGraphicsApi::backend()` — Vulkan on Linux, Metal on macOS) and with
/// the same power preference, so a host with no ICD gets a sentence instead of
/// a `RecvError`. What this arm does **not** cover is
/// `CUBECL_WGPU_DEFAULT_DEVICE` naming an index the host does not have, which
/// selects by enumeration rather than by preference; that is the case
/// `catch_unwind` in [`gpu_client`] is for.
///
/// # Errors
///
/// [`GpuError::NoAdapter`], naming the backend it asked for.
#[cfg(feature = "gpu-wgpu")]
fn probe_availability() -> Result<(), GpuError> {
    use cubecl_wgpu::GraphicsApi;

    let backend: wgpu::Backend = cubecl_wgpu::AutoGraphicsApi::backend();
    let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let request = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    });
    match cubecl::future::block_on(request) {
        Ok(_) => Ok(()),
        Err(error) => {
            log::warn!("wgpu found no {backend} adapter: {error}");
            Err(GpuError::NoAdapter {
                backend: backend.to_str(),
            })
        }
    }
}

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
/// on the same device queue, so the frontend synchronises once per
/// [`crate::frontend::tracker::PatchTracker::track`] call rather than once per
/// stage.
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
) -> Result<LaneBackends<P>, crate::frontend::tracker::TrackerError> {
    // The guard is around the whole of it, not around the client alone: from
    // here to the returned backends every line allocates, launches or reads on
    // the device, and CubeCL panics rather than returning an error when one of
    // those meets a device that is not there (decision D32).
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
            let tracker: LanePatchTracker<P> = GpuPatchTracker::new(
                client.clone(),
                capacity,
                num_levels,
                max_iterations,
                max_recovered_dist2,
            )?;
            let builder: LanePyramidBuilder = GpuPyramidBuilder::new(client.clone(), P::OFFSETS);
            let mut scanner: GpuCornerScan<GpuRuntime> = GpuCornerScan::new(client);
            // The two stages are handed the same frame, so they read the same
            // upload: the builder publishes level 0 per camera and the scanner
            // reads it. This is the one line that makes it one upload per camera
            // per frameset instead of two (see [`Level0`]).
            scanner.share_level0(builder.level0_table());
            Ok((builder, tracker, Box::new(scanner)))
        },
    )
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

/// Run `stage`, turning a panic inside it into `fault`.
///
/// The seam between CubeCL's panics and decision D32. A `ServerError` on a
/// download is mapped by [`read_failed`], but that is only the half of the
/// failure CubeCL returns: `cubecl-runtime`'s client unwraps a failed worker
/// submission, so on a lost device `create_from_slice`, a launch and a read all
/// **panic** on the calling thread. On the per-frame path that unwind crosses
/// the released GIL and reaches Python as a `PanicException`, past
/// `process_frame`'s transactional restore; on the bring-up path it is the
/// `RecvError` the review reproduced. Both land here as the typed error the
/// caller already documents.
///
/// It costs nothing on the path that does not panic — `catch_unwind` is a
/// landing pad the happy path never enters — and the interleaved A/B in the
/// report's "Review fixes (round 2)" is the measurement rather than the claim.
fn guarded<T, E: From<GpuError>>(
    fault: GpuError,
    stage: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(test)]
        fire_if_armed(GUARDED_REGION);
        stage()
    }));
    match outcome {
        Ok(result) => result,
        Err(payload) => {
            let reason: &str = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("no message");
            log::warn!("{fault}: the runtime's own message was: {reason}");
            Err(fault.into())
        }
    }
}

// Where a test can make this module panic as a lost device would (test-only).
// The failure `guarded` is for cannot be produced on a healthy card — a device
// that is gone, or a staging allocation refused under memory pressure — so one
// thread-local stands in for it. It names a site rather than being a bare flag
// because two questions need asking separately: what a panic inside a guarded
// region does, and what one inside the storage probe does, the probe being what
// the re-review found running outside the guard. A doc comment cannot sit on a
// macro invocation.
#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// A fault site: the inside of a guarded region.
#[cfg(test)]
const GUARDED_REGION: &str = "a guarded region";

/// A fault site: [`probe_storage`], which allocates, launches and reads.
#[cfg(test)]
const STORAGE_PROBE: &str = "the storage probe";

/// Panic if a test armed `site`, and disarm it.
#[cfg(test)]
fn fire_if_armed(site: &'static str) {
    FAULT.with(|armed| {
        if armed.get() == Some(site) {
            armed.set(None);
            panic!("the device is gone");
        }
    });
}

/// Make the next arrival at `site` panic (test-only).
#[cfg(test)]
fn arm_fault_at(site: &'static str) {
    FAULT.with(|armed| armed.set(Some(site)));
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
#[cfg(feature = "gpu-core")]
pub fn probe_storage<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> Result<(), GpuError> {
    use cubecl::prelude::CubeElement;

    // This function allocates, launches and reads, and until this round it ran
    // outside every catch: the fault site is here so a test can say that it no
    // longer does (test-only).
    #[cfg(test)]
    fire_if_armed(STORAGE_PROBE);

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
            return Err(GpuError::ShortRead {
                what: "the storage probe",
                actual: bytes.len(),
                expected,
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
///
/// Private for the reason `cuda_client` is: unprobed and unguarded, and
/// [`gpu_client`] is the only public way to a client.
#[cfg(feature = "gpu-wgpu")]
fn wgpu_client() -> cubecl::prelude::ComputeClient<cubecl_wgpu::WgpuRuntime> {
    use cubecl::prelude::Runtime;
    cubecl_wgpu::WgpuRuntime::client(&cubecl_wgpu::WgpuDevice::default())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

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

    /// A panic inside a stage is a typed error, not an unwind into Python.
    ///
    /// What a lost device does to a per-frame call, on the real stage and the
    /// real client: `arm_fault_at` panics where the runtime would, at the
    /// top of the guarded region, and what comes back is the stage's own error
    /// type. The unguarded call on either side of it is the control — the path
    /// works, so the middle line is measuring the guard and not a broken build.
    #[test]
    fn a_panic_inside_a_stage_is_a_typed_error() {
        use crate::pyramid::PyramidBuilder;

        let mut builder: GpuPyramidBuilder<GpuRuntime> =
            GpuPyramidBuilder::new(gpu_client().unwrap(), &[[0.0, 0.0]]);
        let mut pyramid: GpuPyramid<GpuRuntime> = builder.allocate(64, 64, 2).unwrap();
        let image: crate::image::ImageU16 = crate::image::ImageU16::zeros(64, 64).unwrap();
        builder.build(0, &image, &mut pyramid).unwrap();

        arm_fault_at(GUARDED_REGION);
        let error: crate::pyramid::PyramidError =
            builder.build(0, &image, &mut pyramid).unwrap_err();
        assert!(
            matches!(
                error,
                crate::pyramid::PyramidError::Gpu(GpuError::DeviceLost {
                    what: "pyramid build"
                })
            ),
            "a panicking stage gave {error}"
        );

        // One call, and only that one: the flag is consumed where it fires.
        builder.build(0, &image, &mut pyramid).unwrap();
    }

    /// A panic anywhere in the bring-up is a typed error, not an unwind.
    ///
    /// The re-review's second gap: the guard ended at the client, and
    /// `probe_storage` and the three backend constructors — which allocate,
    /// launch and read — ran after it. The fault is armed at the storage probe,
    /// which is inside the region only if `gpu_backends` guards the whole of
    /// what it does, and what comes back is the constructor's own error type.
    #[test]
    fn a_panic_after_the_client_is_built_is_a_typed_error() {
        arm_fault_at(STORAGE_PROBE);
        let error: crate::frontend::tracker::TrackerError =
            gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0).unwrap_err();
        assert!(
            matches!(
                error,
                crate::frontend::tracker::TrackerError::Gpu(GpuError::ClientPanicked { .. })
            ),
            "a panic in the storage probe gave {error}"
        );

        // And the same call with nothing armed builds the three backends, so
        // what the line above measures is the guard.
        gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0).unwrap();
    }

    /// A library cudarc would panic on is a typed error naming it.
    ///
    /// The mapping, asked for a name no host has; the real absence is a child
    /// process that has genuinely lost `libnvrtc`
    /// (`a_cuda_host_that_cannot_load_nvrtc_is_a_typed_error`, which had to
    /// inhibit the binary's own `RUNPATH` to make it lose it). The two
    /// libraries this lane does need are asserted present in the same test, so
    /// a host that has lost one fails here rather than in the mapping.
    #[cfg(all(feature = "gpu", not(feature = "gpu-wgpu")))]
    #[test]
    fn a_library_the_runtime_cannot_load_is_a_typed_error() {
        assert_eq!(
            first_missing_library(&["slam-rs-no-such-library"]),
            Some(GpuError::MissingLibrary {
                library: "slam-rs-no-such-library"
            })
        );
        assert_eq!(first_missing_library(&CUDA_LIBRARIES), None);
        // And the message says what to do about it, because it is what a user
        // on a fleet machine reads instead of a panic.
        let message: String = GpuError::MissingLibrary { library: "nvrtc" }.to_string();
        assert!(message.contains("nvrtc"), "{message}");
        assert!(message.contains("LD_LIBRARY_PATH"), "{message}");
    }
}
