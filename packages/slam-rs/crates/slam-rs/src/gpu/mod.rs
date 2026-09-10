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
mod track;
mod trig;

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
    /// runtime whose shader compilation fails panics on its own worker
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
    /// A pyramid buffer is longer than the `u32` its device metadata carries.
    ///
    /// The metadata array is `u32` because a level base is an **index**: the
    /// per-patch kernels add it to a pixel offset, so it has to arrive exactly,
    /// and every field in that array is an index into one of the two pyramid
    /// buffers. Refusing a buffer past `u32::MAX` therefore refuses every field
    /// at once. Nothing on this lane can reach it — that buffer would be 8 GB of
    /// `u16`, past every device's binding limit — but a silent truncation is a
    /// wrong trajectory rather than a refusal, and this lane does not do that
    /// (decision D32).
    #[error(
        "a pyramid buffer of {pixels} pixels is past the u32 its device metadata \
         carries, so the kernels could not index it"
    )]
    BufferTooLong {
        /// Pixels the buffer would have held.
        pixels: usize,
    },
    /// The runtime cannot store an element width the kernels bind.
    ///
    /// See [`probe_storage`]: shader compilation failures can be
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
    /// No wgpu adapter for the backend this build runs on.
    ///
    /// The Vulkan loader found no ICD, or the device has no adapter.
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
/// CubeCL 0.10 expects an adapter on its worker thread, so bring-up can
/// panic instead of returning an error (decision D32).
///
/// 1. `probe_availability` asks wgpu's fallible `request_adapter` API before
///    any client exists, so common failures name what is missing.
/// 2. The construction itself runs inside the module's `guarded`, because a
///    probe can only anticipate what it knows to ask.
///
/// **A caught panic still prints its own message to stderr**, and that is the
/// deliberate half of the trade. A quiet panic hook around the construction was
/// tried; the hook is process-global and unsynchronised, so it also
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
/// [`GpuError::NoAdapter`] from the probe, and
/// [`GpuError::ClientPanicked`] from anything it did not anticipate.
#[cfg(feature = "gpu-core")]
pub fn gpu_client() -> Result<cubecl::prelude::ComputeClient<GpuRuntime>, GpuError> {
    probe_availability()?;
    guarded(
        GpuError::ClientPanicked {
            runtime: RUNTIME_NAME,
        },
        || Ok(wgpu_client()),
    )
}

/// What this build's lane is called in an error a user reads.
#[cfg(feature = "gpu-wgpu")]
const RUNTIME_NAME: &str = "wgpu";

/// The same runtime as the name a machine reads: the portable lane's.
#[cfg(feature = "gpu-wgpu")]
pub const BACKEND_NAME: &str = "wgpu";

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
            // (see [`ReadRelay`]).
            let relay: ReadRelay = ReadRelay::default();
            scanner.share_reads(relay.clone());
            tracker.share_reads(relay);
            Ok((builder, tracker, Box::new(scanner)))
        },
    )
}

/// Buffers one stage has launched, offered to whichever stage downloads next,
/// and the bytes that download left behind.
///
/// A read on this lane costs about 0.12 ms of host time before it moves a byte
/// (D77), so two buffers that no arithmetic connects are still worth **one**
/// read between them. The corner scanner's cell keys and the tracker's temporal
/// results are exactly that pair: the selection kernels read the frame alone,
/// so they can be launched before the frameset has decided anything, and the
/// tracker's `collect` was going to synchronise anyway. The scanner stages its
/// handles here, `collect` appends them to its own download, and the scanner
/// takes the tail (D78).
///
/// Shared explicitly by [`gpu_backends`] rather than kept in a process-wide
/// static like [`QUEUED`]: an over-count there drains early, which is only
/// conservative, where a crossed relay would hand one frontend another's
/// pixels. A stage that finds nothing here downloads for itself, which is what
/// the first frameset of a run — no temporal pass, so no `collect` — does.
///
/// A `Mutex` for the reason [`Level0Table`] is one: [`crate::frontend::detect::CornerScan`]
/// is `Send + Sync`, and this is taken twice a frameset by two stages on the
/// frontend's own thread, never contended. A poisoned lock is not an error
/// here — every method degrades to "nothing was staged", and the stager then
/// reads for itself.
#[cfg(feature = "gpu-core")]
#[derive(Debug, Clone, Default)]
pub struct ReadRelay(std::sync::Arc<std::sync::Mutex<RelayInner>>);

/// [`ReadRelay`]'s contents: at most one frameset's worth, in one direction.
#[cfg(feature = "gpu-core")]
#[derive(Debug, Default)]
struct RelayInner {
    /// Launched and waiting for someone to download.
    staged: Vec<cubecl::server::Handle>,
    /// What a download left for the stage that staged it.
    delivered: Option<Vec<cubecl::bytes::Bytes>>,
}

#[cfg(feature = "gpu-core")]
impl ReadRelay {
    /// Offer `handles` to the next download, replacing anything unclaimed:
    /// there is one producer and one frameset in flight.
    pub(super) fn stage(&self, handles: Vec<cubecl::server::Handle>) {
        if let Ok(mut inner) = self.0.lock() {
            inner.staged = handles;
            inner.delivered = None;
        }
    }

    /// Take what was staged, to append to a download this stage is making.
    pub(super) fn take_staged(&self) -> Vec<cubecl::server::Handle> {
        self.0
            .lock()
            .map(|mut inner| std::mem::take(&mut inner.staged))
            .unwrap_or_default()
    }

    /// Leave a download's tail for the stage that staged it.
    pub(super) fn deliver(&self, bytes: Vec<cubecl::bytes::Bytes>) {
        if let Ok(mut inner) = self.0.lock() {
            inner.delivered = Some(bytes);
        }
    }

    /// Handles launched and not yet downloaded by anyone.
    ///
    /// Public because it is the only deterministic way to see the mechanism
    /// work: the values alone cannot tell a carried download from one the
    /// stager made itself, and the seam counters are process-wide statics that
    /// a second test thread moves under the assertion.
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.0.lock().map(|inner| inner.staged.len()).unwrap_or(0)
    }

    /// Buffers a download left here and the stager has not taken yet.
    #[must_use]
    pub fn carried(&self) -> usize {
        self.0
            .lock()
            .map(|inner| inner.delivered.as_ref().map_or(0, Vec::len))
            .unwrap_or(0)
    }

    /// Take the bytes a download left, or `None` when none did — a stage that
    /// gets `None` still holds its own handles and reads them itself.
    pub(super) fn take_delivered(&self) -> Option<Vec<cubecl::bytes::Bytes>> {
        self.0.lock().ok().and_then(|mut inner| {
            inner.staged.clear();
            inner.delivered.take()
        })
    }
}

/// Tasks CubeCL 0.10's client-to-server channel holds before a producer spins
/// (`cubecl-common`'s `CHANNEL_MAX_TASK`).
#[cfg(feature = "gpu-core")]
const CHANNEL_TASKS: usize = 32;

/// The most tasks one stage enqueues between two reports: a tracking pass's
/// three uploads and six launches.
#[cfg(feature = "gpu-core")]
const STAGE_TASKS: usize = 9;

/// Tasks handed to the runtime's server since the channel was last known empty.
///
/// One counter for the process, which is what the channel is: CubeCL caches one
/// client per device, and this pipeline is single-threaded. A second frontend on
/// another thread would make this over-count, which drains early — the safe
/// direction.
#[cfg(feature = "gpu-core")]
static QUEUED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Record that a stage enqueued `tasks`, handing the queue to the runtime's
/// server thread when another stage would not fit in it.
///
/// CubeCL 0.10's client-to-server channel is [`CHANNEL_TASKS`] deep, and a
/// producer that fills it does not block: it spins for 524,288 iterations, then
/// yields 4,096 times, then sleeps in 75 microsecond steps
/// (`cubecl-common`'s `SPIN_BUDGET_CLIENT`). That is tuned for a producer and a
/// server on different cores. This pipeline is pinned to one — the benchmark and
/// the fleet both run it with a single-CPU affinity — so the spin is the
/// server's own core and the queue cannot drain until the producer gives it up.
///
/// It only became reachable when the frontend stopped waiting per camera. A
/// two-camera frameset enqueues 28 tasks between its downloads and stays under
/// the cliff; a four-camera frameset enqueues 56, and fell off it hard enough to
/// spend 2.9 ms a frameset spinning inside one launch. Reporting here keeps the
/// queue under the ceiling at any camera count, at about one flush a frameset on
/// a stereo rig and three on a four-camera one.
///
/// The flush is not extra work: it is the encoding and submission the next
/// download would have paid for, moved earlier.
///
/// # Errors
///
/// [`GpuError::DeviceReadFailed`] when the runtime refuses the flush, which on
/// this path means the device is gone.
#[cfg(feature = "gpu-core")]
pub(super) fn queued<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    tasks: usize,
) -> Result<(), GpuError> {
    use std::sync::atomic::Ordering;

    let total: usize = QUEUED.fetch_add(tasks, Ordering::Relaxed) + tasks;
    if total + STAGE_TASKS < CHANNEL_TASKS {
        return Ok(());
    }
    QUEUED.store(0, Ordering::Relaxed);
    client
        .flush()
        .map_err(|error| read_failed("the queued launches", &error))
}

/// The channel is empty: a download has waited for everything that was in it.
#[cfg(feature = "gpu-core")]
pub(super) fn drained() {
    QUEUED.store(0, std::sync::atomic::Ordering::Relaxed);
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
/// `process_frame`'s transactional restore; on the bring-up path it surfaces as
/// a `RecvError`. Both land here as the typed error the caller already
/// documents.
///
/// It costs nothing on the path that does not panic — `catch_unwind` is a
/// landing pad the happy path never enters — and three interleaved A/B pairs on
/// MIO07/1500 measured the guarded build at 5.77-5.86 ms against 5.92-5.99,
/// build-to-build noise in the guardless build's favour nowhere.
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
// region does, and what one inside the storage probe does, the probe being the
// site that has to be inside the guard for `gpu_backends` to cover the whole of
// what it does. A doc comment cannot sit on a macro invocation.
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

/// A fault site: the corner scan's download, which is where a real device
/// failure lands **after** the scan has recorded the new frame's geometry. The
/// guarded region's own site fires before the body runs and so cannot ask what
/// a half-finished scan leaves behind.
#[cfg(test)]
const CORNER_SCAN_READ: &str = "the corner scan's read";

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
/// Shader compilation failures can be **silent**, and are not
/// visible in `client.properties()`, which describes the device rather than the
/// compiler that will run on it:
///
/// * `cubecl-wgpu`'s **WGSL** compiler panics on `u16` and `u8` —
///   "U16 is not a valid WgpuElement" — on the worker thread, returning zeros. The pyramid is `u16` and the candidate image is `u8`, so on
///   the portable lane without `cubecl-wgpu/spirv` every kernel here silently
///   produces nothing. Measured on this host: the pyramid came back all zeros
///   and the detector found no corners, with no error anywhere.
///
/// One frame on the device, and how many pixels it holds.
///
/// `create_from_slice` is CubeCL 0.10's only host-to-device write, it allocates
/// a buffer the size of the slice, and it copies the payload **twice** on the
/// host before the bus sees it (`slice.to_vec()`, then
/// `Bytes::from_bytes_vec(data.to_vec())` inside `do_create_from_slices`). So
/// the upload is exactly as long as the frame and nothing more: an unstrided
/// frame goes straight out of the caller's buffer with no staging copy at all,
/// and only a strided one — dav1d's shape — is repacked row by row into
/// `scratch`, which the caller owns so the per-frame path never allocates.
///
/// Both the pyramid builder's level-0 upload and the corner scanner's own frame
/// upload are this, which is why it is here and not in either.
#[cfg(feature = "gpu-core")]
pub(super) fn upload_frame<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    image: &crate::image::ImageU16,
    scratch: &mut Vec<u16>,
) -> (cubecl::server::Handle, usize) {
    use cubecl::prelude::CubeElement;

    let (width, height): (usize, usize) = (image.width(), image.height());
    let pixels: usize = width * height;
    if image.stride() == width {
        return (
            seam::UPLOAD
                .measure(|| client.create_from_slice(u16::as_bytes(&image.data()[..pixels]))),
            pixels,
        );
    }
    scratch.clear();
    scratch.reserve(pixels);
    for y in 0..height {
        scratch.extend_from_slice(image.row(y));
    }
    (
        seam::UPLOAD.measure(|| client.create_from_slice(u16::as_bytes(scratch))),
        pixels,
    )
}

/// So the check is the one thing that cannot lie: write a known pattern, copy
/// it **on the device**, read it back. Four widths, 256 elements each, once at
/// construction — microseconds, and it is what a fleet machine whose driver
/// silently mishandles a width will fail on instead of producing a trajectory
/// out of zeros (decisions D21, D32).
///
/// # Errors
///
/// [`GpuError::StorageRoundTrip`] naming the width that did not survive,
/// [`GpuError::ShortRead`] or [`GpuError::DeviceReadFailed`] from the read back,
/// and [`GpuError::DeviceLost`] if any of it panics: this is public, so it is a
/// device operation a caller reaches without going through [`gpu_backends`] and
/// its guard, and it carries its own (decision D32).
#[cfg(feature = "gpu-core")]
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

    guarded(
        GpuError::DeviceLost {
            what: "the storage probe",
        },
        || {
            // The fault site is inside the guard, so a test can say the guard is
            // what turns a panicking probe into an error (test-only).
            #[cfg(test)]
            fire_if_armed(STORAGE_PROBE);

            const COUNT: usize = 256;
            // Patterns whose every byte differs from its neighbours, so a
            // truncation, a widening or a packing slip all show up rather than
            // cancelling.
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
        },
    )
}

/// A [`cubecl_wgpu::WgpuRuntime`] client on the default device.
///
/// The portable lane: Vulkan on the Spark and the Pi 5, Metal on macOS. Select
/// the adapter with `CUBECL_WGPU_DEFAULT_DEVICE`; `WGPU_BACKEND` and
/// `WGPU_ADAPTER_NAME` are ignored by cubecl-wgpu.
///
/// Private because it is unprobed and unguarded, and
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
    /// Two guards, because the bring-up has two layers now. `gpu_backends`
    /// itself is guarded from its first line to the returned backends, and a
    /// fault at the top of that region — before any client exists — comes back
    /// as `ClientPanicked`. Inside it, `probe_storage` and the three
    /// constructors each carry their own guard, because each is also a public
    /// entry a caller reaches on its own, and a fault at the probe's site comes
    /// back as that guard's `DeviceLost`. Either way nothing unwinds past the
    /// constructor (decision D32).
    #[test]
    fn a_panic_after_the_client_is_built_is_a_typed_error() {
        arm_fault_at(GUARDED_REGION);
        let outer: crate::frontend::tracker::TrackerError =
            gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0, 2).unwrap_err();
        assert!(
            matches!(
                outer,
                crate::frontend::tracker::TrackerError::Gpu(GpuError::ClientPanicked { .. })
            ),
            "a panic in the outer region gave {outer}"
        );

        arm_fault_at(STORAGE_PROBE);
        let probe: crate::frontend::tracker::TrackerError =
            gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0, 2).unwrap_err();
        assert!(
            matches!(
                probe,
                crate::frontend::tracker::TrackerError::Gpu(GpuError::DeviceLost {
                    what: "the storage probe"
                })
            ),
            "a panic in the storage probe gave {probe}"
        );

        // And the same call with nothing armed builds the three backends, so
        // what the lines above measure is the guards.
        gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0, 2).unwrap();
    }

    /// A panic in an exported constructor is a typed error, not an unwind.
    ///
    /// Three of them touch the device before they return — the corner scanner
    /// uploads the FAST ring, the patch set allocates its store and its
    /// positions, the tracker allocates both transform buffers — and each is a
    /// public entry a caller outside `gpu_backends` can reach. Armed at the
    /// guard, each returns its own error type; unarmed, each builds, so what
    /// the armed lines measure is the guard and not a broken build (decision
    /// D32).
    #[test]
    fn a_panic_in_an_exported_constructor_is_a_typed_error() {
        use crate::frontend::patterns::Pattern51;
        use crate::frontend::tracker::TrackerError;

        let client = gpu_client().unwrap();

        arm_fault_at(GUARDED_REGION);
        let scan: GpuError = GpuCornerScan::<GpuRuntime>::new(client.clone()).unwrap_err();
        assert_eq!(
            scan,
            GpuError::DeviceLost {
                what: "corner scan setup"
            }
        );

        arm_fault_at(GUARDED_REGION);
        let patches: TrackerError =
            GpuPatches::<Pattern51, GpuRuntime>::new(client.clone(), 64, 4).unwrap_err();
        assert!(
            matches!(
                patches,
                TrackerError::Gpu(GpuError::DeviceLost {
                    what: "patch allocation"
                })
            ),
            "a panicking patch allocation gave {patches}"
        );

        arm_fault_at(GUARDED_REGION);
        let tracker: TrackerError =
            GpuPatchTracker::<Pattern51, GpuRuntime>::new(client.clone(), 64, 4, 5, 4.0, 2)
                .unwrap_err();
        assert!(
            matches!(
                tracker,
                TrackerError::Gpu(GpuError::DeviceLost {
                    what: "tracker allocation"
                })
            ),
            "a panicking tracker allocation gave {tracker}"
        );

        GpuCornerScan::<GpuRuntime>::new(client.clone()).unwrap();
        GpuPatches::<Pattern51, GpuRuntime>::new(client.clone(), 64, 4).unwrap();
        GpuPatchTracker::<Pattern51, GpuRuntime>::new(client, 64, 4, 5, 4.0, 2).unwrap();
    }

    /// A panic in the public storage probe is a typed error, not an unwind.
    ///
    /// [`probe_storage`] is public and it allocates, launches and downloads, so
    /// it is a device operation a caller reaches without going through
    /// `gpu_backends` and its guard. The fault is armed at the probe's own
    /// site — the same one `a_panic_after_the_client_is_built_is_a_typed_error`
    /// uses through the constructor path — and here the call is direct.
    #[test]
    fn a_panic_in_the_public_storage_probe_is_a_typed_error() {
        let client = gpu_client().unwrap();

        arm_fault_at(STORAGE_PROBE);
        assert_eq!(
            probe_storage(&client).unwrap_err(),
            GpuError::DeviceLost {
                what: "the storage probe"
            }
        );

        // Unarmed the same probe passes on this host, so the line above is the
        // guard and not a runtime that cannot store these widths.
        probe_storage(&client).unwrap();
    }

    /// A panic in an exported read is a typed error, not an unwind.
    ///
    /// The two downloads that are not on the per-frame path — a pyramid level
    /// and the patch store, both of them how generic code and the tolerance
    /// suite read a buffer a GPU backend owns. Neither is inside a per-frame
    /// stage, so neither was covered by the stage guards.
    #[test]
    fn a_panic_in_an_exported_read_is_a_typed_error() {
        use crate::frontend::patterns::Pattern51;
        use crate::frontend::tracker::TrackerError;
        use crate::pyramid::{Pyramid, PyramidBuilder, PyramidError};

        let client = gpu_client().unwrap();
        let builder: GpuPyramidBuilder<GpuRuntime> =
            GpuPyramidBuilder::new(client.clone(), &[[0.0, 0.0]]);
        let pyramid: GpuPyramid<GpuRuntime> = builder.allocate(64, 64, 2).unwrap();
        let patches: GpuPatches<Pattern51, GpuRuntime> = GpuPatches::new(client, 64, 3).unwrap();
        let mut level: crate::image::ImageU16 = crate::image::ImageU16::default();

        arm_fault_at(GUARDED_REGION);
        let read: PyramidError = pyramid.copy_level_into(0, &mut level).unwrap_err();
        assert!(
            matches!(
                read,
                PyramidError::Gpu(GpuError::DeviceLost {
                    what: "a pyramid level read"
                })
            ),
            "a panicking level read gave {read}"
        );

        arm_fault_at(GUARDED_REGION);
        let store: TrackerError = patches.read_store().unwrap_err();
        assert!(
            matches!(
                store,
                TrackerError::Gpu(GpuError::DeviceLost {
                    what: "the patch store read"
                })
            ),
            "a panicking store read gave {store}"
        );

        // Both reads succeed unarmed, so the two lines above are the guards.
        pyramid.copy_level_into(0, &mut level).unwrap();
        patches.read_store().unwrap();
    }
}
