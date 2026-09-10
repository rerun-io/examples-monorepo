//! Device bring-up, guarded errors, storage validation, and fault injection.
use super::submission::{self, drained, empty, read_failed};
use super::{GpuRuntime, kernels};

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
pub(super) const RUNTIME_NAME: &str = "wgpu";

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

/// Run `stage`, turning a panic inside it into `fault`.
///
/// The seam between CubeCL's panics and decision D32. A `ServerError` on a
/// download is mapped by [`super::submission::read_failed`], but that is only the half of the
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
pub(super) fn guarded<T, E: From<GpuError>>(
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
pub(super) const CORNER_SCAN_READ: &str = "the corner scan's read";

/// A fault site: the blocking read's own download. Unlike the sites above this
/// one is not a panic — [`read_blocking`] turns it into the `ServerError` a
/// lost device returns — because what it is armed for is the queue accounting
/// that runs *after* a failed read.
#[cfg(test)]
pub(super) const BLOCKING_READ: &str = "the blocking read";

/// Whether a test armed `site`, disarming it. The caller decides what the fault
/// means; [`fire_if_armed`] panics, [`download`] returns an error.
#[cfg(test)]
pub(super) fn armed(site: &'static str) -> bool {
    FAULT.with(|fault| {
        let hit: bool = fault.get() == Some(site);
        if hit {
            fault.set(None);
        }
        hit
    })
}

/// Panic if a test armed `site`, and disarm it.
#[cfg(test)]
pub(super) fn fire_if_armed(site: &'static str) {
    FAULT.with(|armed| {
        if armed.get() == Some(site) {
            armed.set(None);
            panic!("the device is gone");
        }
    });
}

/// Make the next arrival at `site` panic (test-only).
#[cfg(test)]
pub(super) fn arm_fault_at(site: &'static str) {
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
/// device operation a caller reaches without going through [`super::gpu_backends`] and
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
        let source: cubecl::server::Handle = submission::upload(client, N::as_bytes(pattern));
        let target: cubecl::server::Handle = empty(client, expected);
        kernels::launch_probe::<N, R>(client, (&source, count), (&target, count), count);
        let bytes = client
            .read_one(target)
            .map_err(|error| read_failed("the storage probe", &error))?;
        drained(client);
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
mod tests;
