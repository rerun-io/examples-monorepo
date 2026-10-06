//! Device bring-up, guarded errors and capability validation.
use cubecl::prelude::*;
use std::mem::{size_of, size_of_val};

use crate::kernels;
use crate::transfer::{self, read_failed};
#[cfg(feature = "wgpu")]
use crate::GpuRuntime;

/// What can go wrong bringing up or running a GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
    /// A frame was begun before the previous frame scope ended.
    #[error("nested GPU frontend frame")]
    NestedFrame,
    /// The fused kernel requires working subgroup operations.
    #[error(
        "GPU KLT requires working power-of-two subgroups of at least 16 lanes; the subgroup probe failed, use the CPU lane"
    )]
    SubgroupRoundTrip,
    /// A device read came back with the wrong number of bytes.
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
    #[error("reading {what} from the device failed")]
    DeviceReadFailed {
        /// Which buffer.
        what: &'static str,
    },
    /// A GPU stage panicked instead of returning an error.
    #[error(
        "the GPU {what} failed on the device; the log carries the runtime's own \
         message, and the CPU frontend runs without a GPU"
    )]
    DeviceLost {
        /// Which stage was running.
        what: &'static str,
    },
    /// A pyramid buffer is longer than the `u32` its device metadata carries.
    #[error(
        "a pyramid buffer of {pixels} pixels is past the u32 its device metadata \
         carries, so the kernels could not index it"
    )]
    BufferTooLong {
        /// Pixels the buffer would have held.
        pixels: usize,
    },
    /// The runtime cannot store an element width the kernels bind.
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
    #[error(
        "wgpu found no {backend} adapter on this host: install a {backend} driver, \
         or run the CPU frontend, which needs no adapter"
    )]
    NoAdapter {
        /// The graphics backend cubecl-wgpu would have used.
        backend: &'static str,
    },
    /// Constructing the CubeCL client panicked.
    #[error(
        "building the {runtime} client panicked; the log carries the runtime's own \
         message, and the CPU frontend runs without a GPU"
    )]
    ClientPanicked {
        /// Which runtime was being built.
        runtime: &'static str,
    },
}

/// Construct the default wgpu client after a fallible adapter probe.
///
/// Select a device with `CUBECL_WGPU_DEFAULT_DEVICE` before the first call.
/// Initialization and its result are cached for the process, including failures.
/// Panic messages remain visible.
///
/// # Errors
/// Returns [`GpuError::NoAdapter`] or [`GpuError::ClientPanicked`].
#[cfg(feature = "wgpu")]
pub fn gpu_client() -> Result<cubecl::prelude::ComputeClient<GpuRuntime>, GpuError> {
    static CLIENT: std::sync::OnceLock<Result<ComputeClient<GpuRuntime>, GpuError>> =
        std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        guarded(GpuError::ClientPanicked { runtime: RUNTIME_NAME }, || {
            probe_availability()?;
            Ok(wgpu_client())
        })
    }).clone()
}

/// What this build's lane is called in an error a user reads.
#[cfg(feature = "wgpu")]
pub const RUNTIME_NAME: &str = "wgpu";

/// The same runtime as the name a machine reads: the portable lane's.
#[cfg(feature = "wgpu")]
pub const BACKEND_NAME: &str = "wgpu";

/// Probe the same graphics backend and power preference used by CubeCL.
#[cfg(feature = "wgpu")]
fn probe_availability() -> Result<(), GpuError> {
    #[cfg(test)]
    PROBE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
        apply_limit_buckets: false,
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

/// Run a fallible operation, converting an unwind into the supplied typed error.
///
/// # Errors
/// Returns the operation error or `fault` if the operation panics.
pub fn guarded<T, E: From<GpuError>>(
    fault: GpuError,
    stage: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(stage));
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

/// Copy known u8, u16, u32 and f32 patterns on the device to verify storage support.
///
/// # Errors
/// Returns a typed read, storage mismatch or device panic error.
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
        let source: cubecl::server::Handle = transfer::upload(client, N::as_bytes(pattern));
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
            #[cfg(test)]
            PROBE_FAULT.with(|fault| {
                if fault.replace(false) {
                    panic!("the device is gone");
                }
            });

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
/// Uses Vulkan on Linux and Metal on macOS. Select
/// the adapter with `CUBECL_WGPU_DEFAULT_DEVICE`; `WGPU_BACKEND` and
/// `WGPU_ADAPTER_NAME` are ignored by cubecl-wgpu.
///
/// Private because it is unprobed and unguarded, and
/// [`gpu_client`] is the only public way to a client.
#[cfg(feature = "wgpu")]
fn wgpu_client() -> cubecl::prelude::ComputeClient<cubecl_wgpu::WgpuRuntime> {
    use cubecl::prelude::Runtime;
    cubecl_wgpu::WgpuRuntime::client(&cubecl_wgpu::WgpuDevice::default())
}

#[cube(launch)]
fn subgroup_probe(out: &mut [u32]) {
    let lane = UNIT_POS_PLANE;
    let base = ABSOLUTE_POS * 5usize;
    out[base] = PLANE_DIM;
    out[base + 1usize] = plane_sum(lane + 1u32);
    out[base + 2usize] = plane_broadcast(lane, 3u32);
    out[base + 3usize] = plane_shuffle_xor(lane, 7u32);
    out[base + 4usize] = plane_shuffle(lane, (lane / 8u32) * 8u32 + 2u32);
}

/// Verify the operations and group width used by fused KLT on the actual device.
///
/// # Errors
/// Returns [`GpuError::SubgroupRoundTrip`] if any operation or width is unsupported.
pub fn probe_subgroups<R: Runtime>(client: &ComputeClient<R>) -> Result<usize, GpuError> {
    guarded(GpuError::SubgroupRoundTrip, || {
        if !client
            .features()
            .plane
            .contains(cubecl::ir::features::Plane::Ops)
            || client.properties().hardware.plane_size_min < 16
        {
            return Err(GpuError::SubgroupRoundTrip);
        }
        let units = client.properties().hardware.plane_size_max.max(64);
        let output = client.empty(units as usize * 5 * size_of::<u32>());

        // SAFETY: The probe launches `units` threads and allocates five u32 outputs
        // per unit.
        unsafe {
            subgroup_probe::launch::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(units),
                BufferArg::from_raw_parts(output.clone(), units as usize * 5),
            );
        }
        let bytes = client.read_one(output).map_err(|error| {
            read_failed("the subgroup probe", &error);
            GpuError::SubgroupRoundTrip
        })?;

        let values = u32::from_bytes(&bytes);
        if values.len() != units as usize * 5 {
            return Err(GpuError::SubgroupRoundTrip);
        }
        for (index, value) in values.as_chunks::<5>().0.iter().enumerate() {
            let size = value[0];
            if size < 16 || !size.is_power_of_two() {
                return Err(GpuError::SubgroupRoundTrip);
            }
            let lane = index as u32 % size;
            if *value != [size, size * (size + 1) / 2, 3, lane ^ 7, lane / 8 * 8 + 2] {
                return Err(GpuError::SubgroupRoundTrip);
            }
        }
        Ok(values[0] as usize)
    })
}

#[cfg(all(test, feature = "wgpu"))]
static PROBE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    static PROBE_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    #[test]
    fn client_initialization_probes_once_per_process() {
        gpu_client().unwrap();
        gpu_client().unwrap();
        assert_eq!(PROBE_COUNT.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
    #[test]
    fn panic_and_error_are_preserved_as_typed_failures() {
        let fault = GpuError::DeviceLost {
            what: "test operation",
        };
        assert_eq!(
            guarded::<(), GpuError>(fault, || panic!("lost device")),
            Err(fault)
        );
        let read = GpuError::DeviceReadFailed { what: "test read" };
        assert_eq!(guarded::<(), GpuError>(fault, || Err(read)), Err(read));
        assert_eq!(guarded::<u32, GpuError>(fault, || Ok(7)), Ok(7));
    }
    #[cfg(feature = "wgpu")]
    #[test]
    fn a_panic_in_the_public_storage_probe_is_a_typed_error() {
        let client = gpu_client().unwrap();
        PROBE_FAULT.with(|fault| fault.set(true));
        assert_eq!(
            probe_storage(&client),
            Err(GpuError::DeviceLost {
                what: "the storage probe"
            })
        );
        probe_storage(&client).unwrap();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn actual_device_subgroups_pass_the_contract() {
        assert!(probe_subgroups(&gpu_client().unwrap()).unwrap() >= 16);
    }
}
