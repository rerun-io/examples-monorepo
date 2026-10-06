//! Host/device transfer primitives.
use crate::runtime::GpuError;
/// Upload a host slice through one aligned allocation.
pub fn upload<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: &[u8],
) -> cubecl::server::Handle {
    // Copy directly into aligned owned storage; create_from_slice copies
    // through two Vecs before allocating this same aligned storage.
    let mut data = cubecl::bytes::Bytes::from_elems(Vec::<u8>::new());
    data.extend_from_byte_slice(bytes);
    client.create(data)
}

/// A failed device read as a typed error, with the runtime's own reason logged.
///
/// [`GpuError`] is `Copy`, so it cannot carry the `ServerError`'s reason and
/// backtrace; the warning is where they are kept, and the returned variant is
/// what the stage errors carry to the caller.
pub fn read_failed(what: &'static str, error: &cubecl::server::ServerError) -> GpuError {
    log::warn!("reading {what} from the device failed: {error}");
    GpuError::DeviceReadFailed { what }
}
