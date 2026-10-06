//! Host/device transfer primitives.
use crate::runtime::GpuError;
/// Upload a host slice through one aligned allocation.
///
/// # Arguments
/// * `client` - Destination device and current stream.
/// * `bytes` - Host bytes copied into CubeCL-owned aligned storage.
///
/// # Errors
/// Converts device allocation/upload panics to a typed error.
pub fn upload<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: &[u8],
) -> Result<cubecl::server::Handle, GpuError> {
    // Copy directly into aligned owned storage; create_from_slice copies
    // through two Vecs before allocating this same aligned storage.
    crate::runtime::guarded(GpuError::DeviceLost { what: "upload" }, || {
        Ok(upload_inner(client, bytes))
    })
}

pub(crate) fn upload_inner<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: &[u8],
) -> cubecl::server::Handle {
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

/// Storage binding alignment and maximum size, in bytes.
pub fn binding_limits<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> (usize, usize) {
    let memory = &client.properties().memory;
    (
        (memory.alignment as usize).max(256),
        memory.max_page_size as usize,
    )
}

/// Download handles together, converting device failures and unwinds to typed errors.
///
/// # Errors
/// Returns a typed read or device-loss error. Callers validate operation-specific lengths.
pub fn read_buffers<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    read_with_lookahead(client, handles, what, || Ok(()))
}
/// Submit a readback, enqueue independent work, then wait for the copied bytes.
///
/// # Arguments
/// * `client`, `handles` - Device buffers, in desired return order.
/// * `what` - Operation name used in failures.
/// * `after_copy` - Independent device work queued after the copy submission.
///
/// The callback must not alter the buffers before their copy has been submitted.
/// CubeCL's `read_async` submits that copy before returning its future.
/// Operation-specific element counts remain the caller's responsibility.
///
/// # Errors
/// Preserves callback failures and converts read failures and device panics.
pub fn read_with_lookahead<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
    after_copy: impl FnOnce() -> Result<(), GpuError>,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    crate::runtime::guarded(GpuError::DeviceLost { what }, || {
        read_inner(client, handles, what, after_copy)
    })
}

pub(crate) fn read_inner<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
    after_copy: impl FnOnce() -> Result<(), GpuError>,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    #[cfg(test)]
    if crate::fault::armed("blocking read") {
        return Err(GpuError::DeviceReadFailed { what });
    }
    let expected = handles.len();
    let pending = client.read_async(handles);
    after_copy()?;
    let bytes =
        cubecl::future::reader::read_sync(pending).map_err(|error| read_failed(what, &error))?;
    if bytes.len() != expected {
        return Err(GpuError::DeviceReadFailed { what });
    }
    Ok(bytes)
}

/// Run device work on the client's exclusive server thread.
///
/// # Arguments
/// * `client` - Device whose current stream is preserved during the call.
/// * `what` - Operation name used in typed errors.
/// * `body` - Work borrowing caller data; nested client operations run directly.
///
/// # Errors
/// Preserves body errors and converts device communication failures and panics.
///
/// ```no_run
/// # #[cfg(feature = "wgpu")] {
/// use kornia_staging_gpu::{runtime::{gpu_client, GpuError}, transfer::execute_exclusive};
/// let client = gpu_client()?;
/// let value = execute_exclusive(&client, "example", || Ok::<_, GpuError>(42))?;
/// assert_eq!(value, 42);
/// # }
/// # Ok::<(), kornia_staging_gpu::runtime::GpuError>(())
/// ```
pub fn execute_exclusive<R, T, E>(
    client: &cubecl::prelude::ComputeClient<R>,
    what: &'static str,
    body: impl FnOnce() -> Result<T, E> + Send,
) -> Result<T, E>
where
    R: cubecl::prelude::Runtime,
    T: Send + 'static,
    E: From<GpuError> + Send + 'static,
{
    crate::runtime::guarded(GpuError::DeviceLost { what }, || {
        client
            .exclusive(|| crate::runtime::guarded(GpuError::DeviceLost { what }, body))
            .map_err(|error| E::from(read_failed(what, &error)))?
    })
}
