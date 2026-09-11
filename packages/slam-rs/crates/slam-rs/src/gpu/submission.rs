//! Mandatory per-device reservation before frontend submissions.

use super::{GpuError, seam};

/// CubeCL 0.11.0-pre.3's private `custom_channel::CHANNEL_MAX_TASK` is 32
/// (`cubecl-common/src/device/handle/channel.rs`).
/// Recheck this value when upgrading CubeCL; it is not exported by the runtime.
pub const CHANNEL_TASKS: usize = 32;

thread_local! {
    // One producer per device is required. Separate producer threads must not
    // submit to the same device through this helper. This is a performance
    // budget for the single-threaded frontend, not a concurrency guarantee.
    static QUEUED: std::cell::RefCell<Vec<((std::any::TypeId, usize), usize)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// Reserve a stage BEFORE it submits. Each upload, allocation and kernel is a
/// one-task stage, so even a pyramid larger than the budget is split safely.
/// Leave one channel slot for the blocking flush or download itself.
///
/// CubeCL clones share `utilities.properties` (0.11.0-pre.3 `client.rs`), so its
/// address identifies their device's queue. A stale address can only retain an
/// old count and cause an early flush. Runtime type separates backend types.
/// Each device must have one producer thread; a read resets only that device.
/// Which device's queue a count belongs to: the runtime type, and the address
/// of the properties every clone of that device's client shares.
type QueueKey = (std::any::TypeId, usize);

/// `client`'s device, as [`QUEUED`] keys one. Both the reservation and the
/// drain have to agree on it exactly, so neither spells it out.
fn queue_key<R: cubecl::prelude::Runtime>(client: &cubecl::prelude::ComputeClient<R>) -> QueueKey {
    (
        std::any::TypeId::of::<R>(),
        std::ptr::from_ref(client.properties()) as usize,
    )
}

pub(super) fn reserve<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    tasks: usize,
) {
    assert!(
        tasks < CHANNEL_TASKS,
        "split stages larger than the queue budget"
    );
    let key: QueueKey = queue_key(client);
    QUEUED.with(|queues| {
        let mut queues = queues.borrow_mut();
        let index = queues
            .iter()
            .position(|(id, _)| *id == key)
            .unwrap_or_else(|| {
                queues.push((key, 0));
                queues.len() - 1
            });
        let count = &mut queues[index].1;
        if *count + tasks >= CHANNEL_TASKS {
            // Same failure contract as CubeCL's uploads and launches: guarded()
            // at the public stage boundary converts this to a typed GPU error.
            client
                .flush()
                .unwrap_or_else(|error| panic!("GPU queue flush failed: {error}"));
            *count = 0;
        }
        *count += tasks;
        seam::queue_reserved(*count);
    });
}

/// A blocking read has consumed this producer's outstanding tasks on this device.
pub(super) fn drained<R: cubecl::prelude::Runtime>(client: &cubecl::prelude::ComputeClient<R>) {
    let key: QueueKey = queue_key(client);
    QUEUED.with(|queues| {
        if let Some((_, count)) = queues.borrow_mut().iter_mut().find(|(id, _)| *id == key) {
            *count = 0;
        }
    });
}

/// Allocation also submits one initialize-memory task in CubeCL 0.10.0.
pub(super) fn empty<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: usize,
) -> cubecl::server::Handle {
    reserve(client, 1);
    client.empty(bytes)
}

/// Tasks this producer has outstanding on `client`'s device (test-only).
#[cfg(test)]
pub(super) fn queued_tasks<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> usize {
    let key: QueueKey = queue_key(client);
    QUEUED.with(|queues| {
        queues
            .borrow()
            .iter()
            .find(|(id, _)| *id == key)
            .map_or(0, |(_, count)| *count)
    })
}

/// Count one launch.
pub(super) fn launch<R: cubecl::prelude::Runtime>(client: &cubecl::prelude::ComputeClient<R>) {
    reserve(client, 1);
    seam::LAUNCH.count();
}

/// Reserve and measure the one task that uploads a host slice.
pub(super) fn upload<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: &[u8],
) -> cubecl::server::Handle {
    reserve(client, 1);
    seam::UPLOAD.measure(|| client.create_from_slice(bytes))
}

/// A failed device read as a typed error, with the runtime's own reason logged.
///
/// [`GpuError`] is `Copy`, so it cannot carry the `ServerError`'s reason and
/// backtrace; the warning is where they are kept, and the returned variant is
/// what the stage errors carry to the caller.
pub(super) fn read_failed(what: &'static str, error: &cubecl::server::ServerError) -> GpuError {
    log::warn!("reading {what} from the device failed: {error}");
    GpuError::DeviceReadFailed { what }
}

/// The download itself, or the `ServerError` a lost device returns when a test
/// armed [`BLOCKING_READ`].
///
/// Its own function so the injection cannot reach past the read into the queue
/// accounting [`read_blocking`] does after it, which is what the test is about.
#[cfg(feature = "gpu-core")]
fn download<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handles: Vec<cubecl::server::Handle>,
) -> Result<Vec<cubecl::bytes::Bytes>, cubecl::server::ServerError> {
    #[cfg(test)]
    if super::runtime::armed(super::runtime::BLOCKING_READ) {
        return Err(cubecl::server::ServerError::Generic {
            reason: "the device is gone".to_owned(),
            backtrace: Default::default(),
        });
    }
    cubecl::future::reader::read_sync(client.read_async(handles))
}

/// One blocking download of every handle at once, and the protocol around it.
///
/// Four per-frame stages wait on the device — the corner scan's candidate image
/// and bitmask, the two cell-key reads, and the tracker batch's packed results
/// (D78) — and each has to do the same three things in the same order: block on
/// the **fallible** `read_sync` rather than `client.read`, which is
/// `read_sync(..).expect("TODO")` and would unwind out of the frontend with the
/// GIL detached (D32); tell [`drained`] this producer's queued tasks are gone;
/// and turn a `ServerError` into the typed [`GpuError::DeviceReadFailed`] the
/// stage error carries. Here so the queue accounting and the failure mapping
/// cannot drift apart one stage at a time.
///
/// What each buffer *is* stays with the stage that asked for it: how many
/// buffers it expects, how long each must be, whether a relay delivery answered
/// instead (D78), and where a test arms a panicking read.
///
/// `meter` is the seam counter of the stage that **issued** the read, not of
/// whoever's data the buffers hold — a tracker download that carries the corner
/// scanner's keys is one tracker read (D78).
#[cfg(feature = "gpu-core")]
pub(super) fn read_blocking<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
    meter: &seam::Meter,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    let outcome: Result<Vec<cubecl::bytes::Bytes>, cubecl::server::ServerError> =
        meter.measure(|| {
            let bytes = download(client, handles);
            // Unconditional, and inside the meter: the wait consumed this
            // producer's tasks whether the download answered or failed, and a
            // count left standing would flush the next stage early.
            drained(client);
            bytes
        });
    outcome.map_err(|error| read_failed(what, &error))
}

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
            upload(client, u16::as_bytes(&image.data()[..pixels])),
            pixels,
        );
    }
    scratch.clear();
    scratch.reserve(pixels);
    for y in 0..height {
        scratch.extend_from_slice(image.row(y));
    }
    (upload(client, u16::as_bytes(scratch)), pixels)
}
