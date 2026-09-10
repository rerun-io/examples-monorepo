//! Mandatory per-device reservation before frontend submissions.

use super::seam;

/// CubeCL 0.10.0's private `custom_channel::CHANNEL_MAX_TASK` is 32.
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
/// CubeCL clones share `utilities.properties` (0.10.0 `client.rs`), so its
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
