//! Fault sites for the frame scheduler's rollback tests.
use super::*;

// Thread-local fault sites exercise consumer frame rollback.
#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// A fault site: the inside of a guarded region.
#[cfg(test)]
const GUARDED_REGION: &str = "a guarded region";

/// A fault site: the blocking read's own download. Unlike the sites above this
/// one is not a panic: the download returns a typed device error.
#[cfg(test)]
pub(super) const BLOCKING_READ: &str = "the blocking read";

/// Whether a test armed `site`, disarming it. The caller decides what the fault
/// means; [`fire_if_armed`] panics, [`super::submission::read_with_lookahead`] returns an error.
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

pub(super) fn guarded<T, E: From<GpuError>>(
    fault: GpuError,
    body: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    kornia_staging_gpu::runtime::guarded(fault, || {
        fire_if_armed(GUARDED_REGION);
        body()
    })
}
#[cfg(test)]
mod tests;
