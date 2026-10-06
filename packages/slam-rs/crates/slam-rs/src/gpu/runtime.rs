//! Fault sites for the frame scheduler's rollback tests.

// Thread-local fault sites exercise consumer frame rollback.
thread_local! {
    static FAULT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// A fault site: the blocking read's own download. Unlike the sites above this
/// one is not a panic: the download returns a typed device error.
pub(super) const BLOCKING_READ: &str = "the blocking read";

/// Whether a test armed `site`, disarming it. The caller decides what the fault
/// means; [`fire_if_armed`] panics, [`super::submission::read_with_lookahead`] returns an error.
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
pub(super) fn fire_if_armed(site: &'static str) {
    FAULT.with(|armed| {
        if armed.get() == Some(site) {
            armed.set(None);
            panic!("the device is gone");
        }
    });
}

/// Make the next arrival at `site` panic (test-only).
pub(super) fn arm_fault_at(site: &'static str) {
    FAULT.with(|armed| armed.set(Some(site)));
}

mod tests;
