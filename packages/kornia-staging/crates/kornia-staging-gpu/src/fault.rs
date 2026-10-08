//! Thread-local fault sites shared by runtime and operation tests.
thread_local! { static SITE: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) }; }
#[cfg(feature = "wgpu")]
pub(crate) fn arm(site: &'static str) {
    SITE.with(|value| value.set(Some(site)));
}
pub(crate) fn armed(site: &'static str) -> bool {
    SITE.with(|value| {
        let hit = value.get() == Some(site);
        if hit {
            value.set(None);
        }
        hit
    })
}
pub(crate) fn fire(site: &'static str) {
    if armed(site) {
        panic!("the device is gone");
    }
}
