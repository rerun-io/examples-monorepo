//! Page flags of the web viewer, read from the page's query string: `force_no_subgroups=1` behaves as if the adapter
//! lacked "subgroups" (the splat kernels use their workgroup-memory variant); `profile=1` logs GPU timestamps of the
//! splat frame and motion pass every few seconds.

use re_viewer::external::re_log;

/// `<name>=1` in the page's query string.
pub fn query_flag(name: &str) -> bool {
    let wanted = format!("{name}=1");
    web_sys::window()
        .and_then(|window| window.location().search().ok())
        .is_some_and(|search| {
            search
                .trim_start_matches('?')
                .split('&')
                .any(|pair| pair == wanted)
        })
}

/// Turn on GPU timestamps when the page asks for `profile=1` and the browser may write them.
pub fn enable_profile_from_page() {
    if !query_flag("profile") {
        return;
    }
    let user_agent = web_sys::window()
        .and_then(|window| window.navigator().user_agent().ok())
        .unwrap_or_default();
    if crate::profile::timestamps_allowed(&user_agent) {
        crate::profile::ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    } else {
        re_log::info!(
            "gsplat profile: GPU timestamps off on Apple GPUs (they fail the browser's Metal command buffer); load and decode lines only"
        );
    }
}
