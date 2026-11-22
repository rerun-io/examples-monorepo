//! Web entry point of the custom gsplat viewer.
//!
//! Links the stock `re_viewer` web entry point (`WebHandle`, unchanged JS API) and, through the
//! `re_viewer::web_hooks` added by our patch, requests the WebGPU subgroup feature (when offered and
//! the page URL has no `force_no_subgroups=1`) and registers the compute splat renderer and
//! visualizers on the new app. Without subgroups the splat kernels use their workgroup-memory variant.
use wasm_bindgen::prelude::wasm_bindgen;

/// Runs once when the module loads, before JavaScript constructs `WebHandle`.
#[wasm_bindgen(start)]
pub fn install_gsplat_hooks() {
    re_viewer::web_hooks::WGPU_OPTIONS
        .set(|options| {
            if let re_viewer::external::eframe::egui_wgpu::WgpuSetup::CreateNew(setup) =
                &mut options.wgpu_setup
            {
                let force_no_subgroups = force_no_subgroups();
                let compute =
                    gsplat_viewer::setup::compute_device_descriptor_with(!force_no_subgroups);
                setup.device_descriptor = std::sync::Arc::new(move |adapter| {
                    let descriptor = compute(adapter);
                    let renderer = if descriptor.required_features.contains(wgpu::Features::SUBGROUP) {
                        "gsplat compute (subgroups on)"
                    } else if gsplat_core::check_adapter(adapter.features(), &adapter.limits()).is_ok() {
                        "gsplat compute (no subgroups)"
                    } else {
                        "stock Rerun splats (adapter lacks the compute limits)"
                    };
                    re_log::info!(
                        "gsplat web viewer: adapter {:?} backend {:?}, subgroups offered: {}, force_no_subgroups: {force_no_subgroups}, active splat renderer: {renderer}",
                        adapter.get_info().name,
                        adapter.get_info().backend,
                        adapter.features().contains(wgpu::Features::SUBGROUP),
                    );
                    descriptor
                });
            }
        })
        .ok();
    re_viewer::web_hooks::SETUP_APP
        .set(
            |app| match gsplat_viewer::setup::register_compute_splats(app) {
                Ok(()) => re_log::info!("gsplat web viewer: compute splat renderer registered"),
                Err(err) => {
                    re_log::error!("gsplat web viewer: compute splat registration failed: {err}")
                }
            },
        )
        .ok();
}

/// `force_no_subgroups=1` in the page's query string: behave as if the adapter lacked "subgroups".
fn force_no_subgroups() -> bool {
    web_sys::window()
        .and_then(|window| window.location().search().ok())
        .is_some_and(|search| {
            search
                .trim_start_matches('?')
                .split('&')
                .any(|pair| pair == "force_no_subgroups=1")
        })
}
