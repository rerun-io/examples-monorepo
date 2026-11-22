//! Web entry point of the custom gsplat viewer.
//!
//! Links the stock `re_viewer` web entry point (`WebHandle`, unchanged JS API) and, through the
//! `re_viewer::web_hooks` added by our patch, requests the WebGPU subgroup feature and registers
//! the compute splat renderer and visualizers on the new app.
use wasm_bindgen::prelude::wasm_bindgen;

/// Runs once when the module loads, before JavaScript constructs `WebHandle`.
#[wasm_bindgen(start)]
pub fn install_gsplat_hooks() {
    re_viewer::web_hooks::WGPU_OPTIONS
        .set(|options| {
            if let re_viewer::external::eframe::egui_wgpu::WgpuSetup::CreateNew(setup) =
                &mut options.wgpu_setup
            {
                let compute = gsplat_viewer::setup::compute_device_descriptor();
                setup.device_descriptor = std::sync::Arc::new(move |adapter| {
                    let descriptor = compute(adapter);
                    let renderer = if descriptor.required_features.contains(wgpu::Features::SUBGROUP) {
                        "gsplat compute (subgroups on)"
                    } else {
                        "stock Rerun splats (adapter lacks subgroups or compute limits)"
                    };
                    re_log::info!(
                        "gsplat web viewer: adapter {:?} backend {:?}, subgroups offered: {}, active splat renderer: {renderer}",
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
        .set(|app| match gsplat_viewer::setup::register_compute_splats(app) {
            Ok(()) => re_log::info!("gsplat web viewer: compute splat renderer registered"),
            Err(err) => re_log::error!("gsplat web viewer: compute splat registration failed: {err}"),
        })
        .ok();
}
