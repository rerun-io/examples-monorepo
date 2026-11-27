//! Web entry point of the canonical viewer: links the stock `re_viewer` web entry point (`WebHandle`, unchanged JS
//! API) and points the patched viewer's hooks at the canonical extensions before JavaScript constructs `WebHandle`.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::wasm_bindgen;

/// Runs once when the module loads.
#[wasm_bindgen(start)]
pub fn start() {
    canonical_viewer::install_hooks();
}
