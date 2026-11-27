//! The contract between the canonical Rerun viewer and one extension crate.
//!
//! An extension crate exports one `pub const EXTENSION: Extension` and has no `main` of its own; the canonical viewer
//! lists it in `EXTENSIONS` and builds it into the native, headless and web viewers.

pub use re_viewer::external::re_renderer::external::wgpu;

/// What one extension adds to the viewer.
pub struct Extension {
    /// Names the extension in logs and tests.
    pub name: &'static str,

    /// Asks for the wgpu features and limits the extension uses, out of what the adapter offers.
    ///
    /// Receives the adapter's features and limits and the descriptor built so far (Rerun's, then each earlier
    /// extension's). It may only add features and raise limits; the viewer drops a change that removes something
    /// already requested or asks for more than the adapter offers.
    pub device_needs:
        Option<fn(&wgpu::Features, &wgpu::Limits, &mut wgpu::DeviceDescriptor<'static>)>,

    /// Registers the extension's renderers, view-class extensions, fallbacks and loaders on a new viewer.
    pub register: fn(&mut re_viewer::App) -> anyhow::Result<()>,
}
