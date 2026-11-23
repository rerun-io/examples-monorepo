//! `AnimatedSogSplats3D`: an animated-sog `.asog` v1 clip logged as its file bytes beside a static
//! `GaussianSplats3D` preview (one mid-clip pose) on the same entity.
//!
//! When the payload is present the compute visualizer draws it instead of the preview rows. Default (GPU) path: the
//! file is entropy-decoded once per payload row on the CPU (zlib streams only), the static part is de-quantised by
//! `sog_dequant.wgsl`, the motion by `asog_motion.wgsl`, and the motion is evaluated per view by `trigs_deform.wgsl`
//! whenever the time cursor moves: no CPU work and no upload on a time change. The CPU decoder in `gsplat_core::asog`
//! is the tests' oracle, not a render path.
//! Time: `t [s] = (frame - frame_origin) / frameRate` with the manifest's frameRate (see `crate::trigs::clip_seconds`).
use re_viewer::external::{re_log, re_renderer, re_sdk_types, re_viewer_context};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const ARCHETYPE: &str = "AnimatedSogSplats3D";

pub(crate) const FIELDS: [&str; 3] = ["asog", "frame_origin", "active_threshold"];

pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; 3] {
    FIELDS.map(|field| {
        re_sdk_types::ComponentIdentifier::from_archetype_field(ARCHETYPE.into(), field)
    })
}

/// What the visualizer needs before the payload is decoded: the clip's frame rate and the capacity inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Info {
    pub fps: f64,
    pub count: usize,
    pub max_degree: u32,
}

/// One entity's `.asog` payload plus the preview rows (bounds, and the native fallback).
pub(crate) struct Payload<'a> {
    pub bytes: &'a [u8],
    pub active_threshold: f32,
    pub info: Info,
    pub preview: &'a gsplat_core::native::NativeSplats<'a>,
}

impl Payload<'_> {
    /// Entropy decode (zlib) on the CPU; static and motion de-quantisation in `sog_dequant.wgsl` +
    /// `asog_motion.wgsl`; the motion is then evaluated per view. Logs the load-time split.
    pub(crate) fn upload(
        &self,
        ctx: &re_renderer::RenderContext,
        core: &gsplat_core::Renderer,
    ) -> Result<Arc<gsplat_core::Scene>, gsplat_core::Error> {
        re_tracing::profile_function!();
        let start = web_time::Instant::now();
        let parts = gsplat_core::asog::decode_parts(self.bytes, self.active_threshold)?;
        let entropy = start.elapsed();
        let start = web_time::Instant::now();
        let scene = core.upload_asog(&ctx.queue, &parts)?;
        #[cfg(not(target_arch = "wasm32"))]
        ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let split = format!(
            "upload + static and motion de-quantise GPU {:.1} ms",
            ms(start.elapsed())
        );
        re_log::info!(
            "ASOG decode: {} splats, SH degree {}, zlib CPU {:.1} ms, {split}; motion pass on the GPU",
            re_format::format_uint(scene.len()),
            parts.sh.as_ref().map_or(0, |s| s.0),
            ms(entropy),
        );
        Ok(scene)
    }
}

/// Manifest facts by payload generation (row ids + threshold), kept while drawn; nothing is decoded here.
#[derive(Default)]
pub(crate) struct AsogCache {
    frame: u64,
    infos: HashMap<u64, (Result<Info, String>, u64)>,
}

impl re_viewer_context::Cache for AsogCache {
    fn name(&self) -> &'static str {
        "AnimatedSogSplats3D manifest"
    }

    fn begin_frame(&mut self) {
        self.infos.retain(|_, (_, seen)| *seen + 1 >= self.frame);
        self.frame += 1;
    }

    fn purge_memory(&mut self) {
        *self = Self::default();
    }
}

impl re_byte_size::MemUsageTreeCapture for AsogCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        // A few numbers per payload.
        re_byte_size::MemUsageNode::new().into_tree()
    }
}

impl AsogCache {
    /// Header + manifest only, once per `generation`.
    pub(crate) fn info(&mut self, generation: u64, bytes: &[u8]) -> Result<Info, String> {
        let frame = self.frame;
        let (info, seen) = self.infos.entry(generation).or_insert_with(|| {
            let info = gsplat_core::asog::parse(bytes)
                .map(|c| {
                    let m = &c.manifest;
                    Info {
                        fps: m.timeline.frame_rate,
                        count: m.count,
                        // An upper bound: a coefficient count of no degree fails in the decode.
                        max_degree: m.sh_n.as_ref().map_or(0, |sh| sh.bands().unwrap_or(4)),
                    }
                })
                .map_err(|e| e.to_string());
            (info, frame)
        });
        *seen = frame;
        info.clone()
    }
}
