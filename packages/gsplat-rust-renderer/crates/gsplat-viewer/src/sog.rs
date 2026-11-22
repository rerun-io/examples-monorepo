//! `SogSplats3D`: a `PlayCanvas` SOG (v2 / v3) logged once as its stored WebP planes + meta.json, on the same entity
//! as a small static `GaussianSplats3D` preview (what stock viewers draw).
//!
//! The compute visualizer draws the payload instead of the preview rows. It decodes once, when the payload's row ids
//! first appear (the scene cache's vacant branch): WebP (L1) on the CPU, then de-quantisation (L2) in
//! `sog_dequant.wgsl`, or on the CPU with `GSPLAT_SOG_DECODE=cpu` (A/B pixels, and the path without the GPU kernel).
//! The preview rows draw only when the payload is unusable (bad meta.json, or the native capacity fallback).
use gsplat_core::native::NativeSplats;
use gsplat_core::sog::{Meta, SogWebp};
use re_viewer::external::{re_log, re_renderer, re_sdk_types};
use std::sync::Arc;

pub(crate) const ARCHETYPE: &str = "SogSplats3D";

/// meta.json, then the WebP planes by role (the last two are optional).
pub(crate) const FIELDS: [&str; 8] = [
    "meta",
    "means_l",
    "means_u",
    "quats",
    "scales",
    "sh0",
    "shN_centroids",
    "shN_labels",
];

pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; 8] {
    FIELDS.map(|field| {
        re_sdk_types::ComponentIdentifier::from_archetype_field(ARCHETYPE.into(), field)
    })
}

/// Where L2 runs; the GPU kernel is the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dequantize {
    Gpu,
    Cpu,
}

impl Dequantize {
    /// `GSPLAT_SOG_DECODE=cpu` selects the CPU decoder (native only; the web build has no environment).
    pub(crate) fn from_env() -> Self {
        static PATH: std::sync::OnceLock<Dequantize> = std::sync::OnceLock::new();
        *PATH.get_or_init(|| match std::env::var("GSPLAT_SOG_DECODE").as_deref() {
            Ok("cpu") => Self::Cpu,
            _ => Self::Gpu,
        })
    }
}

/// One entity's SOG payload plus the preview rows it replaces.
pub(crate) struct Payload<'a> {
    pub meta: &'a Meta,
    pub webp: SogWebp<'a>,
    /// The same entity's `GaussianSplats3D` rows: bounds, and the native fallback.
    pub preview: &'a NativeSplats<'a>,
}

impl Payload<'_> {
    pub(crate) fn count(&self) -> usize {
        self.meta.count
    }

    /// An upper bound of the SH degree before the planes are decoded (old v2 files omit `shN.bands`: v2 has <= 3).
    pub(crate) fn max_degree(&self) -> u32 {
        match (&self.meta.sh_n, self.webp.sh_centroids, self.webp.sh_labels) {
            (Some(sh), Some(_), Some(_)) => sh.bands.unwrap_or(match self.meta.version {
                3 => 4,
                _ => 3,
            }),
            _ => 0,
        }
    }

    /// L1 on the CPU, L2 on the GPU (default) or the CPU, then the scene. Logs the load-time split.
    pub(crate) fn upload(
        &self,
        ctx: &re_renderer::RenderContext,
        core: &gsplat_core::Renderer,
        path: Dequantize,
    ) -> Result<Arc<gsplat_core::Scene>, gsplat_core::Error> {
        re_tracing::profile_function!();
        let start = web_time::Instant::now();
        let decoded = gsplat_core::sog::decode_planes(self.meta, &self.webp)?;
        let webp = start.elapsed();
        let start = web_time::Instant::now();
        let (scene, dequantize) = match path {
            Dequantize::Gpu => {
                let scene =
                    core.upload_sog(&ctx.queue, &decoded.quantization, &decoded.planes())?;
                (scene, None)
            }
            Dequantize::Cpu => {
                let splats =
                    gsplat_core::sog::dequantize(&decoded.quantization, &decoded.planes())?;
                let dequantize = start.elapsed();
                (core.upload(&splats)?, Some(dequantize))
            }
        };
        // Time the GPU work too (one wait at load; the web build cannot block).
        #[cfg(not(target_arch = "wasm32"))]
        ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        let total = start.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let split = dequantize.map_or_else(
            || format!("upload + de-quantise GPU {:.1} ms", ms(total)),
            |cpu| {
                format!(
                    "de-quantise CPU {:.1} ms, upload {:.1} ms",
                    ms(cpu),
                    ms(total.saturating_sub(cpu))
                )
            },
        );
        re_log::info!(
            "SOG decode: {} splats, SH degree {}, WebP {:.1} ms, {split}",
            re_format::format_uint(scene.len()),
            decoded.sh.as_ref().map_or(0, |s| s.bands),
            ms(webp),
        );
        Ok(scene)
    }
}
