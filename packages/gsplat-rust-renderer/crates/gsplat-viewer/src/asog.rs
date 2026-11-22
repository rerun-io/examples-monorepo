//! `AnimatedSogSplats3D`: an animated-sog `.asog` v1 clip logged as its file bytes beside a static
//! `GaussianSplats3D` preview (one mid-clip pose) on the same entity.
//!
//! When the payload is present the compute visualizer draws it instead of the preview rows. Default (GPU) path: the
//! file is entropy-decoded once per payload row on the CPU (zlib streams only), the static part is de-quantised by
//! `sog_dequant.wgsl`, the motion by `asog_motion.wgsl`, and the motion is evaluated per view by `trigs_deform.wgsl`
//! whenever the time cursor moves: no CPU work and no upload on a time change. `GSPLAT_MOTION=cpu` selects the CPU
//! oracle path (decode + `AsogMotion::evaluate_into` per time change + a scene upload), for A/B pixels.
//! Time: `t [s] = (frame - frame_origin) / frameRate` with the manifest's frameRate (see `crate::trigs::clip_seconds`).
use gsplat_core::Splats;
use gsplat_core::asog::Asog;
use re_viewer::external::{re_log, re_log_types, re_renderer, re_sdk_types, re_viewer_context};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const ARCHETYPE: &str = "AnimatedSogSplats3D";

pub(crate) const FIELDS: [&str; 3] = ["asog", "frame_origin", "active_threshold"];

pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; 3] {
    FIELDS.map(|field| {
        re_sdk_types::ComponentIdentifier::from_archetype_field(ARCHETYPE.into(), field)
    })
}

/// Where the motion is evaluated; the GPU pass is the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MotionPath {
    Gpu,
    Cpu,
}

impl MotionPath {
    /// `GSPLAT_MOTION=cpu` selects the CPU oracle path (native only; the web build has no environment).
    pub(crate) fn from_env() -> Self {
        static PATH: std::sync::OnceLock<MotionPath> = std::sync::OnceLock::new();
        *PATH.get_or_init(|| match std::env::var("GSPLAT_MOTION").as_deref() {
            Ok("cpu") => Self::Cpu,
            _ => Self::Gpu,
        })
    }
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
    /// `asog_motion.wgsl` (or both on the CPU with `GSPLAT_SOG_DECODE=cpu`); the motion is then evaluated per view.
    /// Logs the load-time split.
    pub(crate) fn upload(
        &self,
        ctx: &re_renderer::RenderContext,
        core: &gsplat_core::Renderer,
        path: crate::sog::Dequantize,
    ) -> Result<Arc<gsplat_core::Scene>, gsplat_core::Error> {
        re_tracing::profile_function!();
        let start = web_time::Instant::now();
        let parts = gsplat_core::asog::decode_parts(self.bytes, self.active_threshold)?;
        let entropy = start.elapsed();
        let start = web_time::Instant::now();
        let (scene, cpu) = match path {
            crate::sog::Dequantize::Gpu => (core.upload_asog(&ctx.queue, &parts)?, None),
            crate::sog::Dequantize::Cpu => {
                let splats = gsplat_core::sog::dequantize(&parts.quantization, &parts.planes())?;
                let motion = parts.motion();
                let cpu = start.elapsed();
                (
                    core.upload_animated(&splats, gsplat_core::Motion::Asog(&motion))?,
                    Some(cpu),
                )
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        let total = start.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let split = cpu.map_or_else(
            || {
                format!(
                    "upload + static and motion de-quantise GPU {:.1} ms",
                    ms(total)
                )
            },
            |cpu| {
                format!(
                    "static and motion de-quantise CPU {:.1} ms, upload {:.1} ms",
                    ms(cpu),
                    ms(total.saturating_sub(cpu))
                )
            },
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

struct Entry {
    decoded: Result<Arc<Asog>, String>,
    /// The last evaluation: clip time bits and the splats at that time.
    evaluated: Option<(u32, Arc<Splats>)>,
    last_frame: u64,
}

/// Decoded payloads by payload generation (row ids + threshold), kept while drawn.
#[derive(Default)]
pub(crate) struct AsogCache {
    frame: u64,
    entries: HashMap<u64, Entry>,
    /// Manifest facts per payload generation (GPU path; nothing decoded).
    infos: HashMap<u64, (Result<Info, String>, u64)>,
}

impl re_viewer_context::Cache for AsogCache {
    fn name(&self) -> &'static str {
        "AnimatedSogSplats3D decode"
    }

    fn begin_frame(&mut self) {
        self.entries
            .retain(|_, entry| entry.last_frame + 1 >= self.frame);
        self.infos.retain(|_, (_, seen)| *seen + 1 >= self.frame);
        self.frame += 1;
    }

    fn purge_memory(&mut self) {
        *self = Self::default();
    }
}

impl re_byte_size::MemUsageTreeCapture for AsogCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        let splat_bytes =
            |s: &Splats| (s.transforms.len() * 44 + s.sh_coefficients.len() * 12) as u64;
        let bytes = self
            .entries
            .values()
            .map(|e| {
                e.decoded
                    .as_ref()
                    .map_or(0, |a| 2 * splat_bytes(&a.canonical))
                    + e.evaluated.as_ref().map_or(0, |(_, s)| splat_bytes(s))
            })
            .sum::<u64>();
        re_byte_size::MemUsageNode::new()
            .with_child("Decoded .asog and evaluated splats", bytes)
            .into_tree()
    }
}

/// The payload's splats at the view's time cursor and the clip time they were evaluated at.
pub(crate) struct Evaluated {
    pub splats: Arc<Splats>,
    pub t: f32,
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
                        max_degree: m.sh_n.as_ref().map_or(0, |sh| match sh.coeffs {
                            0..=3 => 1,
                            4..=8 => 2,
                            9..=15 => 3,
                            _ => 4,
                        }),
                    }
                })
                .map_err(|e| e.to_string());
            (info, frame)
        });
        *seen = frame;
        info.clone()
    }

    /// Decode `bytes` once per `generation`; evaluate again only when the clip time changes.
    pub(crate) fn evaluate(
        &mut self,
        generation: u64,
        bytes: &[u8],
        active_threshold: f32,
        time_type: re_log_types::TimeType,
        cursor: re_log_types::TimeInt,
        frame_origin: f64,
    ) -> Result<Evaluated, String> {
        let frame = self.frame;
        let entry = self.entries.entry(generation).or_insert_with(|| {
            re_tracing::profile_scope!("asog decode");
            let decoded = gsplat_core::asog::decode(bytes, active_threshold)
                .map(Arc::new)
                .map_err(|e| e.to_string());
            if let Ok(asog) = &decoded {
                re_log::debug!(
                    "Decoded .asog: {} splats, SH degree {}",
                    re_format::format_uint(asog.motion.len()),
                    asog.canonical.sh_degree
                );
            }
            Entry {
                decoded,
                evaluated: None,
                last_frame: frame,
            }
        });
        entry.last_frame = frame;
        let asog = entry.decoded.clone()?;
        let t = crate::trigs::clip_seconds(
            time_type,
            cursor,
            asog.manifest.timeline.frame_rate,
            frame_origin,
        );
        if let Some((bits, splats)) = &entry.evaluated
            && *bits == t.to_bits()
        {
            return Ok(Evaluated {
                splats: splats.clone(),
                t,
            });
        }
        re_tracing::profile_scope!("asog evaluate");
        // Reuse the previous buffers (and its copied SH) when nobody else holds them.
        let mut splats = entry
            .evaluated
            .take()
            .and_then(|(_, s)| Arc::try_unwrap(s).ok())
            .unwrap_or_default();
        asog.motion.evaluate_into(&asog.canonical, t, &mut splats);
        let splats = Arc::new(splats);
        entry.evaluated = Some((t.to_bits(), splats.clone()));
        Ok(Evaluated { splats, t })
    }
}
