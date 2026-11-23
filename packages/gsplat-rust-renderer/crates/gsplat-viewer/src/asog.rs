//! `AnimatedSogSplats3D`: an animated-sog `.asog` v1 clip logged as its file bytes beside a static
//! `GaussianSplats3D` preview (one mid-clip pose) on the same entity.
//!
//! When the payload is present the compute visualizer draws it instead of the preview rows. Default (GPU) path: the
//! file is entropy-decoded once per payload row on the CPU (zlib streams only), the static part is de-quantised by
//! `sog_dequant.wgsl`, the motion by `asog_motion.wgsl`, and the motion is evaluated per view by `trigs_deform.wgsl`
//! whenever the time cursor moves: no CPU work and no upload on a time change. The CPU decoder in `gsplat_core::asog`
//! is the tests' oracle, not a render path.
//! Web: when the page installs [`BROWSER_INFLATE`] (the browser has `DecompressionStream`), the zlib streams are
//! inflated by the browser, all at once, while the frame loop goes on: the entity draws nothing from our renderer
//! until then; the inflate's continuation runs the undo + assembly (main thread, outside a frame) and asks for the
//! frame that uploads the scene.
//! Time: `t [s] = (frame - frame_origin) / frameRate` with the manifest's frameRate (see `crate::trigs::clip_seconds`).
use re_viewer::external::{
    parking_lot::Mutex, re_log, re_renderer, re_sdk_types, re_viewer_context,
};
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

/// Called with the inflated streams (or the browser's error), from the browser's event loop.
pub type InflateDone = Box<dyn FnOnce(Result<Vec<Vec<u8>>, String>)>;

/// The browser's inflate: zlib members in (in order), [`InflateDone`] called when they are inflated.
pub type BrowserInflate = fn(Vec<Vec<u8>>, InflateDone);

/// Set by the web viewer when the browser has `DecompressionStream`.
///
/// Unset (native, or a browser without it): the payload decodes synchronously in [`Payload::upload`].
pub static BROWSER_INFLATE: std::sync::OnceLock<BrowserInflate> = std::sync::OnceLock::new();

/// Streams inflated outside the frame thread and assembled: what [`Payload::upload`] then uploads.
pub(crate) struct Inflated {
    pub parts: gsplat_core::asog::AsogParts,
    /// Payload seen -> inflated streams back; the undo + assembly in the continuation (main thread, outside a frame);
    /// assembled -> the frame that uploads them.
    pub inflate: std::time::Duration,
    pub assemble: std::time::Duration,
    pub wait: std::time::Duration,
}

/// One entity's `.asog` payload plus the preview rows (bounds, and the native fallback).
pub(crate) struct Payload<'a> {
    pub bytes: &'a [u8],
    pub active_threshold: f32,
    pub info: Info,
    pub preview: &'a gsplat_core::native::NativeSplats<'a>,
    /// Decoded by the browser inflate; `None` = decode here.
    pub inflated: Option<&'a Inflated>,
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
        if let Some(inflated) = self.inflated {
            let start = web_time::Instant::now();
            let scene = core.upload_asog(&ctx.queue, &inflated.parts)?;
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            re_log::info!(
                "ASOG decode: {} splats, SH degree {}, browser inflate {:.1} ms (concurrent, off the frame), undo + \
                 assembly {:.1} ms (main thread, in the inflate's continuation), uploaded {:.1} ms later, upload + static and motion de-quantise GPU {:.1} ms (submit only)",
                re_format::format_uint(scene.len()),
                inflated.parts.sh.as_ref().map_or(0, |s| s.0),
                ms(inflated.inflate),
                ms(inflated.assemble),
                ms(inflated.wait),
                ms(start.elapsed()),
            );
            if crate::profile::requested() {
                re_log::info!("ASOG content hash {:016x}", inflated.parts.content_hash());
            }
            return Ok(scene);
        }
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

/// A browser inflate in flight: the slot its continuation fills with the assembled streams (or the error), and the
/// time it filled it.
type Slot = Arc<Mutex<Option<(Result<Inflated, String>, web_time::Instant)>>>;

enum Load {
    Pending(Slot),
    Ready(Arc<Inflated>),
    /// Uploaded (or failed): later uploads of this payload decode in [`Payload::upload`].
    Done,
}

/// What the visualizer does with a payload this frame.
pub(crate) enum Browser {
    /// No browser inflate (native, or no `DecompressionStream`), or already uploaded: decode in the upload.
    Unused,
    /// The browser is inflating: draw nothing from our renderer yet.
    Pending,
    Ready(Arc<Inflated>),
}

/// Manifest facts and browser inflates by payload generation (row ids + threshold), kept while drawn.
#[derive(Default)]
pub(crate) struct AsogCache {
    frame: u64,
    infos: HashMap<u64, (Result<Info, String>, u64)>,
    loads: HashMap<u64, (Load, u64)>,
}

impl re_viewer_context::Cache for AsogCache {
    fn name(&self) -> &'static str {
        "AnimatedSogSplats3D manifest"
    }

    fn begin_frame(&mut self) {
        self.infos.retain(|_, (_, seen)| *seen + 1 >= self.frame);
        self.loads.retain(|_, (_, seen)| *seen + 1 >= self.frame);
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

    /// The browser inflate of `bytes`: started on the first call; its continuation assembles the streams and calls
    /// `repaint`; later calls take the result. An inflate error falls back to [`Browser::Unused`], so the upload's own
    /// decode reports it as without the browser inflate.
    pub(crate) fn browser(
        &mut self,
        generation: u64,
        bytes: &[u8],
        active_threshold: f32,
        repaint: impl Fn() + 'static,
    ) -> Browser {
        let Some(inflate) = BROWSER_INFLATE.get() else {
            return Browser::Unused;
        };
        let frame = self.frame;
        let (load, seen) = self.loads.entry(generation).or_insert_with(|| {
            let Ok(members) = gsplat_core::asog::zlib_members(bytes) else {
                return (Load::Done, frame);
            };
            let slot = Slot::default();
            let filled = slot.clone();
            let started = web_time::Instant::now();
            // The continuation owns a copy of the file: it assembles the streams itself and asks for the frame that
            // uploads them, so nothing waits for a frame to come by.
            let file = bytes.to_vec();
            inflate(
                members.iter().map(|m| m.data.to_vec()).collect(),
                Box::new(move |result| {
                    let inflate = started.elapsed();
                    let start = web_time::Instant::now();
                    let assembled = result.and_then(|inflated| {
                        gsplat_core::asog::decode_parts_inflated(&file, active_threshold, inflated)
                            .map_err(|e| e.to_string())
                    });
                    let assembled = assembled.map(|parts| Inflated {
                        parts,
                        inflate,
                        assemble: start.elapsed(),
                        wait: std::time::Duration::ZERO,
                    });
                    *filled.lock() = Some((assembled, web_time::Instant::now()));
                    repaint();
                }),
            );
            (Load::Pending(slot), frame)
        });
        *seen = frame;
        if let Load::Pending(slot) = load {
            let Some((assembled, back)) = slot.lock().take() else {
                return Browser::Pending;
            };
            *load = match assembled {
                Ok(inflated) => Load::Ready(Arc::new(Inflated {
                    wait: back.elapsed(),
                    ..inflated
                })),
                Err(error) => {
                    re_log::warn!("ASOG browser inflate failed ({error}); decoding in wasm");
                    Load::Done
                }
            };
        }
        match load {
            Load::Ready(inflated) => Browser::Ready(inflated.clone()),
            Load::Pending(..) | Load::Done => Browser::Unused,
        }
    }

    /// The payload's scene is uploaded: drop the assembled streams.
    pub(crate) fn uploaded(&mut self, generation: u64) {
        if let Some((load, _)) = self.loads.get_mut(&generation) {
            *load = Load::Done;
        }
    }
}
