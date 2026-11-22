//! `AnimatedSogSplats3D`: an animated-sog `.asog` v1 clip logged as its file bytes beside a static
//! `GaussianSplats3D` preview (one mid-clip pose) on the same entity.
//!
//! When the payload is present the compute visualizer draws it instead of the preview rows: the file is decoded once
//! per payload row (`gsplat_core::asog`, CPU), the motion is evaluated on the CPU whenever the clip time changes
//! (`gsplat_core::trigs::AsogMotion`, f32, straight into gsplat-core `Splats`: no 8-bit colour or opacity) and the
//! result goes through the ordinary scene upload. Time: `t [s] = (frame - frame_origin) / frameRate` with the
//! manifest's frameRate (see `crate::trigs::clip_seconds`).
use gsplat_core::Splats;
use gsplat_core::asog::Asog;
use re_viewer::external::{re_log, re_log_types, re_sdk_types, re_viewer_context};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const ARCHETYPE: &str = "AnimatedSogSplats3D";

pub(crate) const FIELDS: [&str; 3] = ["asog", "frame_origin", "active_threshold"];

pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; 3] {
    FIELDS.map(|field| {
        re_sdk_types::ComponentIdentifier::from_archetype_field(ARCHETYPE.into(), field)
    })
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
}

impl re_viewer_context::Cache for AsogCache {
    fn name(&self) -> &'static str {
        "AnimatedSogSplats3D decode"
    }

    fn begin_frame(&mut self) {
        self.entries
            .retain(|_, entry| entry.last_frame + 1 >= self.frame);
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
