//! `TrigsMotion3D`: a TRiGS motion model logged once beside a static `GaussianSplats3D` on the same entity.
//!
//! The splat components hold the canonical rows (stock viewers draw them as they are). When the motion fields
//! are present, the compute visualizer uploads the canonical rows and the motion once and each view evaluates the
//! model at its time cursor in `trigs_deform.wgsl` (one dispatch per time change, no re-upload). `GSPLAT_MOTION=cpu`
//! (and the capacity fallback) evaluates on the CPU instead (`gsplat_core::trigs`), drops rows at or below opacity
//! 1/255 and uploads the result like a per-frame row.
//! Time: `t [s] = (frame - frame_origin) / fps` on a sequence timeline; on a duration or timestamp timeline
//! `t = seconds - frame_origin / fps`.
use gsplat_core::native::{NativeSplats, at_or_last};
use gsplat_core::trigs::TrigsMotion;
use re_viewer::external::{re_log_types, re_sdk_types};

pub(crate) const ARCHETYPE: &str = "TrigsMotion3D";

/// Opacity at or below this rounds to alpha 0 (the per-frame export culls the same rows).
const MIN_OPACITY: f32 = 1.0 / 255.0;

/// Per-row fields, then the three one-value fields.
pub(crate) const FIELDS: [&str; 12] = [
    "opacities",
    "times",
    "log_durations",
    "twist_base",
    "twist_c0",
    "twist_c1",
    "twist_c2",
    "anchors",
    "is_static",
    "fps",
    "frame_origin",
    "active_threshold",
];

pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; 12] {
    FIELDS.map(|field| {
        re_sdk_types::ComponentIdentifier::from_archetype_field(ARCHETYPE.into(), field)
    })
}

/// Clip time in seconds at the view's time cursor.
pub(crate) fn clip_seconds(
    time_type: re_log_types::TimeType,
    cursor: re_log_types::TimeInt,
    fps: f64,
    frame_origin: f64,
) -> f32 {
    let value = cursor.as_i64() as f64;
    let seconds = match time_type {
        re_log_types::TimeType::Sequence => (value - frame_origin) / fps,
        _ => value * 1e-9 - frame_origin / fps,
    };
    seconds as f32
}

/// The visible rows at `t`, as owned native arrays (RGB and SH from the canonical rows, alpha = evaluated opacity).
pub(crate) struct Evaluated {
    pub centers: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub quaternions: Vec<[f32; 4]>,
    pub colors: Vec<u32>,
    /// SH degrees 1-3 of the visible rows; empty when the canonical rows carry none.
    pub sh: Vec<[[half::f16; 3]; 15]>,
}

impl Evaluated {
    pub(crate) fn new(canonical: &NativeSplats<'_>, motion: &TrigsMotion<'_>, t: f32) -> Self {
        re_tracing::profile_function!();
        let (rows, values) = gsplat_core::trigs::deform_visible(
            canonical.centers,
            canonical.quaternions,
            motion,
            t,
            MIN_OPACITY,
        );
        let mut out = Self {
            centers: Vec::with_capacity(rows.len()),
            scales: Vec::with_capacity(rows.len()),
            quaternions: Vec::with_capacity(rows.len()),
            colors: Vec::with_capacity(rows.len()),
            sh: Vec::new(),
        };
        for (&row, value) in rows.iter().zip(&values) {
            let i = row as usize;
            out.centers.push(value.center);
            out.quaternions.push(value.quaternion);
            out.scales.push(at_or_last(canonical.scales, i, [0.01; 3]));
            let rgb = at_or_last(canonical.colors, i, u32::MAX) & 0xFFFF_FF00;
            let alpha = (value.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
            out.colors.push(rgb | alpha);
            if !canonical.sh.is_empty() {
                out.sh
                    .push(at_or_last(canonical.sh, i, [[half::f16::ZERO; 3]; 15]));
            }
        }
        out
    }

    pub(crate) fn native<'a>(&'a self, canonical: &NativeSplats<'a>) -> NativeSplats<'a> {
        NativeSplats {
            centers: &self.centers,
            scales: &self.scales,
            quaternions: &self.quaternions,
            colors: &self.colors,
            sh: &self.sh,
            degree: canonical.degree,
        }
    }
}
