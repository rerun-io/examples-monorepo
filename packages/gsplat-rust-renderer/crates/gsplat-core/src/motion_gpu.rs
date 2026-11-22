//! Motion on the GPU (`trigs_deform.wgsl`): one compute dispatch per time change writes a view's transforms and raw
//! opacity from the scene's canonical rows and per-row motion, before the projector reads them.
//!
//! Two forms behind one kernel flag, with the CPU references in [`crate::trigs`]:
//! - [`Motion::Twist`]: `TrigsMotion3D` rows (twist base + Bezier residual, `TrigsMotion::deform_row`);
//! - [`Motion::Asog`]: the animated-sog O/g form with its clip table (`AsogMotion::row`).
//!
//! The scene keeps the canonical transforms, SH and the packed motion (uploaded once). Each view owns its evaluated
//! transforms + raw opacity, so views at different times do not race. Rows with zero weight get raw opacity -100.
use crate::Error;
use crate::asog::{AsogParts, MOTION_CODES};
use crate::gpu::{Kernel, bind, storage, uniform};
use crate::trigs::{AsogMotion, ClipWindow, TrigsMotion};

/// Per-row motion for [`crate::Renderer::upload_animated`].
#[derive(Clone, Copy)]
pub enum Motion<'a> {
    Twist(&'a TrigsMotion<'a>),
    Asog(&'a AsogMotion),
}

/// f32 per row of the twist form: opacity, mu, log sigma, static (> 0.5), base (6), c0 (6), c1 (6), c2 (6), anchor (3).
pub(crate) const TWIST_STRIDE: usize = 31;
/// f32 per row of the asog form: opacity, mu, sigma, flags (u32 bits: static | clip << 1), anchor (3), O0..O2 (9),
/// G0..G4 (5).
pub(crate) const ASOG_STRIDE: usize = 21;
const CLIP_STRIDE: usize = 8;
const WG: u32 = 256;

const TWIST: u32 = 0;
const ASOG: u32 = 1;

/// The packed motion of a scene (immutable, shared by its views).
pub(crate) struct GpuMotion {
    form: u32,
    pub(crate) params: wgpu::Buffer,
    clips: wgpu::Buffer,
    clip_count: u32,
    threshold: f32,
}

impl GpuMotion {
    pub(crate) fn gpu_byte_size(&self) -> u64 {
        self.params.size() + self.clips.size()
    }
}

impl Motion<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Twist(m) => m.len(),
            Self::Asog(m) => m.len(),
        }
    }

    /// Host packing of the per-row motion and the clip table.
    pub(crate) fn pack(&self) -> (u32, Vec<f32>, Vec<f32>, f32) {
        match self {
            Self::Twist(m) => {
                let mut out = Vec::with_capacity(m.len() * TWIST_STRIDE);
                for i in 0..m.len() {
                    out.extend([
                        m.opacities[i],
                        m.times[i],
                        m.log_durations[i],
                        m.is_static.get(i).copied().unwrap_or(0.0),
                    ]);
                    for twist in [m.twist_base, m.twist_c0, m.twist_c1, m.twist_c2] {
                        out.extend_from_slice(&twist[i]);
                    }
                    out.extend_from_slice(&m.anchors[i]);
                }
                (TWIST, out, vec![0.0; CLIP_STRIDE], m.active_threshold)
            }
            Self::Asog(m) => {
                let mut out = Vec::with_capacity(m.len() * ASOG_STRIDE);
                for i in 0..m.len() {
                    let flags = u32::from(m.is_static[i]) | (m.clip[i] << 1);
                    out.extend([
                        m.opacities[i],
                        m.times[i],
                        m.sigmas[i],
                        f32::from_bits(flags),
                    ]);
                    out.extend_from_slice(&m.anchors[i]);
                    for o in &m.omega[i] {
                        out.extend_from_slice(o);
                    }
                    out.extend_from_slice(&m.g[i]);
                }
                (ASOG, out, pack_clips(&m.clips), m.active_threshold)
            }
        }
    }

    /// Upload the packed motion for a scene of `n` rows.
    pub(crate) fn upload(
        &self,
        device: &wgpu::Device,
        n: usize,
        limit: u64,
    ) -> Result<GpuMotion, Error> {
        if self.len() != n {
            return Err(Error::Input("motion rows differ from the splat rows"));
        }
        if let Self::Twist(m) = self
            && !m.is_consistent(n)
        {
            return Err(Error::Input("inconsistent TrigsMotion3D row lengths"));
        }
        if let Self::Asog(m) = self
            && (m.clip.iter().any(|&c| c as usize >= m.clips.len())
                || [
                    m.opacities.len(),
                    m.sigmas.len(),
                    m.anchors.len(),
                    m.omega.len(),
                    m.g.len(),
                    m.is_static.len(),
                    m.clip.len(),
                ]
                .iter()
                .any(|&len| len != n))
        {
            return Err(Error::Input(
                "inconsistent asog motion rows or clip indices",
            ));
        }
        let (form, params, clips, threshold) = self.pack();
        let upload = |label, values: &[f32]| {
            crate::scene::upload(device, label, bytemuck::cast_slice(values), limit)
        };
        Ok(GpuMotion {
            form,
            params: upload("motion parameters", &params)?,
            clips: upload("motion clips", &clips)?,
            clip_count: u32::try_from(clips.len() / CLIP_STRIDE)?,
            threshold: threshold.clamp(0.0, 0.999),
        })
    }
}

/// The clip table as `CLIP_STRIDE` f32 per clip.
fn pack_clips(clips: &[ClipWindow]) -> Vec<f32> {
    clips
        .iter()
        .flat_map(|c| {
            [
                c.offset,
                c.active_start,
                c.active_end,
                c.core_start,
                c.core_end,
                c.fade_in,
                c.fade_out,
                0.0,
            ]
        })
        .collect()
}

/// De-quantise the `.asog` motion integers on the GPU (`asog_motion.wgsl`, one pass submitted on `queue`) into the
/// packed asog rows of a scene whose canonical transforms `canonical` already holds (written by `sog_dequant.wgsl`).
pub(crate) fn dequantize_asog(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &Kernel,
    parts: &AsogParts,
    canonical: &wgpu::Buffer,
    limit: u64,
) -> Result<GpuMotion, Error> {
    let m = &parts.manifest;
    let n = m.count;
    let count = u32::try_from(n)?;
    if parts.motion_codes.len() != n * MOTION_CODES || m.clips.is_empty() {
        return Err(Error::Input("asog motion codes differ from the splat rows"));
    }
    let row_bytes = (n * ASOG_STRIDE * 4) as u64;
    if row_bytes > limit {
        return Err(Error::Capacity {
            required: row_bytes,
            limit,
        });
    }
    let codes = crate::scene::upload(
        device,
        "asog motion codes",
        bytemuck::cast_slice(&parts.motion_codes),
        limit,
    )?;
    let rows = storage(device, "motion parameters", row_bytes);
    let groups = count.div_ceil(WG).max(1);
    let x = groups.min(65535);
    let (mo, ls) = (&m.motion, &m.logsigma);
    let f = |v: f64| (v as f32).to_bits();
    let words = [
        count,
        x * WG,
        u32::try_from(mo.mref_samples.max(1))?,
        0,
        f(ls.min),
        f(ls.max - ls.min),
        f(f64::from((1u32 << ls.bits.min(31)) - 1)),
        f(mo.qa / 1000.0),
        f(mo.qp / 1000.0),
        f(mo.r_min),
        f(mo.mid_o),
        f(mo.mid_g),
        f(mo.sm),
        f(m.mu.scale),
        0,
        0,
    ];
    let params = uniform(device, &words);
    let group = bind(
        device,
        &kernel.layout,
        &[
            (0, params.as_entire_binding()),
            (1, codes.as_entire_binding()),
            (2, canonical.as_entire_binding()),
            (3, rows.as_entire_binding()),
        ],
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ASOG motion de-quantisation"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("asog_motion_dequant"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
    }
    queue.submit([encoder.finish()]);
    let clips = pack_clips(&m.clip_windows());
    Ok(GpuMotion {
        form: ASOG,
        params: rows,
        clips: crate::scene::upload(device, "motion clips", bytemuck::cast_slice(&clips), limit)?,
        clip_count: u32::try_from(clips.len() / CLIP_STRIDE)?,
        threshold: parts.active_threshold.clamp(0.0, 0.999),
    })
}

/// A view's evaluated transforms + raw opacity and the per-frame-slot uniforms of the motion pass.
pub(crate) struct Deform {
    pub transforms: wgpu::Buffer,
    pub opacity: wgpu::Buffer,
    slots: [(wgpu::Buffer, wgpu::BindGroup); 3],
    params: [u32; 8],
    /// Workgroups in x and y (x <= 65,535).
    groups: [u32; 2],
    /// Time (bits) of the last encoded pass; another encode at the same time dispatches nothing.
    last: Option<u32>,
}

impl Deform {
    pub(crate) fn new(
        device: &wgpu::Device,
        kernel: &Kernel,
        motion: &GpuMotion,
        canonical: &wgpu::Buffer,
        n: u32,
        limit: u64,
    ) -> Result<Self, Error> {
        let bytes = u64::from(n) * 40;
        if bytes > limit {
            return Err(Error::Capacity {
                required: bytes,
                limit,
            });
        }
        let transforms = storage(device, "evaluated transforms", bytes);
        let opacity = storage(device, "evaluated raw opacity", u64::from(n) * 4);
        let groups = n.div_ceil(WG).max(1);
        let x = groups.min(65535);
        let params = [
            n,
            motion.form,
            x * WG,
            motion.clip_count,
            0,
            motion.threshold.to_bits(),
            0,
            0,
        ];
        let slots = std::array::from_fn(|_| {
            let buffer = uniform(device, &params);
            let group = bind(
                device,
                &kernel.layout,
                &[
                    (0, buffer.as_entire_binding()),
                    (1, canonical.as_entire_binding()),
                    (2, motion.params.as_entire_binding()),
                    (3, motion.clips.as_entire_binding()),
                    (4, transforms.as_entire_binding()),
                    (5, opacity.as_entire_binding()),
                ],
            );
            (buffer, group)
        });
        Ok(Self {
            transforms,
            opacity,
            slots,
            params,
            groups: [x, groups.div_ceil(x)],
            last: None,
        })
    }

    pub(crate) fn gpu_byte_size(&self) -> u64 {
        self.transforms.size() + self.opacity.size()
    }

    /// Encode the motion pass for `time` (seconds) in frame slot `slot`, unless the last pass had the same time.
    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        kernel: &Kernel,
        slot: usize,
        time: f32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if self.last == Some(time.to_bits()) {
            return;
        }
        self.last = Some(time.to_bits());
        let (buffer, group) = &self.slots[slot];
        let mut params = self.params;
        params[4] = time.to_bits();
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("trigs_deform"),
            timestamp_writes,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
    }
}

#[cfg(test)]
mod tests;
