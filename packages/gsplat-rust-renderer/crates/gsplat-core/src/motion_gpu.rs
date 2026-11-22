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
use crate::gpu::{Kernel, bind, storage, uniform};
use crate::trigs::{AsogMotion, TrigsMotion};

/// Per-row motion for [`crate::Renderer::upload_animated`] and [`crate::Renderer::upload_sog`].
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
    params: wgpu::Buffer,
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
                let clips = m
                    .clips
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
                    .collect();
                (ASOG, out, clips, m.active_threshold)
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
            let bytes: &[u8] = bytemuck::cast_slice(values);
            if bytes.len() as u64 > limit {
                return Err(Error::Capacity {
                    required: bytes.len() as u64,
                    limit,
                });
            }
            Ok(crate::gpu::upload(
                device,
                label,
                bytes,
                wgpu::BufferUsages::STORAGE,
            ))
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

/// A view's evaluated transforms + raw opacity and the per-frame-slot uniforms of the motion pass.
pub(crate) struct Deform {
    pub transforms: wgpu::Buffer,
    pub opacity: wgpu::Buffer,
    slots: [(wgpu::Buffer, wgpu::BindGroup); 3],
    params: [u32; 8],
    groups: u32,
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
            groups,
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
        let x = self.groups.min(65535);
        pass.dispatch_workgroups(x, self.groups.div_ceil(x), 1);
    }
}

#[cfg(test)]
mod tests {
    //! E4: the motion pass read back against the CPU references of both forms.
    use super::*;
    use crate::Splats;
    use crate::test_utils::{gpu, read};
    use crate::trigs::{CULLED_RAW_OPACITY, ClipWindow};

    /// Deterministic uniform floats in [-1, 1) (xorshift).
    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        }

        fn v3(&mut self, scale: f32) -> [f32; 3] {
            std::array::from_fn(|_| self.next() * scale)
        }
    }

    /// Canonical rows: centres within 3 m, random unit wxyz rotations, log scales.
    fn canonical(n: usize, rng: &mut Rng) -> Splats {
        let transforms = (0..n)
            .map(|_| {
                let c = rng.v3(3.0);
                let q: [f32; 4] = std::array::from_fn(|_| rng.next());
                let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-3);
                let mut q = q.map(|v| v / norm);
                if q[0] < 0.0 {
                    q = q.map(|v| -v);
                }
                let s = rng.v3(2.0).map(|v| v - 4.0);
                [c[0], c[1], c[2], q[0], q[1], q[2], q[3], s[0], s[1], s[2]]
            })
            .collect::<Vec<_>>();
        Splats {
            raw_opacities: vec![0.0; n],
            sh_coefficients: vec![[0.0; 3]; n],
            sh_degree: 0,
            min_scale: None,
            transforms,
        }
    }

    /// Run the motion pass on `canonical` at `time` and read back (transforms, raw opacity).
    fn run(canonical: &Splats, motion: Motion<'_>, times: &[f32]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let (device, queue) = gpu();
        let kernels = crate::kernels::Kernels::new(&device);
        let n = canonical.transforms.len();
        let canonical_buffer = crate::gpu::upload(
            &device,
            "canonical",
            bytemuck::cast_slice(&canonical.transforms),
            wgpu::BufferUsages::STORAGE,
        );
        let gpu_motion = motion.upload(&device, n, u64::MAX).unwrap();
        let mut deform = Deform::new(
            &device,
            &kernels.trigs_deform,
            &gpu_motion,
            &canonical_buffer,
            n as u32,
            u64::MAX,
        )
        .unwrap();
        times
            .iter()
            .map(|&time| {
                let mut encoder = device.create_command_encoder(&Default::default());
                deform.encode(&mut encoder, &queue, &kernels.trigs_deform, 0, time, None);
                queue.submit([encoder.finish()]);
                (
                    read(&device, &queue, &deform.transforms, n * 10),
                    read(&device, &queue, &deform.opacity, n),
                )
            })
            .collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Rotation angle between two unit wxyz quaternions (either sign): 2 asin |vec(conj(a) b)| in f64, well
    /// conditioned for small angles (acos of the dot product is not).
    fn angle(a: [f32; 4], b: [f32; 4]) -> f32 {
        let [aw, ax, ay, az] = a.map(f64::from);
        let [bw, bx, by, bz] = b.map(f64::from);
        let x = aw * bx - ax * bw - ay * bz + az * by;
        let y = aw * by + ax * bz - ay * bw - az * bx;
        let z = aw * bz - ax * by + ay * bx - az * bw;
        (2.0 * (x * x + y * y + z * z).sqrt().min(1.0).asin()) as f32
    }

    #[derive(Debug, Default)]
    struct Worst {
        center: f32,
        rotation: f32,
        opacity: f32,
        scale: f32,
        culled: usize,
        culled_mismatch: usize,
    }

    /// Compare one readback against the expected rows (centre, wxyz, raw opacity or -100).
    fn compare(
        gpu: &(Vec<f32>, Vec<f32>),
        expected: &[([f32; 3], [f32; 4], f32)],
        canonical: &Splats,
    ) -> Worst {
        let mut worst = Worst::default();
        for (i, (center, quat, raw)) in expected.iter().enumerate() {
            let t = &gpu.0[i * 10..i * 10 + 10];
            for k in 0..3 {
                worst.center = worst.center.max((t[k] - center[k]).abs());
                worst.scale = worst
                    .scale
                    .max((t[7 + k] - canonical.transforms[i][7 + k]).abs());
            }
            worst.rotation = worst.rotation.max(angle([t[3], t[4], t[5], t[6]], *quat));
            let g = gpu.1[i];
            // Culled rows carry exactly CULLED_RAW_OPACITY (-100); live logits stay within +-13.9.
            let (cpu_culled, gpu_culled) = (*raw <= CULLED_RAW_OPACITY, g <= CULLED_RAW_OPACITY);
            if cpu_culled || gpu_culled {
                worst.culled += usize::from(cpu_culled);
                // A row exactly at the cut may land on either side in f32: allow it only with a tiny weight.
                if cpu_culled != gpu_culled {
                    worst.culled_mismatch += 1;
                    let other = if gpu_culled { *raw } else { g };
                    worst.opacity = worst.opacity.max(sigmoid(other));
                }
            } else {
                worst.opacity = worst.opacity.max((sigmoid(g) - sigmoid(*raw)).abs());
            }
        }
        worst
    }

    fn assert_within(worst: &Worst, form: &str) {
        println!("trigs_deform {form}: {worst:?}");
        // Design E4: 1e-5 m, 1e-5 rad; opacity in linear units; scales are copies.
        assert!(worst.center <= 1e-5, "{form} centre {}", worst.center);
        assert!(worst.rotation <= 1e-5, "{form} rotation {}", worst.rotation);
        assert!(worst.opacity <= 1e-5, "{form} opacity {}", worst.opacity);
        assert_eq!(worst.scale, 0.0, "{form} scales");
        assert!(
            worst.culled > 0,
            "{form}: the test must include culled rows"
        );
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn gpu_twist_form_matches_the_cpu_reference() {
        let n = 70_001;
        let mut rng = Rng(7);
        let canonical = canonical(n, &mut rng);
        let opacities: Vec<f32> = (0..n).map(|_| rng.next() * 0.5 + 0.5).collect();
        let times: Vec<f32> = (0..n).map(|_| rng.next() * 2.0 + 2.0).collect();
        let log_durations: Vec<f32> = (0..n).map(|_| rng.next() * 1.5 - 1.5).collect();
        let mut twist = |scale: f32| -> Vec<[f32; 6]> {
            (0..n)
                .map(|_| std::array::from_fn(|_| rng.next() * scale))
                .collect()
        };
        let mut base = twist(1.0);
        let (c0, c1, c2) = (twist(0.5), twist(0.5), twist(0.5));
        // Some rows with |omega| ~ 0 (gauge switch and the exp Taylor branch).
        for row in base.iter_mut().step_by(97) {
            row[3..].fill(0.0);
        }
        let anchors: Vec<[f32; 3]> = (0..n).map(|_| rng.v3(3.0)).collect();
        let is_static: Vec<f32> = (0..n)
            .map(|i| if i % 31 == 0 { 1.0 } else { 0.0 })
            .collect();
        let motion = TrigsMotion {
            opacities: &opacities,
            times: &times,
            log_durations: &log_durations,
            twist_base: &base,
            twist_c0: &c0,
            twist_c1: &c1,
            twist_c2: &c2,
            anchors: &anchors,
            is_static: &is_static,
            active_threshold: 0.01,
        };
        let eval_times = [0.5f32, 2.0, 3.25, 4.1];
        let readbacks = run(&canonical, Motion::Twist(&motion), &eval_times);
        for (time, readback) in eval_times.iter().zip(&readbacks) {
            let expected: Vec<_> = (0..n)
                .map(|i| {
                    let c = &canonical.transforms[i];
                    let d =
                        motion.deform_row(i, [c[0], c[1], c[2]], [c[4], c[5], c[6], c[3]], *time);
                    let [x, y, z, w] = d.quaternion;
                    let raw = if d.opacity > 0.0 {
                        crate::sog::logit(f64::from(d.opacity))
                    } else {
                        CULLED_RAW_OPACITY
                    };
                    (d.center, [w, x, y, z], raw)
                })
                .collect();
            assert_within(
                &compare(readback, &expected, &canonical),
                &format!("twist t={time}"),
            );
        }
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn gpu_asog_form_matches_the_cpu_reference() {
        let n = 70_001;
        let mut rng = Rng(11);
        let canonical = canonical(n, &mut rng);
        let mut omega = |scale: f32| -> Vec<[[f32; 3]; 3]> {
            (0..n)
                .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| rng.next() * scale)))
                .collect()
        };
        let mut o = omega(2.0);
        for row in o.iter_mut().step_by(89) {
            *row = [[0.0; 3]; 3];
        }
        let motion = AsogMotion {
            opacities: (0..n).map(|_| rng.next() * 0.5 + 0.5).collect(),
            times: (0..n).map(|_| rng.next() * 2.0 + 2.0).collect(),
            sigmas: (0..n).map(|_| (rng.next() * 1.5 - 1.5).exp()).collect(),
            anchors: (0..n).map(|_| rng.v3(3.0)).collect(),
            omega: o,
            g: (0..n)
                .map(|_| std::array::from_fn(|_| rng.next() * 0.3))
                .collect(),
            is_static: (0..n).map(|i| i % 29 == 0).collect(),
            clip: (0..n).map(|i| u32::from(i % 3 == 0)).collect(),
            clips: vec![
                ClipWindow {
                    offset: 0.0,
                    active_start: 0.0,
                    active_end: 4.84,
                    core_start: 0.0,
                    core_end: 4.84,
                    fade_in: 0.0,
                    fade_out: 0.0,
                },
                ClipWindow {
                    offset: 0.5,
                    active_start: 0.5,
                    active_end: 4.5,
                    core_start: 1.0,
                    core_end: 4.0,
                    fade_in: 0.5,
                    fade_out: 0.5,
                },
            ],
            active_threshold: 0.05,
        };
        let eval_times = [0.25f32, 0.75, 2.42, 4.2, 4.8];
        let readbacks = run(&canonical, Motion::Asog(&motion), &eval_times);
        for (time, readback) in eval_times.iter().zip(&readbacks) {
            let evaluated = motion.evaluate(&canonical, *time);
            let expected: Vec<_> = evaluated
                .transforms
                .iter()
                .zip(&evaluated.raw_opacities)
                .map(|(t, raw)| ([t[0], t[1], t[2]], [t[3], t[4], t[5], t[6]], *raw))
                .collect();
            assert_within(
                &compare(readback, &expected, &canonical),
                &format!("asog t={time}"),
            );
        }
    }

    /// Median of a sample, in ms.
    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// Load split, ms per time change (GPU motion pass vs the CPU path: evaluate + upload) and frame time with the
    /// time cursor moving every frame vs still, for a whole `.asog` (`GSPLAT_ASOG_FILE`); run with
    /// `--release -- --ignored --nocapture` under the GPU lock.
    #[test]
    #[ignore = "needs GSPLAT_ASOG_FILE; GPU timing"]
    fn asog_file_timing() {
        // Not a contract: the GPU lane (`--ignored`) runs it too, so it passes without its file.
        let Ok(path) = std::env::var("GSPLAT_ASOG_FILE") else {
            println!("asog_file_timing: GSPLAT_ASOG_FILE is unset; nothing timed");
            return;
        };
        let bytes = std::fs::read(&path).expect("read .asog");
        let (device, queue) = gpu();
        let wait = || {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        let renderer = crate::Renderer::new(&device).unwrap();
        let ms = |start: std::time::Instant| start.elapsed().as_secs_f64() * 1e3;

        let start = std::time::Instant::now();
        let parts = crate::asog::decode_parts(&bytes, crate::trigs::ASOG_ACTIVE_THRESHOLD).unwrap();
        let entropy = ms(start);
        let start = std::time::Instant::now();
        let scene = renderer
            .upload_sog(
                &queue,
                &parts.quantization,
                &parts.planes(),
                Some(Motion::Asog(&parts.motion)),
            )
            .unwrap();
        wait();
        let gpu_load = ms(start);
        let start = std::time::Instant::now();
        let canonical = crate::sog::dequantize(&parts.quantization, &parts.planes()).unwrap();
        let cpu_dequant = ms(start);
        let n = scene.len();
        let fps = parts.manifest.timeline.frame_rate;
        let frames = parts.manifest.timeline.total_frames.unwrap_or(120).max(2);
        let times: Vec<f32> = (0..frames).map(|f| (f as f64 / fps) as f32).collect();
        println!(
            "{path}: {n} splats, SH degree {}, load: zlib + motion de-quantise CPU {entropy:.1} ms, static de-quantise + \
             upload GPU {gpu_load:.1} ms (CPU de-quantise alone {cpu_dequant:.1} ms), scene {} B",
            canonical.sh_degree,
            scene.gpu_byte_size()
        );

        // The motion pass alone, per time change: wall (submit + wait) and GPU timestamps.
        let kernels = crate::kernels::Kernels::new(&device);
        let mut deform = Deform::new(
            &device,
            &kernels.trigs_deform,
            scene.motion().unwrap(),
            scene.transforms().unwrap(),
            n,
            u64::MAX,
        )
        .unwrap();
        let queries = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                })
            });
        let resolve = crate::test_utils::upload(&device, &[0u64; 2]);
        let (mut wall, mut kernel) = (Vec::new(), Vec::new());
        for &t in times.iter().chain(&times) {
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            deform.encode(
                &mut encoder,
                &queue,
                &kernels.trigs_deform,
                0,
                t + 1e-3 * wall.len() as f32,
                queries
                    .as_ref()
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
            );
            if let Some(q) = &queries {
                encoder.resolve_query_set(q, 0..2, &resolve, 0);
            }
            queue.submit([encoder.finish()]);
            wait();
            wall.push(ms(start));
            if queries.is_some() {
                let ticks: Vec<u64> = read(&device, &queue, &resolve, 2);
                kernel.push(
                    (ticks[1] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1e6,
                );
            }
        }
        println!(
            "motion pass per time change ({} changes): wall submit+wait median {:.3} ms, GPU kernel median {:.3} ms",
            wall.len(),
            median(wall),
            if kernel.is_empty() {
                f64::NAN
            } else {
                median(kernel)
            }
        );

        // The CPU path per time change: evaluate on the CPU + a scene upload (what the viewer did before).
        let mut out = Splats::default();
        let (mut evaluate, mut upload) = (Vec::new(), Vec::new());
        for &t in times.iter().step_by(4) {
            let start = std::time::Instant::now();
            parts.motion.evaluate_into(&canonical, t, &mut out);
            evaluate.push(ms(start));
            let start = std::time::Instant::now();
            drop(renderer.upload(&out).unwrap());
            wait();
            upload.push(ms(start));
        }
        println!(
            "CPU path per time change ({} changes): evaluate median {:.2} ms + upload median {:.2} ms",
            evaluate.len(),
            median(evaluate),
            median(upload)
        );

        // Whole frames at 1920x1080, eye 3 m before the centroid looking +z (data frame: x right, y down, z forward).
        let mut centroid = glam::Vec3::ZERO;
        for row in &canonical.transforms {
            centroid += glam::Vec3::new(row[0], row[1], row[2]);
        }
        centroid /= n as f32;
        let mut camera = crate::test_utils::pinhole_camera(1);
        camera.size = glam::UVec2::new(1920, 1080);
        camera.fov_x = 1.2;
        camera.fov_y = 2.0 * ((1.2f64 * 0.5).tan() * 1080.0 / 1920.0).atan();
        camera.position = centroid - glam::Vec3::new(0.0, 0.0, 3.0);
        let target = crate::test_utils::upload(&device, &vec![[0.0f32; 4]; 1920 * 1080]);
        let mut view = renderer.create_view(&scene, 1 << 20).unwrap();
        let mut frame = |time: f32| {
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            let options = crate::RenderOptions {
                time,
                ..Default::default()
            };
            renderer
                .render(
                    &queue,
                    &mut encoder,
                    &mut view,
                    &camera,
                    &options,
                    crate::Target::Float(target.clone()),
                )
                .unwrap();
            queue.submit([encoder.finish()]);
            wait();
            let stats = view.poll_feedback().unwrap().unwrap();
            (ms(start), stats)
        };
        // Warm up until the intersection buffer fits every frame.
        for _ in 0..3 {
            for &t in &times {
                frame(t);
            }
        }
        let moving: Vec<_> = times.iter().map(|&t| frame(t)).collect();
        let visible = moving.iter().map(|(_, s)| s.visible).max().unwrap_or(0);
        let still: Vec<_> = (0..times.len())
            .map(|_| frame(times[times.len() / 2]))
            .collect();
        println!(
            "frame 1920x1080 (submit + wait), {} frames: cursor moving every frame median {:.3} ms, still median {:.3} ms; \
             max visible {visible}; rerenders {}",
            times.len(),
            median(moving.iter().map(|(m, _)| *m).collect()),
            median(still.iter().map(|(m, _)| *m).collect()),
            moving
                .iter()
                .chain(&still)
                .filter(|(_, s)| s.needs_rerender)
                .count()
        );
    }

    #[test]
    fn packing_strides_match_the_shader() {
        let source = crate::shader::resolve("trigs_deform.wgsl");
        assert!(source.contains(&format!("TWIST_STRIDE: u32 = {TWIST_STRIDE}u;")));
        assert!(source.contains(&format!("ASOG_STRIDE: u32 = {ASOG_STRIDE}u;")));
        let motion = AsogMotion {
            opacities: vec![0.5],
            times: vec![1.0],
            sigmas: vec![0.1],
            anchors: vec![[1.0, 2.0, 3.0]],
            omega: vec![[[4.0; 3], [5.0; 3], [6.0; 3]]],
            g: vec![[7.0, 8.0, 9.0, 10.0, 11.0]],
            is_static: vec![true],
            clip: vec![2],
            clips: vec![],
            active_threshold: 0.05,
        };
        let (_, packed, _, _) = Motion::Asog(&motion).pack();
        assert_eq!(packed.len(), ASOG_STRIDE);
        assert_eq!(packed[3].to_bits(), 1 | (2 << 1));
        assert_eq!(packed[20], 11.0);
    }
}
