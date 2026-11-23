//! The motion pass read back against its CPU reference.
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
fn run(canonical: &Splats, motion: &AsogMotion, times: &[f32]) -> Vec<(Vec<f32>, Vec<f32>)> {
    let (device, queue) = gpu();
    let kernels = crate::kernels::Kernels::new(&device);
    let n = canonical.transforms.len();
    let canonical_buffer = crate::gpu::upload(
        &device,
        "canonical",
        bytemuck::cast_slice(&canonical.transforms),
        wgpu::BufferUsages::STORAGE,
    );
    let params = crate::scene::upload(
        &device,
        "motion parameters",
        bytemuck::cast_slice(&pack(motion)),
        u64::MAX,
    )
    .unwrap();
    let gpu_motion = GpuMotion::new(
        &device,
        params,
        &motion.clips,
        motion.active_threshold,
        u64::MAX,
    )
    .unwrap();
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
    /// |centre| (m) of the row with the worst centre error.
    center_at: f32,
    /// Worst centre error relative to max(1 m, |centre|): f32 keeps ~6e-8 of the magnitude.
    center_relative: f32,
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
        let magnitude = center.iter().map(|v| v * v).sum::<f32>().sqrt();
        for k in 0..3 {
            let error = (t[k] - center[k]).abs();
            if error > worst.center {
                worst.center = error;
                worst.center_at = magnitude;
            }
            worst.center_relative = worst.center_relative.max(error / magnitude.max(1.0));
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
    // Bounds: 1e-5 m, 1e-5 rad; opacity in linear units; scales are copies. The centre bound is relative to
    // max(1 m, |centre|): both sides are f32, and a rotation rounding of ~3e-6 rad over a 3.5 m lever arm moves a
    // centre by ~1e-5 m (asog row 20871 at t = 2.42: f32 CPU 1.9e-5 m and GPU 8.1e-6 m from an f64 evaluation).
    assert!(
        worst.center_relative <= 1e-5,
        "{form} centre {} m at |centre| {} m (relative {})",
        worst.center,
        worst.center_at,
        worst.center_relative
    );
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
    let readbacks = run(&canonical, &motion, &eval_times);
    for (time, readback) in eval_times.iter().zip(&readbacks) {
        let evaluated = motion.evaluate(&canonical, *time);
        let expected: Vec<_> = evaluated
            .transforms
            .iter()
            .zip(&evaluated.raw_opacities)
            .map(|(t, raw)| ([t[0], t[1], t[2]], [t[3], t[4], t[5], t[6]], *raw))
            .collect();
        let w = compare(readback, &expected, &canonical);
        assert_within(&w, &format!("asog t={time}"));
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
    let scene = renderer.upload_asog(&queue, &parts).unwrap();
    wait();
    let gpu_load = ms(start);
    let start = std::time::Instant::now();
    let canonical = crate::sog::dequantize(&parts.quantization, &parts.planes()).unwrap();
    let motion = parts.motion();
    let cpu_dequant = ms(start);
    let n = scene.len();
    let fps = parts.manifest.timeline.frame_rate;
    let frames = parts.manifest.timeline.total_frames.unwrap_or(120).max(2);
    let times: Vec<f32> = (0..frames).map(|f| (f as f64 / fps) as f32).collect();
    println!(
        "{path}: {n} splats, SH degree {}, load: entropy (zlib) CPU {entropy:.1} ms, upload + static and motion \
         de-quantise GPU {gpu_load:.1} ms (CPU static + motion de-quantise alone {cpu_dequant:.1} ms), scene {} B",
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
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
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
            kernel
                .push((ticks[1] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1e6);
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

    // The CPU path per time change: evaluate on the CPU + a scene upload (what a CPU render path would cost).
    let mut out = Splats::default();
    let (mut evaluate, mut upload) = (Vec::new(), Vec::new());
    for &t in times.iter().step_by(4) {
        let start = std::time::Instant::now();
        motion.evaluate_into(&canonical, t, &mut out);
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
fn packing_stride_matches_the_shader() {
    let source = crate::shader::resolve("trigs_deform.wgsl");
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
    let packed = pack(&motion);
    assert_eq!(packed.len(), ASOG_STRIDE);
    assert_eq!(packed[3].to_bits(), 1 | (2 << 1));
    assert_eq!(packed[20], 11.0);
}
