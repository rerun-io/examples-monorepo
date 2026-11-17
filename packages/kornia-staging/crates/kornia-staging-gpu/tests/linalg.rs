//! The global BA's device PCG against the CPU lane (DESIGN §7.4, PR e1).
#![cfg(feature = "wgpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kornia_staging_3d::camera::{CameraModelKind, Pinhole};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use kornia_staging_gpu::linalg::{device_dot, device_matvec, fixed_order_dot, probe_pcg, GpuPcg};
use kornia_staging_gpu::runtime::gpu_client;
use kornia_staging_slam::sqrt_ba::{
    apply_left_increment, pcg, solve_global_with, BlockJacobi, GbaCamera, GbaLandmark,
    GbaObservation, GbaOptions, GbaProblem, GbaRigCamera, PcgLane, PcgOptions, PcgTermination,
    ReducedBlock, ReducedSystem, RelPoseConstraint,
};
use nalgebra::{Matrix3, Matrix6, UnitQuaternion, Vector3, Vector6};

fn wobble(i: usize, k: usize) -> f64 {
    (((i * 7919 + k * 104_729) % 9973) as f64 * 0.618_033_988_7).sin()
}

/// Blocks with 1-4 keyframes and row counts on both sides of the eight lanes.
struct Blocks {
    values: Vec<Vec<f32>>,
    keyframes: Vec<Vec<usize>>,
    rows: Vec<usize>,
    poses: usize,
}

fn blocks(poses: usize, count: usize, seed: usize) -> Blocks {
    let mut out = Blocks {
        values: Vec::new(),
        keyframes: Vec::new(),
        rows: Vec::new(),
        poses,
    };
    for l in 0..count {
        let mut kfs: Vec<usize> = (0..1 + (l + seed) % 4)
            .map(|i| (l * (2 * i + 3) + seed + i) % poses)
            .collect();
        kfs.sort_unstable();
        kfs.dedup();
        let rows = 1 + (l * 5 + seed) % 23;
        // Stored with a 3-row head and a stride, as the landmark blocks are.
        let stride = rows + 3;
        out.values.push(
            (0..stride * 6 * kfs.len())
                .map(|e| wobble(l + seed, e) as f32)
                .collect(),
        );
        out.keyframes.push(kfs);
        out.rows.push(rows);
    }
    out
}

fn system(b: &Blocks, lambda: f32) -> ReducedSystem<'_, f32> {
    ReducedSystem {
        keyframes: b.poses,
        lambda,
        blocks: b
            .values
            .iter()
            .zip(&b.keyframes)
            .zip(&b.rows)
            .map(|((values, keyframes), &rows)| ReducedBlock {
                keyframes,
                values,
                stride: rows + 3,
                offset: 3,
                rows,
            })
            .collect(),
    }
}

/// The diagonal 6x6 blocks of `H`, in f64.
fn preconditioner(system: &ReducedSystem<'_, f32>) -> BlockJacobi<f32> {
    let mut diagonal =
        vec![Matrix6::<f64>::identity() * f64::from(system.lambda); system.keyframes];
    for block in &system.blocks {
        for (s, &kf) in block.keyframes.iter().enumerate() {
            for i in 0..6 {
                for j in 0..6 {
                    diagonal[kf][(i, j)] += block
                        .column(6 * s + i)
                        .iter()
                        .zip(block.column(6 * s + j))
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>();
                }
            }
        }
    }
    BlockJacobi::from_blocks(&diagonal)
}

fn vector(n: usize, k: usize) -> Vec<f32> {
    (0..n).map(|i| wobble(i, k) as f32).collect()
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

#[test]
fn block_gemv_and_pose_reduce_match_the_cpu_reference() {
    let client = gpu_client().unwrap();
    let b = blocks(37, 400, 1);
    let sys = system(&b, 1e-3);
    let x = vector(6 * b.poses, 2);
    let mut expected = vec![0.0f32; x.len()];
    sys.matvec(&x, &mut expected);
    let got = device_matvec(&client, &sys, &x, None).unwrap();
    assert!(
        max_rel(&got, &expected) < 1e-5,
        "{}",
        max_rel(&got, &expected)
    );
    let again = device_matvec(&client, &sys, &x, None).unwrap();
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "two device runs differ"
    );
}

#[test]
fn paging_below_the_binding_limit_changes_nothing() {
    let client = gpu_client().unwrap();
    let b = blocks(19, 300, 4);
    let sys = system(&b, 1e-2);
    let x = vector(6 * b.poses, 5);
    let whole = device_matvec(&client, &sys, &x, None).unwrap();
    // The largest block here is 23 rows x 24 columns = 2,208 bytes.
    let paged = device_matvec(&client, &sys, &x, Some(4096)).unwrap();
    assert_eq!(
        whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        paged.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // A block larger than a page is refused by name, never truncated.
    let error = device_matvec(&client, &sys, &x, Some(1024)).unwrap_err();
    assert!(error.to_string().contains("binding limit"), "{error}");
}

#[test]
fn the_device_dot_is_the_fixed_order_sum() {
    let client = gpu_client().unwrap();
    for groups in [1usize, 255, 256, 257, 4000] {
        let (a, b) = (vector(6 * groups, 1), vector(6 * groups, 2));
        let device = device_dot(&client, &a, &b).unwrap();
        let host = fixed_order_dot(&a, &b);
        let exact: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let magnitude: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from((x * y).abs()))
            .sum();
        // Compensated: the products' own rounding (half an ulp each) plus
        // one rounding of the result; FMA contraction stays inside that bound.
        let eps = f64::from(f32::EPSILON);
        for (what, value) in [("device", device), ("host order", host)] {
            assert!(
                (f64::from(value) - exact).abs() <= eps * (0.5 * magnitude + exact.abs()),
                "{groups}: {what} {value} vs {exact}"
            );
        }
        assert_eq!(
            device.to_bits(),
            device_dot(&client, &a, &b).unwrap().to_bits()
        );
    }
    // Exact products that cancel: a plain f32 sum loses every small term
    // next to 1e8 (one ulp there is 8); the compensated sum keeps them all.
    for groups in [1usize, 300, 4000] {
        let a: Vec<f32> = (0..groups)
            .flat_map(|k| [1e8, 1.0, -1e8, 0.5, 0.25 * (k % 3) as f32, -0.125])
            .collect();
        let b = vec![1.0f32; a.len()];
        let exact: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
        for (what, value) in [
            ("device", device_dot(&client, &a, &b).unwrap()),
            ("host order", fixed_order_dot(&a, &b)),
        ] {
            assert!(
                (f64::from(value) - exact).abs() <= f64::from(f32::EPSILON) * exact.abs(),
                "{groups}: {what} {value} vs {exact}"
            );
        }
    }
}

fn cpu_pcg(
    sys: &ReducedSystem<'_, f32>,
    b: &[f32],
    options: &PcgOptions,
) -> (Vec<f32>, usize, PcgTermination) {
    let mut x = vec![0.0f32; b.len()];
    let summary = pcg(
        |v, out| sys.matvec(v, out),
        &preconditioner(sys),
        b,
        &mut x,
        options,
    );
    (x, summary.iterations, summary.termination)
}

fn gpu_pcg(
    lane: &mut GpuPcg<kornia_staging_gpu::GpuRuntime>,
    sys: &ReducedSystem<'_, f32>,
    b: &[f32],
    options: &PcgOptions,
) -> (Vec<f32>, usize, PcgTermination) {
    let mut x = vec![0.0f32; b.len()];
    let solved = lane
        .solve(sys, &preconditioner(sys), b, &mut x, options)
        .unwrap();
    (x, solved.summary.iterations, solved.summary.termination)
}

#[test]
fn the_device_pcg_follows_the_cpu_lane_iteration_by_iteration() {
    let client = gpu_client().unwrap();
    let blocks = blocks(23, 500, 7);
    let sys = system(&blocks, 1e-2);
    let b = vector(6 * blocks.poses, 9);
    let mut lane = GpuPcg::new(client);
    // Across the first step, beta, and the residual replacement at 8 and 16.
    for its in [1usize, 2, 7, 8, 9, 16, 17] {
        let options = PcgOptions {
            max_iterations: its,
            eta: f64::NEG_INFINITY,
        };
        let (cpu, cpu_its, cpu_term) = cpu_pcg(&sys, &b, &options);
        let (gpu, gpu_its, gpu_term) = gpu_pcg(&mut lane, &sys, &b, &options);
        assert_eq!((gpu_its, gpu_term), (cpu_its, cpu_term), "{its}");
        assert_eq!(gpu_term, PcgTermination::MaxIterations);
        assert!(max_rel(&gpu, &cpu) < 1e-4, "{its}: {}", max_rel(&gpu, &cpu));
        assert_eq!(lane.last_solve().syncs, its.div_ceil(8) + 1, "{its}");
        // One exclusive section per iteration and one per state read.
        assert_eq!(lane.last_solve().sections, its + its.div_ceil(8), "{its}");
    }
}

#[test]
fn the_device_pcg_stops_where_the_cpu_lane_stops() {
    let client = gpu_client().unwrap();
    let blocks = blocks(31, 700, 3);
    let sys = system(&blocks, 1e-3);
    let b = vector(6 * blocks.poses, 4);
    let mut lane = GpuPcg::new(client.clone());
    for eta in [0.1, 1e-3] {
        let options = PcgOptions {
            max_iterations: 500,
            eta,
        };
        let (cpu, cpu_its, cpu_term) = cpu_pcg(&sys, &b, &options);
        let (gpu, gpu_its, gpu_term) = gpu_pcg(&mut lane, &sys, &b, &options);
        assert_eq!(gpu_term, cpu_term, "eta {eta}");
        assert!(
            gpu_its.abs_diff(cpu_its) <= 1,
            "eta {eta}: {gpu_its} vs {cpu_its}"
        );
        assert!(
            max_rel(&gpu, &cpu) < 1e-3,
            "eta {eta}: {}",
            max_rel(&gpu, &cpu)
        );
        // Two runs and a paged run are bit-equal.
        let (again, ..) = gpu_pcg(&mut lane, &sys, &b, &options);
        let mut paged_lane = GpuPcg::new(client.clone()).with_max_binding_bytes(8192);
        let (paged, ..) = gpu_pcg(&mut paged_lane, &sys, &b, &options);
        assert!(paged_lane.last_solve().pages > 1);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&gpu), bits(&again));
        assert_eq!(bits(&gpu), bits(&paged));
    }
}

#[test]
fn the_device_pcg_reports_zero_rhs_and_indefinite_like_the_cpu_lane() {
    let client = gpu_client().unwrap();
    let blocks = blocks(5, 40, 2);
    let mut lane = GpuPcg::new(client);
    let sys = system(&blocks, 1e-2);
    let zero = vec![0.0f32; 30];
    let (x, its, term) = gpu_pcg(&mut lane, &sys, &zero, &PcgOptions::default());
    assert_eq!((its, term), (0, PcgTermination::ZeroRhs));
    assert!(x.iter().all(|v| *v == 0.0));
    // No blocks and a negative damping: p' A p < 0 on the first iteration.
    let empty = Blocks {
        values: Vec::new(),
        keyframes: Vec::new(),
        rows: Vec::new(),
        poses: 5,
    };
    let negative = ReducedSystem {
        lambda: -1.0,
        ..system(&empty, 0.0)
    };
    let b = vector(30, 1);
    let cpu = {
        let mut x = vec![0.0f32; 30];
        let pre = BlockJacobi::<f32>::from_blocks(&vec![Matrix6::identity(); 5]);
        pcg(
            |v, out| negative.matvec(v, out),
            &pre,
            &b,
            &mut x,
            &PcgOptions::default(),
        )
    };
    let mut x = vec![0.0f32; 30];
    let pre = BlockJacobi::<f32>::from_blocks(&vec![Matrix6::identity(); 5]);
    let gpu = lane
        .solve(&negative, &pre, &b, &mut x, &PcgOptions::default())
        .unwrap()
        .summary;
    assert_eq!(cpu.termination, PcgTermination::Indefinite);
    assert_eq!(gpu, cpu);
}

#[test]
fn the_back_end_probe_passes_on_this_device() {
    probe_pcg(&gpu_client().unwrap()).unwrap();
}

/// A pinhole camera whose point Jacobian is NaN right of `x > poison` (camera
/// frame): rows the host drops in stage 1, for every lane alike.
#[derive(Debug, Clone, Copy)]
struct Poisoned {
    pinhole: CameraModelKind<f32>,
    poison: f32,
}

impl GbaCamera<f32> for Poisoned {
    fn project(&self, p: [f32; 3], jacobian: Option<&mut [[f32; 3]; 2]>) -> ([f32; 2], bool) {
        let poisoned = p[0] / p[2] > self.poison;
        let (pixel, valid) = match jacobian {
            Some(j) => {
                let out = GbaCamera::project(&self.pinhole, p, Some(j));
                if poisoned {
                    j[0][0] = f32::NAN;
                }
                out
            }
            None => GbaCamera::project(&self.pinhole, p, None),
        };
        (pixel, valid)
    }
}

/// Keyframes on a circle around a point cloud, two cameras, noisy start.
fn scene(n_kf: usize, n_lm: usize) -> (GbaProblem<f32>, Vec<GbaRigCamera<f32, Poisoned>>) {
    let pinhole = CameraModelKind::Pinhole(Pinhole::new([300.0f32, 300.0, 320.0, 240.0]).unwrap());
    let rig: Vec<GbaRigCamera<f32, Poisoned>> = (0..2)
        .map(|c| GbaRigCamera {
            model: Poisoned {
                pinhole,
                poison: 0.6,
            },
            imu_from_camera: RigidTransform::new(
                Rotation3::exp(&Vector3::new(0.02 * c as f32, 0.15 * c as f32 - 0.2, 0.01)),
                Vector3::new(0.08 * c as f32 - 0.1, 0.01 * c as f32, 0.02),
            ),
        })
        .collect();
    let imu_from_world: Vec<RigidTransform<f64>> = (0..n_kf)
        .map(|k| {
            let (s, c) = (0.25 * k as f64).sin_cos();
            let m = Matrix3::from_columns(&[
                Vector3::new(-s, c, 0.0),
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(-c, -s, 0.0),
            ]);
            RigidTransform::new(
                Rotation3::from_unit_quaternion(UnitQuaternion::from_matrix(&m)),
                Vector3::new(5.0 * c, 5.0 * s, 0.3 * (k as f64).sin()),
            )
            .inverse()
        })
        .collect();
    let mut landmarks = Vec::new();
    for l in 0..n_lm {
        let p = Vector3::new(1.5 * wobble(l, 1), 1.5 * wobble(l, 2), wobble(l, 3));
        let mut observations = Vec::new();
        for (k, pose) in imu_from_world.iter().enumerate() {
            for (c, cam) in rig.iter().enumerate() {
                if (k + c + l) % 3 == 0 {
                    continue;
                }
                let p_c = cam.imu_from_camera.cast::<f64>().inverse() * (*pose * p);
                let (px, valid) = GbaCamera::project(
                    &cam.model.pinhole,
                    [p_c.x as f32, p_c.y as f32, p_c.z as f32],
                    None,
                );
                if valid && (0.0..640.0).contains(&px[0]) && (0.0..480.0).contains(&px[1]) {
                    observations.push(GbaObservation {
                        keyframe: k,
                        camera: c,
                        pixel: [
                            f64::from(px[0]) + 0.3 * wobble(l, 10 * k + c),
                            f64::from(px[1]) + 0.3 * wobble(l, 10 * k + c + 5),
                        ],
                    });
                }
            }
        }
        if observations.len() >= 2 {
            landmarks.push(GbaLandmark {
                position: p,
                observations,
            });
        }
    }
    let rel_poses = (1..n_kf)
        .map(|k| {
            RelPoseConstraint::with_sigmas(
                k,
                k - 1,
                imu_from_world[k] * imu_from_world[k - 1].inverse(),
                0.05,
                0.01,
            )
        })
        .collect();
    let mut problem = GbaProblem {
        imu_from_world,
        landmarks,
        rel_poses,
    };
    for (k, pose) in problem.imu_from_world.iter_mut().enumerate().skip(1) {
        let xi = Vector6::from_fn(|i, _| wobble(k, 50 + i) * if i < 3 { 0.03 } else { 0.01 });
        *pose = apply_left_increment(pose, &xi);
    }
    for (l, landmark) in problem.landmarks.iter_mut().enumerate() {
        landmark.position += Vector3::from_fn(|i, _| 0.05 * wobble(l, 70 + i));
    }
    (problem.cast(), rig)
}

#[test]
fn the_global_ba_on_the_device_lane_lands_in_the_cpu_band() {
    let client = gpu_client().unwrap();
    let (problem, rig) = scene(14, 160);
    let options = GbaOptions::default();
    let mut cpu = problem.clone();
    let cpu_report = solve_global_with(&mut cpu, &rig, &options, None, None).unwrap();
    let mut lane = GpuPcg::new(client).with_max_binding_bytes(16 * 1024);
    let mut gpu = problem.clone();
    let gpu_report = solve_global_with(&mut gpu, &rig, &options, None, Some(&mut lane)).unwrap();
    eprintln!(
        "cpu {} attempts {} CG, cost {:.6e}; gpu {} attempts {} CG, cost {:.6e}",
        cpu_report.iterations.len(),
        cpu_report.pcg_iterations(),
        cpu_report.final_cost.error,
        gpu_report.iterations.len(),
        gpu_report.pcg_iterations(),
        gpu_report.final_cost.error
    );
    // Non-finite rows go in stage 1 on the host: from one state, both lanes drop the same rows.
    let first = GbaOptions {
        max_iterations: 1,
        ..GbaOptions::default()
    };
    let (mut a, mut b) = (problem.clone(), problem.clone());
    let cpu_first = solve_global_with(&mut a, &rig, &first, None, None).unwrap();
    let gpu_first = solve_global_with(&mut b, &rig, &first, None, Some(&mut lane)).unwrap();
    assert!(cpu_first.dropped_rows > 0, "the scene has no poisoned rows");
    assert_eq!(gpu_first.dropped_rows, cpu_first.dropped_rows);
    let (c, g) = (cpu_report.final_cost.error, gpu_report.final_cost.error);
    assert!((g - c).abs() <= 1e-3 * c, "final cost gpu {g} vs cpu {c}");
    assert!(g < 0.5 * gpu_report.initial_cost.error);
    let rms = (cpu
        .imu_from_world
        .iter()
        .zip(&gpu.imu_from_world)
        .map(|(a, b)| f64::from((a.inverse().translation - b.inverse().translation).norm_squared()))
        .sum::<f64>()
        / cpu.imu_from_world.len() as f64)
        .sqrt();
    assert!(rms < 1e-3, "keyframe position RMS {rms} m");
    assert!(gpu_report.iterations.iter().all(|i| i.pcg_syncs > 0));
    // Two device runs are bit-equal.
    let mut again = problem.clone();
    let again_report =
        solve_global_with(&mut again, &rig, &options, None, Some(&mut lane)).unwrap();
    assert_eq!(again_report.final_cost, gpu_report.final_cost);
    assert_eq!(again, gpu);
}
