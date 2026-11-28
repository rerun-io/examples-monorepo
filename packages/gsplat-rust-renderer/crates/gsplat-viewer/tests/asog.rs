//! `gsplat_core::asog` + the `.asog` motion form against the asog reference decoder (`asog.decode.read` and
//! `Asog.evaluate`, f64): `tests/fixtures/asog_reference.rrd` holds 2,048 rows of a trained `.asog` clip (every row
//! with near-zero omega and every rank-deficient / inexact pinv row, then random rows) as `AnimatedSogSplats3D`,
//! and the reference decode under `/expected` (written by `tools/asog_reference_fixture.py` in the open4dgs env).
//!
//! Rust decode + evaluate == reference (centre <= 1e-5 m, rotation <= 1e-5 rad, weight <= 1e-6, same visible
//! set).
use gsplat_core::asog::decode;
use gsplat_core::trigs::{ASOG_ACTIVE_THRESHOLD, CULLED_RAW_OPACITY};
mod common;

use common::{chunks, row};
use re_viewer::external::re_chunk::Chunk;
use re_viewer::external::re_sdk_types::external::arrow::array::AsArray as _;
use re_viewer::external::re_sdk_types::external::arrow::datatypes::{
    Float32Type, Float64Type, UInt8Type,
};
use std::collections::BTreeMap;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/asog_reference.rrd"
);

/// `Expected:<field>` as f64 (stored as f32 or f64).
fn expected(all: &BTreeMap<String, Vec<Chunk>>, entity: &str, field: &str) -> Vec<f64> {
    let values =
        row(&all[entity], "Expected", field).unwrap_or_else(|| panic!("Expected:{field} missing"));
    if let Some(f) = values.as_primitive_opt::<Float32Type>() {
        f.values().iter().map(|&v| f64::from(v)).collect()
    } else {
        values.as_primitive::<Float64Type>().values().to_vec()
    }
}

struct Fixture {
    all: BTreeMap<String, Vec<Chunk>>,
    asog: gsplat_core::asog::Asog,
    times: Vec<f64>,
}

fn fixture() -> Fixture {
    let all = chunks(FIXTURE);
    let entity = &all["/asog"];
    let bytes = row(entity, "AnimatedSogSplats3D", "asog").expect("AnimatedSogSplats3D:asog");
    let threshold = row(entity, "AnimatedSogSplats3D", "active_threshold")
        .expect("AnimatedSogSplats3D:active_threshold");
    let threshold = threshold.as_primitive::<Float32Type>().value(0);
    assert_eq!(threshold, ASOG_ACTIVE_THRESHOLD);
    let asog = decode(bytes.as_primitive::<UInt8Type>().values(), threshold).expect("decode");
    let times = expected(&all, "/expected/canonical", "times");
    Fixture { all, asog, times }
}

/// Worst |a - b| / max(|b|, 1).
fn worst_rel(a: impl IntoIterator<Item = f64>, b: &[f64]) -> f64 {
    let a: Vec<f64> = a.into_iter().collect();
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// Rotation angle between unit-ish wxyz quaternions (sign-invariant), in f64.
fn angle(q: [f32; 4], r: &[f64]) -> f64 {
    let q = q.map(f64::from);
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q = q.map(|v| v / n);
    // d = conj(r) * q
    let (rw, rx, ry, rz) = (r[0], -r[1], -r[2], -r[3]);
    let [qw, qx, qy, qz] = q;
    let w = rw * qw - rx * qx - ry * qy - rz * qz;
    let x = rw * qx + rx * qw + ry * qz - rz * qy;
    let y = rw * qy - rx * qz + ry * qw + rz * qx;
    let z = rw * qz + rx * qy - ry * qx + rz * qw;
    2.0 * (x * x + y * y + z * z).sqrt().atan2(w.abs())
}

#[test]
fn decode_matches_the_reference_read() {
    let f = fixture();
    let c = "/expected/canonical";
    let splats = &f.asog.canonical;
    let m = &f.asog.motion;
    let n = splats.transforms.len();
    assert_eq!(n, 2048);
    let col = |range: std::ops::Range<usize>| {
        splats.transforms.iter().flat_map(move |t| {
            t[range.clone()]
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>()
        })
    };
    let checks = [
        ("means", worst_rel(col(0..3), &expected(&f.all, c, "means"))),
        ("quats", worst_rel(col(3..7), &expected(&f.all, c, "quats"))),
        (
            "log_scales",
            worst_rel(col(7..10), &expected(&f.all, c, "log_scales")),
        ),
        (
            "opacity",
            worst_rel(
                m.opacities.iter().map(|&v| f64::from(v)),
                &expected(&f.all, c, "opacity"),
            ),
        ),
        (
            "sh0",
            worst_rel(
                splats
                    .sh_coefficients
                    .chunks(16)
                    .flat_map(|s| s[0].map(f64::from)),
                &expected(&f.all, c, "sh0"),
            ),
        ),
        ("sh_rest", {
            let want = expected(&f.all, c, "sh_rest");
            let rows = want.len() / 45;
            worst_rel(
                splats.sh_coefficients[..16 * rows]
                    .chunks(16)
                    .flat_map(|s| {
                        s[1..]
                            .iter()
                            .flat_map(|v| v.map(f64::from))
                            .collect::<Vec<_>>()
                    }),
                &want,
            )
        }),
        (
            "mu",
            worst_rel(
                m.times.iter().map(|&v| f64::from(v)),
                &expected(&f.all, c, "mu"),
            ),
        ),
        (
            "sigma",
            worst_rel(
                m.sigmas.iter().map(|&v| f64::from(v)),
                &expected(&f.all, c, "sigma"),
            ),
        ),
        (
            "anchor",
            worst_rel(
                m.anchors.iter().flatten().map(|&v| f64::from(v)),
                &expected(&f.all, c, "anchor"),
            ),
        ),
        (
            "omega",
            worst_rel(
                m.omega.iter().flatten().flatten().map(|&v| f64::from(v)),
                &expected(&f.all, c, "omega"),
            ),
        ),
        (
            "g",
            worst_rel(
                m.g.iter().flatten().map(|&v| f64::from(v)),
                &expected(&f.all, c, "g"),
            ),
        ),
    ];
    println!("decode, worst relative difference (floor 1): {checks:?}");
    assert_eq!(splats.sh_degree, 3);
    for (name, worst) in checks {
        assert!(worst <= 1e-6, "{name}: {worst}");
    }
    let statics = expected(&f.all, c, "static");
    assert!(
        m.is_static
            .iter()
            .zip(&statics)
            .all(|(&s, &e)| s == (e > 0.5))
    );
}

#[test]
fn evaluate_matches_the_reference() {
    let f = fixture();
    let m = &f.asog.motion;
    let canonical = &f.asog.canonical;
    for (k, &t) in f.times.iter().enumerate() {
        let entity = format!("/expected/t{k}");
        let positions = expected(&f.all, &entity, "positions");
        let rotations = expected(&f.all, &entity, "rotations");
        let weights = expected(&f.all, &entity, "weights");
        let clips = m.clip_states(t as f32);
        let evaluated = m.evaluate(canonical, t as f32);
        let (mut centre, mut rotation, mut weight, mut mismatched, mut visible) =
            (0.0f64, 0.0f64, 0.0f64, 0, 0);
        for i in 0..m.len() {
            let r = m.row(i, &canonical.transforms[i], &clips);
            let w_ref = weights[i];
            weight = weight.max((f64::from(r.weight) - w_ref).abs());
            if (r.weight > 0.0) != (w_ref > 0.0) {
                mismatched += 1;
                continue;
            }
            // The Splats output is the same row; culled rows carry the culled raw opacity.
            let out = evaluated.transforms[i];
            assert_eq!(
                &out[..7],
                &[
                    r.center[0],
                    r.center[1],
                    r.center[2],
                    r.quaternion[0],
                    r.quaternion[1],
                    r.quaternion[2],
                    r.quaternion[3]
                ]
            );
            assert_eq!(
                evaluated.raw_opacities[i].to_bits() == CULLED_RAW_OPACITY.to_bits(),
                r.weight <= 0.0
            );
            if w_ref <= 0.0 {
                continue;
            }
            visible += 1;
            let d = (0..3)
                .map(|a| (f64::from(r.center[a]) - positions[3 * i + a]).powi(2))
                .sum::<f64>()
                .sqrt();
            centre = centre.max(d);
            rotation = rotation.max(angle(r.quaternion, &rotations[4 * i..4 * i + 4]));
        }
        println!(
            "evaluate t={t}: {visible} visible, {mismatched} visibility mismatches, worst centre {centre:.3e} m, rotation {rotation:.3e} rad, opacity {weight:.3e}"
        );
        assert_eq!(mismatched, 0, "t={t}");
        assert!(
            centre <= 1e-5 && rotation <= 1e-5 && weight <= 1e-6,
            "t={t}"
        );
    }
}

/// Hash of every array `sog_dequant.wgsl` and `asog_motion.wgsl` read (texel planes, SH planes, motion codes,
/// codebooks and bounds).
fn parts_hash(p: &gsplat_core::asog::AsogParts) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut h = std::hash::DefaultHasher::new();
    (
        &p.means_l,
        &p.means_u,
        &p.quats,
        &p.scales,
        &p.sh0,
        &p.sh,
        &p.motion_codes,
    )
        .hash(&mut h);
    let q = &p.quantization;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    (
        q.count,
        q.means_min.map(f64::to_bits),
        q.means_max.map(f64::to_bits),
        bits(&q.scales),
        bits(&q.sh0),
        bits(&q.sh_rest),
    )
        .hash(&mut h);
    h.finish()
}

/// The parallel inflate of `decode_parts` gives the serial result bit for bit: on the fixture, and on
/// `GSPLAT_ASOG_FILE` when it is set (whole trained clips).
#[test]
fn parallel_inflate_gives_the_serial_parts() {
    let all = chunks(FIXTURE);
    let fixture =
        row(&all["/asog"], "AnimatedSogSplats3D", "asog").expect("AnimatedSogSplats3D:asog");
    let mut files = vec![(
        "fixture".to_owned(),
        fixture.as_primitive::<UInt8Type>().values().to_vec(),
    )];
    if let Ok(path) = std::env::var("GSPLAT_ASOG_FILE") {
        files.push((
            path.clone(),
            std::fs::read(&path).expect("GSPLAT_ASOG_FILE"),
        ));
    }
    for (name, bytes) in files {
        let parallel =
            gsplat_core::asog::decode_parts(&bytes, ASOG_ACTIVE_THRESHOLD).expect("decode_parts");
        let serial =
            gsplat_core::asog::decode_parts_serial(&bytes, ASOG_ACTIVE_THRESHOLD).expect("serial");
        let (a, b) = (parts_hash(&parallel), parts_hash(&serial));
        println!(
            "{name}: {} rows, parallel {a:016x}, serial {b:016x}",
            parallel.quantization.count
        );
        assert_eq!(a, b, "{name}: parallel and serial inflate differ");
    }
}

/// `g` codes beyond i32 (a clip whose angular speed nearly reaches zero) reach the GPU as f32 bits: its motion rows
/// match the CPU oracle for a row whose five `g` codes are 3 x `i32::MAX`.
#[test]
#[ignore = "integration: GPU"]
fn gpu_load_path_reads_g_codes_beyond_i32() {
    let f = fixture();
    let bytes =
        row(&f.all["/asog"], "AnimatedSogSplats3D", "asog").expect("AnimatedSogSplats3D:asog");
    let mut parts = gsplat_core::asog::decode_parts(
        bytes.as_primitive::<UInt8Type>().values(),
        ASOG_ACTIVE_THRESHOLD,
    )
    .expect("decode_parts");
    let wide = (f64::from(i32::MAX) * 3.0) as f32;
    parts.motion_codes[15..20].fill(wide.to_bits().cast_signed());
    let load = gsplat_core::test_utils::asog_gpu_load(&parts, &[]);
    // The packed row: opacity, mu, sigma, flags, anchor (3), omega (9), g (5).
    let (gpu, cpu) = (&load.gpu_rows[16..21], &load.cpu_rows[16..21]);
    assert!(
        cpu.iter().all(|v| v.is_finite()) && cpu.iter().any(|&v| v != 0.0),
        "row 0 g: {cpu:?}"
    );
    for (a, b) in gpu.iter().zip(cpu) {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs(),
            "row 0 g: gpu {gpu:?}, cpu {cpu:?}"
        );
    }
}

/// Load path: the GPU de-quantises the fixture (`sog_dequant.wgsl` + `asog_motion.wgsl`) and evaluates it
/// (`trigs_deform.wgsl`); its motion rows match the CPU oracle `AsogParts::motion` (f64, rounded once) within f32
/// rounding, and the evaluated rows match the asog reference itself within the bounds of `evaluate_matches_the_reference`.
#[test]
#[ignore = "integration: GPU"]
fn gpu_load_path_matches_the_cpu_oracle_and_the_reference() {
    let f = fixture();
    let entity = &f.all["/asog"];
    let bytes = row(entity, "AnimatedSogSplats3D", "asog").expect("AnimatedSogSplats3D:asog");
    let parts = gsplat_core::asog::decode_parts(
        bytes.as_primitive::<UInt8Type>().values(),
        ASOG_ACTIVE_THRESHOLD,
    )
    .expect("decode_parts");
    let times: Vec<f32> = f.times.iter().map(|&t| t as f32).collect();
    let load = gsplat_core::test_utils::asog_gpu_load(&parts, &times);
    let (gpu, cpu, evaluated) = (load.gpu_rows, load.cpu_rows, load.evaluated);
    assert_eq!(gpu.len(), cpu.len());
    // Per column of the packed row: worst |gpu - cpu| / max(|cpu|, 1e-3 x the column's largest |cpu|).
    let stride = gpu.len() / parts.manifest.count;
    let mut worst = vec![0.0f64; stride];
    for (col, worst_col) in worst.iter_mut().enumerate() {
        let values = |v: &[f32]| {
            v.iter()
                .skip(col)
                .step_by(stride)
                .copied()
                .collect::<Vec<f32>>()
        };
        let (g, c) = (values(&gpu), values(&cpu));
        if col == 3 {
            assert!(
                g.iter().zip(&c).all(|(a, b)| a.to_bits() == b.to_bits()),
                "flags bits"
            );
            continue;
        }
        let floor = 1e-3 * c.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
        for (a, b) in g.iter().zip(&c) {
            let d =
                (f64::from(*a) - f64::from(*b)).abs() / f64::from(b.abs()).max(floor).max(1e-30);
            *worst_col = worst_col.max(d);
        }
    }
    println!(
        "GPU load: worst relative |gpu - cpu| per motion column (opacity, mu, sigma, flags, anchor, O, G): {:?}",
        worst.iter().map(|w| format!("{w:.1e}")).collect::<Vec<_>>()
    );
    assert!(worst.iter().all(|&w| w <= 1e-5), "motion de-quantisation");
    let opacities = &parts.motion().opacities;
    for (k, (t, (transforms, raw))) in f.times.iter().zip(&evaluated).enumerate() {
        let entity = format!("/expected/t{k}");
        let positions = expected(&f.all, &entity, "positions");
        let rotations = expected(&f.all, &entity, "rotations");
        let weights = expected(&f.all, &entity, "weights");
        let (mut centre, mut rotation, mut weight, mut mismatched, mut visible) =
            (0.0f64, 0.0f64, 0.0f64, 0, 0);
        for i in 0..parts.manifest.count {
            let culled = raw[i] <= CULLED_RAW_OPACITY;
            if culled != (weights[i] <= 0.0) {
                mismatched += 1;
                continue;
            }
            if culled {
                continue;
            }
            visible += 1;
            // Linear opacity: sigmoid(raw) against the reference opacity x weight (clamped as the kernel does).
            let alpha = 1.0 / (1.0 + (-f64::from(raw[i])).exp());
            let expected_alpha = (f64::from(opacities[i]) * weights[i]).clamp(1e-6, 1.0 - 1e-6);
            weight = weight.max((alpha - expected_alpha).abs());
            let r = &transforms[10 * i..10 * i + 10];
            let d = (0..3)
                .map(|a| (f64::from(r[a]) - positions[3 * i + a]).powi(2))
                .sum::<f64>()
                .sqrt();
            centre = centre.max(d);
            rotation = rotation.max(angle(
                [r[3], r[4], r[5], r[6]],
                &rotations[4 * i..4 * i + 4],
            ));
        }
        println!(
            "GPU load t={t}: {visible} visible, {mismatched} visibility mismatches, GPU vs reference worst centre {centre:.3e} m, rotation {rotation:.3e} rad, opacity {weight:.3e}"
        );
        assert_eq!(mismatched, 0, "t={t}");
        assert!(
            centre <= 1e-5 && rotation <= 1e-5 && weight <= 1e-5,
            "t={t}"
        );
    }
}

/// Decode + per-time evaluate cost of a whole `.asog` (`GSPLAT_ASOG_FILE`); run with `--release -- --ignored`.
#[test]
#[ignore = "needs GSPLAT_ASOG_FILE; timing"]
fn full_file_timing() {
    // Not a contract: the GPU lane (`--ignored`) runs this target too, so it passes without its file.
    let Ok(path) = std::env::var("GSPLAT_ASOG_FILE") else {
        println!("full_file_timing: GSPLAT_ASOG_FILE is unset; nothing timed");
        return;
    };
    let bytes = std::fs::read(&path).expect("read .asog");
    let start = std::time::Instant::now();
    let asog = decode(&bytes, ASOG_ACTIVE_THRESHOLD).expect("decode");
    let decode_ms = start.elapsed().as_secs_f64() * 1e3;
    let m = &asog.motion;
    let fps = asog.manifest.timeline.frame_rate;
    let mut out = gsplat_core::Splats::default();
    m.evaluate_into(&asog.canonical, 0.0, &mut out);
    let mut ms = Vec::new();
    for frame in [0.0, 30.0, 60.0, 90.0, 120.0] {
        let start = std::time::Instant::now();
        m.evaluate_into(&asog.canonical, (frame / fps) as f32, &mut out);
        ms.push(start.elapsed().as_secs_f64() * 1e3);
        let visible = out
            .raw_opacities
            .iter()
            .filter(|&&o| o > CULLED_RAW_OPACITY)
            .count();
        println!("frame {frame}: {visible} rows with weight > 0");
    }
    println!(
        "{path}: {} splats, SH degree {}, decode {decode_ms:.1} ms, evaluate per time {ms:.2?} ms",
        m.len(),
        asog.canonical.sh_degree
    );
}
