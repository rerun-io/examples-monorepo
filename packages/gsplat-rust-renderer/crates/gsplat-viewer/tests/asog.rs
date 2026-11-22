//! `gsplat_core::asog` + the `.asog` motion form against the asog reference decoder (`asog.decode.read` and
//! `Asog.evaluate`, f64): `tests/fixtures/asog_reference.rrd` holds 2,048 rows of the run C band `.asog` (every row
//! with near-zero omega and every rank-deficient / inexact pinv row, then random rows) as `AnimatedSogSplats3D`,
//! and the reference decode under `/expected` (written by `tools/asog_reference_fixture.py` in the open4dgs env).
//!
//! E1: Rust decode + evaluate == reference (centre <= 1e-5 m, rotation <= 1e-5 rad, weight <= 1e-6, same visible
//! set). E2: on well-conditioned rows the twist form fed the pinv translation equals the `.asog` form.
use gsplat_core::asog::decode;
use gsplat_core::trigs::{ASOG_ACTIVE_THRESHOLD, CULLED_RAW_OPACITY, TrigsMotion};
use re_viewer::external::re_chunk::{Chunk, ComponentIdentifier};
use re_viewer::external::re_sdk_types::external::arrow::array::{
    Array as _, ArrayRef, AsArray as _, FixedSizeListArray,
};
use re_viewer::external::re_sdk_types::external::arrow::datatypes::{
    Float32Type, Float64Type, UInt8Type,
};
use std::collections::BTreeMap;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/asog_reference.rrd"
);

/// Every chunk of an .rrd by entity path.
fn chunks(path: &str) -> BTreeMap<String, Vec<Chunk>> {
    let file = std::io::BufReader::new(std::fs::File::open(path).expect("open rrd"));
    let mut out: BTreeMap<String, Vec<Chunk>> = BTreeMap::new();
    for msg in re_log_encoding::DecoderApp::decode_eager(file).expect("rrd header") {
        if let re_log_msg::LogMsg::ArrowMsg(_, arrow) = msg.expect("rrd message") {
            let chunk = Chunk::from_arrow_msg(&arrow).expect("chunk");
            out.entry(chunk.entity_path().to_string())
                .or_default()
                .push(chunk);
        }
    }
    out
}

/// Row 0 of `archetype:field` (fixed-size lists flattened).
fn row(chunks: &[Chunk], archetype: &'static str, field: &str) -> ArrayRef {
    let id = ComponentIdentifier::from_archetype_field(archetype.into(), field);
    let array = chunks
        .iter()
        .find_map(|c| c.raw_component_array(id))
        .unwrap_or_else(|| panic!("{archetype}:{field} missing"));
    let row = match array.as_list_opt::<i32>() {
        Some(list) => list.value(0),
        None => array.clone(),
    };
    match row.as_any().downcast_ref::<FixedSizeListArray>() {
        Some(fixed) => fixed.values().clone(),
        None => row,
    }
}

/// `Expected:<field>` as f64 (stored as f32 or f64).
fn expected(all: &BTreeMap<String, Vec<Chunk>>, entity: &str, field: &str) -> Vec<f64> {
    let values = row(&all[entity], "Expected", field);
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
    let bytes = row(entity, "AnimatedSogSplats3D", "asog");
    let threshold = row(entity, "AnimatedSogSplats3D", "active_threshold");
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
fn e1_evaluate_matches_the_reference() {
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
            "E1 t={t}: {visible} visible, {mismatched} visibility mismatches, worst centre {centre:.3e} m, rotation {rotation:.3e} rad, weight {weight:.3e}"
        );
        assert_eq!(mismatched, 0, "t={t}");
        assert!(
            centre <= 1e-5 && rotation <= 1e-5 && weight <= 1e-6,
            "t={t}"
        );
    }
}

#[test]
fn e2_twist_form_with_pinv_translation_equals_the_asog_form() {
    let f = fixture();
    let m = &f.asog.motion;
    let canonical = &f.asog.canonical;
    let nu = expected(&f.all, "/expected/twist", "nu");
    let well = expected(&f.all, "/expected/twist", "well_conditioned");
    let n = m.len();
    let control = |k: usize| -> Vec<[f32; 6]> {
        (0..n)
            .map(|i| {
                let v = &nu[9 * i + 3 * k..9 * i + 3 * k + 3];
                let o = m.omega[i][k];
                [v[0] as f32, v[1] as f32, v[2] as f32, o[0], o[1], o[2]]
            })
            .collect()
    };
    let (c0, c1, c2) = (control(0), control(1), control(2));
    let log_durations: Vec<f32> = m.sigmas.iter().map(|s| s.ln()).collect();
    let base = vec![[0.0f32; 6]; n];
    let is_static: Vec<f32> = m
        .is_static
        .iter()
        .map(|&s| if s { 1.0 } else { 0.0 })
        .collect();
    let twist = TrigsMotion {
        opacities: &m.opacities,
        times: &m.times,
        log_durations: &log_durations,
        twist_base: &base,
        twist_c0: &c0,
        twist_c1: &c1,
        twist_c2: &c2,
        anchors: &m.anchors,
        is_static: &is_static,
        active_threshold: m.active_threshold,
    };
    let rows: Vec<usize> = (0..n).filter(|&i| well[i] > 0.5).collect();
    assert!(rows.len() > 1000);
    for &t in &f.times {
        let t = t as f32;
        let clips = m.clip_states(t);
        let (mut centre, mut rotation, mut opacity, mut checked) = (0.0f64, 0.0f64, 0.0f32, 0);
        for &i in &rows {
            let tr = &canonical.transforms[i];
            let a = m.row(i, tr, &clips);
            if a.weight <= 0.0 {
                continue;
            }
            checked += 1;
            let b = twist.deform_row(i, [tr[0], tr[1], tr[2]], [tr[4], tr[5], tr[6], tr[3]], t);
            let d = (0..3)
                .map(|k| f64::from(a.center[k] - b.center[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            centre = centre.max(d);
            let [x, y, z, w] = b.quaternion.map(f64::from);
            rotation = rotation.max(angle(a.quaternion, &[w, x, y, z]));
            opacity = opacity.max((m.opacities[i] * a.weight - b.opacity).abs());
        }
        println!(
            "E2 t={t}: {checked} rows, worst centre {centre:.3e} m, rotation {rotation:.3e} rad, opacity {opacity:.3e}"
        );
        assert!(
            centre <= 1e-5 && rotation <= 1e-5 && opacity <= 1e-6,
            "t={t}"
        );
    }
}

/// Decode + per-time evaluate cost of a whole `.asog` (`GSPLAT_ASOG_FILE`); run with `--release -- --ignored`.
#[test]
#[ignore = "needs GSPLAT_ASOG_FILE; timing"]
fn full_file_timing() {
    let path = std::env::var("GSPLAT_ASOG_FILE").expect("GSPLAT_ASOG_FILE");
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
