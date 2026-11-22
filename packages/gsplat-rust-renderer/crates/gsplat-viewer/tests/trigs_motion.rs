//! `gsplat_core::trigs` against the trainer: `tests/fixtures/trigs_motion_1000.rrd` holds 1,000 rows of the
//! 4DAnyone run C checkpoint (50 forced static, 100 with zero or tiny rotation) and `TrigsTrainer._deform` at five
//! times (written by the open4dgs scratch script `k2/dump_fixture.py`, CPU torch f32).
use gsplat_core::native::NativeSplats;
use gsplat_core::trigs::{TrigsMotion, deform_visible};
mod common;

use common::{chunks, row};
use re_viewer::external::re_chunk::Chunk;
use re_viewer::external::re_sdk_types::external::arrow::array::AsArray as _;
use re_viewer::external::re_sdk_types::external::arrow::datatypes::{
    Float32Type, Float64Type, UInt32Type,
};

fn floats(chunks: &[Chunk], archetype: &'static str, field: &str) -> Vec<f32> {
    let values =
        row(chunks, archetype, field).unwrap_or_else(|| panic!("{archetype}:{field} missing"));
    if let Some(f) = values.as_primitive_opt::<Float32Type>() {
        f.values().to_vec()
    } else {
        values
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect()
    }
}

fn group<const N: usize>(flat: &[f32]) -> Vec<[f32; N]> {
    flat.as_chunks::<N>().0.to_vec()
}

/// Canonical splats and motion of one `GaussianSplats3D` + `TrigsMotion3D` entity, owned.
struct Model {
    centers: Vec<[f32; 3]>,
    quaternions: Vec<[f32; 4]>,
    scales: Vec<[f32; 3]>,
    colors: Vec<u32>,
    scalars: [Vec<f32>; 4],
    twists: [Vec<[f32; 6]>; 4],
    anchors: Vec<[f32; 3]>,
    active_threshold: f32,
}

impl Model {
    fn read(model: &[Chunk], with_appearance: bool) -> Self {
        let m = |field| floats(model, "TrigsMotion3D", field);
        let splat = |field| floats(model, "GaussianSplats3D", field);
        Self {
            centers: group::<3>(&splat("centers")),
            quaternions: group::<4>(&splat("quaternions")),
            scales: if with_appearance {
                group::<3>(&splat("scales"))
            } else {
                Vec::new()
            },
            colors: if with_appearance {
                row(model, "GaussianSplats3D", "colors")
                    .expect("GaussianSplats3D:colors")
                    .as_primitive::<UInt32Type>()
                    .values()
                    .to_vec()
            } else {
                Vec::new()
            },
            scalars: ["opacities", "times", "log_durations", "is_static"].map(m),
            twists: ["twist_base", "twist_c0", "twist_c1", "twist_c2"].map(|f| group::<6>(&m(f))),
            anchors: group::<3>(&m("anchors")),
            active_threshold: m("active_threshold")[0],
        }
    }

    fn motion(&self) -> TrigsMotion<'_> {
        TrigsMotion {
            opacities: &self.scalars[0],
            times: &self.scalars[1],
            log_durations: &self.scalars[2],
            is_static: &self.scalars[3],
            twist_base: &self.twists[0],
            twist_c0: &self.twists[1],
            twist_c1: &self.twists[2],
            twist_c2: &self.twists[3],
            anchors: &self.anchors,
            active_threshold: self.active_threshold,
        }
    }
}

#[test]
fn deform_matches_trainer() {
    let chunks = chunks(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/trigs_motion_1000.rrd"
    ));
    let model = Model::read(&chunks["/model"], false);
    let motion = model.motion();
    assert_eq!(model.centers.len(), 1000);
    assert!(motion.is_consistent(1000));
    let mut worst = [0.0f32; 3];
    for k in 0..5 {
        let expected = &chunks[&format!("/expected/{k}")];
        let e = |field| floats(expected, "Expected", field);
        let t = e("t")[0];
        let (ec, eq, eo) = (
            group::<3>(&e("centers")),
            group::<4>(&e("quaternions")),
            e("opacities"),
        );
        for i in 0..1000 {
            let d = motion.deform_row(i, model.centers[i], model.quaternions[i], t);
            let dc = (0..3)
                .map(|j| (d.center[j] - ec[i][j]).abs())
                .fold(0.0, f32::max);
            let dq = (0..4)
                .map(|j| (d.quaternion[j] - eq[i][j]).abs())
                .fold(0.0, f32::max);
            let dop = (d.opacity - eo[i]).abs();
            worst = [worst[0].max(dc), worst[1].max(dq), worst[2].max(dop)];
            assert!(
                dc < 1e-5 && dq < 1e-5 && dop < 1e-5,
                "row {i} t {t}: centre {:?} vs {:?}, quat {:?} vs {:?}, opacity {} vs {}",
                d.center,
                ec[i],
                d.quaternion,
                eq[i],
                d.opacity,
                eo[i]
            );
        }
    }
    eprintln!(
        "max abs error: centre {:e} m, quaternion {:e}, opacity {:e}",
        worst[0], worst[1], worst[2]
    );
}

/// CPU cost per time change on a real compact recording (`TRIGS_COMPACT_RRD`, entity /world/person/splats):
/// the motion pre-pass (gamma cull + deform + gather into native arrays) and the `to_core` conversion that
/// per-frame rows also pay. Run with `--release --ignored --nocapture`.
#[test]
#[ignore = "needs TRIGS_COMPACT_RRD; timing"]
fn prepass_timing() {
    let path = std::env::var("TRIGS_COMPACT_RRD").expect("TRIGS_COMPACT_RRD");
    let chunks = chunks(&path);
    let model = Model::read(&chunks["/world/person/splats"], true);
    let motion = model.motion();
    let fps = floats(&chunks["/world/person/splats"], "TrigsMotion3D", "fps")[0];
    let (mut prepass, mut convert, mut counts) = (Vec::new(), Vec::new(), Vec::new());
    for frame in 0..121 {
        let t = frame as f32 / fps;
        let start = std::time::Instant::now();
        let (rows, values) =
            deform_visible(&model.centers, &model.quaternions, &motion, t, 1.0 / 255.0);
        let centers: Vec<[f32; 3]> = values.iter().map(|v| v.center).collect();
        let quaternions: Vec<[f32; 4]> = values.iter().map(|v| v.quaternion).collect();
        let scales: Vec<[f32; 3]> = rows.iter().map(|&r| model.scales[r as usize]).collect();
        let colors: Vec<u32> = rows
            .iter()
            .zip(&values)
            .map(|(&r, v)| {
                (model.colors[r as usize] & 0xFFFF_FF00)
                    | (v.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
            })
            .collect();
        let mid = std::time::Instant::now();
        let native = NativeSplats {
            centers: &centers,
            scales: &scales,
            quaternions: &quaternions,
            colors: &colors,
            sh: &[],
            degree: 0,
        };
        let core = std::hint::black_box(native.to_core());
        let end = std::time::Instant::now();
        assert_eq!(core.transforms.len(), rows.len());
        prepass.push((mid - start).as_secs_f64() * 1e3);
        convert.push((end - mid).as_secs_f64() * 1e3);
        counts.push(rows.len());
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    counts.sort_unstable();
    eprintln!(
        "rows {} | visible per frame min {} median {} max {} | pre-pass ms median {:.2} max {:.2} | to_core ms median {:.2} max {:.2}",
        model.centers.len(),
        counts[0],
        counts[counts.len() / 2],
        counts[counts.len() - 1],
        median(&mut prepass),
        prepass.iter().copied().fold(0.0, f64::max),
        median(&mut convert),
        convert.iter().copied().fold(0.0, f64::max),
    );
}
