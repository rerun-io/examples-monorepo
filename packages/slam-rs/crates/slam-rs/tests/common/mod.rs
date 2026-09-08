//! What more than one integration test needs.
//!
//! Integration tests are separate crates, so a `tests/common/mod.rs` declared
//! with `mod common;` is the only way to share code between them. Each test
//! binary compiles its own copy and uses part of it, which is why the module
//! allows dead code: the alternative is a `cfg` per item per test. The
//! `unwrap`/`expect` allows are the module's own rather than each including
//! binary's, because a fixture that does not parse is a broken checkout and
//! panicking on it is the report.

#![allow(dead_code, clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use nalgebra::{SMatrix, Vector3};
use slam_rs::calib::{Calibration, CameraModel};
use slam_rs::config::VioConfig;
use slam_rs::image::ImageU16;
use slam_rs::lie::{LieScalar, Se3};

const MSDMI: &str = include_str!("../fixtures/msdmi_calib.json");
const MSDMG: &str = include_str!("../fixtures/msdmg_calib.json");
const ROBOCAP: &str = include_str!("../fixtures/robocap-basalt-calib.json");

// ── the fixture directory and the two files every VIO lane reads ───────────

/// `crates/slam-rs/tests/fixtures`.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// `packages/slam-rs/configs`, the shipped VIO configs.
///
/// The crate reads the package's files rather than a copy of its own: these are
/// the ones `reference_segments.toml` names and the C++ reference runs loaded,
/// so a lane that drifted from them would compare against a config nothing ran.
pub fn configs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

/// The MSDMI config, which every VIO lane and the whole-pipeline test share.
pub fn config() -> VioConfig {
    VioConfig::from_json_str(&std::fs::read_to_string(configs().join("msdmi_config.json")).unwrap())
        .unwrap()
}

/// The MSDMI calibration, always `f64`: the estimator casts it to its own
/// scalar, so a lane that runs `f32` still reads the file's doubles.
pub fn calibration() -> Calibration<f64> {
    Calibration::from_json_str(
        &std::fs::read_to_string(fixtures().join("msdmi_calib.json")).unwrap(),
    )
    .unwrap()
}

/// One of the three shipped calibrations by name, compiled in.
///
/// Read at compile time rather than through [`fixtures`], because five test
/// binaries want the *text* — `Calibration::from_json_str` is what several of
/// them are testing — and each had its own `include_str!` of the same file.
///
/// # Panics
///
/// On a name that is not one of the three.
pub fn calibration_text(name: &str) -> &'static str {
    match name {
        "msdmi" => MSDMI,
        "msdmg" => MSDMG,
        "robocap" => ROBOCAP,
        other => panic!("no calibration fixture named {other}"),
    }
}

/// A fixed-size matrix flattened **row major**, which is how every C++ dump
/// prints one.
pub fn row_major<const R: usize, const C: usize, S: LieScalar>(m: &SMatrix<S, R, C>) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(R * C);
    for r in 0..R {
        for c in 0..C {
            out.push(m[(r, c)].to_f64());
        }
    }
    out
}

// ── the PGM framesets ─────────────────────────────────────────────

/// One camera's image as it sits in a PGM: the raw 8-bit raster and its shape.
///
/// The raster, not an `ImageU16`: `flow_parity` widens it the way basalt's
/// camera source does, while `vio_parity` hands the bytes straight to
/// [`slam_rs::ImageView`], and one of those wrapping the other is the only
/// difference between them.
pub struct Pgm {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// `frame_<NNN>_cam<C>.pgm` under `directory`, in `tools/dump_flow.cpp`'s
/// layout.
pub fn read_pgm(directory: &Path, frame: usize, camera: usize) -> Pgm {
    let path: PathBuf = directory.join(format!("frame_{frame:03}_cam{camera}.pgm"));
    let bytes: Vec<u8> = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

    // "P5\n<w> <h>\n255\n" then the raster; the writer emits exactly that.
    let mut fields: Vec<usize> = Vec::new();
    let mut cursor: usize = 2;
    while fields.len() < 3 {
        while bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let start: usize = cursor;
        while !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        fields.push(
            std::str::from_utf8(&bytes[start..cursor])
                .unwrap()
                .parse()
                .unwrap(),
        );
    }
    cursor += 1;
    assert_eq!(fields[2], 255, "{} is not an 8-bit PGM", path.display());
    Pgm {
        width: fields[0],
        height: fields[1],
        pixels: bytes[cursor..].to_vec(),
    }
}

/// How many consecutive framesets `directory` covers, up to `limit`: a
/// frameset counts only when every camera's PGM is there.
pub fn available_framesets(directory: &Path, cameras: usize, limit: usize) -> usize {
    (0..limit)
        .take_while(|frame| {
            (0..cameras).all(|camera| {
                directory
                    .join(format!("frame_{frame:03}_cam{camera}.pgm"))
                    .exists()
            })
        })
        .count()
}

/// A textured 8-bit field with fine detail, so FAST has plenty to find: the
/// smooth plane-wave texture above gives almost no corners.
///
/// One LCG plus a `sin`/`cos` wave, so it is the same field on every machine.
pub fn cornered_bytes(width: usize, height: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; width * height];
    let mut state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let wave: i32 = ((x as f64 / 11.0).sin() * 60.0 + (y as f64 / 7.0).cos() * 50.0) as i32;
            let noise: i32 = (state >> 24) as i32 / 4;
            out[y * width + x] = (128 + wave + noise).clamp(0, 255) as u8;
        }
    }
    out
}

// ---- frontend fixtures (S25 FRONT) ----------------------------------------
//
// The frontend's synthetic rig and its two frames. `flow_rig` and `flow_config`
// were byte-identical in `tests/frame_allocations.rs` and in
// `src/frontend/flow.rs`'s test module, and the allocation test only measures
// the frame the flow tests describe while the two stay in step.

use slam_rs::calib::{CalibAccelBias, CalibGyroBias, PinholeParams};
use slam_rs::config::MatchingGuessType;
use slam_rs::lie::So3;
use slam_rs::pyramid::{CpuPyramidBuilder, PyramidBuilder, PyramidU16};
use std::collections::BTreeMap;

/// The synthetic rig's frame size, shared by `flow_rig` and `dotted_image`.
pub const FLOW_WIDTH: usize = 200;
/// The synthetic rig's frame height.
pub const FLOW_HEIGHT: usize = 200;

/// `count` identical pinhole cameras 5 cm apart along `x`, all seeing a
/// `FLOW_WIDTH` x `FLOW_HEIGHT` frame.
pub fn flow_rig(count: usize) -> Calibration<f64> {
    let intrinsics: CameraModel<f64> = CameraModel::Pinhole(PinholeParams {
        fx: 180.0,
        fy: 180.0,
        cx: FLOW_WIDTH as f64 / 2.0,
        cy: FLOW_HEIGHT as f64 / 2.0,
    });
    Calibration {
        t_i_c: (0..count)
            .map(|index| Se3::new(So3::identity(), Vector3::new(0.05 * index as f64, 0.0, 0.0)))
            .collect(),
        intrinsics: vec![intrinsics; count],
        resolution: vec![[FLOW_WIDTH as u32, FLOW_HEIGHT as u32]; count],
        vignette: Vec::new(),
        cam_time_offset_ns: 0,
        calib_accel_bias: CalibAccelBias::default(),
        calib_gyro_bias: CalibGyroBias::default(),
        imu_update_rate: 200.0,
        gyro_noise_std: Vector3::repeat(1e-4),
        accel_noise_std: Vector3::repeat(1e-3),
        gyro_bias_std: Vector3::repeat(1e-5),
        accel_bias_std: Vector3::repeat(1e-4),
        unknown: BTreeMap::new(),
    }
}

/// basalt's shipped configuration, with the matching guess set to the same
/// pixel so that `flow_rig`'s cameras — which see identical frames — really do
/// match.
pub fn flow_config() -> VioConfig {
    VioConfig {
        optical_flow_matching_guess_type: MatchingGuessType::SamePixel,
        ..VioConfig::default()
    }
}

/// Bright 5x5 squares on a regular lattice, the whole frame shifted by `shift`
/// pixels: four strong FAST corners each, and enough texture in between for the
/// KLT to follow them.
pub fn dotted_image(shift: i32) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(FLOW_WIDTH, FLOW_HEIGHT).expect("a valid geometry");
    for y in 0..FLOW_HEIGHT {
        for x in 0..FLOW_WIDTH {
            let fx: f64 = f64::from(x as i32 - shift);
            let fy: f64 = f64::from(y as i32);
            let base: f64 = 18_000.0 + 5_000.0 * (fx * 0.07).sin() * (fy * 0.05).cos();
            image.set(x, y, base as u16);
        }
    }
    let mut cy: usize = 14;
    while cy + 5 < FLOW_HEIGHT {
        let mut cx: usize = 14;
        while cx + 5 < FLOW_WIDTH {
            for dy in 0..5 {
                for dx in 0..5 {
                    let x: i32 = (cx + dx) as i32 + shift;
                    if x >= 0 && (x as usize) < FLOW_WIDTH {
                        image.set(x as usize, cy + dy, 0xF000);
                    }
                }
            }
            cx += 17;
        }
        cy += 17;
    }
    image
}

/// A CPU pyramid of `image` with `levels` halvings on top of level 0.
pub fn pyramid_of(image: &ImageU16, levels: usize) -> PyramidU16 {
    let mut pyramid: PyramidU16 =
        PyramidU16::with_capacity(image.width(), image.height(), levels).expect("a valid geometry");
    CpuPyramidBuilder::new()
        .build(0, image, &mut pyramid)
        .expect("the geometry the pyramid was allocated for");
    pyramid
}
