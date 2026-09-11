//! Shared integration-test inputs and mathematical references.
//! Each binary uses a subset; item-level dead-code allowances name a consumer.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use nalgebra::{DMatrix, DVector, Vector2, Vector3, Vector6};
use serde::Deserialize;
use slam_rs::calib::{Calibration, CameraModel, Kb4Params};
use slam_rs::config::VioConfig;
use slam_rs::frontend::tracker::PointsSoA;
use slam_rs::image::ImageU16;
use slam_rs::lie::Se3;

#[allow(
    dead_code,
    reason = "used by camera_jacobians; other binaries compile a subset"
)]
const MSDMI: &str = include_str!("../fixtures/msdmi_calib.json");
#[allow(
    dead_code,
    reason = "used by camera_jacobians; other binaries compile a subset"
)]
const MSDMG: &str = include_str!("../fixtures/msdmg_calib.json");
#[allow(
    dead_code,
    reason = "used by camera_jacobians; other binaries compile a subset"
)]
const ROBOCAP: &str = include_str!("../fixtures/robocap_calib.json");

// ── the fixture directory and the two files every VIO lane reads ───────────

/// `crates/slam-rs/tests/fixtures`.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// `packages/slam-rs/configs`, the shipped VIO configs.
///
/// The crate reads the package's files rather than a copy of its own: these are
/// the same files used by the Python entry points.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub fn configs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

/// The MSDMI config, which every VIO lane and the whole-pipeline test share.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub fn config() -> VioConfig {
    VioConfig::from_json_str(&std::fs::read_to_string(configs().join("msdmi_config.json")).unwrap())
        .unwrap()
}

/// The MSDMI calibration, always `f64`: the estimator casts it to its own
/// scalar, so a lane that runs `f32` still reads the file's doubles.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
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
#[allow(
    dead_code,
    reason = "used by camera_jacobians; other binaries compile a subset"
)]
pub fn calibration_text(name: &str) -> &'static str {
    match name {
        "msdmi" => MSDMI,
        "msdmg" => MSDMG,
        "robocap" => ROBOCAP,
        other => panic!("no calibration fixture named {other}"),
    }
}

/// Precision selected by the whole-clip runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    dead_code,
    reason = "used by full_clip; other binaries compile a subset"
)]
pub enum ClipScalar {
    F32,
    F64,
}

#[allow(
    dead_code,
    reason = "used by full_clip; other binaries compile a subset"
)]
impl ClipScalar {
    /// `SLAM_RS_CLIP_SCALAR`, falling back to
    /// `default`. `f32` and `float` mean the same thing, as do `f64` and
    /// `double`.
    ///
    /// # Panics
    ///
    /// On a value that is none of the four.
    pub fn from_env(default: Self) -> Self {
        let Ok(value) = std::env::var("SLAM_RS_CLIP_SCALAR") else {
            return default;
        };
        match value.as_str() {
            "f32" | "float" => Self::F32,
            "f64" | "double" => Self::F64,
            other => panic!("SLAM_RS_CLIP_SCALAR is f32/float or f64/double, not {other}"),
        }
    }

    /// `"f32"` or `"f64"`, which is what a written CSV's name carries.
    pub fn rust_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

/// Clip metadata written beside the input pixels by `tests/tools/dump_clip.py`.
#[derive(Debug, Deserialize)]
#[allow(
    dead_code,
    reason = "used by full_clip; other binaries compile a subset"
)]
pub struct Clip {
    pub segment_id: String,
    pub dataset_name: String,
    /// Added to a frameset timestamp to reach the absolute device clock.
    pub capture_start_time_ns: i64,
    /// `catalog` (the values the C++ was pushed) or `fixture` (the fork file's
    /// doubles).
    pub calibration_source: String,
    pub num_cameras: usize,
    pub framesets: usize,
    pub frame_t_ns: Vec<i64>,
    pub imu_samples: usize,
}

#[allow(
    dead_code,
    reason = "used by full_clip; other binaries compile a subset"
)]
impl Clip {
    /// `clip.json` from a directory `dump_clip.py` wrote.
    ///
    /// # Panics
    ///
    /// When the file is missing or does not parse.
    pub fn read(directory: &Path) -> Self {
        let path: PathBuf = directory.join("clip.json");
        let text: String = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        serde_json::from_str(&text).unwrap()
    }
}

/// The VIO config of the device a clip was captured on, from the committed
/// fixtures, or the file `SLAM_RS_CLIP_CONFIG` names.
///
/// The override exists because a config field is an input like any other: the
/// reference runs load `data/msd/msd*_config.json`, and a lane that builds a
/// default config instead differs from them by whatever that file overrides.
///
/// # Panics
///
/// On a dataset with no pinned config.
#[allow(
    dead_code,
    reason = "used by full_clip; other binaries compile a subset"
)]
pub fn config_for(dataset_name: &str) -> VioConfig {
    if let Some(path) = std::env::var_os("SLAM_RS_CLIP_CONFIG") {
        return VioConfig::from_json_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    }
    let file: &str = match dataset_name {
        "msd-index" => "msdmi_config.json",
        "msd-g2" => "msdmg_config.json",
        other => panic!("no VIO config is pinned for {other}"),
    };
    VioConfig::from_json_str(&std::fs::read_to_string(configs().join(file)).unwrap()).unwrap()
}

// ── the PGM framesets ─────────────────────────────────────────────

/// One camera's image as it sits in a PGM: the raw 8-bit raster and its shape.
///
/// Used as bytes by VIO and widened to u16 by frontend tests.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub struct Pgm {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// `frame_<NNN>_cam<C>.pgm` under `directory`, in `tools/dump_flow.cpp`'s
/// layout.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
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

/// One committed MIO10 frameset as the frontend and the detector take it:
/// 960x960, the msd-index rig's geometry, each PGM byte in the high half of a
/// `u16`.
///
/// The GPU exactness gate and the host-seam bench drive the same three
/// framesets, so the fixture location and the widening live here rather than
/// once per binary. They stay separate binaries: D72 wants the shared CubeCL
/// pool isolated per test process.
#[allow(
    dead_code,
    reason = "used by gpu_detect; other binaries compile a subset"
)]
pub fn mio10_frame(frame: usize, camera: usize) -> ImageU16 {
    let pgm: Pgm = read_pgm(&fixtures().join("flow/frames"), frame, camera);
    let mut image: ImageU16 = ImageU16::zeros(pgm.width, pgm.height).unwrap();
    for y in 0..pgm.height {
        for x in 0..pgm.width {
            image.set(x, y, u16::from(pgm.pixels[y * pgm.width + x]) << 8);
        }
    }
    image
}

/// How many consecutive framesets `directory` covers, up to `limit`: a
/// frameset counts only when every camera's PGM is there.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
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

#[derive(Debug, Deserialize)]
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub struct ImuRow {
    pub t_ns: i64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
}

#[derive(Debug, Deserialize)]
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
struct ImuFixture {
    imu: Vec<ImuRow>,
}

/// The 1,077 uncalibrated samples of the window, in capture order.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub static IMU: LazyLock<Vec<ImuRow>> = LazyLock::new(|| {
    let path: PathBuf = fixtures().join("vio/imu.json");
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let fixture: ImuFixture =
        serde_json::from_str(&text).expect("imu.json does not match the expected shape");
    fixture.imu
});

/// `KannalaBrandtCamera4<Scalar>::getTestProjections()[0]`
/// (`basalt-headers/include/basalt/camera/kannala_brandt_camera4.hpp:487-495`),
/// which is what `test_linearization.cpp:19` puts in both camera slots.
#[allow(
    dead_code,
    reason = "used by linearize_reference; other binaries compile a subset"
)]
pub const KB4_TEST_PROJECTION: [f64; 8] = [
    379.045,
    379.008,
    505.512,
    509.969,
    0.00693023,
    -0.0013828,
    -0.000272596,
    -0.000452646,
];

/// xorshift64*, standing in for Eigen's `Random()`.
///
/// A Rust test that flakes is worse than one that is merely differently
/// arbitrary, so nothing here draws from the system generator.
#[allow(
    dead_code,
    reason = "used by linearize_reference; other binaries compile a subset"
)]
pub struct Rng(u64);

#[allow(
    dead_code,
    reason = "used by linearize_reference; other binaries compile a subset"
)]
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x: u64 = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on `[-1, 1]`, like `Eigen::Matrix::Random()`.
    pub fn symmetric(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    pub fn vector3(&mut self) -> Vector3<f64> {
        Vector3::new(self.symmetric(), self.symmetric(), self.symmetric())
    }

    pub fn vector6(&mut self) -> Vector6<f64> {
        Vector6::from_iterator((0..6).map(|_| self.symmetric()))
    }
}

/// The calibration `get_vo_estimator` builds (`test_linearization.cpp:15-22`):
/// two camera-to-IMU transforms that are small perturbations of identity, and
/// [`KB4_TEST_PROJECTION`] in both camera slots.
///
/// The file supplies only the fields the estimator never touches; everything
/// read is overwritten here.
#[allow(
    dead_code,
    reason = "used by linearize_reference; other binaries compile a subset"
)]
pub fn test_calibration(rng: &mut Rng) -> Calibration<f64> {
    let mut calib: Calibration<f64> = Calibration::from_json_str(MSDMI).unwrap();
    calib.t_i_c = (0..2)
        .map(|_| Se3::<f64>::exp_decoupled(&(rng.vector6() / 100.0)))
        .collect();
    let p: [f64; 8] = KB4_TEST_PROJECTION;
    calib.intrinsics = vec![
        CameraModel::Kb4(Kb4Params {
            fx: p[0],
            fy: p[1],
            cx: p[2],
            cy: p[3],
            k1: p[4],
            k2: p[5],
            k3: p[6],
            k4: p[7],
        });
        2
    ];
    calib
}

/// `H_kk − H_km H_mm⁻¹ H_mk` and `b_k − H_km H_mm⁻¹ b_m`, written out
/// independently of anything under test.
///
/// The reference the square-root marginalization is checked against: the QR of
/// `marginalizeHelperSqrtToSqrt` never forms `JᵀJ`, so squaring its output and
/// comparing with this is the same argument `VoMargSqrtLinearizationTest` makes
/// about the linearization (`test_linearization.cpp:379-388`), one level up.
#[allow(
    dead_code,
    reason = "used by marg_window; other binaries compile a subset"
)]
pub fn dense_schur(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    keep: &[usize],
    marg: &[usize],
) -> (DMatrix<f64>, DVector<f64>) {
    let k: usize = keep.len();
    let m: usize = marg.len();
    let h_kk: DMatrix<f64> = DMatrix::from_fn(k, k, |i, j| h[(keep[i], keep[j])]);
    let h_km: DMatrix<f64> = DMatrix::from_fn(k, m, |i, j| h[(keep[i], marg[j])]);
    let h_mk: DMatrix<f64> = DMatrix::from_fn(m, k, |i, j| h[(marg[i], keep[j])]);
    let h_mm: DMatrix<f64> = DMatrix::from_fn(m, m, |i, j| h[(marg[i], marg[j])]);
    let b_k: DVector<f64> = DVector::from_fn(k, |i, _| b[keep[i]]);
    let b_m: DVector<f64> = DVector::from_fn(m, |i, _| b[marg[i]]);
    let h_mm_inv: DMatrix<f64> = h_mm.try_inverse().expect("the marginalized block inverts");
    let cross: DMatrix<f64> = &h_km * &h_mm_inv;
    (h_kk - &cross * h_mk, b_k - &cross * b_m)
}

// ── synthetic images and patch positions ───────────────────────────────────
//
// Two fields, shared because the tracker's unit tests, the GPU tolerance tests
// and the FAST model test all need the *same* pixels: a band-limited one where
// every patch is well conditioned, and a corner-rich one where FAST has plenty
// to find.

/// Twelve plane waves between 16 and 56 pixels, in fixed pseudo-random
/// directions and phases: band-limited, so a shift really does survive down a
/// pyramid and every patch's `H_se2` is well conditioned.
#[allow(
    dead_code,
    reason = "used by gpu_kernels; other binaries compile a subset"
)]
pub fn texture(x: f64, y: f64) -> f64 {
    const WAVES: [(f64, f64, f64); 12] = [
        (16.0, 0.031, 0.11),
        (19.0, 0.187, 0.37),
        (23.0, 0.311, 0.63),
        (27.0, 0.451, 0.05),
        (31.0, 0.077, 0.81),
        (35.0, 0.229, 0.29),
        (39.0, 0.383, 0.55),
        (43.0, 0.497, 0.73),
        (47.0, 0.143, 0.19),
        (51.0, 0.271, 0.91),
        (54.0, 0.419, 0.43),
        (56.0, 0.353, 0.67),
    ];
    let mut total: f64 = 0.0;
    for (wavelength, direction, phase) in WAVES {
        let angle: f64 = std::f64::consts::TAU * direction;
        let projection: f64 = x * angle.cos() + y * angle.sin();
        total += (std::f64::consts::TAU * (projection / wavelength + phase)).sin();
    }
    total / WAVES.len() as f64
}

/// [`texture`] rendered into a `u16` image, shifted by `(dx, dy)`.
#[allow(
    dead_code,
    reason = "used by gpu_kernels; other binaries compile a subset"
)]
pub fn textured_image(width: usize, height: usize, dx: f32, dy: f32) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(width, height).expect("a valid image geometry");
    for y in 0..height {
        for x in 0..width {
            let value: f64 = texture(x as f64 - f64::from(dx), y as f64 - f64::from(dy));
            let scaled: f64 = (value * 0.4 + 0.5) * 65535.0;
            image.set(x, y, scaled.clamp(0.0, 65535.0) as u16);
        }
    }
    image
}

/// A textured 8-bit field with fine detail, so FAST has plenty to find: the
/// smooth plane-wave texture above gives almost no corners.
///
/// One LCG plus a `sin`/`cos` wave, so it is the same field on every machine.
#[allow(
    dead_code,
    reason = "used by fast_model; other binaries compile a subset"
)]
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

/// [`cornered_bytes`] as a `u16` image, the byte in the high half.
///
/// The detector reads `pixel >> 8`, so this is the same field the CPU sweep and
/// the GPU score kernel see.
#[allow(
    dead_code,
    reason = "used by gpu_detect; other binaries compile a subset"
)]
pub fn cornered_image(width: usize, height: usize) -> ImageU16 {
    let bytes: Vec<u8> = cornered_bytes(width, height);
    let mut image: ImageU16 = ImageU16::zeros(width, height).expect("a valid image geometry");
    for y in 0..height {
        for x in 0..width {
            image.set(x, y, u16::from(bytes[y * width + x]) << 8);
        }
    }
    image
}

/// A grid of source positions well inside a `size` x `size` frame, spaced so no
/// two patches overlap and every one is far enough from the border for the
/// coarsest level's 52-tap pattern.
#[allow(
    dead_code,
    reason = "used by gpu_kernels; other binaries compile a subset"
)]
pub fn grid_positions(size: usize) -> PointsSoA {
    let mut positions: PointsSoA = PointsSoA::with_capacity(256);
    let mut y: usize = 96;
    while y + 96 < size {
        let mut x: usize = 96;
        while x + 96 < size {
            positions.push(Vector2::new(x as f32 + 0.37, y as f32 - 0.21));
            x += 71;
        }
        y += 71;
    }
    positions
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
#[allow(
    dead_code,
    reason = "used by flow_frontend; other binaries compile a subset"
)]
pub const FLOW_WIDTH: usize = 200;
/// The synthetic rig's frame height.
#[allow(
    dead_code,
    reason = "used by flow_frontend; other binaries compile a subset"
)]
pub const FLOW_HEIGHT: usize = 200;

/// `count` identical pinhole cameras 5 cm apart along `x`, all seeing a
/// `FLOW_WIDTH` x `FLOW_HEIGHT` frame.
#[allow(
    dead_code,
    reason = "used by flow_frontend; other binaries compile a subset"
)]
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
#[allow(
    dead_code,
    reason = "used by flow_frontend; other binaries compile a subset"
)]
pub fn flow_config() -> VioConfig {
    VioConfig {
        optical_flow_matching_guess_type: MatchingGuessType::SamePixel,
        ..VioConfig::default()
    }
}

/// Bright 5x5 squares on a regular lattice, the whole frame shifted by `shift`
/// pixels: four strong FAST corners each, and enough texture in between for the
/// KLT to follow them.
#[allow(
    dead_code,
    reason = "used by flow_frontend; other binaries compile a subset"
)]
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
#[allow(
    dead_code,
    reason = "used by klt_tracker; other binaries compile a subset"
)]
pub fn pyramid_of(image: &ImageU16, levels: usize) -> PyramidU16 {
    let mut pyramid: PyramidU16 =
        PyramidU16::with_capacity(image.width(), image.height(), levels).expect("a valid geometry");
    CpuPyramidBuilder::new()
        .build(0, image, &mut pyramid)
        .expect("the geometry the pyramid was allocated for");
    pyramid
}

#[cfg(feature = "gpu-core")]
pub mod gpu;
