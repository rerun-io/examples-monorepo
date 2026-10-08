//! Shared integration-test inputs and mathematical references.
//! Each binary uses a subset; item-level dead-code allowances name a consumer.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use kornia_image::Image;
use nalgebra::{DMatrix, DVector, Vector3, Vector6};
use serde::Deserialize;
use slam_rs::calib::{BasaltCamera, Calibration, Kb4Params};
use slam_rs::config::VioConfig;
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

/// `crates/slam-rs/tests/fixtures`, or portable assets staged for a remote GPU run.
#[allow(
    dead_code,
    reason = "used by vio_pipeline; other binaries compile a subset"
)]
pub fn fixtures() -> PathBuf {
    if let Some(root) = std::env::var_os("SLAM_RS_TEST_ASSETS") {
        return PathBuf::from(root).join("fixtures");
    }
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
    if let Some(root) = std::env::var_os("SLAM_RS_TEST_ASSETS") {
        return PathBuf::from(root).join("configs");
    }
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
    /// Calibration values from catalog f32 statics or the fixture JSON doubles.
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
pub use kornia_staging_imgproc::test_fixtures::Pgm;

#[allow(dead_code)]
pub fn shared_frames() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../kornia-staging/fixtures/frames")
}
#[allow(dead_code)]
pub fn read_pgm(directory: &Path, frame: usize, camera: usize) -> Pgm {
    kornia_staging_imgproc::test_fixtures::read_pgm(
        &directory.join(format!("frame_{frame:03}_cam{camera}.pgm")),
    )
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
/// which is what puts in both camera slots.
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

/// Seeded xorshift64* keeps synthetic tests deterministic.
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

    /// Uniform on `[-1, 1]`.
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

/// The calibration `get_vo_estimator` builds :
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
        BasaltCamera::Kb4(Kb4Params {
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
/// about the linearization, one level up.
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

#[allow(
    unused_imports,
    reason = "integration binaries use different fixture subsets"
)]
pub use images::{cornered_bytes, cornered_image, texture, textured_image};
use kornia_staging_imgproc::test_fixtures as images;

/// A grid of source positions well inside a `size` x `size` frame, spaced so no
/// two patches overlap and every one is far enough from the border for the
/// coarsest level's 52-tap pattern.
#[allow(
    dead_code,
    reason = "used by gpu_kernels; other binaries compile a subset"
)]
#[allow(unused_imports, reason = "integration binaries use different fixtures")]
pub use gpu_flow::grid_positions;
use kornia_staging_imgproc::test_fixtures as gpu_flow;

// ---- frontend fixtures (S25 FRONT) ----------------------------------------
//
// The frontend's synthetic rig and its two frames. `flow_rig` and `flow_config`
// were byte-identical in `tests/frame_allocations.rs` and in
// `src/frontend/flow.rs`'s test module, and the allocation test only measures
// the frame the flow tests describe while the two stay in step.

use slam_rs::calib::{CalibAccelBias, CalibGyroBias, PinholeParams};
use slam_rs::config::MatchingGuessType;
use slam_rs::lie::So3;
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
    let intrinsics: BasaltCamera<f64> = BasaltCamera::Pinhole(PinholeParams {
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

/// Use the shipped configuration with same-pixel matching guesses for identical camera frames.
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
pub fn dotted_image(shift: i32) -> Image<u16, 1> {
    let mut image: Image<u16, 1> =
        slam_rs::image::zeros(FLOW_WIDTH, FLOW_HEIGHT).expect("a valid geometry");
    for y in 0..FLOW_HEIGHT {
        for x in 0..FLOW_WIDTH {
            let fx: f64 = f64::from(x as i32 - shift);
            let fy: f64 = f64::from(y as i32);
            let base: f64 = 18_000.0 + 5_000.0 * (fx * 0.07).sin() * (fy * 0.05).cos();
            image.set_pixel(x, y, 0, base as u16).unwrap();
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
                        image.set_pixel(x as usize, cy + dy, 0, 0xF000).unwrap();
                    }
                }
            }
            cx += 17;
        }
        cy += 17;
    }
    image
}

pub mod flow;

#[cfg(feature = "gpu-core")]
pub mod gpu;
