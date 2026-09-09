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

use nalgebra::SMatrix;
use slam_rs::lie::LieScalar;

const MSDMI: &str = include_str!("../fixtures/msdmi_calib.json");
const MSDMG: &str = include_str!("../fixtures/msdmg_calib.json");
const ROBOCAP: &str = include_str!("../fixtures/robocap-basalt-calib.json");

// ── the fixture directory and the two files every VIO lane reads ───────────

/// `crates/slam-rs/tests/fixtures`.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
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
