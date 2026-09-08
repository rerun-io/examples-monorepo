//! The msd-g2 rotated intrinsics (decision D30), pinned against the catalog.
//!
//! The msd-g2 recordings are stored rotated into portrait: cam0 and cam1 by 90
//! degrees clockwise, cam2 and cam3 by 270. The catalog carries the intrinsics
//! rotated to match, so a port that reads the catalog must reproduce the same
//! arithmetic the writer used — otherwise every reprojection on that dataset is
//! off by a transpose, which is exactly the kind of error a trajectory gate
//! reports as "the estimator is slightly worse".
//!
//! `tests/reference/msd/msd-g2__MGO_others__MGO09_short_1_updown/run.json` is
//! what the basalt C++ reference run actually pushed: per camera, the rotated
//! `fx, fy, cx, cy`, the rotated distortion vector, the rotated resolution and
//! `image_rotation_cw_deg`. The unrotated truth is `fixtures/msdmg_calib.json`,
//! the same file the C++ reads. This test derives one from the other.
//!
//! The mapping, for a 90-degree clockwise image rotation of a `W x H` image:
//!
//! ```text
//! u' = (H - 1) - v      v' = u
//! fx' = fy              fy' = fx
//! cx' = (H - 1) - cy    cy' = cx
//! p1' = p2              p2' = -p1
//! ```
//!
//! and for 270 degrees clockwise:
//!
//! ```text
//! u' = v                v' = (W - 1) - u
//! fx' = fy              fy' = fx
//! cx' = cy              cy' = (W - 1) - cx
//! p1' = -p2             p2' = p1
//! ```
//!
//! The radial coefficients and `rpmax` are invariant because `r'^2` is. In three
//! dimensions the same rotation is `(x, y, z) -> (-y, x, z)` for 90 clockwise
//! and `(x, y, z) -> (y, -x, z)` for 270, which is what the second half of the
//! test checks: the rotated camera looking at the rotated point lands on the
//! rotated pixel.

#![allow(clippy::unwrap_used)]

use nalgebra::{SVector, Vector2, Vector4};
use serde::Deserialize;
use slam_rs::calib::{Calibration, CameraModel};
use slam_rs::camera::{Camera, PinholeRadtan8};

mod common;

const RUN: &str =
    include_str!("../../../tests/reference/msd/msd-g2__MGO_others__MGO09_short_1_updown/run.json");

#[derive(Debug, Deserialize)]
struct Run {
    cameras: Vec<CatalogCamera>,
}

/// What the catalog reports per camera, as the reference run recorded it.
#[derive(Debug, Deserialize)]
struct CatalogCamera {
    index: usize,
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// `[k1, k2, p1, p2, k3, k4, k5, k6, rpmax]`.
    distortion: Vec<f32>,
    image_rotation_cw_deg: f64,
}

/// The rotated model and its resolution, from the unrotated pair.
fn rotate_clockwise(
    camera: &PinholeRadtan8<f64>,
    resolution: [u32; 2],
    degrees: f64,
) -> (PinholeRadtan8<f64>, [u32; 2]) {
    let p: SVector<f64, 12> = camera.params();
    let width: f64 = f64::from(resolution[0]);
    let height: f64 = f64::from(resolution[1]);

    let (cx, cy, p1, p2) = match degrees {
        90.0 => (height - 1.0 - p[3], p[2], p[7], -p[6]),
        270.0 => (p[3], width - 1.0 - p[2], -p[7], p[6]),
        other => panic!("unsupported rotation {other}"),
    };

    let rotated: SVector<f64, 12> = SVector::<f64, 12>::from([
        p[1], p[0], cx, cy, p[4], p[5], p1, p2, p[8], p[9], p[10], p[11],
    ]);
    (
        PinholeRadtan8::new(rotated, camera.rpmax()),
        [resolution[1], resolution[0]],
    )
}

/// `(x, y, z) -> (-y, x, z)` at 90 clockwise, `(y, -x, z)` at 270.
fn rotate_point(point: &Vector4<f64>, degrees: f64) -> Vector4<f64> {
    match degrees {
        90.0 => Vector4::new(-point[1], point[0], point[2], point[3]),
        270.0 => Vector4::new(point[1], -point[0], point[2], point[3]),
        other => panic!("unsupported rotation {other}"),
    }
}

/// The pixel the rotation sends `(u, v)` to, in an image `width x height`.
fn rotate_pixel(uv: &Vector2<f64>, width: f64, height: f64, degrees: f64) -> Vector2<f64> {
    match degrees {
        90.0 => Vector2::new(height - 1.0 - uv[1], uv[0]),
        270.0 => Vector2::new(uv[1], width - 1.0 - uv[0]),
        other => panic!("unsupported rotation {other}"),
    }
}

fn shipped_cameras() -> Vec<(PinholeRadtan8<f64>, [u32; 2])> {
    let calibration: Calibration<f64> =
        Calibration::from_json_str(common::calibration_text("msdmg")).unwrap();
    calibration
        .intrinsics
        .iter()
        .zip(calibration.resolution.iter())
        .map(|(model, resolution)| match model {
            CameraModel::PinholeRadtan8(p) => (
                PinholeRadtan8::new(
                    SVector::<f64, 12>::from([
                        p.fx, p.fy, p.cx, p.cy, p.k1, p.k2, p.p1, p.p2, p.k3, p.k4, p.k5, p.k6,
                    ]),
                    p.rpmax,
                ),
                *resolution,
            ),
            other => panic!("expected pinhole-radtan8, got {}", other.name()),
        })
        .collect()
}

/// The rotated intrinsics are the in-tree ones put through the mapping, to the
/// last bit of the `float32` the catalog stores them in.
#[test]
fn the_catalog_intrinsics_are_the_rotated_shipped_ones() {
    let run: Run = serde_json::from_str(RUN).unwrap();
    let cameras: Vec<(PinholeRadtan8<f64>, [u32; 2])> = shipped_cameras();
    assert_eq!(run.cameras.len(), 4);

    for catalog in &run.cameras {
        let (camera, resolution) = cameras[catalog.index];
        let (rotated, rotated_resolution) =
            rotate_clockwise(&camera, resolution, catalog.image_rotation_cw_deg);
        assert_eq!(
            rotated_resolution,
            [catalog.width, catalog.height],
            "cam{}: resolution",
            catalog.index
        );

        // The catalog stores float32; the comparison rounds the exact f64
        // arithmetic the same way and then asks for equality, which is a
        // stronger statement than any tolerance would be.
        let p: SVector<f64, 12> = rotated.params();
        let expected: [f32; 12] = [
            catalog.fx,
            catalog.fy,
            catalog.cx,
            catalog.cy,
            catalog.distortion[0],
            catalog.distortion[1],
            catalog.distortion[2],
            catalog.distortion[3],
            catalog.distortion[4],
            catalog.distortion[5],
            catalog.distortion[6],
            catalog.distortion[7],
        ];
        for (index, want) in expected.iter().enumerate() {
            assert_eq!(
                p[index] as f32, *want,
                "cam{} parameter {index}",
                catalog.index
            );
        }
        assert_eq!(rotated.rpmax() as f32, catalog.distortion[8]);
    }
}

/// A point in the rotated camera projects to the rotated pixel.
///
/// This is the statement that matters for the port: whichever way the images are
/// stored, the two calibrations describe the same camera, so an estimator fed
/// the catalog's rotated frames and the catalog's rotated intrinsics sees the
/// same geometry the C++ sees on the raw Monado files.
#[test]
fn a_point_projects_to_the_rotated_pixel() {
    let run: Run = serde_json::from_str(RUN).unwrap();
    let cameras: Vec<(PinholeRadtan8<f64>, [u32; 2])> = shipped_cameras();

    let points: [Vector4<f64>; 5] = [
        Vector4::new(0.0, 0.0, 1.0, 1.0),
        Vector4::new(0.3, -0.2, 1.0, 1.0),
        Vector4::new(-0.9, 0.4, 2.0, 1.0),
        Vector4::new(1.4, 1.1, 3.0, 1.0),
        Vector4::new(-0.05, -1.3, 1.5, 1.0),
    ];

    for catalog in &run.cameras {
        let (camera, resolution) = cameras[catalog.index];
        let degrees: f64 = catalog.image_rotation_cw_deg;
        let (rotated, _) = rotate_clockwise(&camera, resolution, degrees);
        let width: f64 = f64::from(resolution[0]);
        let height: f64 = f64::from(resolution[1]);

        for point in &points {
            let mut landscape: Vector2<f64> = Vector2::zeros();
            assert!(
                camera.project(point, &mut landscape),
                "cam{}: the unrotated camera rejected the point",
                catalog.index
            );

            let mut portrait: Vector2<f64> = Vector2::zeros();
            assert!(
                rotated.project(&rotate_point(point, degrees), &mut portrait),
                "cam{}: the rotated camera rejected the rotated point",
                catalog.index
            );

            let expected: Vector2<f64> = rotate_pixel(&landscape, width, height, degrees);
            assert!(
                (portrait - expected).norm() < 1e-9,
                "cam{}: rotated {} expected {}",
                catalog.index,
                portrait.transpose(),
                expected.transpose()
            );
        }
    }
}
