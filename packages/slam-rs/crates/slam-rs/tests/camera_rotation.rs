//! Rotated MSD-G2 cameras preserve projected pixel geometry.
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

use kornia_staging_3d::camera::{BrownConrady, CameraModel};
use nalgebra::{SVector, Vector2, Vector4};
use proptest::prelude::*;
use slam_rs::calib::{BasaltCamera, Calibration};

mod common;

fn brown(params: SVector<f64, 12>, radius: f64) -> BrownConrady<f64> {
    let mut full = [0.0; 18];
    full[..12].copy_from_slice(params.as_slice());
    BrownConrady::new(full, (radius != 0.0).then_some(radius)).unwrap()
}

/// The rotated model and its resolution, from the unrotated pair.
fn rotate_clockwise(
    camera: &BrownConrady<f64>,
    resolution: [u32; 2],
    degrees: f64,
) -> (BrownConrady<f64>, [u32; 2]) {
    let p = camera.params();
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
        brown(rotated, camera.valid_radius().unwrap_or(0.0)),
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

fn shipped_cameras() -> Vec<(BrownConrady<f64>, [u32; 2])> {
    let calibration: Calibration<f64> =
        Calibration::from_json_str(common::calibration_text("msdmg")).unwrap();
    calibration
        .intrinsics
        .iter()
        .zip(calibration.resolution.iter())
        .map(|(model, resolution)| match model {
            BasaltCamera::PinholeRadtan8(p) => (
                brown(
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

proptest! {
    #[test]
    fn a_point_projects_to_the_rotated_pixel(x in -0.3f64..0.3, y in -0.3f64..0.3, z in 1.0f64..5.0) {
        let point = Vector4::new(x, y, z, 1.0);
        for (camera, resolution) in shipped_cameras() {
            for degrees in [90.0, 270.0] {
                let (rotated, rotated_resolution) = rotate_clockwise(&camera, resolution, degrees);
                prop_assert_eq!(rotated_resolution, [resolution[1], resolution[0]]);
                let landscape = Vector2::from(camera.project([point.x, point.y, point.z]).unwrap());
                let p = rotate_point(&point, degrees);
                let portrait = Vector2::from(rotated.project([p.x,p.y,p.z]).unwrap());
                let expected = rotate_pixel(&landscape, f64::from(resolution[0]), f64::from(resolution[1]), degrees);
                prop_assert!((portrait - expected).norm() < 1e-9);
            }
        }
    }
}
