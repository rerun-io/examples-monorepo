use super::*;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

fn lon_lat(lon: f64, lat: f64) -> [f64; 3] {
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()]
}

#[test]
fn equirectangular_jacobians_and_round_trips_both_precisions() {
    let tile = [300.0, 310.0, 320.0, 240.0, 1.7];
    for params in [
        Equirectangular::<f64>::panorama(1920, 960)
            .unwrap()
            .params(),
        tile,
    ] {
        // The finite difference on lon_max must not step past pi; its column is zero anyway.
        sweep(
            params,
            |[a, b, c, d, e]| Equirectangular::new([a, b, c, d, e.min(PI)]).unwrap(),
            1e-6,
            1e-6,
        );
        let narrow = |[a, b, c, d, e]: [f32; 5]| {
            Equirectangular::new([a, b, c, d, e.min(std::f32::consts::PI)]).unwrap()
        };
        sweep(params.map(|v| v as f32), narrow, 1e-3, 0.08);
    }
}

#[test]
fn the_full_sphere_minus_the_poles_round_trips_with_analytic_jacobians() {
    let camera = Equirectangular::<f64>::panorama(1920, 960).unwrap();
    let narrow = Equirectangular::<f32>::panorama(1920, 960).unwrap();
    let h = 1e-6;
    for lon in (-175..=175).step_by(25) {
        for lat in (-85..=85).step_by(17) {
            let ray = lon_lat(f64::from(lon), f64::from(lat));
            let mut jp = [[0.0; 3]; 2];
            let pixel = camera
                .project_with_jacobians(ray, Some(&mut jp), None)
                .unwrap();
            for i in 0..3 {
                let [mut a, mut b] = [ray; 2];
                a[i] += h;
                b[i] -= h;
                let [pa, pb] = [camera.project_unchecked(a), camera.project_unchecked(b)];
                for r in 0..2 {
                    let numeric = (pa[r] - pb[r]) / (2.0 * h);
                    assert!(
                        (jp[r][i] - numeric).abs() < 1e-5 * (1.0 + numeric.abs()),
                        "{lon} {lat}"
                    );
                }
            }
            let back = camera.unproject(pixel).unwrap();
            let rough = narrow
                .unproject(narrow.project(ray.map(|v| v as f32)).unwrap())
                .unwrap();
            for i in 0..3 {
                assert!((back[i] - ray[i]).abs() < 1e-12);
                assert!((f64::from(rough[i]) - ray[i]).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn the_panorama_matches_colmap_minus_half_a_pixel() {
    let (w, h) = (1920.0, 960.0);
    let camera = Equirectangular::<f64>::panorama(1920, 960).unwrap();
    assert_eq!(camera.params(), [w / TAU, h / PI, 959.5, 479.5, PI]);
    for lon in (-179..=179).step_by(14) {
        for lat in (-88..=88).step_by(11) {
            let [x, y, z] = lon_lat(f64::from(lon), f64::from(lat));
            // COLMAP sensor/models/spherical.h: elevation is up-positive, origin (0.5, 0.5).
            let theta = x.atan2(z);
            let phi = (-y).atan2(x.hypot(z));
            let colmap = [(theta / TAU + 0.5) * w, (0.5 - phi / PI) * h];
            let pixel = camera.project([x, y, z]).unwrap();
            for i in 0..2 {
                assert!((pixel[i] - (colmap[i] - 0.5)).abs() < 1e-9, "{lon} {lat}");
            }
        }
    }
}

#[test]
fn a_hemisphere_window_never_wraps() {
    let lon_max = 96f64.to_radians();
    let camera = Equirectangular::hemisphere_tile(1100, 960, 960.0 / PI, lon_max).unwrap();
    let fx = camera.params()[0];
    let eps = 1e-6;
    let edge = lon_max - eps;
    let right = camera.project([edge.sin(), 0.0, edge.cos()]).unwrap();
    let left = camera.project([-edge.sin(), 0.0, edge.cos()]).unwrap();
    assert!((right[0] - left[0] - 2.0 * edge * fx).abs() < 1e-6);
    let beyond = lon_max + 1e-6;
    assert_eq!(
        camera.project([beyond.sin(), 0.0, beyond.cos()]),
        Err(ProjectionReject::OutsideDomain)
    );
    // Straight back is outside a hemisphere window, on either side of the seam.
    assert_eq!(
        camera.project([0.0, 0.0, -1.0]),
        Err(ProjectionReject::OutsideDomain)
    );
    assert_eq!(
        camera.project([-1e-9, 0.0, -1.0]),
        Err(ProjectionReject::OutsideDomain)
    );
    let [_, _, cx, cy, _] = camera.params();
    assert_eq!(
        camera.unproject([cx + (lon_max + 1e-6) * fx, cy]),
        Err(UnprojectError::OutsideDomain)
    );
    assert!(camera.unproject([cx - (lon_max - 1e-6) * fx, cy]).is_ok());
}

#[test]
fn poles_degenerate_points_and_bad_calibrations_are_rejected() {
    let camera = Equirectangular::<f64>::panorama(1920, 960).unwrap();
    let [fx, fy, cx, cy, _] = camera.params();
    assert_eq!(
        camera.project([0.0, 1.0, 0.0]),
        Err(ProjectionReject::OutsideDomain)
    );
    assert_eq!(
        camera.project([1e-12, -1.0, 1e-12]),
        Err(ProjectionReject::OutsideDomain)
    );
    assert_eq!(
        camera.project([0.0; 3]),
        Err(ProjectionReject::BelowMinDepth)
    );
    assert_eq!(
        camera.project([f64::NAN, 0.0, 1.0]),
        Err(ProjectionReject::NonFinite)
    );
    for lat in [FRAC_PI_2, -FRAC_PI_2, 2.0 * PI] {
        assert_eq!(
            camera.unproject([cx, cy + lat * fy]),
            Err(UnprojectError::OutsideDomain)
        );
    }
    assert_eq!(
        camera.unproject([cx + 3.2 * fx, cy]),
        Err(UnprojectError::OutsideDomain)
    );
    assert_eq!(
        camera.unproject([f64::NAN, cy]),
        Err(UnprojectError::NonFinite)
    );
    // Pixel centres of the first and last rows stay inside the domain.
    assert!(camera.unproject([0.0, 0.0]).is_ok());
    assert!(camera.unproject([1919.0, 959.0]).is_ok());
    let mut ji = [[1.0; 5]; 2];
    camera
        .project_with_jacobians([0.3, 0.2, 0.9], None, Some(&mut ji))
        .unwrap();
    assert_eq!([ji[0][4], ji[1][4]], [0.0, 0.0]);
    for lon_max in [0.0, -1.0, 3.2, f64::NAN] {
        assert_eq!(
            Equirectangular::new([fx, fy, cx, cy, lon_max]),
            Err(InvalidCalibration)
        );
    }
    assert_eq!(
        Equirectangular::<f64>::panorama(0, 960),
        Err(InvalidCalibration)
    );
}
