use super::*;

#[test]
fn fisheye624_matches_projectaria_and_inverts_visible_pixels() {
    #[derive(serde::Deserialize)]
    struct AriaCamera {
        params: [f64; 15],
        points_cam: Vec<[f64; 3]>,
        expected_pixels: Vec<Option<[f64; 2]>>,
    }
    #[derive(serde::Deserialize)]
    struct AriaFixture {
        cameras: Vec<AriaCamera>,
    }
    let fixture: AriaFixture = serde_json::from_str(include_str!(
        "../../../../../fixtures/cameras/fisheye624_projectaria.json"
    ))
    .unwrap();
    for entry in fixture.cameras {
        let params = std::array::from_fn(|i| entry.params[if i == 0 { 0 } else { i - 1 }]);
        let camera = Fisheye624::new(params).unwrap();
        for (point, pixel) in entry.points_cam.into_iter().zip(entry.expected_pixels) {
            let Some(expected) = pixel else { continue };
            let actual = camera.project(point).unwrap();
            for i in 0..2 {
                assert_relative_eq!(
                    actual[i],
                    expected[i],
                    max_relative = 1e-12,
                    epsilon = 1e-12
                );
            }
            let ray = camera.unproject(expected).unwrap();
            let norm = point.iter().map(|v| v * v).sum::<f64>().sqrt();
            let error = ray
                .iter()
                .zip(point)
                .map(|(a, b)| (a - b / norm).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(error < 1e-9, "bearing chord {error}");
            let back = camera.project(ray).unwrap();
            for i in 0..2 {
                assert!((back[i] - expected[i]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn fisheye624_full_derivatives_and_round_trips() {
    let p = [
        220.0, 225.0, 318.0, 240.0, 0.1, -0.02, 0.003, -0.0004, 0.00005, -0.000006, 0.002, -0.003,
        0.0001, -0.00001, -0.0002, 0.00002,
    ];
    sweep(p, |p| Fisheye624::new(p).unwrap(), 1e-6, 2e-6);
    sweep(
        p.map(|v| v as f32),
        |p| Fisheye624::new(p).unwrap(),
        1e-3,
        0.05,
    );
}

#[test]
fn fisheye62_matches_checked_simplecv_fixture() {
    let fixture: ProjectionFixture = serde_json::from_str(include_str!(
        "../../../../../fixtures/cameras/fisheye62_simplecv.json"
    ))
    .unwrap();
    let mut p = [0.0; 16];
    p[..4].copy_from_slice(&fixture.intrinsics);
    p[4..4 + fixture.distortion.len()].copy_from_slice(&fixture.distortion);
    let camera = Fisheye624::new(p).expect("valid camera calibration");
    for (point, pixel) in fixture.points.into_iter().zip(fixture.pixels) {
        let actual = camera.project(point).unwrap();
        for i in 0..2 {
            assert_relative_eq!(actual[i], pixel[i], epsilon = 1e-12);
        }
    }
}

#[test]
fn fisheye624_inverse_rejects_the_excluded_horizon() {
    let f = Fisheye624::fisheye62([1.0f32, 1.0, 0.0, 0.0], [0.0; 6], [0.0; 2])
        .expect("valid camera calibration");
    assert_eq!(
        f.unproject([std::f32::consts::FRAC_PI_2, 0.0]),
        Err(UnprojectError::OutsideDomain)
    );
}
