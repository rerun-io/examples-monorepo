use super::*;

#[test]
fn brown_rational_prism_and_tilt_match_opencv() {
    for text in [
        include_str!("../../../../../fixtures/cameras/brown4_opencv.json"),
        include_str!("../../../../../fixtures/cameras/brown5_opencv.json"),
        include_str!("../../../../../fixtures/cameras/brown8_opencv.json"),
        include_str!("../../../../../fixtures/cameras/brown12_opencv.json"),
        include_str!("../../../../../fixtures/cameras/brown14_opencv.json"),
    ] {
        let fixture: ProjectionFixture = serde_json::from_str(text).unwrap();
        let mut params = [0.0; 18];
        params[..4].copy_from_slice(&fixture.intrinsics);
        params[4..4 + fixture.distortion.len()].copy_from_slice(&fixture.distortion);
        let camera = BrownConrady::new(params, None).expect("valid camera calibration");
        for (point, expected) in fixture.points.into_iter().zip(fixture.pixels) {
            let actual = camera.project(point).unwrap();
            for i in 0..2 {
                assert_relative_eq!(actual[i], expected[i], epsilon = 2e-10);
            }
            let back = camera.project(camera.unproject(expected).unwrap()).unwrap();
            for i in 0..2 {
                assert_relative_eq!(back[i], expected[i], epsilon = 1e-8);
            }
        }
    }
}

#[test]
fn brown_jacobians_include_prism_and_tilt() {
    let params = [
        430.0, 415.0, 320.0, 240.0, 0.03, -0.007, 0.003, -0.004, 0.001, 0.002, -0.001, 0.0002,
        0.0007, -0.0003, -0.0008, 0.0004, 0.04, -0.03,
    ];
    sweep(
        params,
        |p| BrownConrady::new(p, None).expect("valid camera calibration"),
        1e-6,
        1e-5,
    );
    sweep(
        params.map(|v| v as f32),
        |p| BrownConrady::new(p, Some(1.0)).expect("valid camera calibration"),
        1e-3,
        0.06,
    );
    let camera = BrownConrady::new(params, Some(0.1)).expect("valid camera calibration");
    assert_eq!(
        camera.project([1.0, 0.0, 1.0]),
        Err(ProjectionReject::OutsideDomain)
    );
    assert_eq!(
        camera.unproject([800.0, 200.0]),
        Err(UnprojectError::OutsideDomain)
    );
}
