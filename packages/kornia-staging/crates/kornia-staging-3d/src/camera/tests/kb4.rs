use super::*;

#[test]
fn kb4_jacobians_and_round_trips_both_precisions() {
    sweep(
        [400.0, 410.0, 320.0, 240.0, 0.01, -0.001, 0.0001, 0.0],
        |p| KannalaBrandt4::new(p).unwrap(),
        1e-6,
        1e-6,
    );
    sweep(
        [400.0_f32, 410.0, 320.0, 240.0, 0.01, -0.001, 0.0001, 0.0],
        |p| KannalaBrandt4::new(p).unwrap(),
        1e-3,
        0.08,
    );
}

#[test]
fn s66_edge_stays_on_the_physical_branch() {
    let camera = right_front();
    assert!((camera.max_angle().to_degrees() - 84.0).abs() < 1.5);
    for pixel in [
        [100.0, 80.0],
        [946.0, 540.0],
        [1200.0, 300.0],
        [300.0, 900.0],
        [1700.0, 100.0],
    ] {
        let ray = camera.unproject(pixel).unwrap();
        assert!(ray[2].acos() < camera.max_angle());
        let back = camera.project(ray).unwrap();
        for i in 0..2 {
            assert_relative_eq!(back[i], pixel[i], epsilon = 1e-9);
        }
    }
    assert_eq!(
        camera.unproject([0.0, 0.0]),
        Err(UnprojectError::OutsideDomain)
    );
    let edge = camera.boundary_bearing([0.0, 0.0]).unwrap();
    assert_relative_eq!(edge[2].acos(), camera.max_angle(), epsilon = 1e-12);
}

#[test]
fn angular_scale_and_stationary_branch_regressions() {
    let cam = KannalaBrandt4::new([400.0f32, 400.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("valid camera calibration");
    let a = cam.project([0.002, 0.0, 0.004]).unwrap();
    let b = cam.project([0.2, 0.0, 0.4]).unwrap();
    assert!((a[0] - b[0]).abs() < 1e-4);
    let cam = KannalaBrandt4::new([400.0f64, 400.0, 0.0, 0.0, -2.0 / 3.0, 0.2, 0.0, 0.0])
        .expect("valid camera calibration");
    assert!(cam.max_angle() > 1.2);
    let ray = [1.2f64.sin(), 0.0, 1.2f64.cos()];
    let back = cam.unproject(cam.project(ray).unwrap()).unwrap();
    for i in 0..3 {
        assert!((back[i] - ray[i]).abs() < 1e-10);
    }
}

#[test]
fn angular_peak_roundtrip_and_singular_derivative() {
    let cam = KannalaBrandt4::new([400.0f64, 400.0, 0.0, 0.0, -1.0 / 3.0, 0.0, 0.0, 0.0])
        .expect("valid camera calibration");
    let theta = cam.max_angle();
    for azimuth in [0.0f64, 0.3, 1.1, 2.6] {
        let ray = [
            theta.sin() * azimuth.cos(),
            theta.sin() * azimuth.sin(),
            theta.cos(),
        ];
        let pixel = cam.project(ray).unwrap();
        let back = cam.unproject(pixel).unwrap();
        for i in 0..3 {
            assert!((back[i] - ray[i]).abs() < 1e-12);
        }
        assert_eq!(
            cam.unproject_with_jacobians(pixel, Some(&mut [[0.0; 2]; 3]), None),
            Err(UnprojectError::Singular)
        );
    }
}

#[test]
fn kb4_pi_endpoint_and_adjacent_radii_are_rejected() {
    macro_rules! check {
        ($s:ty, $pi:path) => {{
            let pi: $s = $pi;
            let camera =
                KannalaBrandt4::<$s>::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
            // The adjacent radii also fall inside the inverse's endpoint snapping tolerance.
            for radius in [pi.next_down(), pi, pi.next_up()] {
                assert_eq!(
                    camera.unproject([radius, 0.0]),
                    Err(UnprojectError::OutsideDomain)
                );
                assert_eq!(
                    camera.boundary_bearing([radius, 0.0]),
                    Err(UnprojectError::OutsideDomain)
                );
            }
            let interior = pi - 32.0 * crate::camera::inverse_epsilon::<$s>();
            let ray = camera.unproject([interior, 0.0]).unwrap();
            assert!(ray[0] > 0.0);
            assert!(
                (camera.project(ray).unwrap()[0] - interior).abs()
                    <= 32.0 * crate::camera::inverse_epsilon::<$s>()
            );
            // A polynomial peak before pi remains a supported saturation boundary.
            let peaked =
                KannalaBrandt4::<$s>::new([1.0, 1.0, 0.0, 0.0, -1.0 / 3.0, 0.0, 0.0, 0.0]).unwrap();
            let edge = peaked.boundary_bearing([1.0, 0.0]).unwrap();
            let pixel = peaked.project(edge).unwrap();
            for radius in [pixel[0].next_down(), pixel[0], pixel[0].next_up()] {
                let ray = peaked.unproject([radius, 0.0]).unwrap();
                assert!((ray[2] - edge[2]).abs() < 32.0 * crate::camera::inverse_epsilon::<$s>());
            }
        }};
    }
    check!(f32, std::f32::consts::PI);
    check!(f64, std::f64::consts::PI);
}

#[test]
fn kb4_extreme_finite_projections_agree_and_never_return_infinity() {
    macro_rules! check {
        ($s:ty, $tiny:expr) => {{
            let camera =
                KannalaBrandt4::<$s>::new([0.3 * <$s>::MAX, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    .unwrap();
            let tiny: $s = $tiny;
            for point in [
                [tiny, 0.0, -1.0],
                [tiny, 0.0, -4.0 * tiny],
                [tiny, tiny, 1.0],
                [0.0, tiny, -1.0],
                [<$s>::MIN_POSITIVE, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [<$s>::MAX, 0.0, 1.0],
            ] {
                let pixel = camera.project(point);
                assert_eq!(
                    pixel,
                    camera.project_with_jacobians(point, None, None),
                    "{point:?}"
                );
                if let Ok(pixel) = pixel {
                    assert!(pixel.iter().all(|v| v.is_finite()), "{point:?}: {pixel:?}");
                }
            }
        }};
    }
    check!(f32, 4.5e-23);
    check!(f64, 2.7e-162);
}
