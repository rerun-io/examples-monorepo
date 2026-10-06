use super::*;

#[cfg(feature = "kornia-3d-interop")]
#[test]
fn upstream_interop_preserves_distortion() {
    let camera = right_front();
    let upstream: kornia_3d::camera::FisheyeCamera = camera.into();
    assert_eq!(KannalaBrandt4::try_from(upstream).unwrap(), camera);
    let upstream = kornia_3d::camera::PinholeCamera {
        fx: 400.0,
        fy: 410.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.1,
        k2: -0.02,
        p1: 0.001,
        p2: -0.003,
    };
    let brown = BrownConrady::try_from(upstream.clone()).unwrap();
    let back = kornia_3d::camera::PinholeCamera::try_from(brown).unwrap();
    assert_eq!(
        [back.fx, back.fy, back.cx, back.cy, back.k1, back.k2, back.p1, back.p2],
        [
            upstream.fx,
            upstream.fy,
            upstream.cx,
            upstream.cy,
            upstream.k1,
            upstream.k2,
            upstream.p1,
            upstream.p2
        ]
    );
}

#[test]
fn point_derivatives_match_the_independent_symforce_oracles() {
    use super::test_oracles::{
        fisheye62::sym::project_fisheye62_with_jacobian,
        pinhole::sym::project_pinhole_with_jacobian,
    };
    use nalgebra::{SMatrix, SVector};
    let focal = SVector::<f64, 2>::new(220.0, 225.0);
    let principal = SVector::<f64, 2>::new(318.0, 240.0);
    let distortion = SVector::<f64, 8>::from_row_slice(&[
        0.1, -0.02, 0.003, -0.0004, 0.00005, -0.000006, 0.002, -0.003,
    ]);
    let camera = Fisheye624::fisheye62(
        [220.0, 225.0, 318.0, 240.0],
        distortion.as_slice()[..6].try_into().unwrap(),
        [0.002, -0.003],
    )
    .expect("valid camera calibration");
    let pinhole = Pinhole::new([220.0, 225.0, 318.0, 240.0]).expect("valid camera calibration");
    for point in [
        [0.0, 0.0, 1.0],
        [0.3, -0.2, 0.8],
        [-0.9, 0.7, 0.5],
        [0.2, 0.1, -0.5],
        [1e-7, -2e-7, 1.0],
    ] {
        let mut expected_j = SMatrix::<f64, 2, 3>::zeros();
        let mut actual_j = [[0.0; 3]; 2];
        let expected = project_fisheye62_with_jacobian(
            &point.into(),
            &focal,
            &principal,
            &distortion,
            Some(&mut expected_j),
        );
        let actual = camera.project_unchecked_with_jacobians(point, Some(&mut actual_j), None);
        for row in 0..2 {
            assert_relative_eq!(actual[row], expected[row], epsilon = 1e-6);
            for col in 0..3 {
                assert_relative_eq!(actual_j[row][col], expected_j[(row, col)], epsilon = 1e-4);
            }
        }
        let expected =
            project_pinhole_with_jacobian(&point.into(), &focal, &principal, Some(&mut expected_j));
        let actual = pinhole.project_unchecked_with_jacobians(point, Some(&mut actual_j), None);
        for row in 0..2 {
            assert_relative_eq!(actual[row], expected[row], epsilon = 1e-6);
            for col in 0..3 {
                assert_relative_eq!(actual_j[row][col], expected_j[(row, col)], epsilon = 1e-4);
            }
        }
    }
}

#[test]
fn extreme_finite_inputs_never_succeed_with_a_zero_bearing() {
    let p = Pinhole::new([1.0f64, 1.0, 0.0, 0.0]).expect("valid camera calibration");
    let b = p.unproject([1e200, 0.0]).unwrap();
    assert!((b[0] * b[0] + b[1] * b[1] + b[2] * b[2] - 1.0).abs() < 1e-12);
    let p = Pinhole::new([1.0f32, 1.0, 0.0, 0.0]).expect("valid camera calibration");
    assert!(p.unproject([1e20, 0.0]).unwrap()[0] > 0.99);
    let k = KannalaBrandt4::new([1.0f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("valid camera calibration");
    assert!(k.project([1e200, 0.0, 1.0]).is_err());
    assert!(k.boundary_bearing([1e200, 0.0]).is_err());
    let f = Fisheye624::fisheye62([1.0f64, 1.0, 0.0, 0.0], [0.0; 6], [0.0; 2])
        .expect("valid camera calibration");
    assert!(f.project([1e200, 0.0, 1.0]).is_err());
    let p = Pinhole::new([1.0f64, 1.0, 0.0, 0.0]).expect("valid camera calibration");
    assert!(p.unproject([f64::MAX, f64::MAX]).is_err());
    let f = Fisheye624::fisheye62([1.0f64, 1.0, 0.0, 0.0], [0.0; 6], [0.0; 2])
        .expect("valid camera calibration");
    assert!(f.project([1.0, 0.0, 1e-200]).is_err());
}

#[test]
fn constructors_reject_invalid_calibration() {
    assert!(Pinhole::new([0.0, 100.0, 0.0, 0.0]).is_err());
    assert!(KannalaBrandt4::new([100.0, 100.0, 0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0]).is_err());
    let mut brown = [0.0; 18];
    brown[..2].fill(100.0);
    assert!(BrownConrady::new(brown, Some(-1.0)).is_err());
    let mut fisheye = [0.0; 16];
    fisheye[..2].fill(100.0);
    fisheye[15] = f64::INFINITY;
    assert!(Fisheye624::new(fisheye).is_err());
}
