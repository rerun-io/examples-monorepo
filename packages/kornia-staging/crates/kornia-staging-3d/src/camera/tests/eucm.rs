use super::double_sphere::{bearing, rear_round_trip, EUROC_EUCM, TUMVI_EUCM};
use super::*;

#[test]
fn eucm_jacobians_and_round_trips_both_precisions() {
    for params in [
        TUMVI_EUCM,
        EUROC_EUCM,
        [400.0, 410.0, 320.0, 240.0, 0.4, 0.8],
    ] {
        sweep(params, |p| Eucm::new(p).unwrap(), 1e-6, 1e-6);
        sweep(
            params.map(|v| v as f32),
            |p| Eucm::new(p).unwrap(),
            1e-3,
            0.08,
        );
    }
}

#[test]
fn eucm_tumvi_admits_rear_rays_up_to_its_domain() {
    let camera = Eucm::new(TUMVI_EUCM).unwrap();
    rear_round_trip(
        camera,
        Eucm::new(TUMVI_EUCM.map(|v| v as f32)).unwrap(),
        120.0,
    );
    // The validity cone of this calibration ends near 127 degrees.
    assert_eq!(
        camera.project(bearing(132.0, 200.0)),
        Err(ProjectionReject::OutsideDomain)
    );
    assert_eq!(
        camera.project([0.0; 3]),
        Err(ProjectionReject::BelowMinDepth)
    );
    assert_eq!(
        camera.project([0.0, f64::INFINITY, 1.0]),
        Err(ProjectionReject::NonFinite)
    );
    assert_eq!(
        camera.unproject([f64::NAN, 0.0]),
        Err(UnprojectError::NonFinite)
    );
}

#[test]
fn eucm_rejects_at_and_beyond_the_domain_edge() {
    // alpha = 0.75, beta = 1 give w = 1/3 and an inverse disk r^2 < 2, both exact.
    let camera = Eucm::new([100.0f64, 100.0, 0.0, 0.0, 0.75, 1.0]).unwrap();
    let (pixel, status) = camera.project_with_status([2.0, 2.0, -1.0], None, None);
    assert_eq!(status, Err(ProjectionReject::OutsideDomain));
    assert!(pixel.iter().all(|v| v.is_finite()));
    assert_eq!(
        camera.project([2.0, 2.0, -1.1]),
        Err(ProjectionReject::OutsideDomain)
    );
    assert!(camera.project([2.0, 2.0, -0.9]).is_ok());
    assert_eq!(
        camera.unproject([100.0, 100.0]),
        Err(UnprojectError::OutsideDomain)
    );
    assert_eq!(
        camera.unproject([100.0, 101.0]),
        Err(UnprojectError::OutsideDomain)
    );
    assert!(camera.unproject([100.0, 99.0]).is_ok());
    for bad in [
        [1.0, 1.0, 0.0, 0.0, 0.5, 0.0],
        [1.0, 1.0, 0.0, 0.0, -0.1, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.1, 1.0],
        [1.0, -1.0, 0.0, 0.0, 0.5, 1.0],
    ] {
        assert_eq!(Eucm::new(bad), Err(InvalidCalibration));
    }
}
