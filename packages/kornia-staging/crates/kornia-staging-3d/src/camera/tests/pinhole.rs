use super::*;

#[test]
fn pinhole_projects_and_returns_unit_bearings() {
    let camera = Pinhole::new([400.0, 410.0, 320.0, 240.0]).expect("valid camera calibration");
    assert_eq!(camera.project([1.0, 2.0, 4.0]).unwrap(), [420.0, 445.0]);
    let ray = camera.unproject([420.0, 445.0]).unwrap();
    assert_relative_eq!(ray[0], 1.0 / 21.0_f64.sqrt(), epsilon = 1e-14);
    assert_eq!(
        camera.project([0.0, 0.0, -1.0]),
        Err(ProjectionReject::BelowMinDepth)
    );
    assert_eq!(
        camera.project([f64::NAN, 0.0, 1.0]),
        Err(ProjectionReject::NonFinite)
    );
}

#[test]
fn pinhole_jacobians_and_round_trips_both_precisions() {
    sweep(
        [458.0, 457.0, 367.0, 248.0],
        |p| Pinhole::new(p).unwrap(),
        1e-6,
        1e-6,
    );
    sweep(
        [458.0_f32, 457.0, 367.0, 248.0],
        |p| Pinhole::new(p).unwrap(),
        1e-2,
        1e-2,
    );
}
