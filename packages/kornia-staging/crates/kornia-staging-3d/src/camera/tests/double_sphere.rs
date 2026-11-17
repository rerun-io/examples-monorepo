use super::*;

// Basalt calibrations, cam0 of `data/tum/tumvi_512_{ds,eucm}_calib.json` and
// `data/euroc/euroc_{ds,eucm}_calib.json` in the rootslam fork.
#[allow(clippy::excessive_precision)]
pub(super) const TUMVI_DS: [f64; 6] = [
    158.28600034966976,
    158.2743455478755,
    254.96116578191652,
    256.8894394501779,
    -0.17213086034353242,
    0.5931177593944744,
];
#[allow(clippy::excessive_precision)]
pub(super) const EUROC_DS: [f64; 6] = [
    349.7560023050409,
    348.72454229977035,
    365.89440762590147,
    249.32995565708703,
    -0.2409573942178872,
    0.566996899163044,
];
#[allow(clippy::excessive_precision)]
pub(super) const TUMVI_EUCM: [f64; 6] = [
    191.14799836282188,
    191.13150963902817,
    254.9585771534443,
    256.88154645599445,
    0.6291060881178562,
    1.0418067381860867,
];
#[allow(clippy::excessive_precision)]
pub(super) const EUROC_EUCM: [f64; 6] = [
    460.76484651566466,
    459.4051018049483,
    365.8937161309615,
    249.33499869752444,
    0.5903365915227143,
    1.127468196965374,
];

/// Unit bearing at polar angle `theta` and azimuth `phi`, degrees.
pub(super) fn bearing(theta: f64, phi: f64) -> [f64; 3] {
    let (t, p) = (theta.to_radians(), phi.to_radians());
    [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()]
}

/// Bearings up to `limit` degrees off the axis project and come back in both precisions.
pub(super) fn rear_round_trip<M64: CameraModel<f64>, M32: CameraModel<f32>>(
    a: M64,
    b: M32,
    limit: f64,
) {
    for theta in (0..=limit as i32).step_by(5) {
        for phi in (0..360).step_by(30) {
            let ray = bearing(f64::from(theta), f64::from(phi));
            let back = a.unproject(a.project(ray).unwrap()).unwrap();
            let narrow = b
                .unproject(b.project(ray.map(|v| v as f32)).unwrap())
                .unwrap();
            for i in 0..3 {
                assert!((back[i] - ray[i]).abs() < 1e-9, "{theta} {phi}: {back:?}");
                assert!(
                    (f64::from(narrow[i]) - ray[i]).abs() < 2e-4,
                    "{theta} {phi}: {narrow:?}"
                );
            }
        }
    }
}

#[test]
fn double_sphere_jacobians_and_round_trips_both_precisions() {
    for params in [TUMVI_DS, EUROC_DS, [400.0, 410.0, 320.0, 240.0, 0.3, 0.4]] {
        sweep(params, |p| DoubleSphere::new(p).unwrap(), 1e-6, 1e-6);
        sweep(
            params.map(|v| v as f32),
            |p| DoubleSphere::new(p).unwrap(),
            1e-3,
            0.08,
        );
    }
}

#[test]
fn double_sphere_tumvi_admits_rear_rays_up_to_its_domain() {
    let camera = DoubleSphere::new(TUMVI_DS).unwrap();
    rear_round_trip(
        camera,
        DoubleSphere::new(TUMVI_DS.map(|v| v as f32)).unwrap(),
        120.0,
    );
    // The validity cone of this calibration ends near 125 degrees.
    assert_eq!(
        camera.project(bearing(130.0, 40.0)),
        Err(ProjectionReject::OutsideDomain)
    );
    assert_eq!(
        camera.project([0.0, 0.0, -1.0]),
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
    assert_eq!(
        camera.unproject([f64::INFINITY, 0.0]),
        Err(UnprojectError::NonFinite)
    );
}

#[test]
fn double_sphere_rejects_at_and_beyond_the_domain_edge() {
    // alpha = 0.75 and xi = 0 give w2 = 1/3 and an inverse disk r^2 < 2, both exact.
    let camera = DoubleSphere::new([100.0f64, 100.0, 0.0, 0.0, 0.0, 0.75]).unwrap();
    // |(2, 2, -1)| = 3, so z = -w2 |p| lies on the edge.
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
        [1.0, 1.0, 0.0, 0.0, 1.5, 0.5],
        [1.0, 1.0, 0.0, 0.0, 0.0, -0.1],
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.1],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
    ] {
        assert_eq!(DoubleSphere::new(bad), Err(InvalidCalibration));
    }
}

#[test]
fn double_sphere_rejects_a_domain_bound_that_is_not_finite() {
    // xi = -1, alpha = 0.5: w1 = 1 and w2 = 0 / sqrt(0).
    assert_eq!(
        DoubleSphere::new([1.0f64, 1.0, 0.0, 0.0, -1.0, 0.5]),
        Err(InvalidCalibration)
    );
    assert_eq!(
        DoubleSphere::new([1.0f32, 1.0, 0.0, 0.0, -1.0, 0.5]),
        Err(InvalidCalibration)
    );
    // Its neighbours keep a finite bound and are accepted.
    for ok in [
        [1.0f64, 1.0, 0.0, 0.0, -1.0, 0.6],
        [1.0, 1.0, 0.0, 0.0, -0.99, 0.5],
    ] {
        assert!(DoubleSphere::new(ok).unwrap().domain_bound().is_finite());
    }
}

#[test]
fn ds_and_eucm_calibrations_of_one_lens_agree_inside_80_degrees() {
    let ds = DoubleSphere::new(TUMVI_DS).unwrap();
    let eucm = Eucm::new(TUMVI_EUCM).unwrap();
    let mut worst = 0.0f64;
    for theta in (0..=80).step_by(5) {
        for phi in (0..360).step_by(15) {
            let ray = bearing(f64::from(theta), f64::from(phi));
            let [a, b] = [ds.project(ray).unwrap(), eucm.project(ray).unwrap()];
            worst = worst.max((a[0] - b[0]).hypot(a[1] - b[1]));
        }
    }
    // Measured 0.02 px: two fits of the same lens.
    assert!(worst < 0.1, "DS and EUCM disagree by {worst} px");
}

#[cfg(feature = "serde")]
#[test]
fn basalt_ds_and_eucm_round_trip_through_the_staged_kind() {
    use crate::camera::formats::BasaltCamera;
    for text in [
        r#"{"camera_type":"ds","intrinsics":{"fx":158.3,"fy":158.2,"cx":255.0,"cy":256.9,"xi":-0.17,"alpha":0.59}}"#,
        r#"{"camera_type":"eucm","intrinsics":{"fx":191.1,"fy":191.2,"cx":255.0,"cy":256.9,"alpha":0.63,"beta":1.04}}"#,
    ] {
        let source: BasaltCamera<f64> = serde_json::from_str(text).unwrap();
        let camera = CameraModelKind::try_from(&source).unwrap();
        assert!(matches!(
            camera,
            CameraModelKind::DoubleSphere(_) | CameraModelKind::Eucm(_)
        ));
        assert_eq!(BasaltCamera::try_from(camera).unwrap(), source);
        let json = serde_json::to_string(&camera).unwrap();
        assert_eq!(
            serde_json::from_str::<CameraModelKind<f64>>(&json).unwrap(),
            camera
        );
    }
    let bad = r#"{"camera_type":"ds","intrinsics":{"fx":1.0,"fy":1.0,"cx":0.0,"cy":0.0,"xi":2.0,"alpha":0.5}}"#;
    let source: BasaltCamera<f64> = serde_json::from_str(bad).unwrap();
    assert!(CameraModelKind::try_from(&source).is_err());
}
