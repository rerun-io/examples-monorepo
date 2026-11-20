use super::double_sphere::{bearing, rear_round_trip};
use super::*;

// Lens 0 of the X4 Air V3 and V6 factory strings at 1920 px (`formats::insta360`), rounded.
const X4AIR_V3: ([f64; 5], [f64; 5]) = (
    [1834.6, 1834.2, 960.242, 965.681, 2.31494],
    [0.96273077, -2.06431413, 4.31214333, 0.00172895, -0.00121632],
);
#[allow(clippy::excessive_precision)]
const X4AIR_V6: ([f64; 5], [f64; 13]) = (
    [1834.9, 1834.5, 958.292, 965.869, 2.31494],
    [
        0.93861496,
        -1.68402135,
        2.19166017,
        3.85762072,
        0.0,
        -0.00495962,
        0.00156076,
        -0.00330722,
        0.00397486,
        0.00827898,
        -0.00125315,
        0.02416405,
        -0.00143547,
    ],
);
const HALF_FIELD: f64 = 100.0 * std::f64::consts::PI / 180.0;

fn v3<S: Scalar>() -> MeiDistortion<S> {
    let (_, d) = X4AIR_V3;
    MeiDistortion::V3 {
        k: [c(d[0]), c(d[1]), c(d[2])],
        p: [c(d[3]), c(d[4])],
    }
}

fn v6<S: Scalar>() -> MeiDistortion<S> {
    MeiDistortion::V6 {
        d: X4AIR_V6.1.map(c),
    }
}

fn camera<S: Scalar>(param: [f64; 5], distortion: MeiDistortion<S>) -> Mei<S> {
    Mei::new(param.map(c), c(HALF_FIELD), distortion).unwrap()
}

#[test]
fn mei_jacobians_and_round_trips_both_precisions() {
    for (param, d64, d32) in [
        (X4AIR_V3.0, v3::<f64>(), v3::<f32>()),
        (X4AIR_V6.0, v6(), v6()),
        (
            [400.0, 410.0, 320.0, 240.0, 0.8],
            MeiDistortion::V3 {
                k: [0.1, -0.05, 0.01],
                p: [0.002, -0.001],
            },
            MeiDistortion::V3 {
                k: [0.1, -0.05, 0.01],
                p: [0.002, -0.001],
            },
        ),
    ] {
        sweep(param, |p| Mei::new(p, HALF_FIELD, d64).unwrap(), 1e-6, 1e-6);
        sweep(
            param.map(|v| v as f32),
            |p| Mei::new(p, HALF_FIELD as f32, d32).unwrap(),
            1e-3,
            0.08,
        );
    }
}

#[test]
fn mei_admits_rear_rays_up_to_its_half_field() {
    for (param, d64, d32) in [
        (X4AIR_V3.0, v3::<f64>(), v3::<f32>()),
        (X4AIR_V6.0, v6(), v6()),
    ] {
        let wide = camera(param, d64);
        rear_round_trip(wide, camera(param, d32), 95.0);
        // Pixels come back within 1e-9 px over the whole field, the rear rays included.
        for theta in (0..=99).step_by(3) {
            for phi in (0..360).step_by(20) {
                let pixel = wide
                    .project(bearing(f64::from(theta), f64::from(phi)))
                    .unwrap();
                let again = wide.project(wide.unproject(pixel).unwrap()).unwrap();
                assert!(
                    (again[0] - pixel[0]).abs() < 1e-9 && (again[1] - pixel[1]).abs() < 1e-9,
                    "{theta} {phi}: {pixel:?} {again:?}"
                );
            }
        }
    }
}

#[test]
fn mei_rejects_at_and_beyond_the_field_edge() {
    let camera = camera(X4AIR_V3.0, v3::<f64>());
    // Just inside the edge projects; just past it does not.
    assert!(camera.project(bearing(99.99, 30.0)).is_ok());
    let (pixel, status) = camera.project_with_status(bearing(100.5, 30.0), None, None);
    assert_eq!(status, Err(ProjectionReject::OutsideDomain));
    assert!(pixel.iter().all(|v| v.is_finite()));
    assert_eq!(
        camera.project(bearing(180.0, 0.0)),
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
    // A pixel beyond the image circle (radius about 972 px at 100 degrees) has no ray in the field.
    let outside = camera.unproject([960.0 + 990.0, 965.0]);
    assert!(
        matches!(
            outside,
            Err(UnprojectError::OutsideDomain | UnprojectError::NoConvergence)
        ),
        "{outside:?}"
    );
    let d = v3::<f64>();
    for (param, half) in [
        ([1.0, 0.0, 0.0, 0.0, 1.0], HALF_FIELD),
        ([1.0, 1.0, 0.0, 0.0, -0.1], HALF_FIELD),
        ([1.0, 1.0, f64::NAN, 0.0, 1.0], HALF_FIELD),
        ([1.0, 1.0, 0.0, 0.0, 1.0], 0.0),
        ([1.0, 1.0, 0.0, 0.0, 1.0], 3.2),
        // xi = 2 folds the sphere beyond acos(-1/2) = 120 degrees.
        ([1.0, 1.0, 0.0, 0.0, 2.0], 2.1),
    ] {
        assert_eq!(
            Mei::new(param, half, d),
            Err(InvalidCalibration),
            "{param:?} {half}"
        );
    }
    assert_eq!(
        Mei::new(
            [1.0, 1.0, 0.0, 0.0, 1.0],
            1.0,
            MeiDistortion::V6 {
                d: [f64::INFINITY; 13]
            }
        ),
        Err(InvalidCalibration)
    );
}

#[test]
fn mei_with_xi_at_most_one_rejects_a_field_past_the_denominator_fold() {
    // For xi <= 1 the map folds where z + xi |P| = 0, at acos(-xi): 143.13 degrees for xi = 0.8.
    let param = [400.0, 410.0, 320.0, 240.0, 0.8];
    let d = v3::<f64>();
    for (xi, half_deg) in [(0.8, 160.0), (0.8, 143.2), (1.0, 180.0), (0.0, 91.0)] {
        let param = [param[0], param[1], param[2], param[3], xi];
        assert_eq!(
            Mei::new(param, f64::to_radians(half_deg), d),
            Err(InvalidCalibration),
            "xi {xi} half field {half_deg}"
        );
    }
    let fold_deg = (-0.8f64).acos().to_degrees();
    let camera = Mei::new(param, (fold_deg - 0.1).to_radians(), d).unwrap();
    let pixel = camera.project(bearing(fold_deg - 0.2, 30.0)).unwrap();
    assert!(pixel.iter().all(|v| v.is_finite()), "{pixel:?}");
    assert_eq!(
        camera.project(bearing(150.0, 30.0)),
        Err(ProjectionReject::OutsideDomain)
    );
}

#[test]
fn mei_is_one_camera_model_kind_and_round_trips_through_serde() {
    let mei = camera(X4AIR_V6.0, v6::<f64>());
    let kind = CameraModelKind::Mei(mei);
    let point = bearing(97.0, 40.0);
    assert_eq!(kind.project(point), mei.project(point));
    #[cfg(feature = "serde")]
    {
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(
            serde_json::from_str::<CameraModelKind<f64>>(&json).unwrap(),
            kind
        );
        let mut params: crate::camera::formats::MeiParams<f64> = mei.into();
        params.xi = -1.0;
        assert!(
            serde_json::from_str::<Mei<f64>>(&serde_json::to_string(&params).unwrap()).is_err()
        );
        assert!(matches!(
            crate::camera::formats::BasaltCamera::try_from(kind),
            Err(crate::camera::formats::ConversionError::Unsupported(_))
        ));
    }
}
