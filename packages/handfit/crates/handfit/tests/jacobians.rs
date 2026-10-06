use handfit::model::{LandmarkJacobian, Step};
use handfit::residual::{camera_model, project_camera};
use handfit::{Model, Pose};
use nalgebra::{Rotation3, SMatrix, SVector, Vector2, Vector3};

// A deterministic random stream keeps the tests reproducible without another crate.
struct Random(u64);
impl Random {
    fn signed(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0
    }
}
fn check_errors(name: &str, mut errors: Vec<f64>) {
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    let worst = *errors.last().unwrap();
    println!(
        "{name}: {} columns, median relative {median:.4e}, worst {worst:.4e}",
        errors.len()
    );
    assert!(
        median <= 1e-6 && worst <= 1e-4,
        "{name}: median {median}, worst {worst}"
    );
}

#[test]
fn camera_jacobians_match_random_central_differences() {
    let mut rng = Random(42);
    let mut errors = [Vec::new(), Vec::new()];
    for _ in 0..200 {
        let focal = Vector2::new(600.0 + rng.signed() * 100.0, 610.0 + rng.signed() * 100.0);
        let principal = Vector2::new(900.0 + rng.signed() * 50.0, 500.0 + rng.signed() * 50.0);
        // Exercise positive and negative z, all six radial and both tangential terms.
        let point = Vector3::new(
            rng.signed(),
            rng.signed(),
            rng.signed().signum() * (0.2 + rng.signed().abs()),
        );
        let distortion =
            SVector::<f64, 8>::from_fn(|i, _| rng.signed() * 0.03 * 0.1_f64.powi(i.min(5) as i32));
        for lens in 0..2 {
            let camera =
                camera_model(&focal, &principal, (lens == 1).then_some(&distortion)).unwrap();
            let project = |p: &Vector3<f64>, j: Option<&mut SMatrix<f64, 2, 3>>| {
                project_camera(&camera, p, j)
            };
            let mut jac = SMatrix::<f64, 2, 3>::zeros();
            project(&point, Some(&mut jac));
            for k in 0..3 {
                let mut plus = point;
                let mut minus = point;
                plus[k] += 1e-6;
                minus[k] -= 1e-6;
                let numeric = (project(&plus, None) - project(&minus, None)) / 2e-6;
                errors[lens].push((numeric - jac.column(k)).norm() / numeric.norm().max(1e-8));
            }
        }
    }
    check_errors("pinhole", errors[0].clone());
    check_errors("fisheye62", errors[1].clone());
}

#[test]
fn landmark_jacobian_matches_random_central_differences() {
    let mut rng = Random(97);
    let mut errors = Vec::new();
    for trial in 0..60 {
        let mut axes = SMatrix::<f64, 20, 3>::from_fn(|_, _| rng.signed());
        for i in 0..20 {
            let length = axes.row(i).norm();
            axes.row_mut(i).scale_mut(1.0 / length);
        }
        let model = Model {
            axes,
            pivots: SMatrix::from_fn(|_, _| rng.signed() * 40.0),
            rest: SMatrix::from_fn(|_, _| rng.signed() * 70.0),
            weights: SMatrix::from_fn(|i, k| {
                if i == 20 {
                    [0.85, 0.10, 0.05][k]
                } else if k == 0 {
                    1.0
                } else {
                    0.0
                }
            }),
            limits: SMatrix::zeros(),
        };
        let pose = Pose {
            rotation: Rotation3::from_scaled_axis(Vector3::new(
                rng.signed(),
                rng.signed(),
                rng.signed(),
            ))
            .into_inner(),
            translation: Vector3::new(rng.signed(), rng.signed(), rng.signed()),
            angles: SVector::from_fn(|_, _| {
                if trial % 3 == 0 {
                    rng.signed() * 0.005
                } else if trial % 3 == 1 {
                    rng.signed().signum() * 0.02
                } else {
                    rng.signed()
                }
            }),
        };
        let mirror = if trial % 2 == 0 { 1.0 } else { -1.0 };
        let mut jac = LandmarkJacobian::zeros();
        model.landmarks(&pose, mirror, &Step::zeros(), Some(&mut jac));
        for k in 0..26 {
            let mut delta = Step::zeros();
            delta[k] = 1e-6;
            let plus = model.landmarks(&pose, mirror, &delta, None);
            delta[k] = -1e-6;
            let minus = model.landmarks(&pose, mirror, &delta, None);
            let numeric = (plus - minus) / 2e-6;
            errors.push((numeric - jac.column(k)).norm() / numeric.norm().max(1e-8));
        }
    }
    check_errors("landmarks", errors);
}

#[test]
fn camera_optical_axis_and_pinhole_z_safe() {
    let focal = Vector2::new(500.0, 600.0);
    let principal = Vector2::new(320.0, 240.0);
    let mut jac = SMatrix::<f64, 2, 3>::zeros();
    let p = Vector3::new(0.0, 0.0, 1.0);
    assert_eq!(
        project_camera(
            &camera_model(&focal, &principal, Some(&SVector::zeros())).unwrap(),
            &p,
            Some(&mut jac)
        ),
        principal
    );
    assert!((jac[(0, 0)] - 500.0).abs() < 1e-7);
    assert!((jac[(1, 1)] - 600.0).abs() < 1e-7);
    assert_eq!(jac[(0, 2)], 0.0);
    for z in [-0.5e-9, -1e-10, 0.0, 0.5e-9] {
        let p = Vector3::new(1e-9, 2e-9, z);
        let got = project_camera(
            &camera_model(&focal, &principal, None).unwrap(),
            &p,
            Some(&mut jac),
        );
        assert!((got - Vector2::new(820.0, 1440.0)).norm() < 1e-10);
        assert_eq!(jac[(0, 2)], 0.0);
        assert_eq!(jac[(1, 2)], 0.0);
    }
}
