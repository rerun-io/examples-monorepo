use super::*;

#[test]
fn rejected_projection_retains_pixel_and_leaves_jacobians_untouched() {
    use kornia_staging_3d::camera::{Pinhole, ProjectionReject};
    let camera = kornia_staging_3d::camera::CameraModelKind::Pinhole(
        Pinhole::new([1.0, 1.0, 0.0, 0.0]).unwrap(),
    );
    let mut transform = Matrix4::identity();
    transform[(2, 2)] = -1.0;
    let mut residual = Vector2::repeat(42.0);
    let mut projection = Vector4::repeat(13.0);
    let mut pose_j = Matrix2x6::repeat(17.0);
    let mut landmark_j = Matrix2x3::repeat(19.0);
    let result = linearize_point(
        &Vector2::new(5.0, 6.0),
        &Vector2::zeros(),
        1.0,
        &transform,
        &camera,
        &mut residual,
        &mut LinearizePointOut {
            d_res_d_xi: Some(&mut pose_j),
            d_res_d_p: Some(&mut landmark_j),
            proj: Some(&mut projection),
        },
    );
    assert_eq!(result, Err(ProjectionReject::BelowMinDepth));
    assert_eq!(residual, Vector2::zeros());
    assert_eq!(projection, Vector4::new(0.0, 0.0, 13.0, 13.0));
    assert_eq!(pose_j, Matrix2x6::repeat(17.0));
    assert_eq!(landmark_j, Matrix2x3::repeat(19.0));
}

use nalgebra::Vector2;
#[test]
fn the_irls_huber_cost_matches_a_hand_computation() {
    let above: Vector2<f64> = Vector2::new(3.0, 4.0);
    let (weight, cost) = irls_huber_cost(&above, above.norm(), 2.5, 0.5);
    assert_eq!(weight, 0.5);
    assert_eq!(cost, 37.5);

    let above32: Vector2<f32> = Vector2::new(3.0, 4.0);
    let (weight32, cost32) = irls_huber_cost(&above32, above32.norm(), 2.5, 0.5);
    assert_eq!(weight32, 0.5);
    assert_eq!(cost32, 37.5);

    let below: Vector2<f64> = Vector2::new(0.5, 0.5);
    let (weight, cost) = irls_huber_cost(&below, below.norm(), 1.0, 0.5);
    assert_eq!(weight, 1.0);
    assert_eq!(cost, 1.0);

    let below32: Vector2<f32> = Vector2::new(0.5, 0.5);
    let (weight32, cost32) = irls_huber_cost(&below32, below32.norm(), 1.0, 0.5);
    assert_eq!(weight32, 1.0);
    assert_eq!(cost32, 1.0);

    // The threshold is on the raw pixel norm, before `1/sigma`
    // (papers-part2 §13 ): a residual of exactly the threshold is *not*
    // downweighted only because the comparison is strict `<`.
    let at: Vector2<f64> = Vector2::new(1.0, 0.0);
    assert_eq!(irls_huber_cost(&at, at.norm(), 1.0, 0.5).0, 1.0);
    let just_over: Vector2<f64> = Vector2::new(1.0 + f64::EPSILON, 0.0);
    assert!(irls_huber_cost(&just_over, just_over.norm(), 1.0, 0.5).0 < 1.0);
}

fn pose(seed: f64) -> RigidTransform<f64> {
    RigidTransform::exp(&nalgebra::Vector6::new(
        0.12 * seed,
        -0.03 * seed,
        0.04 * seed,
        0.01 * seed,
        0.02 * seed,
        -0.015 * seed,
    ))
}
fn moved(mut p: RigidTransform<f64>, axis: usize, shift: f64) -> RigidTransform<f64> {
    let mut increment = nalgebra::Vector6::zeros();
    increment[axis] = shift;
    p.apply_inc(&increment);
    p
}

#[test]
fn relative_pose_jacobians_match_finite_differences() {
    let (host, target, host_camera, target_camera) = (pose(1.0), pose(2.0), pose(0.3), pose(-0.4));
    let mut jh = Matrix6::zeros();
    let mut jt = jh;
    let reference = compute_rel_pose(
        &host,
        &host_camera,
        &target,
        &target_camera,
        Some(&mut jh),
        Some(&mut jt),
    );
    for at_host in [false, true] {
        for axis in 0..6 {
            let error = |shift| {
                let rel = if at_host {
                    compute_rel_pose(
                        &moved(host, axis, shift),
                        &host_camera,
                        &target,
                        &target_camera,
                        None,
                        None,
                    )
                } else {
                    compute_rel_pose(
                        &host,
                        &host_camera,
                        &moved(target, axis, shift),
                        &target_camera,
                        None,
                        None,
                    )
                };
                let delta = rel * reference.inverse();
                let omega = delta.rotation.log();
                [
                    delta.translation[0],
                    delta.translation[1],
                    delta.translation[2],
                    omega[0],
                    omega[1],
                    omega[2],
                ]
            };
            let plus = error(1e-8);
            let minus = error(-1e-8);
            for row in 0..6 {
                approx::assert_abs_diff_eq!(
                    (plus[row] - minus[row]) / 2e-8,
                    if at_host {
                        jh[(row, axis)]
                    } else {
                        jt[(row, axis)]
                    },
                    epsilon = 1e-7
                );
            }
        }
    }
}

#[test]
fn hosted_jacobians_match_finite_differences_for_brown_and_kb4() {
    use kornia_staging_3d::camera::{BrownConrady, CameraModelKind, KannalaBrandt4};
    fn check(camera: CameraModelKind<f64>) {
        let transform_pose = pose(0.4);
        let transform = |p: RigidTransform<f64>| p.matrix();
        let direction = Vector2::new(0.1, -0.05);
        let distance = 0.1231231;
        let mut residual = Vector2::zeros();
        let mut jp = Matrix2x6::zeros();
        let mut jl = Matrix2x3::zeros();
        linearize_point(
            &Vector2::zeros(),
            &direction,
            distance,
            &transform(transform_pose),
            &camera,
            &mut residual,
            &mut LinearizePointOut {
                d_res_d_xi: Some(&mut jp),
                d_res_d_p: Some(&mut jl),
                proj: None,
            },
        )
        .unwrap();
        for axis in 0..9 {
            let value = |shift: f64| {
                let mut d = direction;
                let mut inv = distance;
                let mut p = transform_pose;
                if axis < 6 {
                    // Coupled left pose update for the relative transform.
                    let mut dt = [0.0; 3];
                    let mut dr = [0.0; 3];
                    if axis < 3 {
                        dt[axis] = shift;
                    } else {
                        dr[axis - 3] = shift;
                    }
                    p = RigidTransform::exp(&nalgebra::Vector6::new(
                        dt[0], dt[1], dt[2], dr[0], dr[1], dr[2],
                    )) * p;
                } else if axis < 8 {
                    d[axis - 6] += shift;
                } else {
                    inv += shift;
                }
                let mut r = Vector2::zeros();
                linearize_point(
                    &Vector2::zeros(),
                    &d,
                    inv,
                    &transform(p),
                    &camera,
                    &mut r,
                    &mut LinearizePointOut::default(),
                )
                .unwrap();
                r
            };
            let plus = value(1e-7);
            let minus = value(-1e-7);
            for row in 0..2 {
                approx::assert_abs_diff_eq!(
                    (plus[row] - minus[row]) / 2e-7,
                    if axis < 6 {
                        jp[(row, axis)]
                    } else {
                        jl[(row, axis - 6)]
                    },
                    epsilon = 1e-5
                );
            }
        }
    }
    check(CameraModelKind::Kb4(
        KannalaBrandt4::new([400.0, 405.0, 320.0, 240.0, 0.02, -0.003, 0.0005, 0.0001]).unwrap(),
    ));
    check(CameraModelKind::BrownConrady(
        BrownConrady::new(
            [
                400.0, 405.0, 320.0, 240.0, 0.02, -0.003, 0.001, -0.0005, 0.0001, 0.002, -0.0002,
                0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            None,
        )
        .unwrap(),
    ));
}
