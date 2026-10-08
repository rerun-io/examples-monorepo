use approx::assert_abs_diff_eq;
use kornia_staging_algebra::lie::RigidTransform;
use nalgebra::Vector6;

#[test]
fn coupled_maps_preserve_the_near_zero_and_signed_angle_branches() {
    for angle in [
        1e-12,
        1e-9,
        1e-5,
        std::f64::consts::PI - 1e-6,
        std::f64::consts::PI + 0.5,
    ] {
        let pose = RigidTransform::exp(&Vector6::new(0.2, -0.3, 1.0, 0.0, 0.0, angle));
        let back = RigidTransform::exp(&pose.log());
        // Preserve the source's cancellation at 1e-9, above its Taylor cutoff.
        // A direct comparison to the untouched implementation is bit-exact.
        let tolerance = if angle == 1e-9 { 5e-10 } else { 1e-10 };
        for (a, b) in pose.matrix().iter().zip(back.matrix().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = tolerance);
        }
    }
}

#[test]
fn inverse_jacobians_keep_both_precision_branches_at_zero_and_pi() {
    use kornia_staging_algebra::lie::{
        left_jacobian_inv_so3, right_jacobian_inv_so3, right_jacobian_so3,
    };
    use nalgebra::{Matrix3, Vector3};
    for angle in [0.0, 1e-9, 1e-3, std::f64::consts::PI] {
        let phi = Vector3::new(angle, 0.0, 0.0);
        assert_abs_diff_eq!(
            right_jacobian_inv_so3(&phi) * right_jacobian_so3(&phi),
            Matrix3::identity(),
            epsilon = 1e-12
        );
    }
    let boundary = std::f32::consts::PI - 1e-5f32.sqrt();
    for angle in [
        f32::from_bits(boundary.to_bits() - 1),
        boundary,
        f32::from_bits(boundary.to_bits() + 1),
        std::f32::consts::PI,
    ] {
        let narrow = Vector3::new(angle, 0.0, 0.0);
        let wide = narrow.map(f64::from);
        assert_abs_diff_eq!(
            right_jacobian_inv_so3(&narrow).map(f64::from),
            right_jacobian_inv_so3(&wide),
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            left_jacobian_inv_so3(&narrow).map(f64::from),
            left_jacobian_inv_so3(&wide),
            epsilon = 1e-3
        );
    }
}
