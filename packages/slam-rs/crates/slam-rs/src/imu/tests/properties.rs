//! Gravity-alignment properties owned by the SLAM initialization policy.
use super::super::gravity_from_first_accel;
use nalgebra::Vector3;
use proptest::prelude::*;

#[test]
fn gravity_init_deviation_stays_within_its_bound() {
    for tilt in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] {
        let a = Vector3::new(tilt, -tilt, -1.0).normalize();
        assert!(
            (gravity_from_first_accel(&a) * a - Vector3::z()).norm() <= 2.0 * (2e-12_f64).sqrt()
        );
        let a = a.map(|v| v as f32);
        assert!(
            (gravity_from_first_accel(&a) * a - Vector3::z()).norm() <= 2.0 * (2e-5_f32).sqrt()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn a_rotation_maps_one_direction_to_the_other(
        a in prop::array::uniform3(-20.0f64..20.0),
        b in prop::array::uniform3(-20.0f64..20.0),
        antiparallel in any::<bool>()
    ) {
        let a = Vector3::from(a);
        let b = if antiparallel { -a } else { Vector3::from(b) };
        prop_assume!(a.norm() > 0.01 && b.norm() > 0.01);
        // The public API aligns to +Z. Compose two such rotations to test arbitrary pairs.
        let rotation = gravity_from_first_accel(&b).inverse() * gravity_from_first_accel(&a);
        prop_assert!((rotation * a.normalize() - b.normalize()).norm() < 6e-6);
    }
}
