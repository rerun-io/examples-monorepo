use kornia_algebra::{Vec2F32, Vec3AF32, Vec3F64, SE3F32, SO3F32, SO3F64};
use kornia_algebra::{Vec2F64, Vec4F32, Vec4F64};
use kornia_staging_3d::pose::{self as geometry, StereographicError};
use proptest::prelude::*;
#[test]
fn south_pole_is_a_typed_failure() {
    assert_eq!(
        geometry::stereographic_project((Vec4F64::new(0.0, 0.0, -1.0, 0.0)).to_array())
            .map(Vec2F64::from_array),
        Err(StereographicError::SouthPole)
    );
    assert_eq!(
        geometry::stereographic_project(
            (Vec4F64::from_array(geometry::stereographic_unproject(
                (Vec2F64::ZERO).to_array()
            )))
            .to_array()
        )
        .map(Vec2F64::from_array)
        .unwrap(),
        Vec2F64::ZERO
    );
}

#[test]
fn behind_camera_parallel_and_nonfinite_bearings() {
    let rotation = SO3F64::IDENTITY;
    let translation = Vec3F64::new(0.1, 0.0, 0.0);
    let behind = Vec3F64::new(0.0, 0.0, -2.0);
    let point = geometry::triangulate_bearing(
        &nalgebra::Vector3::from((-behind.normalize()).to_array()),
        &nalgebra::Vector3::from(
            (-(rotation.inverse() * (behind - translation)).normalize()).to_array(),
        ),
        &kornia_staging_algebra::lie::RigidTransform::new(
            kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(rotation.to_array()),
            (translation.to_array()).into(),
        ),
    )
    .map(Into::into)
    .map(Vec4F64::from_array)
    .unwrap();
    assert!(point.w < 0.0);
    assert!(geometry::triangulate_bearing(
        &nalgebra::Vector3::from((Vec3F64::new(0.0, 0.0, 1.0)).to_array()),
        &nalgebra::Vector3::from((Vec3F64::new(0.0, 0.0, 1.0)).to_array()),
        &kornia_staging_algebra::lie::RigidTransform::new(
            kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(rotation.to_array()),
            (translation.to_array()).into()
        )
    )
    .map(Into::into)
    .map(Vec4F64::from_array)
    .is_none());
    assert!(geometry::triangulate_bearing(
        &nalgebra::Vector3::from((Vec3F64::ZERO).to_array()),
        &nalgebra::Vector3::from((Vec3F64::ZERO).to_array()),
        &kornia_staging_algebra::lie::RigidTransform::new(
            kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(rotation.to_array()),
            (translation.to_array()).into()
        )
    )
    .map(Into::into)
    .map(Vec4F64::from_array)
    .is_none());
    assert!(geometry::triangulate_bearing(
        &nalgebra::Vector3::from((Vec3F64::new(f64::NAN, 0.0, 1.0)).to_array()),
        &nalgebra::Vector3::from((Vec3F64::new(0.0, 0.0, 1.0)).to_array()),
        &kornia_staging_algebra::lie::RigidTransform::new(
            kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(rotation.to_array()),
            (translation.to_array()).into()
        )
    )
    .map(Into::into)
    .map(Vec4F64::from_array)
    .is_none());
    let pose = SE3F32::new(SO3F32::IDENTITY, Vec3AF32::new(0.1, 0.0, 0.0));
    assert!(geometry::triangulate_bearing(
        &nalgebra::Vector3::from((Vec3AF32::new(0.0, 0.0, 1.0)).to_array()),
        &nalgebra::Vector3::from((Vec3AF32::new(0.0, 0.0, 1.0)).to_array()),
        &kornia_staging_algebra::lie::RigidTransform::new(
            kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(pose.r.to_array()),
            (pose.t.to_array()).into()
        )
    )
    .map(Into::into)
    .map(Vec4F32::from_array)
    .is_none());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn project_ignores_the_length(x in -3.0f64..3.0, y in -3.0f64..3.0, z in 0.05f64..4.0, scale in 0.1f64..10.0) {
        let p = Vec4F64::new(x, y, z, 1.0);
        let scaled = Vec4F64::new(x * scale, y * scale, z * scale, 1.0);
        let a = geometry::stereographic_project((p).to_array()).map(Vec2F64::from_array).unwrap();
        let b = geometry::stereographic_project((scaled).to_array()).map(Vec2F64::from_array).unwrap();
        prop_assert!((a - b).length() < 1e-12);
    }
    #[test]
    fn stereographic_round_trips(u in -8.0f64..8.0,v in -8.0f64..8.0) {
        let p=Vec2F64::new(u,v);
        let bearing=Vec4F64::from_array(geometry::stereographic_unproject((p).to_array()));
        prop_assert!((Vec3F64::new(bearing.x,bearing.y,bearing.z).length()-1.0).abs()<1e-12);
        prop_assert!((geometry::stereographic_project((bearing).to_array()).map(Vec2F64::from_array).unwrap()-p).length()<1e-11);
        let p=Vec2F32::new(u as f32,v as f32);
        prop_assert!((geometry::stereographic_project((Vec4F32::from_array(geometry::stereographic_unproject((p).to_array()))).to_array()).map(Vec2F32::from_array).unwrap()-p).length()<1e-3);
    }
    #[test]
    fn triangulation_reprojects_to_both_observations(
        x in -0.5f64..0.5,y in -0.5f64..0.5,depth in 0.5f64..20.0,
        baseline in 0.1f64..0.5,rotation in prop::array::uniform3(-0.1f64..0.1),
    ) {
        let rotation = SO3F64::exp(Vec3F64::from_array(rotation));
        let translation = Vec3F64::new(baseline, 0.0, 0.0);
        let point=Vec3F64::new(x,y,depth);
        let f0=point.normalize();
        let f1=(rotation.inverse() * (point - translation)).normalize();
        let result=geometry::triangulate_bearing(&nalgebra::Vector3::from((f0).to_array()), &nalgebra::Vector3::from((f1).to_array()), &kornia_staging_algebra::lie::RigidTransform::new(kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(rotation.to_array()), (translation.to_array()).into())).map(Into::into).map(Vec4F64::from_array).unwrap();
        prop_assert!(result.w>0.0 && result.w<3.0);
        let reconstructed=Vec3F64::new(result.x,result.y,result.z)/result.w;
        prop_assert!((reconstructed.normalize()-f0).length()<1e-9);
        prop_assert!(((rotation.inverse() * (reconstructed - translation)).normalize()-f1).length()<1e-9);
    }
    #[test]
    fn inverse_distance_cutoffs(delta in 1e-4f32..0.01,baseline in 0.05f32..0.2) {
        for (distance,accepted) in [(3.0-delta,true),(3.0+delta,false)] {
            let pose=SE3F32::new(SO3F32::IDENTITY,Vec3AF32::new(baseline,0.0,0.0));
            let point=Vec3AF32::new(0.0,0.0,1.0/distance);
            let result=geometry::triangulate_bearing(&nalgebra::Vector3::from((Vec3AF32::new(0.0,0.0,1.0)).to_array()), &nalgebra::Vector3::from(((pose.inverse()*point).normalize()).to_array()), &kornia_staging_algebra::lie::RigidTransform::new(kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(pose.r.to_array()), (pose.t.to_array()).into())).map(Into::into).map(Vec4F32::from_array).unwrap();
            prop_assert_eq!(result.w>0.0 && result.w<3.0,accepted);
            prop_assert!((result.w-distance).abs()<2e-6);
            let rotation = SO3F64::IDENTITY;
            let translation = Vec3F64::new(f64::from(baseline), 0.0, 0.0);
            let point=Vec3F64::new(0.0,0.0,1.0/f64::from(distance));
            let result=geometry::triangulate_bearing(&nalgebra::Vector3::from((Vec3F64::new(0.0,0.0,1.0)).to_array()), &nalgebra::Vector3::from(((rotation.inverse() * (point - translation)).normalize()).to_array()), &kornia_staging_algebra::lie::RigidTransform::new(kornia_staging_algebra::lie::Rotation3::from_kornia_quaternion(rotation.to_array()), (translation.to_array()).into())).map(Into::into).map(Vec4F64::from_array).unwrap();
            prop_assert_eq!(result.w>0.0 && result.w<3.0,accepted);
            prop_assert!((result.w-f64::from(distance)).abs()<1e-10);
        }
    }
    // Preserve the SLAM property domain and TestConstants<float> step/tolerance.
    #[test]
    fn chart_jacobian_on_nonunit_points_in_f32(
        x in -2.0f64..2.0, y in -2.0f64..2.0, z in 0.2f64..3.0,
    ) {
        let point = Vec4F32::new(x as f32, y as f32, z as f32, 1.0);
        let (_, jacobian) = geometry::stereographic_project_with_jacobian((point).to_array()).map(|(p,j)| (Vec2F32::from_array(p),j)).unwrap();
        for col in 0..3 {
            let mut plus = point.to_array();
            let mut minus = plus;
            plus[col] += 1e-2;
            minus[col] -= 1e-2;
            let numeric = (geometry::stereographic_project((Vec4F32::from_array(plus)).to_array()).map(Vec2F32::from_array).unwrap()
                - geometry::stereographic_project((Vec4F32::from_array(minus)).to_array()).map(Vec2F32::from_array).unwrap()) / 2e-2;
            let analytic = Vec2F32::from_array(jacobian[col]);
            prop_assert!((numeric - analytic).length() <= 1e-2 * (1.0 + analytic.length()));
        }
    }
    #[test]
    fn chart_jacobians_match_finite_differences(u in -1.0f64..1.0,v in -1.0f64..1.0) {
        let point=Vec2F64::new(u,v);
        let (bearing,j)={ let (p,j) = geometry::stereographic_unproject_with_jacobian((point).to_array()); (Vec4F64::from_array(p),j) };
        let (_,jp)=geometry::stereographic_project_with_jacobian((bearing).to_array()).map(|(p,j)| (Vec2F64::from_array(p),j)).unwrap();
        let h=1e-6;
        for col in 0..2 {
            let mut plus=point.to_array();
        let mut minus=plus;
            plus[col]+=h;
            minus[col]-=h;
            let numeric=(Vec4F64::from_array(geometry::stereographic_unproject((Vec2F64::from_array(plus)).to_array()))-Vec4F64::from_array(geometry::stereographic_unproject((Vec2F64::from_array(minus)).to_array())))/(2.0*h);
            for (actual,expected) in numeric.to_array().into_iter().zip(j[col]) {prop_assert!((actual-expected).abs()<1e-8);}
        }
        for col in 0..4 {
            let mut plus=bearing.to_array();
        let mut minus=plus;
            plus[col]+=h;
            minus[col]-=h;
            let numeric=(geometry::stereographic_project((Vec4F64::from_array(plus)).to_array()).map(Vec2F64::from_array).unwrap()-geometry::stereographic_project((Vec4F64::from_array(minus)).to_array()).map(Vec2F64::from_array).unwrap())/(2.0*h);
            for (actual,expected) in numeric.to_array().into_iter().zip(jp[col]) {prop_assert!((actual-expected).abs()<1e-8);}
        }
    }
}
