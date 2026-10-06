//! Landmark residual and estimator Jacobian properties.
#![allow(clippy::unwrap_used)]

use kornia_staging_slam::factors::LinearizePointOut;
use kornia_staging_slam::factors::irls_huber_cost;
use nalgebra::{Matrix2x3, Matrix2x6, Vector2, Vector3, Vector6};
use proptest::prelude::*;
use slam_rs::ba_base::linearize_point;
use slam_rs::calib::Calibration;
use slam_rs::camera::SlamCamera;
use slam_rs::landmark::Landmark;
use slam_rs::lie::{Se3, So3};
use slam_rs::types::{LandmarkId, TimeCamId};

mod common;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn landmark_and_pose_jacobians_match_central_differences(
        direction in prop::array::uniform2(-0.2f64..0.2),
        inv_dist in 0.01f64..2.9,
        translation in prop::array::uniform3(-0.05f64..0.05),
        rotation in prop::array::uniform3(-0.1f64..0.1),
        observation in prop::array::uniform2(100.0f64..500.0),
    ) {
        let pose = Se3::new(So3::exp(&Vector3::from(rotation)), Vector3::from(translation));
        let landmark = Landmark::new(LandmarkId(0), TimeCamId::new(0, 0), Vector2::from(direction), inv_dist);
        let observation = Vector2::from(observation);
        for name in ["msdmi", "msdmg"] {
            let calibration: Calibration<f64> = Calibration::from_json_str(common::calibration_text(name)).unwrap();
            let camera = SlamCamera::from_model(&calibration.intrinsics[0]).unwrap();
            let mut residual = Vector2::zeros();
            let mut jp = Matrix2x3::zeros();
            let mut jx = Matrix2x6::zeros();
            prop_assert!(linearize_point(&observation, &landmark, &pose.matrix(), &camera, &mut residual,
                &mut LinearizePointOut { d_res_d_p: Some(&mut jp), d_res_d_xi: Some(&mut jx), proj: None }), "valid landmark");
            let evaluate = |lm: &Landmark<f64>, t: Se3<f64>| {
                let mut res = Vector2::zeros();
                assert!(linearize_point(&observation, lm, &t.matrix(), &camera, &mut res, &mut LinearizePointOut::default()));
                res
            };
            // Pixel residual derivatives: step 1e-6 and absolute tolerance 1e-5.
            let epsilon = 1e-6;
            for axis in 0..6 {
                let mut step = Vector6::zeros();
                step[axis] = epsilon;
                let numeric = (evaluate(&landmark, Se3::exp(&step) * pose) - evaluate(&landmark, Se3::exp(&-step) * pose)) / (2.0 * epsilon);
                prop_assert!((numeric - jx.column(axis)).amax() < 1e-5);
            }
            for axis in 0..3 {
                let mut plus = landmark.clone();
                let mut minus = landmark.clone();
                if axis < 2 {
                    plus.direction[axis] += epsilon;
                    minus.direction[axis] -= epsilon;
                } else {
                    plus.inv_dist += epsilon;
                    minus.inv_dist -= epsilon;
                }
                let numeric = (evaluate(&plus, pose) - evaluate(&minus, pose)) / (2.0 * epsilon);
                prop_assert!((numeric - jp.column(axis)).amax() < 1e-5);
            }
        }
    }

    #[test]
    fn irls_huber_cost_has_the_closed_form_on_both_sides(
        threshold in 0.1f64..5.0, sigma in 0.1f64..3.0,
        angle in -3.0f64..3.0, ratio in 0.01f64..0.99,
    ) {
        for magnitude in [threshold * ratio, threshold, threshold / ratio] {
            let residual = Vector2::new(angle.cos(), angle.sin()) * magnitude;
            let (_, cost) = irls_huber_cost(&residual, residual.norm(), threshold, sigma);
            let expected = if magnitude <= threshold { 0.5 * magnitude.powi(2) } else { threshold * (magnitude - 0.5 * threshold) } / sigma.powi(2);
            prop_assert!((cost - expected).abs() < 1e-12 * expected.max(1.0));
        }
    }

}
