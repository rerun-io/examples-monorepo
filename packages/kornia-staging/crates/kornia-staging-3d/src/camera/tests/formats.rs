use super::*;

#[cfg(feature = "serde")]
#[test]
fn canonical_tags_and_colmap_pixel_origin_round_trip() {
    use crate::camera::formats::ColmapCamera;
    let source = ColmapCamera {
        model_id: 1,
        params: vec![400.0, 410.0, 320.5, 240.5],
    };
    let camera = CameraModelKind::try_from(&source).unwrap();
    assert_eq!(camera.project([0.0, 0.0, 1.0]).unwrap(), [320.0, 240.0]);
    let json = serde_json::to_string(&camera).unwrap();
    assert!(json.contains("pinhole"));
    assert_eq!(
        serde_json::from_str::<CameraModelKind<f64>>(&json).unwrap(),
        camera
    );
    assert_eq!(camera.to_colmap(1).unwrap(), source);
    for id in [7, 10, 12, 13, 16] {
        assert!(CameraModelKind::try_from(&ColmapCamera {
            model_id: id,
            params: vec![]
        })
        .is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn every_supported_colmap_id_round_trips_and_restrictions_are_checked() {
    use crate::camera::formats::ColmapCamera;
    for (id, count, tied) in [
        (0, 3, true),
        (1, 4, false),
        (2, 4, true),
        (3, 5, true),
        (4, 8, false),
        (5, 8, false),
        (6, 12, false),
        (8, 4, true),
        (9, 5, true),
        (11, 16, false),
    ] {
        let mut params = if tied {
            vec![400.0, 320.5, 240.5]
        } else {
            vec![400.0, 410.0, 320.5, 240.5]
        };
        params.resize(count, 0.001);
        let source = ColmapCamera {
            model_id: id,
            params,
        };
        let camera = CameraModelKind::try_from(&source).unwrap();
        assert_eq!(camera.to_colmap(id).unwrap(), source);
        let text = serde_json::to_string(&camera).unwrap();
        let back: CameraModelKind<f64> = serde_json::from_str(&text).unwrap();
        assert_eq!(back, camera);
        assert_eq!(
            back.unproject([330.0, 250.0]).unwrap(),
            camera.unproject([330.0, 250.0]).unwrap()
        );
    }
    let camera = CameraModelKind::Kb4(right_front());
    assert!(camera.to_colmap(8).is_err());
    let camera = CameraModelKind::BrownConrady(
        BrownConrady::new([1.0; 18], None).expect("valid camera calibration"),
    );
    assert!(camera.to_colmap(6).is_err());
    assert!(CameraModelKind::try_from(&ColmapCamera {
        model_id: 1,
        params: vec![0.0; 4]
    })
    .is_err());
    assert!(CameraModelKind::try_from(&ColmapCamera {
        model_id: 1,
        params: vec![1.0; 3]
    })
    .is_err());
}

#[cfg(feature = "serde")]
#[test]
fn basalt_preserves_supported_named_parameters() {
    use crate::camera::formats::BasaltCamera;
    for text in [
        r#"{"camera_type":"pinhole","intrinsics":{"fx":400.0,"fy":410.0,"cx":320.0,"cy":240.0}}"#,
        r#"{"camera_type":"kb4","intrinsics":{"fx":400.0,"fy":410.0,"cx":320.0,"cy":240.0,"k1":0.1,"k2":-0.01,"k3":0.001,"k4":0.0}}"#,
        r#"{"camera_type":"pinhole-radtan8","intrinsics":{"fx":400.0,"fy":410.0,"cx":320.0,"cy":240.0,"k1":0.1,"k2":-0.01,"p1":0.002,"p2":-0.003,"k3":0.001,"k4":0.001,"k5":0.0,"k6":0.0,"rpmax":2.5}}"#,
    ] {
        let source: BasaltCamera<f64> = serde_json::from_str(text).unwrap();
        let camera = CameraModelKind::try_from(&source).unwrap();
        assert_eq!(BasaltCamera::try_from(camera).unwrap(), source);
    }
}
