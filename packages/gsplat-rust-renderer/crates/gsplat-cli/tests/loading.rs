//! File formats enter through one shared camera/upload boundary.
use gsplat_cli::camera_files::load_frames;
use gsplat_core::CameraModel;

#[tokio::test]
async fn ply_metadata_selects_mip_without_changing_raw_parameters() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("scene.ply");
    std::fs::write(
        &path,
        concat!(
            "ply\n",
            "format ascii 1.0\n",
            "comment splatrendermode: mip\n",
            "element vertex 1\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "end_header\n",
            "0 0 2\n",
        ),
    )
    .unwrap();
    let scene = gsplat_cli::PlyScene::load(&path).await.unwrap();
    assert_eq!(scene.mode, gsplat_core::RenderMode::Mip);
    assert_eq!(
        gsplat_cli::raw_splats(&scene.data).unwrap().transforms[0][2],
        2.0
    );
}

#[tokio::test]
async fn colmap_text_preserves_distortion_and_inverts_pose() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    std::fs::write(
        root.join("cameras.txt"),
        "1 FULL_OPENCV 640 480 400 410 310 230 0.1 0.02 0.001 -0.002 0.003 0.004 0.005 0.006\n",
    )
    .unwrap();
    std::fs::write(
        root.join("images.txt"),
        "1 1 0 0 0 -1 -2 -3 1 nested/frame.jpg\n\n",
    )
    .unwrap();
    let frames = load_frames(root, None).await.unwrap();
    assert_eq!(frames.len(), 1);
    let camera = &frames[0].camera;
    assert_eq!(
        camera.pose().w_axis.truncate(),
        glam::Vec3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        frames[0].file_path,
        std::path::PathBuf::from("nested/frame.jpg")
    );
    assert_eq!(camera.fx, 400.0);
    assert!(matches!(
        camera.model,
        CameraModel::RadialTangential8 {
            k3: 0.003,
            k4: 0.004,
            p1: 0.001,
            p2: -0.002,
            ..
        }
    ));
    let resized = load_frames(root, Some((320, 240))).await.unwrap();
    assert_eq!(resized[0].camera.fx, 200.0);
}

#[test]
fn raw_upload_preserves_precision_and_brush_defaults() {
    let data = brush_serde::import::SplatData {
        means: vec![1.234567, 2.0, 3.0],
        rotations: None,
        log_scales: None,
        sh_coeffs: None,
        raw_opacities: None,
    };
    let raw = gsplat_cli::raw_splats(&data).unwrap();
    assert_eq!(
        raw.transforms,
        vec![[1.234567, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, -4.0, -4.0, -4.0]]
    );
    assert_eq!(raw.sh_coefficients, vec![[0.5; 3]]);
    assert_eq!(raw.raw_opacities, vec![0.0]);
    let invalid = brush_serde::import::SplatData {
        sh_coeffs: Some(vec![0.0; 7]),
        ..data
    };
    assert!(gsplat_cli::raw_splats(&invalid).is_err());
}

#[test]
fn raw_upload_preserves_dc_and_coefficient_major_rgb() {
    let data = brush_serde::import::SplatData {
        means: vec![0.0; 3],
        rotations: None,
        log_scales: None,
        sh_coeffs: Some(vec![-0.25, 0.5, 1.25]),
        raw_opacities: None,
    };
    let raw = gsplat_cli::raw_splats(&data).unwrap();
    assert_eq!(raw.sh_degree, 0);
    assert_eq!(raw.sh_coefficients, [[-0.25, 0.5, 1.25]]);
    let higher = brush_serde::import::SplatData {
        sh_coeffs: Some(vec![
            -0.25, 0.5, 1.25, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, 0.1, 0.2, 0.3,
        ]),
        ..data
    };
    let raw = gsplat_cli::raw_splats(&higher).unwrap();
    assert_eq!(raw.sh_degree, 1);
    assert_eq!(
        raw.sh_coefficients,
        [
            [-0.25, 0.5, 1.25],
            [2.0, 3.0, 4.0],
            [-1.0, -2.0, -3.0],
            [0.1, 0.2, 0.3]
        ]
    );
}

#[tokio::test]
async fn upstream_hexel_cameras_centre_the_principal_point_and_invert_the_view() {
    let temporary = tempfile::tempdir().unwrap();
    // Camera at (1, 2, 3) looking down world +z: world_view_transform is the transposed w2c.
    let record = serde_json::json!({
        "name": "0008", "width": 778, "height": 580, "fov_x": 1.0, "fov_y": 0.8,
        "world_view_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-1, -2, -3, 1]],
        "camera_center": [1, 2, 3],
    });
    for document in [
        serde_json::json!({"format": "hexels-cameras-1", "background": [0, 0, 0], "cameras": [record]}),
        serde_json::json!({"settings": {}, "geometry": {"training": [], "test": [record]}}),
    ] {
        let path = temporary.path().join("cameras.json");
        std::fs::write(&path, document.to_string()).unwrap();
        let frames = load_frames(&path, None).await.unwrap();
        let camera = &frames[0].camera;
        assert_eq!(frames[0].file_path, std::path::PathBuf::from("0008"));
        assert_eq!(
            (camera.width, camera.height, camera.cx, camera.cy),
            (778, 580, 389.0, 290.0)
        );
        assert!((camera.fx - 778.0 / (2.0 * 0.5f32.tan())).abs() < 1e-3);
        assert!((camera.fy - 580.0 / (2.0 * 0.4f32.tan())).abs() < 1e-3);
        assert_eq!(
            camera.pose().w_axis.truncate(),
            glam::Vec3::new(1.0, 2.0, 3.0)
        );
    }
}
