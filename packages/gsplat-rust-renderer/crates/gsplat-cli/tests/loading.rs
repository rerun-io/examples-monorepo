//! File formats enter through one shared camera/upload boundary.
use gsplat_cli::camera::{CameraModel, load_frames};

#[tokio::test]
async fn ply_metadata_selects_mip_without_changing_raw_parameters() {
    let path = std::env::temp_dir().join(format!("gsplat-metadata-{}.ply", std::process::id()));
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
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn colmap_text_preserves_distortion_and_inverts_pose() {
    let root = std::env::temp_dir().join(format!("gsplat-colmap-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
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
    let frames = load_frames(&root, None).await.unwrap();
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
    let resized = load_frames(&root, Some((320, 240))).await.unwrap();
    assert_eq!(resized[0].camera.fx, 200.0);
    std::fs::remove_dir_all(root).unwrap();
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
