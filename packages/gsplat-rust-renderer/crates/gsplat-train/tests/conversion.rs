//! Native component conversion against a pretrained PLY reference.
use gsplat_train::retains;
use std::path::PathBuf;

#[tokio::test]
#[ignore = "golden: requires pretrained Lego PLY and GPU; set GSPLAT_TEST_PLY"]
async fn pretrained_lego_conversion_matches_rerun_loader() {
    let path = PathBuf::from(std::env::var_os("GSPLAT_TEST_PLY").expect("set GSPLAT_TEST_PLY"));
    let expected = rerun::GaussianSplats3D::from_ply_file_path(&path).unwrap();
    let parsed = brush_serde::load_splat_from_ply(tokio::fs::File::open(path).await.unwrap(), None)
        .await
        .unwrap();
    let device = brush_process::burn_init_setup().await;
    let splats = parsed.data.into_splats(
        &device,
        brush_render::gaussian_splats::SplatRenderMode::Default,
    );
    let actual = gsplat_train::read_splats(splats, true).await.unwrap();
    assert_eq!(actual.centers, expected.centers);
    assert_eq!(actual.scales, expected.scales);
    assert_eq!(actual.quaternions, expected.quaternions);
    assert_eq!(actual.colors, expected.colors);
    assert_eq!(actual.sh_coefficients, expected.sh_coefficients);
}

#[test]
fn retention_keeps_first_stride_and_final_once() {
    let steps: Vec<_> = (1..=7000)
        .filter(|&step| retains(step, 7000, 50, 1000))
        .collect();
    assert_eq!(steps, [50, 1000, 2000, 3000, 4000, 5000, 6000, 7000]);
    assert!(retains(37, 37, 50, 1000));
    assert!(!retains(0, 7000, 50, 1000));
}
