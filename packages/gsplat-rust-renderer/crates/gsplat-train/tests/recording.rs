//! Opt-in GPU/asset integration: GSPLAT_LEGO and GSPLAT_TEST_OUTPUT identify inputs.
use std::{collections::BTreeSet, io::BufReader, path::PathBuf, process::Command};

#[test]
#[ignore = "integration: requires a GPU and GSPLAT_LEGO dataset"]
fn lego_training_recording_contains_snapshots_cameras_curves_and_eval_pairs() {
    let dataset = std::env::var_os("GSPLAT_LEGO").expect("set GSPLAT_LEGO to the Lego dataset");
    let output = std::env::var_os("GSPLAT_TEST_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("gsplat-train-{}", std::process::id()))
        });
    std::fs::create_dir_all(&output).unwrap();
    let recording = output.join("training.rrd");
    let unused_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let disconnected_sink = format!(
        "rerun+http://{}/proxy",
        unused_listener.local_addr().unwrap()
    );
    drop(unused_listener);
    assert!(
        Command::new(env!("CARGO_BIN_EXE_gsplat-train"))
            .arg(dataset)
            .args([
                "--total-train-iters",
                "200",
                "--eval-every",
                "100",
                "--refine-every",
                "50",
                "--rerun-log-train-stats-every",
                "5",
                "--save"
            ])
            .arg(&recording)
            .arg("--connect")
            .arg(disconnected_sink)
            .arg("--export-path")
            .arg(output.join("exports"))
            .status()
            .unwrap()
            .success()
    );
    let mut entities = BTreeSet::new();
    let mut snapshots = Vec::new();
    let mut loss_steps = Vec::new();
    let mut learning_rate_steps = Vec::new();
    let mut refine_steps = Vec::new();
    let mut render_steps = Vec::new();
    let expected =
        rerun::GaussianSplats3D::from_ply_file_path(&output.join("exports/export_200.ply"))
            .unwrap();
    for message in re_log_encoding::Decoder::<rerun::log::LogMsg>::decode_eager(BufReader::new(
        std::fs::File::open(recording).unwrap(),
    ))
    .unwrap()
    {
        if let rerun::log::LogMsg::ArrowMsg(_, message) = message.unwrap() {
            let chunk = re_chunk::Chunk::from_arrow_msg(&message).unwrap();
            let entity = chunk.entity_path().to_string();
            entities.insert(entity.clone());
            if entity == "/refine/num_added" {
                refine_steps.extend(
                    chunk
                        .timelines()
                        .values()
                        .find(|t| t.timeline().name().as_str() == "iterations")
                        .unwrap()
                        .times_raw()
                        .iter()
                        .copied(),
                );
            }
            if entity == "/eval/view_0/render" {
                render_steps.extend(
                    chunk
                        .timelines()
                        .values()
                        .find(|t| t.timeline().name().as_str() == "iterations")
                        .unwrap()
                        .times_raw()
                        .iter()
                        .copied(),
                );
            }
            if entity == "/world/splats" {
                let timeline = chunk
                    .timelines()
                    .values()
                    .find(|t| t.timeline().name().as_str() == "iterations")
                    .unwrap();
                snapshots.extend(timeline.times_raw().iter().copied());
                for (row, step) in timeline.times_raw().iter().enumerate() {
                    let sh = expected.sh_coefficients.as_ref().unwrap();
                    let actual_sh = chunk.component_batch_raw(sh.descriptor.component, row);
                    let degree = rerun::GaussianSplats3D::new([[0.0; 3]])
                        .with_spherical_harmonics_degree(if *step == 200 { 3 } else { 0 })
                        .spherical_harmonics_degree
                        .unwrap();
                    assert_eq!(
                        *chunk
                            .component_batch_raw(degree.descriptor.component, row)
                            .unwrap()
                            .unwrap(),
                        *degree.array
                    );
                    if *step != 200 {
                        assert!(actual_sh.is_none(), "intermediate snapshot must be DC-only");
                        continue;
                    }
                    assert_eq!(*actual_sh.unwrap().unwrap(), *sh.array);
                    let colors = expected.colors.as_ref().unwrap();
                    assert_eq!(
                        *chunk
                            .component_batch_raw(colors.descriptor.component, row)
                            .unwrap()
                            .unwrap(),
                        *colors.array
                    );
                    // Both paths must bake the training min-scale filter identically.
                    use rerun::external::arrow::array::{Array, FixedSizeListArray, Float32Array};
                    for batch in [&expected.centers, &expected.scales, &expected.quaternions] {
                        let batch = batch.as_ref().unwrap();
                        let actual = chunk
                            .component_batch_raw(batch.descriptor.component, row)
                            .unwrap()
                            .unwrap();
                        let actual = actual
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .values()
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .unwrap();
                        let expected = batch
                            .array
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .values()
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .unwrap();
                        assert_eq!(actual.len(), expected.len());
                        for (actual, expected) in actual.values().iter().zip(expected.values()) {
                            assert!(
                                actual.to_bits().abs_diff(expected.to_bits()) <= 1,
                                "{}: {actual} != {expected}",
                                batch.descriptor.component
                            );
                        }
                    }
                }
            }
            if entity == "/loss/total" || entity == "/lr/mean" {
                let timeline = chunk
                    .timelines()
                    .values()
                    .find(|t| t.timeline().name().as_str() == "iterations")
                    .unwrap();
                if entity == "/loss/total" {
                    loss_steps.extend(timeline.times_raw().iter().copied());
                } else {
                    learning_rate_steps.extend(timeline.times_raw().iter().copied());
                }
            }
        }
    }
    snapshots.sort_unstable();
    assert_eq!(snapshots, [50, 200]);
    let expected_stats: Vec<i64> = (5..=200).step_by(5).collect();
    loss_steps.sort_unstable();
    learning_rate_steps.sort_unstable();
    assert_eq!(loss_steps, expected_stats);
    assert_eq!(learning_rate_steps, expected_stats);
    refine_steps.sort_unstable();
    assert_eq!(refine_steps, [51, 101, 151]);
    render_steps.sort_unstable();
    assert_eq!(render_steps, [100, 200]);
    for path in [
        "/world/dataset/camera/0",
        "/loss/total",
        "/psnr/eval",
        "/ssim/eval",
        "/lr/mean",
        "/splats/num_splats",
        "/eval/view_3/ground_truth",
        "/eval/view_3/render",
    ] {
        assert!(entities.contains(path), "missing entity {path}");
    }
}

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
