//! GPU readback and logging run off the trainer thread, using captured tensor handles.
use anyhow::Context;
use brush_dataset::{Dataset, scene::SceneView};
use brush_render::gaussian_splats::Splats;
use brush_train::msg::{RefineStats, TrainStepStats};
use rerun::{RecordingStream, Scalars};
use std::{sync::mpsc::Receiver, time::Duration};

pub enum Observation {
    Dataset(Dataset),
    Snapshot {
        iter: u32,
        splats: Splats,
        full_sh: bool,
    },
    Config {
        max_image_size: u32,
    },
    UpAxis(glam::Vec3),
    Step {
        iter: u32,
        stats: TrainStepStats,
        step_duration: Duration,
        num_splats: u32,
    },
    Refine {
        iter: u32,
        refine: RefineStats,
        refine_duration: Duration,
    },
    Eval {
        iter: u32,
        psnr: f32,
        ssim: f32,
        splats: Splats,
    },
}

fn scalar(rec: &RecordingStream, path: &str, value: f64) -> anyhow::Result<()> {
    rec.log(path, &Scalars::single(value))?;
    Ok(())
}

fn refine_stats(
    rec: &RecordingStream,
    stats: &RefineStats,
    duration: Duration,
) -> anyhow::Result<()> {
    for (key, value) in [
        ("num_added", stats.num_added),
        ("num_split_oversized", stats.num_split_oversized),
        ("num_split_high_grad", stats.num_split_high_grad),
        ("num_pruned", stats.num_pruned),
        ("num_pruned_non_finite", stats.num_pruned_non_finite),
    ] {
        scalar(rec, &format!("refine/{key}"), value as f64)?;
    }
    scalar(rec, "train/refine_ms", duration.as_secs_f64() * 1000.0)
}

fn jpeg(
    rec: &RecordingStream,
    path: &str,
    image: image::RgbImage,
    is_static: bool,
) -> anyhow::Result<()> {
    let mut data = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut data, 85).encode_image(&image)?;
    let image = rerun::EncodedImage::from_file_contents(data).with_media_type("image/jpeg");
    if is_static {
        rec.log_static(path, &image)?;
    } else {
        rec.log(path, &image)?;
    }
    Ok(())
}

fn black_ground_truth(
    image: image::DynamicImage,
    alpha: brush_render::AlphaMode,
) -> anyhow::Result<image::RgbImage> {
    let (w, h) = (image.width(), image.height());
    let (packed, _) = brush_dataset::scene::view_to_packed_data(image, alpha);
    let pixels = packed.try_into_vec::<i32>()?;
    let bytes = pixels
        .into_iter()
        .flat_map(|pixel| {
            let b = pixel.to_le_bytes();
            [b[0], b[1], b[2]]
        })
        .collect();
    image::RgbImage::from_raw(w, h, bytes).context("invalid packed GT image")
}

pub async fn run(
    rx: Receiver<Observation>,
    rec: RecordingStream,
    device: brush_process::ProcessDevice,
    compute: bool,
    video: bool,
) {
    let mut max_image_size = 512;
    let mut eval_views: Vec<SceneView> = Vec::new();
    let mut warned_sh4 = false;
    for observation in rx {
        if !rec.is_enabled() {
            continue;
        }
        let result: anyhow::Result<()> = async {
            match observation {
                Observation::Config {
                    max_image_size: size,
                } => max_image_size = size,
                Observation::UpAxis(up) => {
                    let up = scene_up(up);
                    let a = up.abs();
                    let axis = match (a.max_position(), up[a.max_position()] > 0.0) {
                        (0, true) => rerun::ViewCoordinates::RIGHT_HAND_X_UP(),
                        (0, false) => rerun::ViewCoordinates::RIGHT_HAND_X_DOWN(),
                        (1, true) => rerun::ViewCoordinates::RIGHT_HAND_Y_UP(),
                        (1, false) => rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN(),
                        (_, true) => rerun::ViewCoordinates::RIGHT_HAND_Z_UP(),
                        (_, false) => rerun::ViewCoordinates::RIGHT_HAND_Z_DOWN(),
                    };
                    rec.log_static("world", &axis)?;
                    super::dashboard::send(
                        &rec,
                        eval_views.len(),
                        compute,
                        video,
                        Some(up.to_array()),
                    )?;
                }
                Observation::Dataset(dataset) => {
                    eval_views = dataset.eval.map_or_else(Vec::new, |scene| {
                        scene.views.iter().take(4).cloned().collect()
                    });
                    rec.log_static(
                        "world/dataset/camera",
                        &rerun::TextDocument::new(concat!(
                            "Pinhole approximations; lens distortion is omitted from frustums. ",
                            "Evaluation uses each original camera model.",
                        )),
                    )?;
                    for (i, view) in dataset.train.views.iter().enumerate() {
                        let thumbnail = view
                            .image
                            .clone()
                            .with_max_resolution(max_image_size.min(256))
                            .load()
                            .await?;
                        let size = glam::uvec2(thumbnail.width(), thumbnail.height());
                        let camera = view.camera.with_pinhole();
                        let path = format!("world/dataset/camera/{i}");
                        rec.log_static(
                            path.as_str(),
                            &rerun::Transform3D::from_translation_rotation(
                                camera.position,
                                camera.rotation,
                            ),
                        )?;
                        rec.log_static(
                            path.as_str(),
                            &rerun::Pinhole::from_focal_length_and_resolution(
                                camera.focal(size).to_array(),
                                size.as_vec2().to_array(),
                            )
                            .with_principal_point(camera.center(size).to_array()),
                        )?;
                        jpeg(
                            &rec,
                            &format!("{path}/image"),
                            black_ground_truth(thumbnail, view.image.alpha_mode())?,
                            true,
                        )?;
                    }
                    for (i, view) in eval_views.iter().enumerate() {
                        jpeg(
                            &rec,
                            &format!("eval/view_{i}/ground_truth"),
                            black_ground_truth(
                                view.image
                                    .clone()
                                    .with_max_resolution(max_image_size)
                                    .load()
                                    .await?,
                                view.image.alpha_mode(),
                            )?,
                            true,
                        )?;
                    }
                }
                Observation::Snapshot {
                    iter,
                    splats,
                    full_sh,
                } => {
                    rec.set_time_sequence("iterations", iter);
                    if splats.sh_degree() > 3 && !warned_sh4 {
                        eprintln!("Rerun supports SH through degree 3; truncating degree 4.");
                        warned_sh4 = true;
                    }
                    let snapshot = gsplat_train::read_splats(splats, full_sh).await?;
                    rec.log("world/splats", &snapshot)?;
                    // Memory reporting synchronizes the compute server. Keep it at
                    // snapshot cadence, on this worker, rather than the step path.
                    if let Some(usage) = device.memory_pool_usage() {
                        scalar(&rec, "memory/used", usage.bytes_in_use as f64)?;
                        scalar(&rec, "memory/reserved", usage.bytes_reserved as f64)?;
                    }
                }
                Observation::Step {
                    iter,
                    stats,
                    step_duration,
                    num_splats,
                } => {
                    rec.set_time_sequence("iterations", iter);
                    scalar(
                        &rec,
                        "loss/total",
                        stats.loss.into_scalar_async::<f32>().await? as f64,
                    )?;
                    scalar(&rec, "train/step_ms", step_duration.as_secs_f64() * 1000.0)?;
                    scalar(&rec, "splats/num_splats", num_splats as f64)?;
                    scalar(&rec, "splats/splats_visible", stats.num_visible as f64)?;
                    for (key, value) in [
                        ("mean", stats.lr_mean),
                        ("rotation", stats.lr_rotation),
                        ("scale", stats.lr_scale),
                        ("coeffs", stats.lr_coeffs),
                        ("opac", stats.lr_opac),
                    ] {
                        scalar(&rec, &format!("lr/{key}"), value)?;
                    }
                }
                Observation::Refine {
                    iter,
                    refine,
                    refine_duration,
                } => {
                    rec.set_time_sequence("iterations", iter);
                    refine_stats(&rec, &refine, refine_duration)?;
                }
                Observation::Eval {
                    iter,
                    psnr,
                    ssim,
                    splats,
                } => {
                    rec.set_time_sequence("iterations", iter);
                    scalar(&rec, "psnr/eval", psnr as f64)?;
                    scalar(&rec, "ssim/eval", ssim as f64)?;
                    for (i, view) in eval_views.iter().enumerate() {
                        let (w, h) = view
                            .image
                            .clone()
                            .with_max_resolution(max_image_size)
                            .dimensions()
                            .await?;
                        let (render, _) = brush_render::render_splats(
                            splats.clone(),
                            &view.camera,
                            glam::uvec2(w, h),
                            glam::Vec3::ZERO,
                            None,
                            brush_render::TextureMode::Float,
                        )
                        .await;
                        let pixels = render.into_data_async().await?.try_into_vec::<f32>()?;
                        let bytes = pixels
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|p| {
                                p[..3]
                                    .iter()
                                    .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                            })
                            .collect();
                        jpeg(
                            &rec,
                            &format!("eval/view_{i}/render"),
                            image::RgbImage::from_raw(w, h, bytes)
                                .context("invalid render image")?,
                            false,
                        )?;
                    }
                }
            }
            Ok(())
        }
        .await;
        if let Err(error) = result {
            eprintln!("Recording observation warning: {error:#}");
        }
    }
    if let Err(error) = rec.flush_with_timeout(Duration::from_secs(2)) {
        eprintln!("Recording flush warning: {error}");
    }
}

fn scene_up(brush_up: glam::Vec3) -> glam::Vec3 {
    // Brush's Viewer rotates the model from -Y toward this metadata vector.
    // Rerun keeps the model unchanged, so its eye needs the inverse rotation.
    let target = brush_up.try_normalize().unwrap_or(glam::Vec3::NEG_Y);
    glam::Quat::from_rotation_arc(glam::Vec3::NEG_Y, target).inverse() * glam::Vec3::NEG_Y
}

#[cfg(test)]
mod tests {
    #[test]
    fn brush_rotation_target_becomes_world_up() {
        for (brush_up, expected) in [
            (glam::Vec3::NEG_Z, glam::Vec3::Z),
            (glam::Vec3::NEG_Y, glam::Vec3::NEG_Y),
            (glam::Vec3::X, glam::Vec3::NEG_X),
        ] {
            assert!(super::scene_up(brush_up).abs_diff_eq(expected, 1e-6));
        }
    }
}
