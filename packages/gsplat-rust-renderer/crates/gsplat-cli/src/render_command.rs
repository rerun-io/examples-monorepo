//! Headless gsplat-core CLI; flags preserve the Python evaluation harness contract.
use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use gsplat_cli::{
    Output, PlyScene, StandaloneRenderer,
    camera::CameraSpec,
    camera_files::{CameraFrame, load_frames},
    settings::RenderSettings,
    statistics::median,
};
use serde::Serialize;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Render a Gaussian splat or hexel PLY with the shared raw-wgpu core")]
pub(crate) struct Args {
    /// Splat PLY, or hexel PLY (selected by its `comment hexels_format 1` header line).
    #[arg(long)]
    ply: PathBuf,
    /// NeRF transforms JSON, Hexels `cameras.json`/`contract.json`, `CameraSpec` JSON array,
    /// or COLMAP sparse model directory. 2D hexel fits render their image without one.
    #[arg(long)]
    camera: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    frame: usize,
    #[arg(
        long,
        required_unless_present = "output_dir",
        conflicts_with = "output_dir"
    )]
    output: Option<PathBuf>,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Output size; omit both to keep the camera file's size.
    #[arg(long, requires = "height")]
    width: Option<u32>,
    #[arg(long, requires = "width")]
    height: Option<u32>,
    #[arg(long, default_value = "1,1,1", value_parser = parse_background)]
    background: glam::Vec3,
    /// Write 16-bit RGB PNGs (clamped to [0, 1], rounded) instead of 8-bit.
    #[arg(long)]
    png16: bool,
    /// Also write a recording with, per view on the `view` timeline, the camera at
    /// `world/camera` and RGB, median and expected depth, alpha and the world normal
    /// (`0.5 n + 0.5`, zero where alpha is zero) under `world/camera/<surface-name>/`.
    #[arg(long)]
    surface: Option<PathBuf>,
    #[arg(long, default_value = "wgsl", requires = "surface")]
    surface_name: String,
    /// After the renders, time the rendered views with upstream Hexels' protocol for the input
    /// (`NVS_FPS` for 3D scenes, `IMAGE_FPS` for 2D fits) and write the report here.
    #[arg(long)]
    fps: Option<PathBuf>,
    #[command(flatten)]
    settings: RenderSettings,
}

fn parse_background(text: &str) -> Result<glam::Vec3, String> {
    let values = text
        .split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if values.len() != 3 || values.iter().any(|v| !v.is_finite()) {
        return Err("background must be three finite comma-separated floats".into());
    }
    Ok(glam::Vec3::from_slice(&values))
}

/// A 2D fit draws in pixel space; its frame only fixes the image size and the output name.
fn image_frame(ply: &Path, size: glam::UVec2) -> CameraFrame {
    let camera = CameraSpec::from_nerf(glam::Mat4::IDENTITY, 1.0, size.x, size.y);
    let stem = ply.file_stem().unwrap_or_default();
    CameraFrame {
        camera,
        file_path: PathBuf::from(stem),
    }
}

/// One view of the surface recording; depth images are in camera z units (`meter = 1`).
fn log_surface(
    recording: &rerun::RecordingStream,
    renderer: &StandaloneRenderer,
    index: usize,
    frame: &CameraFrame,
    name: &str,
) -> Result<()> {
    let camera = &frame.camera;
    let resolution = [camera.width, camera.height];
    let rgba = renderer.read_rgba()?;
    let surface = renderer.read_surface()?;
    recording.set_time_sequence("view", i64::try_from(index)?);
    let pose = camera.pose();
    recording.log(
        "world/camera",
        &rerun::Transform3D::from_translation_mat3x3(
            pose.w_axis.truncate(),
            glam::Mat3::from_mat4(pose),
        ),
    )?;
    recording.log(
        "world/camera",
        &rerun::Pinhole::new([
            [camera.fx, 0.0, 0.0],
            [0.0, camera.fy, 0.0],
            [camera.cx, camera.cy, 1.0],
        ])
        .with_resolution([camera.width as f32, camera.height as f32]),
    )?;
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            p[..3]
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
    let alpha: Vec<f32> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    let depth = |i: usize| -> Vec<f32> { surface.depth.iter().map(|d| d[i]).collect() };
    let normal: Vec<f32> = surface
        .normal
        .iter()
        .zip(&alpha)
        .flat_map(|(n, a)| {
            n[..3]
                .iter()
                .map(move |v| if *a > 0.0 { 0.5 * v + 0.5 } else { 0.0 })
        })
        .collect();
    let entity = |part: &str| format!("world/camera/{name}/{part}");
    let depth_image = |values: &[f32]| {
        rerun::DepthImage::from_data_type_and_bytes(
            bytemuck::cast_slice::<f32, u8>(values).to_vec(),
            resolution,
            rerun::ChannelDatatype::F32,
        )
        .with_meter(1.0)
    };
    recording.log(entity("rgb"), &rerun::Image::from_rgb24(rgb, resolution))?;
    recording.log(entity("median_depth"), &depth_image(&depth(1)))?;
    recording.log(entity("expected_depth"), &depth_image(&depth(0)))?;
    recording.log(
        entity("alpha"),
        &rerun::Image::from_elements(&alpha, resolution, rerun::ColorModel::L),
    )?;
    recording.log(
        entity("normal"),
        &rerun::Image::from_elements(&normal, resolution, rerun::ColorModel::RGB),
    )?;
    Ok(())
}

/// Warm-up passes over the views, then `trials` timed runs of `timed_passes` passes each.
struct FpsProtocol {
    name: &'static str,
    warmup_passes: usize,
    timed_passes: usize,
    trials: usize,
    /// Wait for every render (upstream 2D decode timing) instead of one GPU wait per trial.
    wait_each: bool,
}

/// Upstream Hexels `scripts/evaluate_nvs_campaign.py` `benchmark` settings.
const NVS_FPS: FpsProtocol = FpsProtocol {
    name: "hexels-paper: 2 warm-up passes, 4 trials x 10 passes, median FPS; up to 3 frames in flight, one GPU wait per trial; no readback",
    warmup_passes: 2,
    timed_passes: 10,
    trials: 4,
    wait_each: false,
};

/// Upstream Hexels 2D decode timing (`quality_and_timing`): one view, every render waited on.
const IMAGE_FPS: FpsProtocol = FpsProtocol {
    name: "hexels-2d-decode: 10 warm-up renders, 5 trials x 20 renders, each waited on; no readback",
    warmup_passes: 10,
    timed_passes: 20,
    trials: 5,
    wait_each: true,
};

#[derive(Debug, Serialize)]
struct FpsTrial {
    seconds: f64,
    frames: usize,
    fps: f64,
}

#[derive(Serialize)]
struct FpsReport {
    protocol: &'static str,
    warmup_passes: usize,
    timed_passes: usize,
    timing_trials: usize,
    views: usize,
    width: u32,
    height: u32,
    adapter: String,
    backend: String,
    trials: Vec<FpsTrial>,
    median_fps: f64,
}

/// Median of the per-trial frame rates (the midpoint for an even trial count, like Python's `statistics.median`).
fn median_fps(trials: &[FpsTrial]) -> Result<f64> {
    ensure!(
        !trials.is_empty() && trials.iter().all(|t| t.fps.is_finite() && t.fps > 0.0),
        "FPS trials must be nonempty, finite and positive"
    );
    Ok(median(&trials.iter().map(|t| t.fps).collect::<Vec<_>>()))
}

/// Warm up (synchronized renders size the intersection buffers), then time whole passes over
/// `frames`, submitted back to back with one GPU wait per trial unless the protocol waits on each.
fn measure_fps(
    renderer: &mut StandaloneRenderer,
    frames: &[&CameraFrame],
    protocol: &FpsProtocol,
) -> Result<Vec<FpsTrial>> {
    for _ in 0..protocol.warmup_passes {
        for frame in frames {
            renderer.render(&frame.camera, Output::Float)?;
        }
    }
    renderer.finish()?;
    (0..protocol.trials)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..protocol.timed_passes {
                for frame in frames {
                    if protocol.wait_each {
                        renderer.render(&frame.camera, Output::Float)?;
                    } else {
                        renderer.submit_float(&frame.camera)?;
                    }
                }
            }
            if !protocol.wait_each {
                renderer.finish_submitted()?;
            }
            let seconds = start.elapsed().as_secs_f64();
            let count = frames.len() * protocol.timed_passes;
            Ok(FpsTrial {
                seconds,
                frames: count,
                fps: count as f64 / seconds,
            })
        })
        .collect()
}

fn frame_output(root: &Path, frame: &CameraFrame) -> Result<PathBuf> {
    ensure!(
        !frame.file_path.components().any(|c| matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )),
        "frame path must stay relative: {:?}",
        frame.file_path
    );
    Ok(root.join(&frame.file_path).with_extension("png"))
}

fn save(renderer: &StandaloneRenderer, size: glam::UVec2, path: &Path, png16: bool) -> Result<()> {
    let pixels = renderer.read_rgba()?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let image = gsplat_cli::parity_image(pixels, size.x, size.y)?;
    if png16 {
        image.to_rgb16().save(path)?;
    } else {
        image.to_rgb8().save(path)?;
    }
    Ok(())
}

pub(crate) async fn run(args: Args) -> Result<()> {
    let resize = args.width.zip(args.height);
    ensure!(
        resize.is_none_or(|(w, h)| w > 0 && h > 0),
        "image dimensions must be positive"
    );
    let scene = PlyScene::load(&args.ply).await?;
    let frames = match (&args.camera, scene.image_size) {
        (Some(camera), None) => load_frames(camera, resize).await?,
        (None, Some(size)) if resize.is_none() => vec![image_frame(&args.ply, size)],
        _ => {
            anyhow::bail!("3D scenes need --camera; 2D hexel fits take neither --camera nor a size")
        }
    };
    let size = glam::UVec2::new(frames[0].camera.width, frames[0].camera.height);
    ensure!(
        frames
            .iter()
            .all(|f| (f.camera.width, f.camera.height) == (size.x, size.y)),
        "mixed camera resolutions; pass --width and --height"
    );
    args.settings.validate()?;
    let mut renderer =
        gsplat_cli::engines::ours(&scene, size, &args.settings, None, args.background).await?;
    eprintln!(
        "{} {}; {} ({:?}); {:?}",
        scene.data.num_splats(),
        scene.hexels.as_ref().map_or_else(
            || "splats".to_owned(),
            |h| format!("{:?} hexels", h.variant)
        ),
        renderer.adapter_info().name,
        renderer.adapter_info().backend,
        args.settings.mode(scene.mode)
    );
    let selected: Vec<(usize, &CameraFrame, PathBuf)> = if let Some(root) = &args.output_dir {
        frames
            .iter()
            .enumerate()
            .map(|(index, frame)| Ok((index, frame, frame_output(root, frame)?)))
            .collect::<Result<_>>()?
    } else {
        let frame = frames
            .get(args.frame)
            .context("frame index is outside the camera file")?;
        let path = args
            .output
            .clone()
            .context("output is required without output-dir")?;
        vec![(args.frame, frame, path)]
    };
    let recording = args
        .surface
        .as_ref()
        .map(|path| -> Result<_> {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            Ok(rerun::RecordingStreamBuilder::new("hexels-surface").save(path)?)
        })
        .transpose()?;
    for (done, (index, frame, path)) in selected.iter().enumerate() {
        let output = if recording.is_some() {
            Output::Surface
        } else {
            Output::Float
        };
        renderer.render(&frame.camera, output)?;
        save(&renderer, size, path, args.png16)?;
        if let Some(recording) = &recording {
            log_surface(recording, &renderer, *index, frame, &args.surface_name)?;
        }
        if (done + 1) % 25 == 0 || done + 1 == selected.len() {
            eprintln!("Rendered {}/{} test frames", done + 1, selected.len());
        }
    }
    if let Some(recording) = recording {
        recording.flush_blocking()?;
    }
    if let Some(path) = &args.fps {
        let timed: Vec<&CameraFrame> = selected.iter().map(|(_, frame, _)| *frame).collect();
        let protocol = if scene.image_size.is_some() {
            &IMAGE_FPS
        } else {
            &NVS_FPS
        };
        let trials = measure_fps(&mut renderer, &timed, protocol)?;
        let median_fps = median_fps(&trials)?;
        eprintln!("Median FPS {median_fps:.1} over {} views", timed.len());
        crate::write_json(
            path,
            &FpsReport {
                protocol: protocol.name,
                warmup_passes: protocol.warmup_passes,
                timed_passes: protocol.timed_passes,
                timing_trials: protocol.trials,
                views: timed.len(),
                width: size.x,
                height: size.y,
                adapter: renderer.adapter_info().name.clone(),
                backend: format!("{:?}", renderer.adapter_info().backend),
                trials,
                median_fps,
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn evaluation_cli_accepts_all_frames_white_background_and_mode_override() {
        let args = Args::try_parse_from([
            "gsplat render",
            "--ply",
            "scene.ply",
            "--camera",
            "transforms_test.json",
            "--output-dir",
            "renders",
            "--width",
            "800",
            "--height",
            "800",
            "--background",
            "1,1,1",
            "--render-mode",
            "mip",
        ])
        .unwrap();
        assert_eq!(args.background, glam::Vec3::ONE);
        assert_eq!(args.width.zip(args.height), Some((800, 800)));
        assert!(!args.png16);
        assert!(matches!(
            args.settings.render_mode,
            gsplat_cli::settings::Mode::Mip
        ));
        assert!(args.output.is_none());
        assert!(args.fps.is_none());
        assert!(parse_background("1,nan,0").is_err());
    }
    #[test]
    fn fps_report_takes_the_median_trial_like_upstream() {
        let args = Args::try_parse_from([
            "gsplat render",
            "--ply",
            "model.ply",
            "--camera",
            "cameras.json",
            "--output-dir",
            "renders/test",
            "--fps",
            "fps.json",
        ])
        .unwrap();
        assert_eq!(args.fps, Some(PathBuf::from("fps.json")));
        let trial = |fps: f64| FpsTrial {
            seconds: 240.0 / fps,
            frames: 240,
            fps,
        };
        let trials = [trial(400.0), trial(100.0), trial(300.0), trial(200.0)];
        assert_eq!(median_fps(&trials).unwrap(), 250.0);
        assert_eq!(median_fps(&trials[..3]).unwrap(), 300.0);
        assert!(median_fps(&[]).is_err());
        assert!(median_fps(&[trial(f64::INFINITY)]).is_err());
    }
    #[test]
    fn output_paths_preserve_subdirectories_and_reject_parent_escape() {
        let camera = CameraSpec::from_nerf(glam::Mat4::IDENTITY, 1.0, 800, 800);
        let mut frame = CameraFrame {
            camera,
            file_path: "./test/r_0".into(),
        };
        assert_eq!(
            frame_output(Path::new("out"), &frame).unwrap(),
            PathBuf::from("out/test/r_0.png")
        );
        frame.file_path = "../escape".into();
        assert!(frame_output(Path::new("out"), &frame).is_err());
    }
}
