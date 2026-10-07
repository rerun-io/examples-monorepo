//! Headless gsplat-core CLI; flags preserve the Python evaluation harness contract.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use gsplat_cli::{
    Output, PlyScene, Renderer,
    camera::{CameraFrame, load_frames},
    settings::RenderSettings,
};
use std::path::{Component, Path, PathBuf};

#[derive(Parser)]
#[command(about = "Render Gaussian splats with the shared raw-wgpu core")]
pub(crate) struct Args {
    #[arg(long)]
    ply: PathBuf,
    /// NeRF transforms JSON, CameraSpec JSON array, or COLMAP sparse model directory.
    #[arg(long)]
    camera: PathBuf,
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
    #[arg(long, default_value_t = 800)]
    width: u32,
    #[arg(long, default_value_t = 800)]
    height: u32,
    #[arg(long, default_value = "1,1,1", value_parser = parse_background)]
    background: glam::Vec3,
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
fn save(renderer: &Renderer, size: glam::UVec2, path: &Path) -> Result<()> {
    let pixels = renderer.read_rgba()?;
    ensure!(
        pixels.iter().all(|v| v.is_finite()),
        "render produced nonfinite pixels"
    );
    let rgb: Vec<u8> = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            p[..3]
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    image::save_buffer(path, &rgb, size.x, size.y, image::ColorType::Rgb8)?;
    Ok(())
}
pub(crate) async fn run(args: Args) -> Result<()> {
    ensure!(
        args.width > 0 && args.height > 0,
        "image dimensions must be positive"
    );
    let scene = PlyScene::load(&args.ply).await?;
    let frames = load_frames(&args.camera, Some((args.width, args.height))).await?;
    let size = glam::UVec2::new(args.width, args.height);
    args.settings.validate()?;
    let mut splats = gsplat_cli::raw_splats(&scene.data)?;
    if let Some(floor) = args.settings.min_scale {
        splats.min_scale = Some(vec![floor; splats.transforms.len()]);
    }
    let options = gsplat_core::RenderOptions {
        background: args.background,
        ..args.settings.options(scene.mode)
    };
    let mut renderer =
        Renderer::new(&splats, options, size, args.settings.initial_capacity).await?;
    eprintln!(
        "{} splats; {} ({:?}); {:?}",
        scene.data.num_splats(),
        renderer.adapter_info().name,
        renderer.adapter_info().backend,
        options.render_mode
    );
    if let Some(root) = args.output_dir {
        for (index, frame) in frames.iter().enumerate() {
            let path = frame_output(&root, frame)?;
            renderer.render(&frame.camera, Output::Float)?;
            save(&renderer, size, &path)?;
            if (index + 1) % 25 == 0 || index + 1 == frames.len() {
                eprintln!("Rendered {}/{} test frames", index + 1, frames.len());
            }
        }
    } else {
        let frame = frames
            .get(args.frame)
            .context("frame index is outside the camera file")?;
        if let Some(path) = args.output {
            renderer.render(&frame.camera, Output::Float)?;
            save(&renderer, size, &path)?;
        }
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
        assert!(matches!(
            args.settings.render_mode,
            gsplat_cli::settings::Mode::Mip
        ));
        assert!(args.output.is_none());
        assert!(parse_background("1,nan,0").is_err());
    }
    #[test]
    fn output_paths_preserve_subdirectories_and_reject_parent_escape() {
        let camera = gsplat_cli::camera::CameraSpec::from_nerf(glam::Mat4::IDENTITY, 1.0, 800, 800);
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
