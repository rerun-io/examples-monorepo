//! Synchronized speed and float Brush-oracle parity runners.
mod speed;
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use glam::{Mat4, Vec2};
use gsplat_bench::{
    camera::{self, CameraPath, CameraSpec},
    renderers::{Adapter, Brush, Implementation, Native, Old, RenderEngine, Scene},
    settings::RenderSettings,
};
use gsplat_eval::{Evaluator, Metrics, Versions, ViewMetrics};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Speed(speed::SpeedArgs),
    Parity(ParityArgs),
}
#[derive(clap::Args)]
pub struct CameraArgs {
    #[arg(long)]
    ply: PathBuf,
    #[arg(long, alias = "cameras", default_value = "orbit:300")]
    path: CameraPath,
    #[arg(long, default_value = "native")]
    res: String,
    #[arg(long, value_delimiter = ',', num_args = 2)]
    radius: Option<Vec<f32>>,
    #[arg(long)]
    elevation: Option<f32>,
    /// World direction above the orbit plane (default +Z).
    #[arg(long, num_args = 3, allow_hyphen_values = true)]
    orbit_up: Option<Vec<f32>>,
    #[arg(long, value_delimiter = ',', num_args = 3, allow_hyphen_values = true)]
    center: Option<Vec<f32>>,
}
#[derive(Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Oracle {
    Brush,
}
#[derive(clap::Args)]
struct ParityArgs {
    #[arg(long = "impl", value_enum)]
    implementation: Implementation,
    #[arg(long, value_enum, default_value = "brush")]
    oracle: Oracle,
    #[command(flatten)]
    camera: CameraArgs,
    #[command(flatten)]
    settings: RenderSettings,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    save_images: Option<PathBuf>,
    #[arg(long)]
    limit: Option<usize>,
}
fn resolution(value: &str) -> Result<Option<(u32, u32)>> {
    if value == "native" {
        return Ok(None);
    }
    let (w, h) = value
        .split_once('x')
        .context("resolution must be WxH or native")?;
    let (w, h) = (w.parse()?, h.parse()?);
    ensure!(w >= 11 && h >= 11, "resolution must be at least 11x11");
    Ok(Some((w, h)))
}
async fn cameras(args: &CameraArgs, scene: &Scene) -> Result<Vec<CameraSpec>> {
    let size = resolution(&args.res)?;
    let mut path = match &args.path {
        CameraPath::Orbit(count) => {
            let (w, h) = size.unwrap_or((800, 800));
            let template = CameraSpec::from_nerf(Mat4::IDENTITY, 0.6911112, w, h);
            let r = args
                .radius
                .as_ref()
                .map(|r| Vec2::new(r[0], r[1]))
                .unwrap_or(Vec2::splat(scene.extent * 2.5));
            ensure!(
                r.is_finite() && r.min_element() > 0.0,
                "invalid orbit radius"
            );
            let center = args
                .center
                .as_deref()
                .map(glam::Vec3::from_slice)
                .unwrap_or(scene.center);
            ensure!(center.is_finite(), "invalid orbit center");
            let mut path = camera::orbit(
                center,
                r,
                args.elevation.unwrap_or(scene.extent * 0.8),
                *count,
                &template,
            );
            if let Some(up) = &args.orbit_up {
                let up = glam::Vec3::from_slice(up);
                ensure!(
                    up.is_finite() && up.length_squared() > 1e-12,
                    "invalid orbit up-vector"
                );
                let rotation = glam::Quat::from_rotation_arc(glam::Vec3::Z, up.normalize());
                let basis = Mat4::from_translation(center)
                    * Mat4::from_quat(rotation)
                    * Mat4::from_translation(-center);
                for camera in &mut path {
                    camera.world_from_camera =
                        (basis * camera.pose()).transpose().to_cols_array_2d();
                }
            }
            path
        }
        CameraPath::TestViews(p) | CameraPath::Colmap(p) => camera::load_frames(p, None)
            .await?
            .into_iter()
            .map(|frame| frame.camera)
            .collect(),
        CameraPath::Specs(p) => serde_json::from_slice::<Vec<CameraSpec>>(&std::fs::read(p)?)?,
    };
    if let Some((w, h)) = size {
        path = path.into_iter().map(|c| c.resized(w, h)).collect();
    }
    ensure!(!path.is_empty(), "empty camera path");
    for c in &path {
        c.validate()?;
        ensure!(
            (c.width, c.height) == (path[0].width, path[0].height),
            "mixed camera resolutions"
        );
    }
    Ok(path)
}
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParityView {
    metrics: ViewMetrics,
    alpha_psnr: f64,
    white_psnr: f64,
    minimum_psnr: f64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParityReport {
    implementation: Implementation,
    oracle: Oracle,
    image_boundary: String,
    settings: RenderSettings,
    ply: PathBuf,
    splats: usize,
    cameras: Vec<CameraSpec>,
    adapter: Adapter,
    oracle_adapter: Adapter,
    versions: Versions,
    views: Vec<ParityView>,
    mean: Metrics,
    mean_alpha_psnr: f64,
    mean_white_psnr: f64,
    worst_five: Vec<ParityView>,
}
struct Evidence {
    view: ParityView,
    ours: Vec<f32>,
    reference: Vec<f32>,
}
fn save_pair(
    dir: &Path,
    name: &str,
    ours: &[f32],
    reference: &[f32],
    w: u32,
    h: u32,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // EXR retains the scored values; PNGs are display-only previews of the same samples.
    for (suffix, pixels) in [("render", ours), ("brush", reference)] {
        let image = gsplat_bench::parity_image(pixels.to_vec(), w, h)?;
        image.save(dir.join(format!("{name}-{suffix}.exr")))?;
        image
            .to_rgb8()
            .save(dir.join(format!("{name}-{suffix}.png")))?;
    }
    let diff: Vec<u8> = ours
        .as_chunks::<4>()
        .0
        .iter()
        .zip(reference.as_chunks::<4>().0.iter())
        .flat_map(|(a, b)| {
            (0..3).map(move |i| {
                ((a[i] - b[i]).abs() * 4.0 * 255.0)
                    .clamp(0.0, 255.0)
                    .round() as u8
            })
        })
        .collect();
    image::RgbImage::from_raw(w, h, diff)
        .context("diff dimensions")?
        .save(dir.join(format!("{name}-diff4.png")))?;
    Ok(())
}
async fn parity<R: RenderEngine>(
    a: ParityArgs,
    scene: Scene,
    cameras: Vec<CameraSpec>,
    mut renderer: R,
) -> Result<()> {
    let mut oracle = Brush::new(&scene, &a.settings).await;
    let evaluator = Evaluator::new(false);
    let mut views = Vec::new();
    let mut worst: Vec<Evidence> = Vec::new();
    for (index, c) in cameras.iter().enumerate() {
        renderer.render(c, true).await?;
        renderer.finish()?;
        let ours = renderer.read_rgba_f32().await?;
        oracle.render(c, true).await?;
        oracle.finish()?;
        let reference = oracle.read_rgba_f32().await?;
        let score = evaluator
            .evaluate_renders(&ours, &reference, c.width, c.height)
            .await?;
        let view = ParityView {
            metrics: ViewMetrics {
                name: format!("{index:03}"),
                psnr: score.rgb.psnr,
                ssim: score.rgb.ssim,
                lpips: None,
            },
            alpha_psnr: score.alpha_psnr,
            white_psnr: score.white_psnr,
            minimum_psnr: score.minimum_psnr(),
        };
        eprintln!(
            "view {index}: RGB {:.6}, alpha {:.6}, white {:.6} dB",
            view.metrics.psnr, view.alpha_psnr, view.white_psnr
        );
        if let Some(dir) = &a.save_images {
            save_pair(
                dir,
                &view.metrics.name,
                &ours,
                &reference,
                c.width,
                c.height,
            )?;
        }
        views.push(view.clone());
        worst.push(Evidence {
            view,
            ours,
            reference,
        });
        worst.sort_by(|a, b| a.view.minimum_psnr.total_cmp(&b.view.minimum_psnr));
        worst.truncate(5);
    }
    if a.save_images.is_none() {
        let dir = a.out.with_extension("images");
        for e in &worst {
            save_pair(
                &dir,
                &e.view.metrics.name,
                &e.ours,
                &e.reference,
                cameras[0].width,
                cameras[0].height,
            )?;
        }
    }
    let mean = gsplat_eval::mean(&views.iter().map(|v| v.metrics.clone()).collect::<Vec<_>>());
    let mean_alpha_psnr = views.iter().map(|v| v.alpha_psnr).sum::<f64>() / views.len() as f64;
    let mean_white_psnr = views.iter().map(|v| v.white_psnr).sum::<f64>() / views.len() as f64;
    write_json(&a.out,&ParityReport {implementation:a.implementation,oracle:a.oracle,settings:a.settings,image_boundary:"in-memory premultiplied RGBA f32; no scoring clipping/quantization; old/native intrinsically use RGBA8 targets".into(),ply:a.camera.ply,splats:scene.data.num_splats(),cameras,adapter:renderer.adapter(),oracle_adapter:oracle.adapter(),versions:Versions::default(),views,mean,mean_alpha_psnr,mean_white_psnr,worst_five:worst.into_iter().map(|e|e.view).collect()})
}
async fn run_parity(a: ParityArgs) -> Result<()> {
    let scene = Scene::load(&a.camera.ply).await?;
    gsplat_bench::settings::validate_backend(&a.settings, a.implementation, scene.mode)?;
    let mut cameras = cameras(&a.camera, &scene).await?;
    if let Some(n) = a.limit {
        ensure!(n > 0, "limit must be positive");
        cameras.truncate(n);
    }
    match a.implementation {
        Implementation::Brush => {
            let r = Brush::new(&scene, &a.settings).await;
            parity(a, scene, cameras, r).await
        }
        Implementation::Ours => {
            let r = gsplat_bench::renderers::ours(
                &scene,
                cameras[0].width,
                cameras[0].height,
                &a.settings,
            )
            .await?;
            parity(a, scene, cameras, r).await
        }
        Implementation::OursOld => {
            let r = Old::new(&scene, cameras[0].width, cameras[0].height)?;
            parity(a, scene, cameras, r).await
        }
        Implementation::Native => {
            let r = Native::new(&a.camera.ply, scene.data.num_splats()).await?;
            parity(a, scene, cameras, r).await
        }
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    let action = Args::parse().command;
    let out = match &action {
        Action::Speed(a) => a.out.clone(),
        Action::Parity(a) => a.out.clone(),
    };
    let result = match action {
        Action::Speed(a) => speed::run(a).await,
        Action::Parity(a) => run_parity(a).await,
    };
    if let Err(error) = &result
        && let Some(gsplat_bench::Error::Unsupported(reason)) =
            error.downcast_ref::<gsplat_bench::Error>()
    {
        #[derive(Serialize)]
        struct UnsupportedReport<'a> {
            status: &'static str,
            reason: &'a str,
            versions: Versions,
        }
        write_json(
            &out,
            &UnsupportedReport {
                status: "unsupported",
                reason,
                versions: Versions::default(),
            },
        )?;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn orbit_up_places_cameras_above_the_scene_and_looks_at_focus() {
        let args = Args::try_parse_from([
            "bench",
            "parity",
            "--impl",
            "native",
            "--ply",
            "unused",
            "--path",
            "orbit:4",
            "--out",
            "unused",
            "--center",
            "1",
            "2",
            "3",
            "--radius",
            "3",
            "3",
            "--elevation",
            "1",
            "--orbit-up",
            "0",
            "-1",
            "0",
        ])
        .unwrap();
        let Action::Parity(args) = args.command else {
            panic!("parity command")
        };
        let scene = Scene {
            data: brush_serde::import::SplatData {
                means: vec![0.0; 3],
                rotations: None,
                log_scales: None,
                sh_coeffs: None,
                raw_opacities: None,
            },
            mode: gsplat_core::RenderMode::Default,
            center: glam::Vec3::ZERO,
            extent: 1.0,
        };
        let path = cameras(&args.camera, &scene).await.unwrap();
        let expected = [
            [4.0, 1.0, 3.0],
            [1.0, 1.0, 6.0],
            [-2.0, 1.0, 3.0],
            [1.0, 1.0, 0.0],
        ];
        for (camera, expected) in path.iter().zip(expected) {
            let pose = Mat4::from_cols_array_2d(&camera.world_from_camera).transpose();
            assert!((pose.w_axis.truncate() - glam::Vec3::from_array(expected)).length() < 1e-5);
            let target = pose
                .inverse()
                .transform_point3(glam::Vec3::new(1.0, 2.0, 3.0));
            assert!(target.x.abs() < 1e-5 && target.y.abs() < 1e-5 && target.z > 0.0);
        }
    }
    #[test]
    fn cli_rejects_unknown_path_and_oracle() {
        assert!(
            Args::try_parse_from([
                "bench", "parity", "--impl", "native", "--ply", "x", "--path", "orbit300", "--out",
                "y"
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "bench", "parity", "--impl", "native", "--ply", "x", "--oracle", "typo", "--out",
                "y"
            ])
            .is_err()
        );
        assert_eq!(resolution("1920x1080").unwrap(), Some((1920, 1080)));
        assert!(resolution("0x800").is_err());
    }
}
