//! Score exported splats against NeRF test images using unclipped float renders.
use anyhow::Result;
use gsplat_cli::camera::load_frames;
use gsplat_cli::engines::{Brush, Engine, PlyScene};
use gsplat_cli::{Convention, Evaluator, Metrics, Provenance, ViewMetrics};
use serde::Serialize;
use std::path::PathBuf;

#[derive(clap::Args)]
pub struct ScoreArgs {
    #[arg(long)]
    ply: PathBuf,
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Serialize)]
struct ScoreReport {
    provenance: Provenance,
    ply: PathBuf,
    transforms: PathBuf,
    convention: Convention,
    image_boundary: &'static str,
    views: Vec<ViewMetrics>,
    mean: Metrics,
}

pub async fn run(args: ScoreArgs) -> Result<()> {
    let transforms = args.dataset.join("transforms_test.json");
    let cameras = load_frames(&transforms, None).await?;
    let scene = PlyScene::load(&args.ply).await?;
    let mut renderer = Engine::Brush(Box::new(Brush::new(&scene, &Default::default()).await));
    let evaluator = Evaluator::new(false);
    let mut views = Vec::with_capacity(cameras.len());
    for view in cameras {
        let rendered = gsplat_cli::parity_image(
            renderer.capture(&view.camera, true).await?,
            view.camera.width,
            view.camera.height,
        )?;
        let image_path = args.dataset.join(&view.file_path);
        let gt = image::open(&image_path)?;
        let metrics = evaluator
            .evaluate_pair(&rendered, &gt, Convention::Brush)
            .await?;
        views.push(ViewMetrics {
            name: image_path
                .strip_prefix(&args.dataset)?
                .to_string_lossy()
                .into_owned(),
            metrics,
        });
    }
    let mean = gsplat_cli::mean(views.iter().map(|v| &v.metrics));
    println!(
        "{} views: PSNR {:.8}, SSIM {:.9}",
        views.len(),
        mean.psnr,
        mean.ssim
    );
    super::write_json(
        &args.out,
        &ScoreReport {
            provenance: Provenance::capture(),
            ply: args.ply,
            transforms,
            convention: Convention::Brush,
            image_boundary: "in-memory RGBA f32; RGB scored without clipping or alpha multiplication",
            views,
            mean,
        },
    )
}
