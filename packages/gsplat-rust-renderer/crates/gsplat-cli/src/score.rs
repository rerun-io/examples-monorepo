//! Score exported splats against NeRF test images using unclipped float renders.
use anyhow::Result;
use gsplat_cli::camera::load_frames;
use gsplat_cli::engines::{Brush, PlyScene, RenderEngine};
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
#[serde(deny_unknown_fields)]
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
    let mut renderer = Brush::new(&scene, &Default::default()).await;
    let evaluator = Evaluator::new(false);
    let mut views = Vec::with_capacity(cameras.len());
    for view in cameras {
        renderer.render(&view.camera, true).await?;
        let rendered = gsplat_cli::parity_image(
            renderer.read_rgba_f32().await?,
            view.camera.width,
            view.camera.height,
        )?;
        let mut image_path = args.dataset.join(&view.file_path);
        if image_path.extension().is_none() {
            image_path.set_extension("png");
        }
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
    let mean = gsplat_cli::mean(&views);
    println!(
        "{} views: PSNR {:.8}, SSIM {:.9}",
        views.len(),
        mean.psnr,
        mean.ssim
    );
    super::write_json(
        &args.out,
        &ScoreReport {
            provenance: Provenance::default(),
            ply: args.ply,
            transforms,
            convention: Convention::Brush,
            image_boundary: "in-memory RGBA f32; RGB scored without clipping or alpha multiplication",
            views,
            mean,
        },
    )
}
