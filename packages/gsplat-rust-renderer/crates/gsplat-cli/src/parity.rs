//! Compare unclipped RGB, alpha, and white composition against a renderer oracle.
use super::{CameraArgs, cameras, write_json};
use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use gsplat_cli::{
    Evaluator, Metrics, Provenance, ViewMetrics,
    camera::CameraSpec,
    engines::{Adapter, Engine, Implementation, PlyScene, RenderEngine},
    settings::RenderSettings,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Oracle {
    Brush,
    Ours,
}
#[derive(clap::Args)]
pub(super) struct Args {
    /// Quantize through the native archetype, optionally from one complete RRD splat row.
    #[arg(long, num_args = 0..=1)]
    pub(super) archetype: Option<Option<PathBuf>>,
    #[arg(long = "impl", value_enum)]
    pub(super) implementation: Implementation,
    #[arg(long, value_enum)]
    oracle: Option<Oracle>,
    #[command(flatten)]
    pub(super) camera: CameraArgs,
    #[command(flatten)]
    pub(super) settings: RenderSettings,
    #[arg(long)]
    pub(super) out: PathBuf,
    #[arg(long)]
    pub(super) save_images: Option<PathBuf>,
    #[arg(long)]
    pub(super) limit: Option<usize>,
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
    archetype: bool,
    oracle: Oracle,
    image_boundary: String,
    settings: RenderSettings,
    ply: PathBuf,
    splats: usize,
    cameras: Vec<CameraSpec>,
    adapter: Adapter,
    oracle_adapter: Adapter,
    provenance: Provenance,
    views: Vec<ParityView>,
    mean: Metrics,
    min_rgb_psnr: f64,
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
    oracle_label: &str,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // EXR retains the scored values; PNGs are display-only previews of the same samples.
    for (suffix, pixels) in [("render", ours), (oracle_label, reference)] {
        let image = gsplat_cli::parity_image(pixels.to_vec(), w, h)?;
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
async fn parity(
    a: Args,
    scene: PlyScene,
    cameras: Vec<CameraSpec>,
    mut renderer: Engine,
    mut oracle: Engine,
    oracle_kind: Oracle,
) -> Result<()> {
    let oracle_label = match oracle_kind {
        Oracle::Brush => "brush",
        Oracle::Ours => "ours-f32",
    };
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
                oracle_label,
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
                oracle_label,
            )?;
        }
    }
    let mean = gsplat_cli::mean(&views.iter().map(|v| v.metrics.clone()).collect::<Vec<_>>());
    let mean_alpha_psnr = views.iter().map(|v| v.alpha_psnr).sum::<f64>() / views.len() as f64;
    let mean_white_psnr = views.iter().map(|v| v.white_psnr).sum::<f64>() / views.len() as f64;
    const IMAGE_BOUNDARY: &str = concat!(
        "in-memory premultiplied RGBA f32; no scoring clipping/quantization; ",
        "native intrinsically uses an RGBA8 target",
    );
    write_json(
        &a.out,
        &ParityReport {
            implementation: a.implementation,
            archetype: a.archetype.is_some(),
            oracle: oracle_kind,
            settings: a.settings,
            image_boundary: IMAGE_BOUNDARY.into(),
            ply: a.camera.ply,
            splats: scene.data.num_splats(),
            cameras,
            adapter: renderer.adapter(),
            oracle_adapter: oracle.adapter(),
            provenance: Provenance::default(),
            min_rgb_psnr: views
                .iter()
                .map(|v| v.metrics.psnr)
                .fold(f64::INFINITY, f64::min),
            views,
            mean,
            mean_alpha_psnr,
            mean_white_psnr,
            worst_five: worst.into_iter().map(|e| e.view).collect(),
        },
    )
}
pub(super) async fn run(a: Args) -> Result<()> {
    ensure!(
        a.archetype.is_none() || a.implementation == Implementation::Ours,
        "--archetype requires --impl ours"
    );
    let scene = PlyScene::load(&a.camera.ply).await?;
    let mut cameras = cameras(&a.camera, &scene).await?;
    if let Some(n) = a.limit {
        ensure!(n > 0, "limit must be positive");
        cameras.truncate(n);
    }
    let source = a
        .archetype
        .as_ref()
        .map(|path| path.as_deref().unwrap_or(a.camera.ply.as_path()));
    let renderer = Engine::new(
        a.implementation,
        &scene,
        &a.settings,
        &cameras[0],
        &a.camera.ply,
        source,
    )
    .await?;
    let oracle_kind = a.oracle.unwrap_or(if source.is_some() {
        Oracle::Ours
    } else {
        Oracle::Brush
    });
    let kind = match oracle_kind {
        Oracle::Ours => Implementation::Ours,
        Oracle::Brush => Implementation::Brush,
    };
    let oracle = Engine::new(kind, &scene, &a.settings, &cameras[0], &a.camera.ply, None).await?;
    parity(a, scene, cameras, renderer, oracle, oracle_kind).await
}
