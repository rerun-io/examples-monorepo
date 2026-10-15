//! Compare unclipped RGB, alpha, and white composition against a renderer oracle.
use super::{CameraArgs, cameras, write_json};
use anyhow::{Context as _, Result, ensure};
use gsplat_cli::PlyScene;
use gsplat_cli::{
    Evaluator, Metrics, Provenance, RenderMetrics,
    camera::CameraSpec,
    engines::{Adapter, Engine, Implementation},
    settings::RenderSettings,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub(super) struct Args {
    /// Quantize through the native archetype, optionally from one complete RRD splat row.
    #[arg(long, num_args = 0..=1)]
    pub(super) archetype: Option<Option<PathBuf>>,
    #[arg(long = "impl", value_enum)]
    pub(super) implementation: Implementation,
    #[arg(long, value_enum)]
    oracle: Option<Implementation>,
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

#[derive(Clone, Serialize)]
struct ParityView {
    name: String,
    #[serde(flatten)]
    metrics: RenderMetrics,
}

#[derive(Serialize)]
struct ParityReport {
    implementation: Implementation,
    archetype: bool,
    oracle: Implementation,
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
// Only scores and paths survive a view; the exact scored pixels live on disk.
struct Evidence {
    worst: Vec<ParityView>,
    staging: Option<tempfile::TempDir>,
}

impl Evidence {
    fn new(save_all: bool) -> Result<Self> {
        Ok(Self {
            worst: Vec::new(),
            staging: if save_all {
                None
            } else {
                Some(tempfile::tempdir()?)
            },
        })
    }

    fn record(
        &mut self,
        view: ParityView,
        ours: &[f32],
        reference: &[f32],
        size: (u32, u32),
        label: &str,
    ) -> Result<()> {
        self.worst.push(view.clone());
        // Stable sort preserves the earlier view on ties, as before.
        self.worst.sort_by(|a, b| {
            a.metrics
                .minimum_psnr()
                .total_cmp(&b.metrics.minimum_psnr())
        });
        let displaced = if self.worst.len() > 5 {
            self.worst.pop()
        } else {
            None
        };
        if let Some(staging) = &self.staging {
            if self
                .worst
                .iter()
                .any(|candidate| candidate.name == view.name)
            {
                save_pair(
                    &staging.path().join(&view.name),
                    &view.name,
                    ours,
                    reference,
                    size.0,
                    size.1,
                    label,
                )?;
            }
            if let Some(displaced) = displaced {
                let path = staging.path().join(displaced.name);
                if path.exists() {
                    std::fs::remove_dir_all(path)?;
                }
            }
        }
        Ok(())
    }

    fn publish(&self, destination: &Path) -> Result<()> {
        if let Some(staging) = &self.staging {
            std::fs::create_dir_all(destination)?;
            for view in &self.worst {
                for file in std::fs::read_dir(staging.path().join(&view.name))? {
                    let file = file?;
                    // Copy permits the temporary directory and destination to be on different filesystems.
                    std::fs::copy(file.path(), destination.join(file.file_name()))?;
                }
            }
        }
        Ok(())
    }
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
    oracle_kind: Implementation,
) -> Result<()> {
    let oracle_label = match oracle_kind {
        Implementation::Brush => "brush",
        Implementation::Ours => "ours-f32",
        Implementation::Native => anyhow::bail!("native cannot be a parity oracle"),
    };
    let evaluator = Evaluator::new(false);
    let mut views = Vec::new();
    let mut evidence = Evidence::new(a.save_images.is_some())?;
    for (index, c) in cameras.iter().enumerate() {
        let ours = renderer.capture(c, true).await?;
        let reference = oracle.capture(c, true).await?;
        let score = evaluator
            .evaluate_renders(&ours, &reference, c.width, c.height)
            .await?;
        let view = ParityView {
            name: format!("{index:03}"),
            metrics: score,
        };
        eprintln!(
            "view {index}: RGB {:.6}, alpha {:.6}, white {:.6} dB",
            view.metrics.rgb.psnr, view.metrics.alpha_psnr, view.metrics.white_psnr
        );
        if let Some(dir) = &a.save_images {
            save_pair(
                dir,
                &view.name,
                &ours,
                &reference,
                c.width,
                c.height,
                oracle_label,
            )?;
        }
        views.push(view.clone());
        evidence.record(view, &ours, &reference, (c.width, c.height), oracle_label)?;
    }
    evidence.publish(&a.out.with_extension("images"))?;
    let mean = gsplat_cli::mean(views.iter().map(|v| &v.metrics.rgb));
    let mean_alpha_psnr =
        views.iter().map(|v| v.metrics.alpha_psnr).sum::<f64>() / views.len() as f64;
    let mean_white_psnr =
        views.iter().map(|v| v.metrics.white_psnr).sum::<f64>() / views.len() as f64;
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
            provenance: Provenance::capture(),
            min_rgb_psnr: views
                .iter()
                .map(|v| v.metrics.rgb.psnr)
                .fold(f64::INFINITY, f64::min),
            views,
            mean,
            mean_alpha_psnr,
            mean_white_psnr,
            worst_five: evidence.worst,
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
        Implementation::Ours
    } else {
        Implementation::Brush
    });
    ensure!(
        oracle_kind != Implementation::Native,
        "native cannot be a parity oracle"
    );
    let oracle = Engine::new(
        oracle_kind,
        &scene,
        &a.settings,
        &cameras[0],
        &a.camera.ply,
        None,
    )
    .await?;
    parity(a, scene, cameras, renderer, oracle, oracle_kind).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(name: &str, psnr: f64) -> ParityView {
        ParityView {
            name: name.into(),
            metrics: RenderMetrics {
                rgb: Metrics {
                    psnr,
                    ssim: 1.0,
                    lpips: None,
                },
                alpha_psnr: psnr,
                white_psnr: psnr,
            },
        }
    }

    #[test]
    fn evidence_keeps_stable_worst_five_and_publishes_exact_scored_pixels() {
        let mut evidence = Evidence::new(false).unwrap();
        let pixels = [1.17, 0.5, -0.2, 0.1];
        for (name, score) in [
            ("000", 10.0),
            ("001", 20.0),
            ("002", 20.0),
            ("003", 30.0),
            ("004", 40.0),
            ("005", 50.0),
            ("006", 5.0),
        ] {
            evidence
                .record(view(name, score), &pixels, &pixels, (1, 1), "brush")
                .unwrap();
        }
        assert_eq!(
            evidence
                .worst
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            ["006", "000", "001", "002", "003"]
        );
        let staging = evidence.staging.as_ref().unwrap().path().to_owned();
        assert!(!staging.join("004").exists());
        assert!(!staging.join("005").exists());
        let destination = tempfile::tempdir().unwrap();
        evidence.publish(destination.path()).unwrap();
        assert_eq!(std::fs::read_dir(destination.path()).unwrap().count(), 25);
        let image = image::open(destination.path().join("000-render.exr")).unwrap();
        assert_eq!(image.to_rgba32f().as_raw(), &pixels);
        drop(evidence);
        assert!(!staging.exists());
    }

    #[test]
    fn evidence_save_all_retains_only_scores_and_failure_removes_staging() {
        let mut evidence = Evidence::new(true).unwrap();
        evidence
            .record(view("000", 10.0), &[], &[], (1, 1), "brush")
            .unwrap();
        assert!(evidence.staging.is_none());
        let mut evidence = Evidence::new(false).unwrap();
        let staging = evidence.staging.as_ref().unwrap().path().to_owned();
        assert!(
            evidence
                .record(
                    view("000", 10.0),
                    &[f32::NAN; 4],
                    &[0.0; 4],
                    (1, 1),
                    "brush"
                )
                .is_err()
        );
        drop(evidence);
        assert!(!staging.exists());
    }
}
