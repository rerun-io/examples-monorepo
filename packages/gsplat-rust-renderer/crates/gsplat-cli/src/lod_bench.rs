//! `gsplat lod-bench`: one process loads a Streamed SOG set once and, per selection mode, renders every camera
//! (PNG + PSNR/SSIM against the first mode), times it and records the selection, the counts and the GPU stages.
//! One process = one GPU job on a shared machine.
use anyhow::{Context as _, Result, ensure};
use gsplat_cli::lod::{CameraSelection, LodArgs, SetReport};
use gsplat_cli::{Evaluator, Output, PlyScene, RenderMetrics, camera_files::load_frames};
use gsplat_core::lod::{LodMode, LodParams};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[command(flatten)]
    lod: LodArgs,
    /// `CameraSpec` JSON array (or any `gsplat render` camera file).
    #[arg(long)]
    camera: PathBuf,
    /// Modes in order: `off`, `cull`, `lod:<tau>`. The first one is the quality reference of the others.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "off,off,cull,lod:1,lod:2,lod:4"
    )]
    modes: Vec<String>,
    /// Timed frames per camera and mode, queued with up to 3 frames in flight (`submit_float`), after `--warmup` untimed ones.
    #[arg(long, default_value_t = 30)]
    frames: usize,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    /// Write PNGs of every camera of every mode under `<out>/<index>-<mode>/`.
    #[arg(long)]
    images: bool,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct CameraResult {
    #[serde(flatten)]
    selection: CameraSelection,
    /// Wall time of the timed frames, queued back to back with frames in flight, divided by their number. A serial
    /// submit-and-wait loop leaves the GPU idle between frames and lets the desktop driver lower its clock.
    frame_ms: f64,
    stages_ms: Option<Vec<(String, f64)>>,
    /// Against the first mode (absent for the first mode).
    vs_reference: Option<RenderMetrics>,
}

#[derive(Serialize)]
struct ModeResult {
    mode: String,
    cameras: Vec<CameraResult>,
}

#[derive(Serialize)]
struct Report {
    set: SetReport,
    adapter: String,
    width: u32,
    height: u32,
    frames: usize,
    warmup: usize,
    modes: Vec<ModeResult>,
}

/// `off`, `cull` or `lod:<tau>` over the `--lod-*` parameters.
fn parse_mode(text: &str, base: LodParams) -> Result<LodParams> {
    let (name, tau) = match text.split_once(':') {
        Some((name, tau)) => (name, Some(tau)),
        None => (text, None),
    };
    let mode = name.parse::<LodMode>().ok();
    let tau_px = match (mode, tau) {
        (Some(LodMode::Off | LodMode::Cull), None) => base.tau_px,
        (Some(LodMode::Lod), Some(tau)) => tau.parse().context("lod:<tau>")?,
        _ => anyhow::bail!("mode {text:?}: off, cull or lod:<tau>"),
    };
    Ok(LodParams {
        mode: mode.unwrap_or_default(),
        tau_px,
        ..base
    })
}

pub(crate) async fn run(a: Args) -> Result<()> {
    let dir = a.lod.lod.clone().context("--lod <dir> is required")?;
    ensure!(!a.modes.is_empty() && a.frames > 0, "need modes and frames");
    let modes = a
        .modes
        .iter()
        .map(|m| parse_mode(m, a.lod.params()))
        .collect::<Result<Vec<_>>>()?;
    let frames = load_frames(&a.camera, None).await?;
    let size = glam::UVec2::new(frames[0].camera.width, frames[0].camera.height);
    let scene = PlyScene::load_lod(&dir, &a.lod)?;
    let settings = gsplat_cli::settings::RenderSettings::default();
    let mut renderer =
        gsplat_cli::engines::ours(&scene, size, &settings, None, glam::Vec3::ZERO).await?;
    let source = scene.lod.as_ref().expect("loaded as a set");
    let set = SetReport::new(source, &a.lod, &renderer)?;
    eprintln!(
        "{} rows, {} leaves, upload {:.2} s, scene {} MB, view {} MB",
        set.scene_rows,
        set.leaves,
        set.upload_seconds,
        set.scene_gpu_bytes >> 20,
        set.view_gpu_bytes >> 20
    );
    let evaluator = Evaluator::new(false);
    let mut reference: Vec<Vec<f32>> = Vec::new();
    let mut results = Vec::new();
    for (index, (text, mode)) in a.modes.iter().zip(&modes).enumerate() {
        renderer.set_lod_params(*mode);
        let mut cameras = Vec::new();
        for (c, frame) in frames.iter().enumerate() {
            let camera = &frame.camera;
            let stats = renderer.render(camera, Output::Float)?;
            let pixels = renderer.read_rgba()?;
            if a.images {
                let path = a.out.join(format!("{index}-{}", text.replace(':', "")));
                std::fs::create_dir_all(&path)?;
                gsplat_cli::parity_image(pixels.clone(), size.x, size.y)?
                    .to_rgb8()
                    .save(path.join(format!("{c:03}.png")))?;
            }
            let vs_reference = if index == 0 {
                reference.push(pixels);
                None
            } else {
                Some(
                    evaluator
                        .evaluate_renders(&pixels, &reference[c], size.x, size.y)
                        .await?,
                )
            };
            let selection = CameraSelection::new(c, &renderer, stats.visible, stats.intersections)?;
            for _ in 0..a.warmup {
                renderer.render(camera, Output::Float)?;
            }
            let start = Instant::now();
            for _ in 0..a.frames {
                renderer.submit_float(camera)?;
            }
            renderer.finish_submitted()?;
            let frame_ms = start.elapsed().as_secs_f64() * 1000.0 / a.frames.max(1) as f64;
            let stages_ms = renderer.stages(camera)?.map(|stages| {
                stages
                    .into_iter()
                    .map(|s| (s.name, s.ms))
                    .collect::<Vec<_>>()
            });
            cameras.push(CameraResult {
                selection,
                frame_ms,
                stages_ms,
                vs_reference,
            });
        }
        eprintln!(
            "mode {text}: median over cameras {:.3} ms per frame",
            gsplat_cli::statistics::median(&cameras.iter().map(|c| c.frame_ms).collect::<Vec<_>>())
        );
        results.push(ModeResult {
            mode: text.clone(),
            cameras,
        });
    }
    crate::write_json(
        &a.out.join("lod-bench.json"),
        &Report {
            set,
            adapter: renderer.adapter_info().name.clone(),
            width: size.x,
            height: size.y,
            frames: a.frames,
            warmup: a.warmup,
            modes: results,
        },
    )
}
