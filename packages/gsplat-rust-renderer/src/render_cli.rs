//! Standalone GPU renderer for Gaussian splats.
//!
//! Renders a PLY file at a camera position from a NeRF transforms JSON,
//! producing a PNG image.  No Rerun dependency — this binary is built with
//! `--no-default-features` to exclude all `re_*` crates.
//!
//! Uses the same GPU compute pipeline as the Rerun viewer (stages 1-8 of the
//! 9-stage pipeline in `gaussian_renderer.rs`; the viewer-only composite blit
//! is replaced by a texture readback) but runs headless via raw `wgpu`.
//!
//! # Usage
//!
//! ```bash
//! gsplat-render \
//!   --ply data/trained/chair.ply \
//!   --camera data/nerf-synthetic/chair/transforms_test.json \
//!   --frame 0 \
//!   --output /tmp/chair_test_0.png \
//!   --width 800 --height 800 \
//!   --background 1,1,1
//! ```

use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use gsplat_lib::gsplat_core::gpu_context::GpuContext;
use gsplat_lib::gsplat_core::gpu_renderer::{GpuRenderResources, GpuRenderer};
use gsplat_lib::gsplat_core::types::RenderOutput;
use gsplat_lib::nerf_camera;
use gsplat_lib::ply_loader;

#[derive(Parser)]
#[command(
    name = "gsplat-render",
    about = "Standalone GPU Gaussian splat renderer"
)]
struct Args {
    /// Path to a 3DGS PLY file.
    #[arg(long)]
    ply: PathBuf,

    /// Path to a NeRF transforms JSON file (e.g. transforms_test.json).
    #[arg(long)]
    camera: PathBuf,

    /// Frame index within the transforms JSON.
    #[arg(long, default_value_t = 0)]
    frame: usize,

    /// Output PNG path (required unless --benchmark; with --benchmark,
    /// optionally saves the last benchmark frame).
    #[arg(
        long,
        required_unless_present_any = ["benchmark", "output_dir"],
        conflicts_with = "output_dir"
    )]
    output: Option<PathBuf>,

    /// Render every camera in the transforms JSON into this directory. Output
    /// paths preserve each frame's relative file path and use a PNG extension.
    #[arg(long, conflicts_with = "benchmark")]
    output_dir: Option<PathBuf>,

    /// Image width in pixels.
    #[arg(long, default_value_t = 800)]
    width: u32,

    /// Image height in pixels.
    #[arg(long, default_value_t = 800)]
    height: u32,

    /// Background color as R,G,B floats (e.g. "1,1,1" for white, "0,0,0" for black).
    #[arg(long, default_value = "1,1,1")]
    background: String,

    /// Run benchmark mode: render N frames and report timing + FPS.
    #[arg(long)]
    benchmark: bool,

    /// Number of frames to render in benchmark mode.
    #[arg(long, default_value_t = 10)]
    num_frames: usize,
}

/// Composite onto `background` and save as an 8-bit RGB PNG.
fn save_rgb8_png(
    output: &RenderOutput,
    background: [f32; 3],
    path: &PathBuf,
) -> anyhow::Result<()> {
    let rgb8: Vec<u8> = output.to_rgb8(background);
    image::save_buffer(
        path,
        &rgb8,
        output.width,
        output.height,
        image::ColorType::Rgb8,
    )?;
    Ok(())
}

fn parse_background(s: &str) -> anyhow::Result<[f32; 3]> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        anyhow::bail!("Background must be 3 comma-separated floats (e.g. 1,1,1)");
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let background: [f32; 3] = parse_background(&args.background)?;

    eprintln!("Loading PLY: {:?}", args.ply);
    let load_start: Instant = Instant::now();
    let cloud = ply_loader::load_ply(&args.ply)?;
    eprintln!(
        "Loaded {} splats in {:.1}ms",
        cloud.len(),
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let all_frames = args
        .output_dir
        .as_ref()
        .map(|_| nerf_camera::load_cameras(&args.camera, args.width, args.height))
        .transpose()?;
    let camera = if let Some(frames) = &all_frames {
        frames
            .first()
            .map(|frame| frame.camera.clone())
            .ok_or_else(|| anyhow::anyhow!("Camera file contains no frames"))?
    } else {
        nerf_camera::load_camera(&args.camera, args.frame, args.width, args.height)?
    };
    eprintln!(
        "Camera: frame {}, {}x{}, position {:?}",
        args.frame, args.width, args.height, camera.world_position
    );

    // ── GPU initialization ───────────────────────────────────────────────
    eprintln!("Initializing GPU...");
    let gpu_start: Instant = Instant::now();
    let ctx: GpuContext = GpuContext::new()?;
    eprintln!(
        "GPU ready in {:.1}ms — {} ({:?})",
        gpu_start.elapsed().as_secs_f64() * 1000.0,
        ctx.adapter_info.name,
        ctx.adapter_info.backend,
    );

    let renderer: GpuRenderer = GpuRenderer::new(&ctx.device);

    // Persistent per-(cloud, resolution) resources: splat upload, scratch
    // buffers, and bind groups happen once here, not per frame (Brush model —
    // its benchmark harness reuses resources the same way).
    let resources_start: Instant = Instant::now();
    let resources: GpuRenderResources =
        GpuRenderResources::new(&ctx.device, &renderer, &cloud, camera.viewport_size_px);
    eprintln!(
        "GPU resources built in {:.1}ms",
        resources_start.elapsed().as_secs_f64() * 1000.0
    );

    if let (Some(output_dir), Some(frames)) = (&args.output_dir, &all_frames) {
        std::fs::create_dir_all(output_dir)?;
        let render_start = Instant::now();
        for (index, frame) in frames.iter().enumerate() {
            let relative_path = Path::new(&frame.file_path);
            if relative_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            }) {
                anyhow::bail!(
                    "Frame {index} has unsafe non-relative file path {:?}",
                    frame.file_path
                );
            }
            let output_path = output_dir.join(relative_path).with_extension("png");
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let output = resources.render(&ctx, &renderer, &frame.camera);
            save_rgb8_png(&output, background, &output_path)?;
            if (index + 1) % 25 == 0 || index + 1 == frames.len() {
                eprintln!("Rendered {}/{} test frames", index + 1, frames.len());
            }
        }
        eprintln!(
            "Saved {} frames under {:?} in {:.1}s",
            frames.len(),
            output_dir,
            render_start.elapsed().as_secs_f64()
        );
    } else if args.benchmark {
        // Warmup frame (pipeline compilation happens here).
        eprintln!("Warming up (compiling GPU pipelines)...");
        let warmup_start: Instant = Instant::now();
        let _warmup = resources.render(&ctx, &renderer, &camera);
        eprintln!(
            "Warmup: {:.1}ms",
            warmup_start.elapsed().as_secs_f64() * 1000.0
        );

        eprintln!("Benchmarking {} frames...", args.num_frames);
        let mut total_ms: f64 = 0.0;
        let mut last_output: Option<RenderOutput> = None;
        for i in 0..args.num_frames {
            let start: Instant = Instant::now();
            let output = resources.render(&ctx, &renderer, &camera);
            let ms: f64 = start.elapsed().as_secs_f64() * 1000.0;
            total_ms += ms;
            eprintln!("  Frame {i}: {ms:.1}ms");
            last_output = Some(output);
        }
        let avg_ms: f64 = total_ms / args.num_frames as f64;
        let fps: f64 = 1000.0 / avg_ms;
        eprintln!(
            "Average: {avg_ms:.3}ms/frame ({fps:.1} FPS), {} splats, {}x{}",
            cloud.len(),
            args.width,
            args.height,
        );
        // Optionally save the last frame — proves buffer reuse stays correct.
        if let (Some(output_path), Some(output)) = (&args.output, &last_output) {
            save_rgb8_png(output, background, output_path)?;
            eprintln!("Saved last benchmark frame to {:?}", output_path);
        }
    } else {
        let output_path = args
            .output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--output is required when not in benchmark mode"))?;

        let start: Instant = Instant::now();
        let output = resources.render(&ctx, &renderer, &camera);
        let render_ms: f64 = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Rendered in {render_ms:.1}ms");

        save_rgb8_png(&output, background, output_path)?;
        eprintln!("Saved to {:?}", output_path);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_frames_cli_accepts_output_directory_without_single_output() {
        let args = Args::try_parse_from([
            "gsplat-render",
            "--ply",
            "scene.ply",
            "--camera",
            "transforms_test.json",
            "--output-dir",
            "renders",
        ])
        .expect("all-frame CLI should accept an output directory");

        assert_eq!(args.output_dir, Some(PathBuf::from("renders")));
        assert!(args.output.is_none());
    }
}
