//! Standalone CPU renderer for Gaussian splats.
//!
//! Renders a PLY file at a camera position from a NeRF transforms JSON,
//! producing a PNG image.  No Rerun dependency — this binary is built with
//! `--no-default-features` to exclude all `re_*` crates.
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

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use gsplat_lib::gsplat_core::software_rasterizer::software_render;
use gsplat_lib::nerf_camera;
use gsplat_lib::ply_loader;

#[derive(Parser)]
#[command(
    name = "gsplat-render",
    about = "Standalone CPU Gaussian splat renderer"
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

    /// Output PNG path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Image width in pixels.
    #[arg(long, default_value_t = 800)]
    width: u32,

    /// Image height in pixels.
    #[arg(long, default_value_t = 800)]
    height: u32,

    /// Background color as R,G,B floats (e.g. "1,1,1" for white, "0,0,0" for black).
    #[arg(long, default_value = "1,1,1")]
    background: String,

    /// Run benchmark mode: render N frames and report timing.
    #[arg(long)]
    benchmark: bool,

    /// Number of frames to render in benchmark mode.
    #[arg(long, default_value_t = 1)]
    num_frames: usize,
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
    let background = parse_background(&args.background)?;

    eprintln!("Loading PLY: {:?}", args.ply);
    let load_start = Instant::now();
    let cloud = ply_loader::load_ply(&args.ply)?;
    eprintln!(
        "Loaded {} splats in {:.1}ms",
        cloud.len(),
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let camera = nerf_camera::load_camera(&args.camera, args.frame, args.width, args.height)?;
    eprintln!(
        "Camera: frame {}, {}x{}, position {:?}",
        args.frame, args.width, args.height, camera.world_position
    );

    if args.benchmark {
        eprintln!("Benchmarking {} frames...", args.num_frames);
        let mut total_ms = 0.0_f64;
        for i in 0..args.num_frames {
            let start = Instant::now();
            let _output = software_render(&cloud, &camera, background);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            total_ms += ms;
            eprintln!("  Frame {i}: {ms:.1}ms");
        }
        eprintln!("Average: {:.1}ms/frame", total_ms / args.num_frames as f64);
    } else {
        let output_path = args
            .output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--output is required when not in benchmark mode"))?;

        let start = Instant::now();
        let output = software_render(&cloud, &camera, background);
        let render_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Rendered in {render_ms:.1}ms");

        let rgb8 = output.to_rgb8(background);
        image::save_buffer(
            output_path,
            &rgb8,
            output.width,
            output.height,
            image::ColorType::Rgb8,
        )?;
        eprintln!("Saved to {:?}", output_path);
    }

    Ok(())
}
