//! Serial, rotated repeat scheduling and median-of-repeat statistics.
use crate::{CameraArgs, cameras, write_json};
use anyhow::{Result, ensure};
use clap::Args;
use gsplat_bench::{
    renderers::{Adapter, Brush, Counts, Engine, Implementation, RenderEngine, Scene, StageTiming},
    statistics::{Statistics, median, summarize},
};
use gsplat_eval::{Evaluator, RenderMetrics, Versions};
use gsplat_render::{camera::CameraSpec, settings::RenderSettings};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
mod host;
#[derive(Args)]
pub struct SpeedArgs {
    #[arg(
        long = "impl",
        value_enum,
        value_delimiter = ',',
        default_value = "brush,ours,native"
    )]
    pub implementations: Vec<Implementation>,
    #[command(flatten)]
    pub camera: CameraArgs,
    #[command(flatten)]
    settings: RenderSettings,
    #[arg(long, default_value_t = 120)]
    warmup: usize,
    #[arg(long, default_value_t = 600)]
    frames: usize,
    #[arg(long, default_value_t = 10.0)]
    min_seconds: f64,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    camera: usize,
    ms: f64,
    counts: Counts,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Repeat {
    repeat: usize,
    order: usize,
    frames: Vec<Frame>,
    statistics: Statistics,
    warmup_frames: usize,
    warmup_seconds: f64,
    measured_seconds: f64,
    loaded_host: bool,
    quiet_host_verified: bool,
    desktop_baseline_gpu_pct: Option<f64>,
    before: Vec<host::HostSample>,
    after: host::HostSample,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendReport {
    implementation: Implementation,
    adapter: Adapter,
    output_format: String,
    cold_start_to_first_frame_ms: f64,
    repeats: Vec<Repeat>,
    median_of_repeat_medians_ms: f64,
    median_of_repeat_p95s_ms: f64,
    stable_within_five_percent: bool,
    pooled: Statistics,
    lane1: Option<Vec<StageTiming>>,
    camera_checks: Vec<CameraCheck>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeedReport {
    complete: bool,
    settings: RenderSettings,
    ply: PathBuf,
    splats: usize,
    cameras: Vec<CameraSpec>,
    versions: Versions,
    cpuset: String,
    minimum_warmup_seconds: f64,
    minimum_measured_seconds: f64,
    minimum_frames: usize,
    repeat_count: usize,
    backends: Vec<BackendReport>,
}
async fn warmup<R: RenderEngine>(
    renderer: &mut R,
    cameras: &[CameraSpec],
    minimum_frames: usize,
) -> Result<(usize, f64)> {
    let start = Instant::now();
    let mut frames = 0;
    loop {
        for camera in cameras {
            renderer.render(camera, false).await?;
            renderer.finish()?;
            frames += 1;
        }
        if frames >= minimum_frames && start.elapsed().as_secs_f64() >= 5.0 {
            return Ok((frames, start.elapsed().as_secs_f64()));
        }
    }
}
async fn repeat<R: RenderEngine>(
    renderer: &mut R,
    cameras: &[CameraSpec],
    a: &SpeedArgs,
    index: usize,
    order: usize,
    admission_deadline: Instant,
) -> Result<Repeat> {
    eprintln!("repeat {}: quiet-host admission", index + 1);
    let (before, loaded_host) = host::wait_quiet(admission_deadline)?;
    let (warmup_frames, warmup_seconds) = warmup(renderer, cameras, a.warmup).await?;
    let start = Instant::now();
    let mut frames = Vec::new();
    loop {
        for (camera, c) in cameras.iter().enumerate() {
            let now = Instant::now();
            let counts = renderer.render(c, false).await?;
            renderer.finish()?;
            frames.push(Frame {
                camera,
                ms: now.elapsed().as_secs_f64() * 1000.0,
                counts,
            });
        }
        if frames.len() >= a.frames.max(cameras.len() * 2)
            && start.elapsed().as_secs_f64() >= a.min_seconds.max(10.0)
        {
            break;
        }
    }
    let measured_seconds = start.elapsed().as_secs_f64();
    let after = host::sample()?;
    let statistics = summarize(&frames.iter().map(|f| f.ms).collect::<Vec<_>>())?;
    eprintln!(
        "repeat {}: median {:.4} ms, p95 {:.4} ms, {} frames",
        index + 1,
        statistics.median_ms,
        statistics.p95_ms,
        frames.len()
    );
    let desktop_baseline_gpu_pct = if cfg!(target_os = "macos") {
        let percentages: Vec<_> = before
            .iter()
            .filter_map(|sample| sample.gpu_percent.map(f64::from))
            .collect();
        (!percentages.is_empty()).then(|| median(&percentages))
    } else {
        None
    };
    Ok(Repeat {
        repeat: index,
        order,
        frames,
        statistics,
        warmup_frames,
        warmup_seconds,
        measured_seconds,
        loaded_host,
        quiet_host_verified: before.iter().all(host::HostSample::quiet_verified),
        desktop_baseline_gpu_pct,
        before,
        after,
    })
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CameraCheck {
    camera: usize,
    metrics: RenderMetrics,
}

async fn check_cameras<R: RenderEngine>(
    renderer: &mut R,
    oracle: &mut Brush,
    cameras: &[CameraSpec],
) -> Result<Vec<CameraCheck>> {
    let evaluator = Evaluator::new(false);
    let mut checks = Vec::new();
    for (index, c) in cameras.iter().enumerate().step_by(30) {
        renderer.capture_next_frame();
        renderer.render(c, false).await?;
        renderer.finish()?;
        let pixels = renderer.read_rgba_f32().await?;
        oracle.render(c, true).await?;
        oracle.finish()?;
        let reference = oracle.read_rgba_f32().await?;
        let metrics = evaluator
            .evaluate_renders(&pixels, &reference, c.width, c.height)
            .await?;
        checks.push(CameraCheck {
            camera: index,
            metrics,
        });
    }
    Ok(checks)
}

pub async fn run(a: SpeedArgs) -> Result<()> {
    ensure!(
        a.repeats >= 3 && a.min_seconds.is_finite() && a.min_seconds >= 10.0,
        "need >=3 repeats and >=10 measured seconds"
    );
    ensure!(!a.implementations.is_empty(), "empty implementation list");
    for (index, kind) in a.implementations.iter().enumerate() {
        ensure!(
            !a.implementations[..index].contains(kind),
            "duplicate implementation"
        );
    }
    let scene = Scene::load(&a.camera.ply).await?;
    let cameras = cameras(&a.camera, &scene).await?;
    let mut engines = Vec::new();
    for &kind in &a.implementations {
        let start = Instant::now();
        let mut engine =
            Engine::new(kind, &scene, &a.settings, &cameras[0], &a.camera.ply, None).await?;
        engine.render(&cameras[0], false).await?;
        engine.finish()?;
        engines.push((
            kind,
            engine,
            start.elapsed().as_secs_f64() * 1000.0,
            Vec::new(),
        ));
    }
    // Rotate backend order for each repeat; each timed run admits ten quiet samples.
    let admission_deadline = Instant::now() + Duration::from_secs(90 * 60);
    for index in 0..a.repeats {
        for order in 0..engines.len() {
            let backend = (index + order) % engines.len();
            let (kind, engine, _, repeats) = &mut engines[backend];
            eprintln!("{kind:?}: round {}, position {}", index + 1, order + 1);
            repeats.push(repeat(engine, &cameras, &a, index, order, admission_deadline).await?);
        }
    }
    let mut backends = Vec::new();
    let mut oracle = Brush::new(&scene, &a.settings).await;
    for (implementation, mut engine, cold_start_to_first_frame_ms, repeats) in engines {
        let medians: Vec<_> = repeats.iter().map(|r| r.statistics.median_ms).collect();
        let median_of_repeat_medians_ms = median(&medians);
        let stable_within_five_percent = medians.iter().enumerate().all(|(i, value)| {
            let others: Vec<_> = medians
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .collect();
            (*value / median(&others) - 1.0).abs() <= 0.05
        });
        backends.push(BackendReport {
            implementation,
            adapter: engine.adapter(),
            output_format: engine.output_format().into(),
            cold_start_to_first_frame_ms,
            median_of_repeat_medians_ms,
            median_of_repeat_p95s_ms: median(
                &repeats
                    .iter()
                    .map(|r| r.statistics.p95_ms)
                    .collect::<Vec<_>>(),
            ),
            stable_within_five_percent,
            pooled: summarize(
                &repeats
                    .iter()
                    .flat_map(|r| r.frames.iter().map(|f| f.ms))
                    .collect::<Vec<_>>(),
            )?,
            lane1: engine.stages(&cameras[0]).await?,
            camera_checks: check_cameras(&mut engine, &mut oracle, &cameras).await?,
            repeats,
        });
    }
    write_json(
        &a.out,
        &SpeedReport {
            complete: true,
            settings: a.settings,
            ply: a.camera.ply,
            splats: scene.data.num_splats(),
            cameras,
            versions: Versions::default(),
            cpuset: host::cpuset()?,
            minimum_warmup_seconds: 5.0,
            minimum_measured_seconds: a.min_seconds,
            minimum_frames: a.frames,
            repeat_count: a.repeats,
            backends,
        },
    )
}
