//! Serial, rotated repeat scheduling and median-of-repeat statistics.
use crate::{CameraArgs, cameras, write_json};
use anyhow::{Result, ensure};
use clap::Args;
use gsplat_bench::{
    camera::CameraSpec,
    renderers::{
        Adapter, Brush, Counts, Implementation, Native, Old, RenderEngine, Scene, StageTiming,
    },
    statistics::{Statistics, median, summarize},
};
use gsplat_eval::{Evaluator, RenderMetrics, Versions};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[path = "host.rs"]
mod host;
#[derive(Args)]
pub struct SpeedArgs {
    #[arg(
        long = "impl",
        value_enum,
        value_delimiter = ',',
        default_value = "brush,ours-old,native"
    )]
    pub implementations: Vec<Implementation>,
    #[command(flatten)]
    pub camera: CameraArgs,
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
    attempt: usize,
    order: usize,
    frames: Vec<Frame>,
    statistics: Statistics,
    warmup_frames: usize,
    warmup_seconds: f64,
    measured_seconds: f64,
    loaded_host: bool,
    quiet_host_verified: bool,
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
    attempts: Vec<Repeat>,
    selected_attempts: Vec<usize>,
    median_of_repeat_medians_ms: f64,
    median_of_repeat_p95s_ms: f64,
    stable_within_five_percent: bool,
    pooled: Option<Statistics>,
    lane1: Option<Vec<StageTiming>>,
    camera_checks: Vec<CameraCheck>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeedReport {
    complete: bool,
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
async fn repeat<R: RenderEngine>(
    renderer: &mut R,
    cameras: &[CameraSpec],
    a: &SpeedArgs,
    index: usize,
    attempt: usize,
    order: usize,
    admission_deadline: Instant,
) -> Result<Repeat> {
    eprintln!(
        "repeat {} attempt {}: quiet-host admission",
        index + 1,
        attempt + 1
    );
    let (before, loaded_host) = host::wait_quiet(admission_deadline)?;
    let start = Instant::now();
    let mut warmup_frames = 0;
    loop {
        for c in cameras {
            renderer.render(c, false).await?;
            renderer.finish()?;
            warmup_frames += 1;
        }
        if warmup_frames >= a.warmup && start.elapsed().as_secs_f64() >= 5.0 {
            break;
        }
    }
    let warmup_seconds = start.elapsed().as_secs_f64();
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
        "repeat {} attempt {}: median {:.4} ms, p95 {:.4} ms, {} frames",
        index + 1,
        attempt + 1,
        statistics.median_ms,
        statistics.p95_ms,
        frames.len()
    );
    Ok(Repeat {
        repeat: index,
        attempt,
        order,
        frames,
        statistics,
        warmup_frames,
        warmup_seconds,
        measured_seconds,
        loaded_host,
        quiet_host_verified: before.iter().all(host::HostSample::quiet_verified),
        before,
        after,
    })
}
async fn initialize<R: RenderEngine>(
    renderer: &mut R,
    c: &CameraSpec,
    start: Instant,
    kind: Implementation,
    format: &str,
    repeats: usize,
) -> Result<BackendReport> {
    renderer.render(c, false).await?;
    renderer.finish()?;
    Ok(BackendReport {
        implementation: kind,
        adapter: renderer.adapter(),
        output_format: format.into(),
        cold_start_to_first_frame_ms: start.elapsed().as_secs_f64() * 1000.0,
        attempts: Vec::new(),
        selected_attempts: vec![0; repeats],
        median_of_repeat_medians_ms: 0.0,
        median_of_repeat_p95s_ms: 0.0,
        stable_within_five_percent: false,
        pooled: None,
        lane1: None,
        camera_checks: Vec::new(),
    })
}
fn outliers(report: &BackendReport) -> Vec<usize> {
    let times: Vec<_> = report
        .selected_attempts
        .iter()
        .map(|&i| report.attempts[i].statistics.median_ms)
        .collect();
    (0..times.len())
        .filter(|&i| {
            let others: Vec<_> = times
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .collect();
            (times[i] / median(&others) - 1.0).abs() > 0.05
        })
        .collect()
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
    packed: bool,
) -> Result<Vec<CameraCheck>> {
    let evaluator = Evaluator::new(false);
    let mut checks = Vec::new();
    for (index, c) in cameras.iter().enumerate().step_by(30) {
        renderer.render(c, !packed).await?;
        renderer.finish()?;
        // Old's readback checks the raw intersection count against capacity.
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
    for (i, kind) in a.implementations.iter().enumerate() {
        ensure!(
            !a.implementations[..i].contains(kind),
            "duplicate implementation"
        );
    }
    let scene = Scene::load(&a.camera.ply).await?;
    let cameras = cameras(&a.camera, &scene).await?;
    let mut brush = None;
    let mut old = None;
    let mut native = None;
    let mut reports = Vec::new();
    for kind in &a.implementations {
        let start = Instant::now();
        let report = match kind {
            Implementation::Brush => {
                let r = brush.insert(Brush::new(&scene).await);
                initialize(r, &cameras[0], start, *kind, "Brush Packed", a.repeats).await?
            }
            Implementation::OursOld => {
                let r = old.insert(Old::new(&scene, cameras[0].width, cameras[0].height)?);
                initialize(r, &cameras[0], start, *kind, "RGBA8Unorm", a.repeats).await?
            }
            Implementation::Native => {
                let r = native.insert(Native::new(&a.camera.ply, scene.data.num_splats()).await?);
                initialize(
                    r,
                    &cameras[0],
                    start,
                    *kind,
                    "Rerun RGBA8UnormSrgb/MSAA4",
                    a.repeats,
                )
                .await?
            }
        };
        reports.push(report);
    }
    let mut suite = SpeedReport {
        complete: false,
        ply: a.camera.ply.clone(),
        splats: scene.data.num_splats(),
        cameras,
        versions: Versions::default(),
        cpuset: host::cpuset()?,
        minimum_warmup_seconds: 5.0,
        minimum_measured_seconds: a.min_seconds,
        minimum_frames: a.frames,
        repeat_count: a.repeats,
        backends: reports,
    };
    // ABC, BCA, CAB; each admitted run has its own ten quiet samples.
    let admission_deadline = Instant::now() + Duration::from_secs(90 * 60);
    for attempt in 0..=2 {
        let pending: Vec<Vec<usize>> = suite
            .backends
            .iter()
            .map(|r| {
                if attempt == 0 {
                    (0..a.repeats).collect()
                } else {
                    outliers(r)
                }
            })
            .collect();
        if pending.iter().all(Vec::is_empty) {
            break;
        }
        for index in 0..a.repeats {
            for order in 0..suite.backends.len() {
                let backend = (index + order) % suite.backends.len();
                if !pending[backend].contains(&index) {
                    continue;
                }
                let report = &mut suite.backends[backend];
                eprintln!(
                    "{:?}: round {}, position {}",
                    report.implementation,
                    index + 1,
                    order + 1
                );
                let run = match report.implementation {
                    Implementation::Brush => {
                        repeat(
                            brush.as_mut().unwrap(),
                            &suite.cameras,
                            &a,
                            index,
                            attempt,
                            order,
                            admission_deadline,
                        )
                        .await?
                    }
                    Implementation::OursOld => {
                        repeat(
                            old.as_mut().unwrap(),
                            &suite.cameras,
                            &a,
                            index,
                            attempt,
                            order,
                            admission_deadline,
                        )
                        .await?
                    }
                    Implementation::Native => {
                        repeat(
                            native.as_mut().unwrap(),
                            &suite.cameras,
                            &a,
                            index,
                            attempt,
                            order,
                            admission_deadline,
                        )
                        .await?
                    }
                };
                report.selected_attempts[index] = report.attempts.len();
                report.attempts.push(run);
                write_json(&a.out, &suite)?;
            }
        }
    }
    for report in &mut suite.backends {
        report.stable_within_five_percent = outliers(report).is_empty();
        let selected: Vec<_> = report
            .selected_attempts
            .iter()
            .map(|&i| &report.attempts[i])
            .collect();
        report.median_of_repeat_medians_ms = median(
            &selected
                .iter()
                .map(|r| r.statistics.median_ms)
                .collect::<Vec<_>>(),
        );
        report.median_of_repeat_p95s_ms = median(
            &selected
                .iter()
                .map(|r| r.statistics.p95_ms)
                .collect::<Vec<_>>(),
        );
        report.pooled = Some(summarize(
            &selected
                .iter()
                .flat_map(|r| r.frames.iter().map(|f| f.ms))
                .collect::<Vec<_>>(),
        )?);
        report.lane1 = match report.implementation {
            Implementation::Brush => brush.as_mut().unwrap().stages(&suite.cameras[0]).await?,
            Implementation::OursOld => old.as_mut().unwrap().stages(&suite.cameras[0]).await?,
            Implementation::Native => native.as_mut().unwrap().stages(&suite.cameras[0]).await?,
        };
    }
    let mut oracle = Brush::new(&scene).await;
    for report in &mut suite.backends {
        report.camera_checks = match report.implementation {
            Implementation::Brush => {
                check_cameras(brush.as_mut().unwrap(), &mut oracle, &suite.cameras, true).await?
            }
            Implementation::OursOld => {
                check_cameras(old.as_mut().unwrap(), &mut oracle, &suite.cameras, false).await?
            }
            Implementation::Native => {
                check_cameras(native.as_mut().unwrap(), &mut oracle, &suite.cameras, false).await?
            }
        };
    }
    suite.complete = true;
    write_json(&a.out, &suite)
}
