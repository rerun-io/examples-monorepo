//! Run Brush's pinned training stream with optional native Rerun observations.
mod dashboard;
mod logging;
mod recorder;
mod test_renders;

use anyhow::{Context as _, ensure};
use brush_process::{
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
};
use clap::Parser;

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};
use tokio_stream::StreamExt as _;

#[derive(Parser)]
#[command(
    about = "Train with Brush; --save, --connect or --spawn enables native Rerun logging. Without a sink, all recording work is disabled."
)]
struct Cli {
    source: brush_process::DataSource,
    #[command(flatten)]
    training: TrainStreamConfig,
    #[arg(long)]
    connect: Option<String>,
    #[arg(long)]
    save: Option<PathBuf>,
    /// Launch the stock viewer. For a custom viewer use --connect.
    #[arg(long, conflicts_with_all = ["save", "connect", "compute_visualizer"])]
    spawn: bool,
    #[arg(long, default_value_t = 50, value_parser = observation_interval)]
    snapshot_first: u32,
    /// Flat layout with a scene spinning about Brush's estimated up axis.
    #[arg(long)]
    video: bool,
    /// Explicit override for a custom viewer; stock Rerun is the default.
    #[arg(long)]
    compute_visualizer: bool,
    /// After the last step, render the eval views with Brush (black background) as 16-bit RGB PNGs here.
    #[arg(long)]
    test_renders: Option<PathBuf>,
    /// Write steps, loop and wall seconds, the final splat count and Brush's eval scores as JSON.
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(serde::Serialize)]
struct Summary {
    steps: u32,
    /// Brush's summed step time: data wait, optimisation and refinement; no evaluation or export.
    loop_seconds: f64,
    /// From the end of loading to the end of the stream, including evaluation and export.
    wall_seconds: f64,
    /// Wall clock at the end of the stream, before the test renders, so memory samples can be split.
    training_end_unix_seconds: f64,
    splats: u32,
    /// Brush's own eval at the last step, after an 8-bit roundtrip of the render.
    brush_eval_psnr: Option<f32>,
    brush_eval_ssim: Option<f32>,
    renderer: &'static str,
}

fn observation_interval(value: &str) -> Result<u32, String> {
    checked_interval(value.parse().map_err(|error| format!("{error}"))?)
}

fn checked_interval(steps: u32) -> Result<u32, String> {
    if steps == 0 || !steps.is_multiple_of(5) {
        return Err(
            "Brush emits observations every five steps; use a positive multiple of 5".into(),
        );
    }
    Ok(steps)
}

fn validate_logging_config(config: &TrainStreamConfig) -> anyhow::Result<()> {
    let rerun = &config.rerun_config;
    ensure!(
        !rerun.rerun_enabled,
        "--rerun-enabled enables Brush's legacy logger and is unsupported; use --save, --connect or --spawn"
    );
    ensure!(
        rerun.rerun_log_distribution_every == 1000,
        "--rerun-log-distribution-every is unsupported by the native recorder"
    );
    checked_interval(rerun.rerun_log_train_stats_every).map_err(anyhow::Error::msg)?;
    if let Some(every) = rerun.rerun_log_splats_every {
        checked_interval(every).map_err(anyhow::Error::msg)?;
    }
    ensure!(
        rerun.rerun_max_img_size > 0,
        "--rerun-max-img-size must be positive"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    validate_logging_config(&cli.training)?;
    let enabled = cli.spawn || cli.save.is_some() || cli.connect.is_some();
    ensure!(
        !cli.compute_visualizer || enabled,
        "--compute-visualizer requires --save or --connect"
    );
    let device = brush_process::burn_init_setup().await;
    let mut recorder = recorder::Recorder::new(&cli, &device)?;
    let (test_renders_dir, summary_path) = (cli.test_renders.clone(), cli.summary.clone());
    let config_error = Arc::new(OnceLock::new());
    let callback_error = config_error.clone();
    let mut process =
        brush_process::create_process_with_device(cli.source, device, async move |initial| {
            let config = brush_process::args_file::merge_configs(&initial, &cli.training);
            if let Err(error) = validate_logging_config(&config) {
                callback_error.set(error.to_string()).ok();
                return None;
            }
            Some(config)
        });
    let mut final_step = 0;
    let mut training_start = None;
    let mut brush_elapsed = Duration::ZERO;
    let mut eval_views = None;
    let mut last_eval = None;
    let training_result: anyhow::Result<()> = async {
        while let Some(event) = process.stream.next().await {
            let event = event?;
            match &event {
                ProcessMessage::DoneLoading => {
                    training_start = Some(Instant::now());
                    println!("Completed loading.");
                }
                ProcessMessage::TrainMessage(TrainMessage::TrainConfig { config }) => {
                    final_step = config.train_config.total_iters();
                }
                ProcessMessage::TrainMessage(TrainMessage::Dataset { dataset }) => {
                    eval_views = dataset.eval.as_ref().map(|scene| scene.views.clone());
                }
                ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    ..
                }) => {
                    brush_elapsed = *total_elapsed;
                    if iter.is_multiple_of(100) || *iter == final_step {
                        println!(
                            "Train iter {iter}: brush_seconds {:.9}",
                            total_elapsed.as_secs_f64()
                        );
                    }
                }
                ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                    iter,
                    avg_psnr,
                    avg_ssim,
                }) => {
                    println!("Eval iter {iter}: PSNR {avg_psnr}, SSIM {avg_ssim}");
                    last_eval = Some((*avg_psnr, *avg_ssim));
                }
                ProcessMessage::Warning { error } => eprintln!("Brush warning: {error:#}"),
                ProcessMessage::StartLoading {
                    training: false, ..
                } => anyhow::bail!("training requires a dataset folder"),
                _ => {}
            }
            if let Some(recorder) = &mut recorder {
                recorder.observe(event, &process.splat_view)?;
            }
        }
        Ok(())
    }
    .await;
    if let Some(recorder) = recorder {
        recorder.finish();
    }
    if let Some(error) = config_error.get() {
        anyhow::bail!("Invalid logging configuration: {error}");
    }
    training_result?;
    let wall = training_start
        .context("training never started")?
        .elapsed()
        .as_secs_f64();
    let training_end = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    println!(
        "Training summary: steps={final_step} wall_seconds={wall:.9} brush_seconds={:.9} logging={enabled}",
        brush_elapsed.as_secs_f64()
    );
    let splats = process
        .splat_view
        .latest()
        .context("training produced no splats")?;
    if let Some(dir) = &test_renders_dir {
        let views = eval_views.context("--test-renders needs eval views (transforms_test.json)")?;
        test_renders::write(&splats, &views, dir).await?;
    }
    if let Some(path) = summary_path {
        let summary = Summary {
            steps: final_step,
            loop_seconds: brush_elapsed.as_secs_f64(),
            wall_seconds: wall,
            training_end_unix_seconds: training_end.as_secs_f64(),
            splats: splats.num_splats(),
            brush_eval_psnr: last_eval.map(|(psnr, _)| psnr),
            brush_eval_ssim: last_eval.map(|(_, ssim)| ssim),
            renderer: test_renders::RENDERER,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&summary)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logging_rejects_unobservable_or_unsupported_brush_options() {
        for args in [
            vec!["--rerun-log-train-stats-every", "7"],
            vec!["--rerun-log-splats-every", "1"],
            vec!["--rerun-enabled"],
            vec!["--rerun-log-distribution-every", "200"],
        ] {
            let config =
                TrainStreamConfig::try_parse_from(std::iter::once("train").chain(args)).unwrap();
            assert!(validate_logging_config(&config).is_err());
        }
        let config = TrainStreamConfig::try_parse_from([
            "train",
            "--rerun-log-train-stats-every",
            "5",
            "--rerun-log-splats-every",
            "500",
        ])
        .unwrap();
        assert!(validate_logging_config(&config).is_ok());
        assert!(Cli::try_parse_from(["train", "lego", "--snapshot-first", "7"]).is_err());
    }
}
