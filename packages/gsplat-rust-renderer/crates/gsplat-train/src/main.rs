//! Run Brush's pinned training stream with optional native Rerun observations.
mod dashboard;
mod logging;

use anyhow::{Context, ensure};
use brush_process::{
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
};
use clap::{CommandFactory, FromArgMatches, Parser};

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};
use tokio_stream::StreamExt;

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
}

fn observation_interval(value: &str) -> Result<u32, String> {
    let steps: u32 = value.parse().map_err(|error| format!("{error}"))?;
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
    observation_interval(&rerun.rerun_log_train_stats_every.to_string())
        .map_err(anyhow::Error::msg)?;
    if let Some(every) = rerun.rerun_log_splats_every {
        observation_interval(&every.to_string()).map_err(anyhow::Error::msg)?;
    }
    ensure!(
        rerun.rerun_max_img_size > 0,
        "--rerun-max-img-size must be positive"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches)?;
    ensure!(
        matches.value_source("rerun_log_distribution_every")
            != Some(clap::parser::ValueSource::CommandLine),
        "--rerun-log-distribution-every is unsupported by the native recorder"
    );
    validate_logging_config(&cli.training)?;
    let enabled = cli.spawn || cli.save.is_some() || cli.connect.is_some();
    ensure!(
        !cli.compute_visualizer || enabled,
        "--compute-visualizer requires --save or --connect"
    );
    let device = brush_process::burn_init_setup().await;
    let (tx, worker) = if enabled {
        let mut sinks: Vec<Box<dyn rerun::sink::LogSink>> = Vec::new();
        if let Some(path) = &cli.save {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            sinks.push(Box::new(rerun::sink::FileSink::new(path)?));
        }
        if let Some(url) = &cli.connect {
            use rerun::sink::LogSink as _;
            let sink = rerun::sink::GrpcSink::new(url.parse()?);
            // An unconnected SDK sink can fill its bounded queue before the
            // final flush is reached. Attach it only after a bounded handshake.
            match sink.flush_blocking(Duration::from_secs(2)) {
                Ok(()) => sinks.push(Box::new(sink)),
                Err(error) => eprintln!(
                    "Live recording disabled: {error}; training and file recording continue"
                ),
            }
        }
        let rec = if cli.spawn {
            rerun::RecordingStreamBuilder::new("gsplat-train").spawn()?
        } else {
            rerun::RecordingStreamBuilder::new("gsplat-train").set_sinks(sinks)?
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let logging_device = device.clone();
        let worker = std::thread::Builder::new()
            .name("gsplat-rerun".into())
            .spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map(|runtime| {
                        runtime.block_on(logging::run(
                            rx,
                            rec,
                            logging_device,
                            cli.compute_visualizer,
                            cli.video,
                        ));
                    })
            })?;
        (Some(tx), Some(worker))
    } else {
        (None, None)
    };
    let config_error = Arc::new(OnceLock::new());
    let callback_error = config_error.clone();
    let mut process =
        brush_process::create_process_with_device(cli.source, device, async move |initial| {
            let config = brush_process::args_file::merge_configs(&initial, &cli.training);
            if let Err(error) = validate_logging_config(&config) {
                let _ = callback_error.set(error.to_string());
                return None;
            }
            Some(config)
        });
    let mut snapshot_every = 1000;
    let mut stats_every = 50;
    let mut final_step = 0;
    let mut splat_count = 0;
    let mut training_start = None;
    let mut brush_elapsed = Duration::ZERO;
    let training_result: anyhow::Result<()> = async {
        while let Some(event) = process.stream.next().await {
            match event? {
                ProcessMessage::DoneLoading => {
                    training_start = Some(Instant::now());
                    println!("Completed loading.");
                }
                ProcessMessage::SplatsUpdated {
                    up_axis,
                    num_splats,
                    ..
                } => {
                    splat_count = num_splats;
                    if let (Some(tx), Some(up)) = (&tx, up_axis) {
                        let _ = tx.send(logging::Observation::UpAxis(up));
                    }
                }
                ProcessMessage::TrainMessage(message) => match message {
                    TrainMessage::TrainConfig { config } => {
                        final_step = config.train_config.total_iters();
                        stats_every = config.rerun_config.rerun_log_train_stats_every;
                        snapshot_every = config.rerun_config.rerun_log_splats_every.unwrap_or(1000);
                        if let Some(tx) = &tx {
                            let _ = tx.send(logging::Observation::Config {
                                max_image_size: config.rerun_config.rerun_max_img_size,
                            });
                        }
                    }
                    TrainMessage::Dataset { dataset } => {
                        if let Some(tx) = &tx {
                            let _ = tx.send(logging::Observation::Dataset(dataset));
                        }
                    }
                    step @ TrainMessage::TrainStep {
                        iter,
                        total_elapsed,
                        ..
                    } => {
                        brush_elapsed = total_elapsed;
                        if iter.is_multiple_of(100) || iter == final_step {
                            println!(
                                "Train iter {iter}: brush_seconds {:.9}",
                                total_elapsed.as_secs_f64()
                            );
                        }
                        if let Some(tx) = &tx {
                            if gsplat_train::retains(
                                iter,
                                final_step,
                                cli.snapshot_first,
                                snapshot_every,
                            ) {
                                let splats = process
                                    .splat_view
                                    .get(0)
                                    .context("missing training splats")?;
                                let _ = tx.send(logging::Observation::Snapshot {
                                    iter,
                                    splats,
                                    full_sh: iter == final_step,
                                });
                            }
                            if iter.is_multiple_of(stats_every) || iter == final_step {
                                let _ = tx.send(logging::Observation::Metrics {
                                    message: step,
                                    num_splats: splat_count,
                                });
                            }
                        }
                    }
                    refine @ TrainMessage::RefineStep { .. } => {
                        if let Some(tx) = &tx {
                            let _ = tx.send(logging::Observation::Metrics {
                                message: refine,
                                num_splats: splat_count,
                            });
                        }
                    }
                    TrainMessage::EvalResult {
                        iter,
                        avg_psnr,
                        avg_ssim,
                    } => {
                        println!("Eval iter {iter}: PSNR {avg_psnr}, SSIM {avg_ssim}");
                        if let Some(tx) = &tx {
                            let splats =
                                process.splat_view.get(0).context("missing eval splats")?;
                            let _ = tx.send(logging::Observation::Eval {
                                iter,
                                psnr: avg_psnr,
                                ssim: avg_ssim,
                                splats,
                            });
                        }
                    }
                    _ => {}
                },
                ProcessMessage::Warning { error } => eprintln!("Brush warning: {error:#}"),
                ProcessMessage::StartLoading {
                    training: false, ..
                } => anyhow::bail!("training requires a dataset folder"),
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    drop(tx);
    if let Some(worker) = worker {
        match worker.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!("Recording warning: {error}"),
            Err(_) => {
                eprintln!("Recording warning: worker panicked; training and exports continued")
            }
        }
    }
    if let Some(error) = config_error.get() {
        anyhow::bail!("Invalid logging configuration: {error}");
    }
    training_result?;
    let wall = training_start
        .context("training never started")?
        .elapsed()
        .as_secs_f64();
    println!(
        "Training summary: steps={final_step} wall_seconds={wall:.9} brush_seconds={:.9} logging={enabled}",
        brush_elapsed.as_secs_f64()
    );
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
