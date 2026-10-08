//! Recording cadence and bounded delivery of GPU tensors to the logging worker.
use crate::{
    Cli,
    logging::{self, Observation},
};
use anyhow::Context;
use brush_process::{
    ProcessDevice,
    message::{ProcessMessage, TrainMessage},
    slot::Slot,
};
use brush_render::gaussian_splats::Splats;
use std::{
    sync::mpsc::{SyncSender, TrySendError},
    thread::JoinHandle,
    time::Duration,
};

pub struct Recorder {
    sender: SyncSender<Observation>,
    worker: JoinHandle<anyhow::Result<()>>,
    final_step: u32,
    splat_count: u32,
    snapshot_first: u32,
    snapshot_every: u32,
    stats_every: u32,
    dropped_steps: u64,
}
impl Recorder {
    pub fn new(cli: &Cli, device: &ProcessDevice) -> anyhow::Result<Option<Self>> {
        if !cli.spawn && cli.save.is_none() && cli.connect.is_none() {
            return Ok(None);
        }
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
        // At most eight pending observations retain tensor handles. Snapshots/evaluations
        // apply backpressure; intermediate step metrics may be dropped when the worker lags.
        let (sender, rx) = std::sync::mpsc::sync_channel(8);
        let device = device.clone();
        let (compute, video) = (cli.compute_visualizer, cli.video);
        let worker = std::thread::Builder::new()
            .name("gsplat-rerun".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                runtime.block_on(logging::run(rx, rec, device, compute, video));
                Ok(())
            })?;
        Ok(Some(Self {
            sender,
            worker,
            final_step: 0,
            splat_count: 0,
            snapshot_first: cli.snapshot_first,
            snapshot_every: 1000,
            stats_every: 50,
            dropped_steps: 0,
        }))
    }
    pub fn finish(self) {
        drop(self.sender);
        if self.dropped_steps > 0 {
            eprintln!(
                "Recording: dropped {} intermediate step observations while the worker was busy",
                self.dropped_steps
            );
        }
        match self.worker.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!("Recording warning: {error:#}"),
            Err(_) => {
                eprintln!("Recording warning: worker panicked; training and exports continued")
            }
        }
    }
    pub fn observe(&mut self, event: ProcessMessage, splats: &Slot<Splats>) -> anyhow::Result<()> {
        let observation = match event {
            ProcessMessage::SplatsUpdated {
                up_axis,
                num_splats,
                ..
            } => {
                self.splat_count = num_splats;
                let Some(up) = up_axis else {
                    return Ok(());
                };
                Observation::UpAxis(up)
            }
            ProcessMessage::TrainMessage(TrainMessage::TrainConfig { config }) => {
                self.final_step = config.train_config.total_iters();
                self.stats_every = config.rerun_config.rerun_log_train_stats_every;
                self.snapshot_every = config.rerun_config.rerun_log_splats_every.unwrap_or(1000);
                Observation::Config {
                    max_image_size: config.rerun_config.rerun_max_img_size,
                }
            }
            ProcessMessage::TrainMessage(TrainMessage::Dataset { dataset }) => {
                Observation::Dataset(dataset)
            }
            ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                iter,
                stats,
                step_duration,
                ..
            }) => {
                if gsplat_train::retains(
                    iter,
                    self.final_step,
                    self.snapshot_first,
                    self.snapshot_every,
                ) {
                    let _ = self.sender.send(Observation::Snapshot {
                        iter,
                        splats: splats.get(0).context("missing training splats")?,
                        full_sh: iter == self.final_step,
                    });
                }
                if !iter.is_multiple_of(self.stats_every) && iter != self.final_step {
                    return Ok(());
                }
                let step = Observation::Step {
                    iter,
                    stats,
                    step_duration,
                    num_splats: self.splat_count,
                };
                if iter == self.final_step {
                    step
                } else {
                    if matches!(self.sender.try_send(step), Err(TrySendError::Full(_))) {
                        self.dropped_steps += 1;
                    }
                    return Ok(());
                }
            }
            ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                iter,
                refine,
                refine_duration,
                ..
            }) => Observation::Refine {
                iter,
                refine,
                refine_duration,
            },
            ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                iter,
                avg_psnr,
                avg_ssim,
            }) => Observation::Eval {
                iter,
                psnr: avg_psnr,
                ssim: avg_ssim,
                splats: splats.get(0).context("missing eval splats")?,
            },
            _ => return Ok(()),
        };
        let _ = self.sender.send(observation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "integration: requires a GPU"]
    async fn saturated_recording_drops_intermediate_metrics_without_blocking_training() {
        let device = brush_process::burn_init_setup().await;
        let loss = burn::tensor::Tensor::<1>::zeros([1], &device);
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        let mut active = Recorder {
            sender,
            worker: std::thread::spawn(|| Ok(())),
            final_step: 100,
            splat_count: 1,
            snapshot_first: 50,
            snapshot_every: 1000,
            stats_every: 5,
            dropped_steps: 0,
        };
        for iter in [5, 10, 15] {
            active
                .observe(
                    ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                        iter,
                        total_elapsed: Duration::ZERO,
                        stats: brush_train::msg::TrainStepStats {
                            num_visible: 1,
                            lr_mean: 0.0,
                            lr_rotation: 0.0,
                            lr_scale: 0.0,
                            lr_coeffs: 0.0,
                            lr_opac: 0.0,
                            loss: loss.clone(),
                        },
                        step_duration: Duration::ZERO,
                        lod_progress: None,
                    }),
                    &Slot::empty(),
                )
                .unwrap();
        }
        assert_eq!(active.dropped_steps, 1);
        let observed: Vec<_> = receiver
            .try_iter()
            .map(|observation| {
                let Observation::Step { iter, .. } = observation else {
                    panic!("unexpected non-step observation");
                };
                iter
            })
            .collect();
        assert_eq!(observed, [5, 10]);
        active.worker.join().unwrap().unwrap();
    }
}
