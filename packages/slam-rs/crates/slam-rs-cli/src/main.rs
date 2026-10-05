//! Native runner for the slam-rs core.
//!
//! `replay` runs the VIO on a clip written by `slam_rs/apis/dump_clip.py`
//! (PGM framesets, `imu.csv`, `calib.json`, `clip.json`) without Python. Every frame is read into memory before the clock starts; the timed loop is the IMU pushes and
//! `Vio::track`. Accuracy is scored outside, against the catalog's ground truth (the trajectory is on the clip's clock).

mod clip;
mod replay;
use clip::load_clip;
use replay::replay;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};

/// slam-rs command line.
#[derive(Debug, Parser)]
#[command(
    name = "slam-rs",
    version,
    about = "Native runner for the slam-rs core"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Lane {
    Cpu,
    Gpu,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the version of the core.
    Version,
    /// Replay a local clip, with every frame already in memory before timing.
    Replay {
        /// The clip directory (`clip.json`, `calib.json`, `imu.csv`, `frame_NNN_camK.pgm`).
        #[arg(long)]
        clip: PathBuf,
        /// Resolved VIO config JSON; `-` reads stdin.
        #[arg(long)]
        config: PathBuf,
        /// The frontend backend.
        #[arg(long, value_enum, default_value = "cpu")]
        lane: Lane,
        /// Frontend worker threads (any count gives the same trajectory, D31).
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Stop after this many framesets.
        #[arg(long)]
        max_framesets: Option<usize>,
        /// Read the frames from this raw file instead of the PGMs: every frameset's cameras in order, each a tight
        /// width x height gray8 raster (`-` reads stdin, e.g. `xz -dc frames.u8.xz | slam-rs replay --frames - ...`).
        #[arg(long)]
        frames: Option<PathBuf>,
        /// Trajectory CSV: `t_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z` on the clip's frameset clock, tracked framesets only.
        #[arg(long)]
        out: PathBuf,
        /// Summary JSON: wall time, per-frameset track times, frontend stage means.
        #[arg(long)]
        summary: PathBuf,
    },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Version => {
            println!("slam-rs {}", slam_rs::VERSION);
            ExitCode::SUCCESS
        }
        Command::Replay {
            clip,
            config,
            lane,
            threads,
            max_framesets,
            frames,
            out,
            summary,
        } => {
            let started = Instant::now();
            let input = load_clip(&clip, max_framesets, frames.as_deref());
            match input.and_then(|input| {
                replay(
                    input,
                    &config,
                    lane,
                    threads,
                    started.elapsed().as_secs_f64(),
                    &out,
                    &summary,
                )
            }) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
