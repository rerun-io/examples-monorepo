//! Native runner for the slam-rs core: the estimator without any Python.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// slam-rs command line.
#[derive(Debug, Parser)]
#[command(
    name = "slam-rs",
    version,
    about = "Run the slam-rs VIO core without Python"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the version of the core.
    Version,
    /// Replay a recorded sequence and write the estimated trajectory.
    Replay {
        /// Directory of grayscale frames, one file per frameset.
        #[arg(long)]
        frames: PathBuf,
        /// CSV of IMU samples: t_ns, gx, gy, gz, ax, ay, az.
        #[arg(long)]
        imu: PathBuf,
        /// CSV to write the trajectory to.
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() {
    match Cli::parse().command {
        Command::Version => println!("slam-rs {}", slam_rs::VERSION),
        Command::Replay { frames, imu, out } => {
            println!("replay is not implemented yet; it would:");
            println!("  read framesets from {}", frames.display());
            println!("  read imu samples from {}", imu.display());
            println!("  write the trajectory to {}", out.display());
        }
    }
}
