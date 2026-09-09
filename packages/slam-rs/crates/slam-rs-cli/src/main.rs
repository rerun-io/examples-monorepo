//! Placeholder CLI for the slam-rs core.
//!
//! `version` is the only subcommand that does anything. Replay runs through the
//! Python tools — `tools/apps/replay.py`, which own the catalog feed, the decode
//! and the evaluation — so a Python-free replay would need all three ported and
//! nothing has asked for one.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

/// slam-rs command line.
#[derive(Debug, Parser)]
#[command(
    name = "slam-rs",
    version,
    about = "Placeholder CLI for the slam-rs core: only `version` does anything, and replay runs through the Python tools"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the version of the core.
    Version,
    /// Not implemented: a placeholder. Replay a sequence with the Python tool
    /// `tools/apps/replay.py --stage vio` instead.
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

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Version => {
            println!("slam-rs {}", slam_rs::VERSION);
            ExitCode::SUCCESS
        }
        // A placeholder, and it stays one: the feed, the decode and the
        // evaluation a replay needs are Python's. Fail loudly rather than
        // exiting zero, which would let a script read a missing or stale
        // trajectory as a finished replay.
        Command::Replay { frames, imu, out } => {
            eprintln!("error: this CLI is a placeholder and replay is not implemented;");
            eprintln!("  the replay tool is Python: tools/apps/replay.py --stage vio");
            eprintln!(
                "  nothing was read from {} or {}, and nothing was written to {}",
                frames.display(),
                imu.display(),
                out.display()
            );
            ExitCode::FAILURE
        }
    }
}
