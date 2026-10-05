//! Native runner for the slam-rs core.
//!
//! `replay` runs the VIO on a catalog segment or a clip written by `slam_rs/apis/dump_clip.py`
//! (PGM framesets, `imu.csv`, `calib.json`, `clip.json`) without Python. Every frame is read into memory before the clock starts; the timed loop is the IMU pushes and
//! `Vio::track`. Accuracy is scored outside, against the catalog's ground truth (the trajectory is on the clip's clock).

mod clip;
mod replay;
use clip::load_clip;
#[cfg(all(test, feature = "catalog"))]
use clip::read_imu;
#[cfg(feature = "catalog")]
use clip::{Clip, ReplayInput};
use replay::replay;
#[cfg(feature = "catalog")]
use slam_rs::catalog_timing::ImuRow;

#[cfg(feature = "catalog")]
mod catalog;
#[cfg(feature = "catalog")]
mod decode;

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
    /// Replay a catalog segment or local clip, with every frame already in memory before timing.
    Replay {
        /// The clip directory (`clip.json`, `calib.json`, `imu.csv`, `frame_NNN_camK.pgm`).
        #[arg(long, required_unless_present = "catalog", conflicts_with = "catalog")]
        clip: Option<PathBuf>,
        /// Read a RoboCap or MSD G2 segment from this Rerun catalog (requires the catalog build feature).
        #[arg(long, requires = "segment")]
        catalog: Option<String>,
        /// Catalog segment ID, including its robocap__ or msd-g2__ prefix.
        #[arg(long, requires = "catalog")]
        segment: Option<String>,
        /// Resolved VIO config JSON; `-` reads stdin (catalog mode can stream config without copying a file).
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
        #[arg(long, requires = "clip", conflicts_with = "catalog")]
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
            catalog,
            segment,
            config,
            lane,
            threads,
            max_framesets,
            frames,
            out,
            summary,
        } => {
            let started = Instant::now();
            let input = match (clip.as_deref(), catalog.as_deref(), segment.as_deref()) {
                (Some(path), None, None) => load_clip(path, max_framesets, frames.as_deref()),
                (None, Some(url), Some(segment)) => {
                    #[cfg(feature = "catalog")]
                    {
                        catalog::load(url, segment, max_framesets)
                            .map_err(|error| format!("catalog: {error:#}"))
                    }
                    #[cfg(not(feature = "catalog"))]
                    {
                        let _ = (url, segment);
                        Err("rebuild slam-rs-cli with --features catalog".into())
                    }
                }
                _ => Err("choose either --clip or --catalog with --segment".into()),
            };
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

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn catalog_replay_accepts_a_segment_without_fixture_files() {
        let parsed = Cli::try_parse_from([
            "slam-rs",
            "replay",
            "--catalog",
            "rerun+http://localhost:51235",
            "--segment",
            "msd-g2__MGO_others__MGO07_mapping_easy",
            "--max-framesets",
            "2",
            "--config",
            "config.json",
            "--out",
            "poses.csv",
            "--summary",
            "summary.json",
        ]);
        assert!(parsed.is_ok(), "{parsed:?}");
    }

    #[test]
    fn replay_rejects_missing_or_conflicting_input_sources() {
        for source in [
            vec![],
            vec!["--catalog", "rerun+http://localhost:51235"],
            vec!["--segment", "msd-g2__clip"],
            vec![
                "--clip",
                "fixture",
                "--catalog",
                "rerun+http://localhost:51235",
                "--segment",
                "msd-g2__clip",
            ],
            vec![
                "--catalog",
                "rerun+http://localhost:51235",
                "--segment",
                "msd-g2__clip",
                "--frames",
                "-",
            ],
        ] {
            let mut args = vec![
                "slam-rs",
                "replay",
                "--config",
                "config.json",
                "--out",
                "poses.csv",
                "--summary",
                "summary.json",
            ];
            args.extend(source);
            assert!(Cli::try_parse_from(&args).is_err(), "accepted {args:?}");
        }
    }
}

#[cfg(all(test, feature = "catalog"))]
mod catalog_tests;
