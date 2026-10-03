//! Native runner for the slam-rs core.
//!
//! `replay` runs the VIO on a clip written by `slam_rs/apis/dump_clip.py` (PGM framesets, `imu.csv`, `calib.json`,
//! `clip.json`) with no Python and no decoder, so a device without the Python stack (the RoboCap) can be timed on the
//! same inputs as the host. Every frame is read into memory before the clock starts; the timed loop is the IMU pushes and
//! `Vio::track`. Accuracy is scored outside, against the catalog's ground truth (the trajectory is on the clip's clock).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::{Backend, ImageView, Vio, VioResult, VioStatus};

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
    /// Replay a clip dumped by `dump_clip.py`, timing `Vio::track` with every frame already in memory.
    Replay {
        /// The clip directory (`clip.json`, `calib.json`, `imu.csv`, `frame_NNN_camK.pgm`).
        #[arg(long)]
        clip: PathBuf,
        /// The resolved VIO config JSON (the dataset config with the profile applied, as `SlamConfig.vio_config_text` writes it).
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

/// The fields of `clip.json` the replay reads.
#[derive(Debug, serde::Deserialize)]
struct Clip {
    segment_id: String,
    num_cameras: usize,
    framesets: usize,
    frame_t_ns: Vec<i64>,
    resolution_wh: Vec<(usize, usize)>,
}

/// One gray8 raster of a PGM.
struct Raster {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

fn read_pgm(path: &Path) -> Result<Raster, String> {
    let bytes: Vec<u8> =
        std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    // "P5\n<w> <h>\n255\n" then the raster: dump_clip.py writes exactly that.
    let mut fields: Vec<usize> = Vec::with_capacity(3);
    let mut cursor: usize = 2;
    while fields.len() < 3 {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let start: usize = cursor;
        while bytes
            .get(cursor)
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            cursor += 1;
        }
        let text = std::str::from_utf8(&bytes[start..cursor])
            .map_err(|_| format!("{}: bad PGM header", path.display()))?;
        fields.push(
            text.parse()
                .map_err(|_| format!("{}: bad PGM header", path.display()))?,
        );
    }
    let (width, height) = (fields[0], fields[1]);
    let pixels: Vec<u8> = bytes[cursor + 1..].to_vec();
    if pixels.len() != width * height {
        return Err(format!(
            "{}: {} bytes of raster for {width}x{height}",
            path.display(),
            pixels.len()
        ));
    }
    Ok(Raster {
        width,
        height,
        pixels,
    })
}

/// One sample of `imu.csv`.
struct ImuRow {
    t_ns: i64,
    gyro: [f64; 3],
    accel: [f64; 3],
}

fn read_imu(path: &Path) -> Result<Vec<ImuRow>, String> {
    let text: String =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    text.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let value = |index: usize| -> Result<f64, String> {
                fields
                    .get(index)
                    .and_then(|field| field.parse().ok())
                    .ok_or_else(|| format!("imu.csv: bad line {line:?}"))
            };
            Ok(ImuRow {
                t_ns: fields[0]
                    .parse()
                    .map_err(|_| format!("imu.csv: bad line {line:?}"))?,
                gyro: [value(1)?, value(2)?, value(3)?],
                accel: [value(4)?, value(5)?, value(6)?],
            })
        })
        .collect()
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank: usize = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[rank]
}

#[allow(clippy::too_many_arguments)]
fn replay(
    clip_dir: &Path,
    config_path: &Path,
    lane: Lane,
    threads: usize,
    max_framesets: Option<usize>,
    frames: Option<&Path>,
    out: &Path,
    summary: &Path,
) -> Result<(), String> {
    let clip: Clip = serde_json::from_str(
        &std::fs::read_to_string(clip_dir.join("clip.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("clip.json: {error}"))?;
    let calibration: Calibration<f64> = Calibration::from_json_str(
        &std::fs::read_to_string(clip_dir.join("calib.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("calib.json: {error:?}"))?;
    let config_text: String = std::fs::read_to_string(config_path)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let config: VioConfig = VioConfig::from_json_str(&config_text)
        .map_err(|error| format!("{}: {error:?}", config_path.display()))?;
    let imu: Vec<ImuRow> = read_imu(&clip_dir.join("imu.csv"))?;
    let framesets: usize = max_framesets.map_or(clip.framesets, |limit| limit.min(clip.framesets));

    // Every frame in one buffer, frameset-major then camera, before the clock starts.
    let sizes: Vec<usize> = clip
        .resolution_wh
        .iter()
        .map(|&(width, height)| width * height)
        .collect();
    if sizes.len() != clip.num_cameras {
        return Err(format!(
            "clip.json: {} resolutions for {} cameras",
            sizes.len(),
            clip.num_cameras
        ));
    }
    let frameset_bytes: usize = sizes.iter().sum();
    let load_started: Instant = Instant::now();
    let mut pixels: Vec<u8> = vec![0; framesets * frameset_bytes];
    match frames {
        Some(path) => {
            use std::io::Read as _;
            let mut reader: Box<dyn std::io::Read> = if path == Path::new("-") {
                Box::new(std::io::stdin().lock())
            } else {
                Box::new(
                    std::fs::File::open(path)
                        .map_err(|error| format!("{}: {error}", path.display()))?,
                )
            };
            reader
                .read_exact(&mut pixels)
                .map_err(|error| format!("{}: {error}", path.display()))?;
        }
        None => {
            for frame in 0..framesets {
                let mut offset: usize = frame * frameset_bytes;
                for (camera, &size) in sizes.iter().enumerate() {
                    let raster: Raster =
                        read_pgm(&clip_dir.join(format!("frame_{frame:03}_cam{camera}.pgm")))?;
                    if (raster.width, raster.height) != clip.resolution_wh[camera] {
                        return Err(format!(
                            "frame {frame} cam {camera}: {}x{} is not the clip's resolution",
                            raster.width, raster.height
                        ));
                    }
                    pixels[offset..offset + size].copy_from_slice(&raster.pixels);
                    offset += size;
                }
            }
        }
    }
    let load_s: f64 = load_started.elapsed().as_secs_f64();

    let backend: Backend = match lane {
        Lane::Cpu => Backend::Cpu,
        Lane::Gpu => Backend::Gpu,
    };
    let options = FrontendOptions {
        threads,
        ..FrontendOptions::default()
    };
    let setup_started: Instant = Instant::now();
    let mut vio: Vio<f32> = Vio::with_backend(config, calibration, options, backend)
        .map_err(|error| format!("Vio: {error}"))?;
    let setup_s: f64 = setup_started.elapsed().as_secs_f64();

    let mut poses: String = String::from("#t_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n");
    let mut track_ms: Vec<f64> = Vec::with_capacity(framesets);
    let mut stage_ns: [u64; 5] = [0; 5];
    let mut tracked: usize = 0;
    let mut cursor: usize = 0;
    let started: Instant = Instant::now();
    for frame in 0..framesets {
        let t_ns: i64 = clip.frame_t_ns[frame];
        // Push the IMU up to and including the first sample after the frameset, as the catalog feed and full_clip.rs do.
        while cursor < imu.len() {
            let row: &ImuRow = &imu[cursor];
            vio.push_imu(row.t_ns, row.gyro, row.accel)
                .map_err(|error| format!("imu: {error}"))?;
            cursor += 1;
            if row.t_ns > t_ns {
                break;
            }
        }
        let mut offset: usize = frame * frameset_bytes;
        let views: Vec<ImageView<'_>> = clip
            .resolution_wh
            .iter()
            .map(|&(width, height)| {
                let view = ImageView {
                    width,
                    height,
                    stride: width,
                    data: &pixels[offset..offset + width * height],
                };
                offset += width * height;
                view
            })
            .collect();
        let call: Instant = Instant::now();
        let result: VioResult = vio
            .track(t_ns, &views)
            .map_err(|error| format!("frameset {frame}: {error}"))?;
        track_ms.push(call.elapsed().as_secs_f64() * 1e3);
        let timings = vio.frontend_timings();
        for (sum, value) in stage_ns.iter_mut().zip([
            timings.pyramid_ns,
            timings.detect_ns,
            timings.track_ns,
            timings.stereo_ns,
            timings.imu_ns,
        ]) {
            *sum += value;
        }
        if result.status == VioStatus::Tracking {
            tracked += 1;
            let pose: [f64; 7] = result.world_from_rig;
            writeln!(
                poses,
                "{t_ns},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
                pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]
            )
            .map_err(|error| error.to_string())?;
        }
    }
    let wall_s: f64 = started.elapsed().as_secs_f64();
    std::fs::write(out, poses).map_err(|error| format!("{}: {error}", out.display()))?;

    let mean_ms: f64 = track_ms.iter().sum::<f64>() / track_ms.len().max(1) as f64;
    let mut sorted: Vec<f64> = track_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let per_frame = |ns: u64| ns as f64 / 1e6 / framesets.max(1) as f64;
    let report = serde_json::json!({
        "segment_id": clip.segment_id,
        "lane": format!("{lane:?}").to_lowercase(),
        "threads": threads,
        "framesets": framesets,
        "tracked": tracked,
        "load_s": load_s,
        "setup_s": setup_s,
        "wall_s": wall_s,
        "wall_ms_per_frameset": wall_s * 1e3 / framesets.max(1) as f64,
        "track_ms": {"mean": mean_ms, "p50": percentile(&sorted, 0.5), "p95": percentile(&sorted, 0.95),
                     "p99": percentile(&sorted, 0.99), "max": sorted.last().copied().unwrap_or(f64::NAN)},
        "frontend_ms_mean": {"pyramid": per_frame(stage_ns[0]), "detect": per_frame(stage_ns[1]), "track": per_frame(stage_ns[2]),
                             "stereo": per_frame(stage_ns[3]), "imu": per_frame(stage_ns[4])},
        "track_ms_per_frame": track_ms,
    });
    std::fs::write(
        summary,
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("{}: {error}", summary.display()))?;
    println!(
        "{}: {tracked}/{framesets} tracked, wall {wall_s:.2} s = {:.2} ms/frameset (track mean {mean_ms:.2}, p50 {:.2}, p95 {:.2})",
        clip.segment_id,
        wall_s * 1e3 / framesets.max(1) as f64,
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95)
    );
    Ok(())
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
            match replay(
                &clip,
                &config,
                lane,
                threads,
                max_framesets,
                frames.as_deref(),
                &out,
                &summary,
            ) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
