//! Replay a `robocap-live-dump/1` clip through the logger alone (independent of the runtime):
//! the dump's frames, area-downsampled to 640x360, with poses and hands from a `--record` JSONL (the Python reference
//! pipeline's, or the runtime's), paced at the clip's rate, streamed to a viewer and/or saved.
//!
//! ```text
//! log_replay --dump <dir> [--record <results.jsonl>] [--viewer rerun+http://<host>:9876/proxy] [--save <file.rrd>]
//!            [--video h264|raw|off] [--encoder mpp|openh264|x264] [--bps 1000000] [--display <asset.rrd>]
//!            [--frames N] [--loops K] [--fps 30] [--no-log] [--write-small <dir>] [--video-cameras 0,1,2,3,4,5]
//!            [--flush-ms 30]
//! ```
//!
//! `--write-small <dir>` writes the downsampled clip as a 640x360 dump (same format, 9x smaller, for copying to a cap) and
//! exits; a dump whose `meta.json` says 640x360 is replayed without downsampling.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use kornia_image::Image;
use kornia_staging_imgproc::resize::resize_area_u8;
use robocap_live::frame::isometry_from_matrix;
use robocap_live::frame::{
    CameraFrame, DumpMeta, FULL_SIZE, FrameMeta, FrameReader, Frameset, Luma, NUM_CAMERAS, Rig,
    SMALL_SIZE, write_small_dump,
};
use robocap_live::hands::{HandFrameResult, HandOutput};
use robocap_live::log::video::{EncoderConfig, EncoderKind};
use robocap_live::log::{FrameLog, Logger, LoggerConfig, VideoMode};
use robocap_live::sched::FrameTimings;
use serde::Deserialize;

type Error = Box<dyn std::error::Error>;

#[derive(Deserialize)]
struct RecordHand {
    #[serde(default)]
    tracked: bool,
    reported: bool,
    landmarks: Option<Vec<[f64; 3]>>,
}

/// One line of the runtime's `--record` JSONL (`sched::RecordWriter`), as far as the replay reads it: SLAM's pose and state,
/// the hands' landmarks, the hand scale and the timings.
#[derive(Deserialize)]
struct RecordLine {
    index: u64,
    world_from_rig: Option<Vec<f64>>,
    #[serde(default)]
    slam_ok: bool,
    #[serde(default)]
    hands: Vec<RecordHand>,
    /// `null` without hands.
    scale: Option<f64>,
    #[serde(default)]
    timings_ms: BTreeMap<String, Option<f64>>,
}

struct Args {
    dump: PathBuf,
    record: Option<PathBuf>,
    viewer: Option<String>,
    save: Option<PathBuf>,
    video: VideoMode,
    encoder: EncoderKind,
    bps: u32,
    display: Option<PathBuf>,
    frames: usize,
    loops: usize,
    fps: f64,
    log: bool,
    write_small: Option<PathBuf>,
    video_cameras: Vec<usize>,
    flush_ms: u64,
}

fn parse_args() -> Result<Args, Error> {
    let mut args = Args {
        dump: PathBuf::new(),
        record: None,
        viewer: None,
        save: None,
        video: VideoMode::H264,
        encoder: EncoderKind::Mpp,
        bps: 1_000_000,
        display: None,
        frames: usize::MAX,
        loops: 1,
        fps: 30.0,
        log: true,
        write_small: None,
        video_cameras: (0..NUM_CAMERAS).collect(),
        flush_ms: 30,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--dump" => args.dump = value()?.into(),
            "--record" => args.record = Some(value()?.into()),
            "--viewer" => args.viewer = Some(value()?),
            "--save" => args.save = Some(value()?.into()),
            "--video" => args.video = value()?.parse()?,
            "--encoder" => args.encoder = value()?.parse()?,
            "--bps" => args.bps = value()?.parse()?,
            "--display" => args.display = Some(value()?.into()),
            "--frames" => args.frames = value()?.parse()?,
            "--loops" => args.loops = value()?.parse()?,
            "--fps" => args.fps = value()?.parse()?,
            "--no-log" => args.log = false,
            "--write-small" => args.write_small = Some(value()?.into()),
            "--video-cameras" => {
                args.video_cameras = value()?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?
            }
            "--flush-ms" => args.flush_ms = value()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if args.dump.as_os_str().is_empty() {
        return Err("--dump <dir> is required".into());
    }
    Ok(args)
}

/// The 640x360 small image of a full frame (production's area /3).
fn area3(full: &Image<u8, 1>) -> Result<Luma, Error> {
    let mut small = Image::from_size_val(SMALL_SIZE, 0u8)?;
    resize_area_u8(full, &mut small)?;
    Ok(Arc::new(small))
}

fn hand_result(line: &RecordLine) -> HandFrameResult {
    let mut result = HandFrameResult {
        scale: line.scale.unwrap_or_default(),
        scale_final: true,
        ..HandFrameResult::default()
    };
    for (slot, hand) in result.hands.iter_mut().zip(&line.hands) {
        *slot = HandOutput {
            tracked: hand.tracked,
            reported: hand.reported,
            landmarks_world: hand
                .landmarks
                .as_ref()
                .and_then(|l| <[[f64; 3]; 21]>::try_from(l.as_slice()).ok()),
            ..HandOutput::default()
        };
    }
    result
}

struct Frame {
    index: u64,
    t_ns: i64,
    cam_t_ns: [i64; NUM_CAMERAS],
    small: [Option<Luma>; NUM_CAMERAS],
}

fn main() -> Result<(), Error> {
    let args = parse_args()?;
    let rig = Rig::load(&args.dump.join("rig.json"))?;
    let records: BTreeMap<u64, RecordLine> = match &args.record {
        Some(path) => {
            let mut map = BTreeMap::new();
            for line in BufReader::new(std::fs::File::open(path)?).lines() {
                let line: RecordLine = serde_json::from_str(&line?)?;
                map.insert(line.index, line);
            }
            map
        }
        None => BTreeMap::new(),
    };
    let meta = DumpMeta::load(&args.dump)?;
    let size = kornia_image::ImageSize {
        width: meta.width as usize,
        height: meta.height as usize,
    };
    let already_small = size == SMALL_SIZE;
    if !already_small && size != FULL_SIZE {
        return Err(
            format!("dump frames are {size:?}; expected {FULL_SIZE:?} or {SMALL_SIZE:?}").into(),
        );
    }
    let load_start = Instant::now();
    let mut reader = FrameReader::open(&args.dump.join("frames.bin"), size)?;
    let mut frames = Vec::new();
    while frames.len() < args.frames {
        let Some(frameset) = reader.next_frameset()? else {
            break;
        };
        let mut small: [Option<Luma>; NUM_CAMERAS] = Default::default();
        for (slot, camera) in small.iter_mut().zip(frameset.cameras.iter()) {
            if let Some(camera) = camera {
                *slot = Some(if already_small {
                    camera.full.clone()
                } else {
                    area3(&camera.full)?
                });
            }
        }
        let cam_t_ns =
            std::array::from_fn(|c| frameset.cameras[c].as_ref().map_or(0, |f| f.meta.pts_ns));
        frames.push(Frame {
            index: frameset.index,
            t_ns: frameset.t_ns,
            cam_t_ns,
            small,
        });
    }
    eprintln!(
        "loaded {} framesets (downsampled to {}x{}) in {:.1} s",
        frames.len(),
        SMALL_SIZE.width,
        SMALL_SIZE.height,
        load_start.elapsed().as_secs_f64()
    );
    if frames.is_empty() {
        return Err("the dump has no framesets".into());
    }
    if let Some(out) = &args.write_small {
        let framesets = frames.iter().map(|frame| Frameset {
            index: frame.index,
            t_ns: frame.t_ns,
            cameras: std::array::from_fn(|c| {
                frame.small[c].as_ref().map(|full| CameraFrame {
                    meta: FrameMeta {
                        seq: frame.index,
                        pts_ns: frame.cam_t_ns[c],
                        source_id: c as u32,
                        turned_180: false,
                    },
                    full: full.clone(),
                })
            }),
        });
        let written = write_small_dump(
            &args.dump,
            out,
            framesets,
            "area /3 to 640x360 by log_replay --write-small",
        )?;
        println!("wrote {written} framesets to {}", out.display());
        return Ok(());
    }

    let options = LoggerConfig {
        viewer: args.viewer.clone(),
        save: args.save.clone(),
        video: args.video,
        encoder: EncoderConfig::for_kind(args.encoder, SMALL_SIZE, 30, args.bps, 30),
        display: args.display.clone(),
        video_cameras: args.video_cameras.clone(),
        preview_flush: Duration::from_millis(args.flush_ms),
        ..LoggerConfig::default()
    };
    let mut logger = if args.log {
        Some(Logger::new(&rig, options)?)
    } else {
        None
    };
    if let Some(logger) = &logger {
        eprintln!("recording id {}", logger.recording_id());
    }

    let period = Duration::from_secs_f64(1.0 / args.fps);
    let clip_ns = frames.last().map_or(0, |f| f.t_ns) - frames[0].t_ns + (1e9 / args.fps) as i64;
    let start = Instant::now();
    let (mut calls, mut call_total, mut call_max, mut late) =
        (0u64, Duration::ZERO, Duration::ZERO, 0u64);
    let mut last_report = Instant::now();
    for lap in 0..args.loops {
        for (n, frame) in frames.iter().enumerate() {
            let due = start + period * (lap * frames.len() + n) as u32;
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            } else if now - due > period {
                late += 1;
            }
            let record = records.get(&frame.index);
            let pose = record
                .and_then(|r| r.world_from_rig.as_deref())
                .and_then(|m| <&[f64; 16]>::try_from(m).ok())
                .and_then(isometry_from_matrix);
            let hands = record.map(hand_result);
            let get = |name: &str| {
                record
                    .and_then(|r| r.timings_ms.get(name).copied().flatten())
                    .unwrap_or(f64::NAN)
            };
            let timings = FrameTimings {
                slam_ms: get("slam_ms"),
                pose_wait_ms: get("pose_wait_ms"),
                hands_ms: get("hands_ms"),
                detnet_ms: get("detnet_ms"),
                crops_ms: get("crops_ms"),
                keynet_ms: get("keynet_ms"),
                fit_ms: get("fit_ms"),
                tracker_ms: get("tracker_ms"),
                pipeline_ms: get("pipeline_ms"),
                ..FrameTimings::downsampled(get("downsample_ms"))
            };
            let status = match record {
                Some(r) if r.slam_ok => "tracking (reference)",
                Some(_) => "lost (reference)",
                None => "no record",
            };
            let Some(logger) = logger.as_mut() else {
                continue;
            };
            let t_ns = frame.t_ns + lap as i64 * clip_ns;
            let call = Instant::now();
            logger.log_frameset(&FrameLog {
                t_ns,
                small: std::array::from_fn(|c| frame.small[c].as_ref()),
                world_from_rig: pose.as_ref(),
                slam_status: status,
                hands: hands.as_ref(),
                timings: &timings,
            })?;
            let took = call.elapsed();
            calls += 1;
            call_total += took;
            call_max = call_max.max(took);
            if last_report.elapsed() > Duration::from_secs(5) {
                last_report = Instant::now();
                eprintln!(
                    "{:.0} s: {:?}",
                    start.elapsed().as_secs_f64(),
                    logger.stats()
                );
            }
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let cpu = robocap_live::log::video::process_cpu_seconds(std::process::id());
    println!(
        "replayed {} framesets in {wall:.1} s ({:.1} fps), late {late}; log_frameset mean {:.3} ms max {:.3} ms; process CPU {:.1} s",
        frames.len() * args.loops,
        (frames.len() * args.loops) as f64 / wall,
        call_total.as_secs_f64() * 1e3 / calls.max(1) as f64,
        call_max.as_secs_f64() * 1e3,
        cpu.unwrap_or(0.0),
    );
    if let Some(logger) = logger.take() {
        let (stats, encoders) = logger.finish()?;
        println!("{stats:?}");
        let bits = stats.preview_payload_bytes as f64 * 8.0 / stats.elapsed_s.max(1e-9) / 1e6;
        println!(
            "preview payload {bits:.2} Mbit/s; video {:.2} Mbit/s over {} samples",
            stats.video_bytes as f64 * 8.0 / wall / 1e6,
            stats.video_samples
        );
        for (camera, e) in encoders.iter().enumerate() {
            println!(
                "encoder {camera}: {} in, {} out, {:.2} Mbit/s, CPU {:.1}% of a core",
                e.frames_in,
                e.samples_out,
                e.bytes_out as f64 * 8.0 / e.wall_seconds.max(1e-9) / 1e6,
                robocap_live::log::encoder_cpu_percent(e)
            );
        }
    }
    Ok(())
}
