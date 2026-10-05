//! SLAM-only lossless bench: the four SLAM cameras of a dump (1920x1080, area /3 here, or an already-small 640x360 dump) and
//! its IMU through [`SlamEstimator`] with the factory calibration of the dump's device (`rig.json`), one frameset after the
//! other, as the runtime's lossless replay feeds it. Prints one JSON summary: tracked framesets, ATE against the dump's
//! `reference_world_from_rig.bin` (kornia-algebra's rigid Umeyama on the positions of the tracked framesets, each matched to
//! the reference pose within 2 ms as `--slam reference` looks it up), and the compute time per frameset with its stage
//! breakdown.
//!
//! ```text
//! slam_bench --dump <dir> [--start S] [--frames N] [--threads 2] [--cpus 6-7] [--uclamp 1024] [--set KEY=VALUE ...]
//!            [--profile fast|live|live30] [--out poses.jsonl] [--write-small <dir>] [--repeat K]
//! ```
//!
//! `--write-small <dir>` writes the four SLAM cameras at 640x360 (with the dump's side files) and exits. `--set` takes the
//! runtime's `--slam-set` keys, `--cpus` its CPU lists. `--profile` picks the VIO profile the `--set` keys apply on top of
//! (default `live`, the runtime's).

use slam_rs::replay::{feed_imu_through, percentile};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use kornia_algebra::Vec3F64;
use kornia_algebra::linalg::rigid::umeyama_f64;
use kornia_image::Image;
use robocap_live::capture::Cap;
use robocap_live::downsample::resize_area_u8;
use robocap_live::frame::{
    CameraFrame, DumpMeta, FULL_SIZE, FrameMeta, FrameReader, Frameset, ImuSample, Luma, NUM_CAMERAS, Rig, SLAM_CAMERAS, SMALL_SIZE,
    read_imu, write_small_dump,
};
use robocap_live::sched::{parse_cpu_list, pin_current_thread, set_uclamp_min};
use robocap_live::frame::matrix_from_isometry;
use robocap_live::slam::{ReferencePoses, SlamEstimator, SlamPose, SlamProfile, parse_override};
use robocap_live::source::replay::read_reference_poses;

type Error = Box<dyn std::error::Error + Send + Sync>;

struct Args {
    dump: PathBuf,
    start: usize,
    frames: usize,
    threads: usize,
    cpus: Option<Vec<usize>>,
    uclamp: Option<u32>,
    overrides: Vec<(String, serde_json::Value)>,
    out: Option<PathBuf>,
    write_small: Option<PathBuf>,
    repeat: usize,
    profile: SlamProfile,
}

fn parse_args() -> Result<Args, Error> {
    let mut args = Args {
        dump: PathBuf::new(),
        start: 0,
        frames: usize::MAX,
        threads: 2,
        cpus: None,
        uclamp: None,
        overrides: Vec::new(),
        out: None,
        write_small: None,
        repeat: 1,
        profile: SlamProfile::Live,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--dump" => args.dump = value()?.into(),
            "--start" => args.start = value()?.parse()?,
            "--frames" => args.frames = value()?.parse()?,
            "--threads" => args.threads = value()?.parse()?,
            "--cpus" => args.cpus = parse_cpu_list(&value()?)?,
            "--uclamp" => args.uclamp = Some(value()?.parse()?),
            "--set" => args.overrides.push(parse_override(&value()?)?),
            "--out" => args.out = Some(value()?.into()),
            "--write-small" => args.write_small = Some(value()?.into()),
            "--repeat" => args.repeat = value()?.parse()?,
            "--profile" => args.profile = value()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if args.dump.as_os_str().is_empty() {
        return Err("--dump <dir> is required".into());
    }
    Ok(args)
}

/// One frameset's SLAM input.
struct Frame {
    index: u64,
    t_ns: i64,
    cam_t_ns: [i64; 4],
    images: [Luma; 4],
}

fn load(args: &Args) -> Result<Vec<Frame>, Error> {
    let size = DumpMeta::load(&args.dump)?.size();
    if size != FULL_SIZE && size != SMALL_SIZE {
        return Err(format!("dump frames are {size:?}; expected {FULL_SIZE:?} or {SMALL_SIZE:?}").into());
    }
    let mut reader = FrameReader::open(&args.dump.join("frames.bin"), size)?;
    let mut frames = Vec::new();
    let mut skipped = 0;
    let mut seen = 0;
    while frames.len() < args.frames {
        let Some(frameset) = reader.next_frameset()? else { break };
        seen += 1;
        if seen <= args.start {
            continue;
        }
        let mut images = Vec::with_capacity(4);
        let mut cam_t_ns = [0; 4];
        for (slot, &camera) in SLAM_CAMERAS.iter().enumerate() {
            let Some(frame) = &frameset.cameras[camera] else { break };
            cam_t_ns[slot] = frame.meta.pts_ns;
            if size == SMALL_SIZE {
                images.push(frame.full.clone());
            } else {
                let mut small = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
                resize_area_u8(&frame.full, &mut small)?;
                images.push(Arc::new(small));
            }
        }
        let Ok(images) = <[Luma; 4]>::try_from(images) else {
            skipped += 1;
            continue;
        };
        frames.push(Frame { index: frameset.index, t_ns: frameset.t_ns, cam_t_ns, images });
    }
    eprintln!("loaded {} framesets ({} without all four SLAM cameras skipped)", frames.len(), skipped);
    Ok(frames)
}

fn write_small(args: &Args, frames: &[Frame], out: &Path) -> Result<(), Error> {
    let framesets = frames.iter().map(|frame| {
        let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
        for (slot, &camera) in SLAM_CAMERAS.iter().enumerate() {
            let meta = FrameMeta { seq: frame.index, pts_ns: frame.cam_t_ns[slot], source_id: camera as u32, turned_180: false };
            cameras[camera] = Some(CameraFrame { meta, full: frame.images[slot].clone() });
        }
        Frameset { index: frame.index, t_ns: frame.t_ns, cameras }
    });
    let written = write_small_dump(&args.dump, out, framesets, "SLAM cameras only, area /3 to 640x360 by slam_bench --write-small")?;
    println!("wrote {written} framesets to {}", out.display());
    Ok(())
}

/// Rigid (rotation + translation, no scale) Umeyama alignment of `est` onto `reference`; RMSE and max error, metres.
fn ate(est: &[Vec3F64], reference: &[Vec3F64]) -> Option<(f64, f64)> {
    if est.len() < 3 {
        return None;
    }
    let (rotation, translation, _) = umeyama_f64(est, reference, false).ok()?;
    let errors: Vec<f64> = est.iter().zip(reference).map(|(e, r)| (rotation * *e + translation - *r).length()).collect();
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / est.len() as f64).sqrt();
    Some((rmse, errors.iter().cloned().fold(0.0, f64::max)))
}

fn quantiles(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!({"count": 0});
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| percentile(&sorted, p);
    serde_json::json!({
        "count": sorted.len(),
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
        "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": q(1.0),
    })
}

fn run(args: &Args, calibration: &str, frames: &[Frame], imu: &[ImuSample]) -> Result<(Vec<SlamPose>, f64), Error> {
    let mut poses = Vec::with_capacity(frames.len());
    let mut next_imu = 0;
    let started = Instant::now();
    let mut slam = SlamEstimator::with_profile(calibration, args.threads, args.profile, &args.overrides)?;
    for frame in frames {
        feed_imu_through(imu, &mut next_imu, frame.t_ns, |s| s.t_ns, |s| slam.push_imu(s))?;
        let images = [&*frame.images[0], &*frame.images[1], &*frame.images[2], &*frame.images[3]];
        if let Some(pose) = slam.track(frame.index, frame.t_ns, images)? {
            poses.push(pose);
        }
    }
    if let Some(pose) = slam.flush()? {
        poses.push(pose);
    }
    Ok((poses, started.elapsed().as_secs_f64()))
}

fn main() -> Result<(), Error> {
    let args = parse_args()?;
    let frames = load(&args)?;
    if let Some(out) = &args.write_small {
        return write_small(&args, &frames, out);
    }
    if frames.is_empty() {
        return Err("the dump has no complete SLAM framesets".into());
    }
    let imu = read_imu(&args.dump.join("imu.bin"))?;
    let device = Rig::load(&args.dump.join("rig.json"))?.device;
    let cap = Cap::from_device(&device).ok_or_else(|| format!("no factory SLAM calibration for device {device:?} (the dump's rig.json)"))?;
    let reference = read_reference_poses(&args.dump).ok().map(|poses| ReferencePoses::new(poses, 0, None));
    if let Some(cpus) = &args.cpus {
        pin_current_thread(cpus)?;
    }
    if let Some(min) = args.uclamp {
        set_uclamp_min(min)?;
    }

    let mut walls = Vec::new();
    let mut poses = Vec::new();
    for _ in 0..args.repeat.max(1) {
        let (run_poses, wall) = run(&args, cap.slam_calibration(), &frames, &imu)?;
        walls.push(wall);
        poses = run_poses;
    }

    let mut est = Vec::new();
    let mut matched = Vec::new();
    if let Some(reference) = &reference {
        for pose in poses.iter().filter(|p| p.ok) {
            if let Some(at) = reference.at(pose.t_ns) {
                let (e, r) = (pose.world_from_rig.translation.vector, at.translation.vector);
                est.push(Vec3F64::new(e.x, e.y, e.z));
                matched.push(Vec3F64::new(r.x, r.y, r.z));
            }
        }
    }
    let ate = ate(&est, &matched);

    let ran: Vec<&SlamPose> = poses.iter().filter(|p| p.compute_ms > 0.0).collect();
    let tracking: Vec<&SlamPose> = ran.iter().copied().filter(|p| p.ok).collect();
    let pick = |set: &[&SlamPose], f: fn(&SlamPose) -> f64| set.iter().map(|p| f(p)).collect::<Vec<f64>>();
    let keyframes: Vec<&SlamPose> = ran.iter().copied().filter(|p| p.stages.keyframe).collect();
    let summary = serde_json::json!({
        "dump": args.dump.display().to_string(),
        "framesets": frames.len(),
        "threads": args.threads,
        "profile": args.profile.as_str(),
        "overrides": args.overrides.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
        "tracked": est.len(),
        "ok": poses.iter().filter(|p| p.ok).count(),
        "resets": poses.last().map_or(0, |p| p.resets),
        "ate_rmse_mm": ate.map(|(rmse, _)| rmse * 1e3),
        "ate_max_mm": ate.map(|(_, max)| max * 1e3),
        "wall_s": walls,
        "hz_lossless": frames.len() as f64 / walls.iter().cloned().fold(f64::INFINITY, f64::min),
        "keyframes": keyframes.len(),
        "compute_ms": quantiles(&pick(&ran, |p| p.compute_ms)),
        "compute_tracking_ms": quantiles(&pick(&tracking, |p| p.compute_ms)),
        "compute_keyframe_ms": quantiles(&pick(&keyframes, |p| p.compute_ms)),
        "frontend_ms": quantiles(&pick(&ran, |p| p.stages.frontend_ms)),
        "pyramid_ms": quantiles(&pick(&ran, |p| p.stages.pyramid_ms)),
        "detect_ms": quantiles(&pick(&ran, |p| p.stages.detect_ms)),
        "track_ms": quantiles(&pick(&ran, |p| p.stages.track_ms)),
        "stereo_ms": quantiles(&pick(&ran, |p| p.stages.stereo_ms)),
        "optimize_ms": quantiles(&pick(&ran, |p| p.stages.optimize_ms)),
        "optimize_keyframe_ms": quantiles(&pick(&keyframes, |p| p.stages.optimize_ms)),
        "marginalize_ms": quantiles(&pick(&ran, |p| p.stages.marginalize_ms)),
        "keyframe_stage_ms": quantiles(&pick(&ran, |p| p.stages.keyframe_ms)),
    });
    println!("{}", serde_json::to_string(&summary)?);
    if let Some(out) = &args.out {
        let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
        for pose in &poses {
            let line = serde_json::json!({
                "index": pose.index,
                "t_ns": pose.t_ns,
                "world_from_rig": matrix_from_isometry(&pose.world_from_rig),
                "slam_ok": pose.ok,
                "slam_status": pose.status.as_str(),
                "landmarks": pose.landmarks,
                "tracked": pose.tracked,
                "compute_ms": pose.compute_ms,
                "stages": pose.stages,
            });
            writeln!(file, "{line}")?;
        }
    }
    Ok(())
}
