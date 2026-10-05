//! Timed replay and its trajectory/summary outputs.

use crate::Lane;
use crate::clip::ReplayInput;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::replay::{feed_imu_through, percentile};
use slam_rs::{Backend, ImageView, Vio, VioResult};
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

#[derive(Default)]
struct ReplayCounters {
    pyramid_ns: u64,
    detect_ns: u64,
    track_ns: u64,
    stereo_ns: u64,
    imu_ns: u64,
    overlap_estimator_ns: u64,
    overlap_wait_ns: u64,
}

pub(super) fn replay(
    input: ReplayInput,
    config_path: &Path,
    lane: Lane,
    threads: usize,
    load_s: f64,
    out: &Path,
    summary: &Path,
) -> Result<(), String> {
    let ReplayInput {
        clip,
        calibration,
        imu,
        pixels,
    } = input;
    let config_text: String = if config_path == Path::new("-") {
        use std::io::Read as _;
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(|error| error.to_string())?;
        text
    } else {
        std::fs::read_to_string(config_path)
            .map_err(|error| format!("{}: {error}", config_path.display()))?
    };
    let config: VioConfig = VioConfig::from_json_str(&config_text)
        .map_err(|error| format!("{}: {error:?}", config_path.display()))?;
    let framesets = clip.framesets;
    let frameset_bytes: usize = clip.resolution_wh.iter().map(|&(w, h)| w * h).sum();

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
    let mut counters = ReplayCounters::default();
    let mut tracked: usize = 0;
    let mut record = |result: &VioResult| -> Result<(), String> {
        if let Some(estimate) = &result.pose {
            tracked += 1;
            let pose = estimate.world_from_rig;
            let t_ns = result.t_ns;
            writeln!(
                poses,
                "{t_ns},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
                pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]
            )
            .map_err(|error| error.to_string())?;
        }
        Ok(())
    };
    let mut cursor: usize = 0;
    let started: Instant = Instant::now();
    for frame in 0..framesets {
        let t_ns: i64 = clip.frame_t_ns[frame];
        feed_imu_through(
            &imu,
            &mut cursor,
            t_ns,
            |row| row.t_ns,
            |row| {
                vio.push_imu(row.t_ns, row.gyro, row.accel)
                    .map_err(|error| format!("imu: {error}"))
            },
        )?;
        let frameset_views = |frame: usize| {
            let mut offset = frame * frameset_bytes;
            clip.resolution_wh
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
                .collect::<Vec<_>>()
        };
        let views = frameset_views(frame);
        let call: Instant = Instant::now();
        let result: VioResult = vio
            .track(t_ns, &views)
            .map_err(|error| format!("frameset {frame}: {error}"))?;
        track_ms.push(call.elapsed().as_secs_f64() * 1e3);
        let overlap = vio.overlap_timings();
        counters.overlap_estimator_ns += overlap.estimator_ns;
        counters.overlap_wait_ns += overlap.wait_ns;
        let timings = vio.frontend_timings();
        counters.pyramid_ns += timings.flow.pyramid_ns;
        counters.detect_ns += timings.flow.detect_ns;
        counters.track_ns += timings.flow.track_ns;
        counters.stereo_ns += timings.flow.stereo_ns;
        counters.imu_ns += timings.imu_ns;
        record(&result)?;
    }
    let flush_started = Instant::now();
    if let Some(result) = vio.flush().map_err(|error| format!("flush: {error}"))? {
        record(&result)?;
        if result.pose.is_some() {
            counters.overlap_estimator_ns += vio.overlap_timings().estimator_ns;
        }
    }
    let flush_ms = flush_started.elapsed().as_secs_f64() * 1e3;
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
        "frontend_ms_mean": {"pyramid": per_frame(counters.pyramid_ns), "detect": per_frame(counters.detect_ns), "track": per_frame(counters.track_ns),
                             "stereo": per_frame(counters.stereo_ns), "imu": per_frame(counters.imu_ns)},
        "overlap_ms_mean": {"estimator": per_frame(counters.overlap_estimator_ns), "wait": per_frame(counters.overlap_wait_ns)},
        "flush_ms": flush_ms,
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
