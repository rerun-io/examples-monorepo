//! Timed replay and its trajectory/summary outputs.

use crate::ReplayOutputs;
use crate::clip::ReplayInput;
use slam_rs::backend::MapResult;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::replay::{feed_imu_through, percentile};
use slam_rs::{
    Backend, BackendRequest, Backends, GbaRequest, ImageView, Vio, VioResult, select_backend,
};
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
    gpu_lookahead_frames: usize,
    gpu_one_wait_frames: usize,
}

pub(super) fn replay(
    input: ReplayInput,
    config_path: &Path,
    (request, gba_request): (BackendRequest, GbaRequest),
    run_gba: bool,
    threads: usize,
    load_s: f64,
    outputs: &ReplayOutputs,
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

    let backends: Backends = select_backend(request, gba_request)
        .map_err(|error| format!("--backend/--ba-backend gpu: {error}"))?;
    let backend: Backend = backends.frontend;
    eprintln!(
        "slam-rs: frontend backend {}, GBA lane {}",
        backend.name(),
        backends.gba.name()
    );
    let options = FrontendOptions {
        threads,
        ..FrontendOptions::default()
    };
    let setup_started: Instant = Instant::now();
    let mut vio: Vio<f32> = Vio::with_backends(config, calibration, options, backends)
        .map_err(|error| format!("Vio: {error}"))?;
    let setup_s: f64 = setup_started.elapsed().as_secs_f64();

    let mut raw: Vec<(i64, [f64; 7])> = Vec::with_capacity(framesets);
    let mut track_ms: Vec<f64> = Vec::with_capacity(framesets);
    let mut counters = ReplayCounters::default();
    let mut tracked: usize = 0;
    let mut record = |result: &VioResult| -> Result<(), String> {
        if let Some(estimate) = &result.pose {
            tracked += 1;
            raw.push((result.t_ns, estimate.world_from_rig));
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
        let next_views = if backend == Backend::Gpu && frame + 1 < framesets {
            frameset_views(frame + 1)
        } else {
            Vec::new()
        };
        let lookahead =
            (!next_views.is_empty()).then(|| (clip.frame_t_ns[frame + 1], next_views.as_slice()));
        let result: VioResult = vio
            .track_with_lookahead(t_ns, &views, lookahead)
            .map_err(|error| format!("frameset {frame}: {error}"))?;
        track_ms.push(call.elapsed().as_secs_f64() * 1e3);
        let overlap = vio.overlap_timings();
        counters.overlap_estimator_ns += overlap.estimator_ns;
        counters.overlap_wait_ns += overlap.wait_ns;
        let timings = vio.frontend_timings();
        counters.gpu_lookahead_frames += usize::from(timings.flow.gpu_lookahead);
        counters.gpu_one_wait_frames += usize::from(timings.flow.gpu_one_wait);
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
    let mean_ms: f64 = track_ms.iter().sum::<f64>() / track_ms.len().max(1) as f64;
    let mut sorted: Vec<f64> = track_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let per_frame = |ns: u64| ns as f64 / 1e6 / framesets.max(1) as f64;
    let report = serde_json::json!({
        "segment_id": clip.segment_id,
        "lane": backend.name(),
        "gba_lane": backends.gba.name(),
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
        "gpu_path_frames": {"lookahead": counters.gpu_lookahead_frames, "one_wait": counters.gpu_one_wait_frames},
        "overlap_ms_mean": {"estimator": per_frame(counters.overlap_estimator_ns), "wait": per_frame(counters.overlap_wait_ns)},
        "flush_ms": flush_ms,
        "track_ms_per_frame": track_ms,
    });
    write_outputs(outputs, &raw, report, || vio.finish_map(run_gba))?;
    println!(
        "{}: {tracked}/{framesets} tracked, wall {wall_s:.2} s = {:.2} ms/frameset (track mean {mean_ms:.2}, p50 {:.2}, p95 {:.2})",
        clip.segment_id,
        wall_s * 1e3 / framesets.max(1) as f64,
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95)
    );
    Ok(())
}

/// Write a replay's files. The raw VIO (`--out-vio`) and the summary go first,
/// so nothing after them loses a tracked run. Then the final trajectory (DESIGN
/// §1, §2): the raw VIO re-anchored on the map `finish_map` leaves, after the
/// global BA when it ran, or the raw VIO itself with the back end off; and the
/// summary again, with the finish. A GBA that failed costs only the GBA: one
/// warning, `--out` re-anchored on the map as loops and PGO left it, and
/// `"gba": {"failed": <reason>}` in the summary.
fn write_outputs(
    outputs: &ReplayOutputs,
    raw: &[(i64, [f64; 7])],
    mut report: serde_json::Value,
    finish_map: impl FnOnce() -> Option<MapResult>,
) -> Result<(), String> {
    let write_summary = |report: &serde_json::Value| -> Result<(), String> {
        let text: String =
            serde_json::to_string_pretty(report).map_err(|error| error.to_string())?;
        std::fs::write(&outputs.summary, text)
            .map_err(|error| format!("{}: {error}", outputs.summary.display()))
    };
    if let Some(path) = &outputs.out_vio {
        std::fs::write(path, trajectory_csv(raw.iter().copied())?)
            .map_err(|error| format!("{}: {error}", path.display()))?;
    }
    write_summary(&report)?;

    let finish_started = Instant::now();
    let map: Option<MapResult> = finish_map();
    let finish_map_s: f64 = finish_started.elapsed().as_secs_f64();
    let gba_failure: Option<String> = map
        .as_ref()
        .and_then(|map| map.gba_failure.as_ref())
        .map(|error| format!("global BA: {error}"));
    if let Some(reason) = &gba_failure {
        eprintln!(
            "slam-rs: warning: the global BA failed ({reason}); --out is re-anchored on the map as loops and PGO left it"
        );
    }
    let final_poses: Vec<[f64; 7]> = match &map {
        Some(map) => map
            .reanchor(raw)
            .map_err(|error| format!("re-anchoring: {error}"))?,
        None => raw.iter().map(|&(_, pose)| pose).collect(),
    };
    let final_rows = raw.iter().map(|&(t_ns, _)| t_ns).zip(final_poses);
    std::fs::write(&outputs.out, trajectory_csv(final_rows)?)
        .map_err(|error| format!("{}: {error}", outputs.out.display()))?;

    let gba = match (map.as_ref().and_then(|map| map.gba.as_ref()), gba_failure) {
        (_, Some(reason)) => serde_json::json!({"failed": reason}),
        (Some(gba), None) => {
            serde_json::json!({"lane": gba.lane.name(), "keyframes": gba.counts.keyframes,
                                                "attempts": gba.report.iterations.len(), "build_s": gba.build_s,
                                                "solve_s": gba.solve_s, "apply_s": gba.apply_s})
        }
        (None, None) => serde_json::Value::Null,
    };
    report["finish_map_s"] = finish_map_s.into();
    report["gba"] = gba;
    write_summary(&report)
}

/// `t_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z` rows from `(t_ns, [tx, ty, tz, qx, qy, qz, qw])`.
fn trajectory_csv(rows: impl Iterator<Item = (i64, [f64; 7])>) -> Result<String, String> {
    let mut text: String = String::from("#t_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n");
    for (t_ns, pose) in rows {
        writeln!(
            text,
            "{t_ns},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
            pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::{ReplayOutputs, trajectory_csv, write_outputs};
    use slam_rs::backend::{GbaDriverError, MapResult};
    use slam_rs::lie::Se3;
    use std::path::{Path, PathBuf};

    const RAW: [(i64, [f64; 7]); 2] = [
        (10, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        (20, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    fn outputs(name: &str) -> ReplayOutputs {
        let dir: PathBuf =
            std::env::temp_dir().join(format!("slam-rs-cli-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["slam_rs.csv", "slam_rs_vio.csv", "summary.json"] {
            let _ = std::fs::remove_file(dir.join(file));
        }
        ReplayOutputs {
            out: dir.join("slam_rs.csv"),
            out_vio: Some(dir.join("slam_rs_vio.csv")),
            summary: dir.join("summary.json"),
        }
    }

    /// The map as loops and PGO left it: one keyframe, 2 m off the raw VIO.
    fn lc_map() -> MapResult {
        let mut moved: Se3<f64> = Se3::identity();
        moved.translation.x = 2.0;
        MapResult {
            keyframe_t_ns: vec![10],
            keyframe_poses_before_gba: vec![moved],
            keyframe_poses: vec![moved],
            gba: None,
            gba_failure: None,
        }
    }

    fn summary(path: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn a_failed_gba_falls_back_to_the_map_after_loops_and_pgo() {
        let outputs = outputs("gba-fails");
        let failed = MapResult {
            gba_failure: Some(GbaDriverError::NoRig),
            ..lc_map()
        };
        write_outputs(&outputs, &RAW, serde_json::json!({"tracked": 2}), || {
            Some(failed)
        })
        .unwrap();
        let raw_csv: String = trajectory_csv(RAW.iter().copied()).unwrap();
        assert_eq!(
            std::fs::read_to_string(outputs.out_vio.as_ref().unwrap()).unwrap(),
            raw_csv
        );
        let lc_final = RAW
            .iter()
            .map(|&(t_ns, _)| t_ns)
            .zip(lc_map().reanchor(&RAW).unwrap());
        assert_eq!(
            std::fs::read_to_string(&outputs.out).unwrap(),
            trajectory_csv(lc_final).unwrap()
        );
        let report = summary(&outputs.summary);
        assert_eq!(report["tracked"], 2);
        assert_eq!(
            report["gba"]["failed"],
            format!("global BA: {}", GbaDriverError::NoRig),
            "{report}"
        );
        assert!(report["finish_map_s"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn the_raw_vio_and_the_summary_are_written_before_the_map_is_finished() {
        let outputs = outputs("finish-fails");
        // A keyframe that is not one of the frames: re-anchoring refuses it.
        let stray = MapResult {
            keyframe_t_ns: vec![15],
            ..lc_map()
        };
        let error = write_outputs(&outputs, &RAW, serde_json::json!({"tracked": 2}), || {
            Some(stray)
        })
        .unwrap_err();
        assert!(error.contains("re-anchoring"), "{error}");
        assert_eq!(
            std::fs::read_to_string(outputs.out_vio.as_ref().unwrap()).unwrap(),
            trajectory_csv(RAW.iter().copied()).unwrap()
        );
        assert_eq!(summary(&outputs.summary)["tracked"], 2);
        assert!(!outputs.out.exists());
    }

    #[test]
    fn without_the_gba_the_summary_says_none() {
        let outputs = outputs("no-gba");
        write_outputs(&outputs, &RAW, serde_json::json!({}), || Some(lc_map())).unwrap();
        assert!(summary(&outputs.summary)["gba"].is_null());
    }
}
