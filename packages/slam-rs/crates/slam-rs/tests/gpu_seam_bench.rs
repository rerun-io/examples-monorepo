//! Where the GPU frontend's host time goes, over real MIO10 framesets, and the
//! queue depth a whole frameset of it reserves.
//!
//! `SLAM_RS_SEAM_BENCH=<framesets>` runs the whole frontend on the device and
//! prints the four stage timers beside the seam counters, so a change to the
//! launch/wait shape can be read in one process without the A/B harness.
//! The timing test is ignored unless selected explicitly. The queue-peak test is
//! an assertion and always runs: the same rig, driven for its task counts
//! rather than its timings.
//!
//! ```bash
//! SLAM_RS_SEAM_BENCH=300 cargo test --release --features gpu-wgpu \
//!   --test gpu_seam_bench -- --ignored --exact the_gpu_frontend_reports_its_host_seam --nocapture
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "gpu-wgpu")]

use slam_rs::frontend::flow::{FrameToFrameOpticalFlow, FrontendOptions, PosePrediction};
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::image::ImageU16;

mod common;

/// The median of a copy of `values`, which is what a stage timer is reported as.
fn median(values: &mut [u64]) -> f64 {
    values.sort_unstable();
    values[values.len() / 2] as f64 / 1e6
}

/// The MSDMI calibration widened to `cameras` by repeating its two, each new
/// camera 5 cm further along the body `x` axis so no two share a pose.
///
/// Both tests here drive rigs wider than the fixture's, which is the only way
/// a four- or eight-camera batch shape is reachable from committed frames.
fn widened(cameras: usize) -> slam_rs::calib::Calibration<f64> {
    let mut calibration = common::calibration();
    while calibration.intrinsics.len() < cameras {
        let source: usize = calibration.intrinsics.len() % 2;
        calibration.intrinsics.push(calibration.intrinsics[source]);
        calibration.resolution.push(calibration.resolution[source]);
        let mut pose = calibration.t_i_c[source];
        pose.translation.x += 0.05 * (calibration.t_i_c.len() as f64);
        calibration.t_i_c.push(pose);
    }
    calibration.intrinsics.truncate(cameras);
    calibration.resolution.truncate(cameras);
    calibration.t_i_c.truncate(cameras);
    calibration
}

#[test]
#[ignore = "timing harness: run alone with --ignored --exact"]
fn the_gpu_frontend_reports_its_host_seam() {
    let Some(count) = std::env::var("SLAM_RS_SEAM_BENCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        println!("skipped: set SLAM_RS_SEAM_BENCH=<framesets> to measure the seam");
        return;
    };

    let mut config = common::config();
    // The vendored config detects on every frameset. `SLAM_RS_SEAM_REDETECT`
    // puts the fast profile's gate on instead (D75), which is the mix that
    // decides whether speculating a selection pays: a frameset that detects
    // saves a read and one that skips wastes the FAST kernels.
    if let Some(ratio) = std::env::var("SLAM_RS_SEAM_REDETECT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
    {
        config.port_redetect_survivor_ratio = ratio;
    }
    // `SLAM_RS_SEAM_CAMERAS` widens the two-camera fixture rig by repeating its
    // cameras, which is how a four-camera rig's batch shape is reachable from
    // the committed 960x960 frames.
    let cameras: usize = std::env::var("SLAM_RS_SEAM_CAMERAS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let calibration = widened(cameras);
    let options: FrontendOptions = FrontendOptions::default();
    let (builder, tracker, scanner) = slam_rs::gpu::gpu_backends::<Pattern51>(
        options.max_keypoints,
        config.optical_flow_levels as usize + 1,
        config.optical_flow_max_iterations as usize,
        config.optical_flow_max_recovered_dist2,
        cameras,
    )
    .unwrap();
    let mut flow = FrameToFrameOpticalFlow::with_backends(
        config,
        &calibration,
        options,
        builder,
        tracker,
        scanner,
    )
    .unwrap();

    // The three committed framesets, walked forward and back so consecutive
    // framesets are a real small motion rather than a jump.
    let frames: Vec<Vec<ImageU16>> = (0..3)
        .map(|frame| {
            (0..cameras)
                .map(|camera| common::mio10_frame(frame, camera % 2))
                .collect()
        })
        .collect();
    let order: [usize; 4] = [0, 1, 2, 1];

    let mut pyramid: Vec<u64> = Vec::with_capacity(count);
    let mut detect: Vec<u64> = Vec::with_capacity(count);
    let mut track: Vec<u64> = Vec::with_capacity(count);
    let mut stereo: Vec<u64> = Vec::with_capacity(count);
    let mut whole: Vec<u64> = Vec::with_capacity(count);
    let mut keypoints: usize = 0;
    // Framesets that ran `addPoints`, which is the only thing the speculated
    // selection is for: the id space only grows where a corner was registered.
    let mut detected: usize = 0;
    // The marginal cost of one more synchronising read, measured where the
    // frontend pays it: `SLAM_RS_SEAM_EXTRA=<k>` adds k four-byte reads to every
    // frameset, on the same client the frontend uses.
    let extra: usize = std::env::var("SLAM_RS_SEAM_EXTRA")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let client = slam_rs::gpu::gpu_client().unwrap();
    let probe = client.empty(16);

    // A tenth of the run is warm-up: the first frameset allocates every buffer.
    let warmup: usize = (count / 10).max(2);
    let mut start = slam_rs::gpu::seam::snapshot();
    for index in 0..count + warmup {
        if index == warmup {
            start = slam_rs::gpu::seam::snapshot();
        }
        let images: &[ImageU16] = &frames[order[index % order.len()]];
        let ids_before: u64 = flow.last_keypoint_id();
        let mark: std::time::Instant = std::time::Instant::now();
        let frame = flow
            .process_frame(
                index as i64 * 50_000_000,
                images,
                &PosePrediction::default(),
                &[],
            )
            .unwrap();
        for _ in 0..extra {
            client.read_one(probe.clone()).unwrap();
        }
        let elapsed: u64 = mark.elapsed().as_nanos() as u64;
        keypoints = frame.cameras[0].len();
        if index >= warmup {
            if flow.last_keypoint_id() != ids_before {
                detected += 1;
            }
            let timings = flow.timings();
            pyramid.push(timings.pyramid_ns);
            detect.push(timings.detect_ns);
            track.push(timings.track_ns);
            stereo.push(timings.stereo_ns);
            whole.push(elapsed);
        }
    }

    println!(
        "MIO10 {count} framesets, {cameras} cameras, {keypoints} keypoints on camera 0, \
         {detected} detected\n\
         medians ms: whole {:.3} = pyramid {:.3} + detect {:.3} + track {:.3} + stereo {:.3}\n\
         {}",
        median(&mut whole),
        median(&mut pyramid),
        median(&mut detect),
        median(&mut track),
        median(&mut stereo),
        seam_line(slam_rs::gpu::seam::snapshot().delta(start), count as u64),
    );
}

/// A whole frameset's reserved tasks stay under the runtime's channel depth,
/// two, four and eight cameras wide.
///
/// `prepared_pyramids_stay_below_the_runtime_channel_depth` (`gpu_kernels.rs`)
/// bounds the staging phase alone, before the relay runs. What D77's ceiling has
/// to hold for is a whole frameset of the shipped wiring, where three groups of
/// launches share the intervals between its downloads: camera 0's selection is
/// speculated **before** the temporal passes, the other cameras' selections go
/// out **behind** the matches, and both ride another stage's read home (D78).
/// Eight cameras is twice the widest rig the configs carry, and the count grows
/// per camera.
///
/// Over the whole frameset, not per stage: a phase that stayed under the ceiling
/// alone can still cross it behind a phase that did not drain. The measure is
/// the reserved peak, so what it bounds is the accounted submissions; an
/// unaccounted one is invisible here and shows up as the runtime's own spin,
/// which is what the A/B harness measures. `queue_peak` is thread-local, so a
/// concurrent test thread cannot move this assertion.
#[test]
fn the_queue_peak_of_a_whole_frameset_stays_below_the_channel_depth() {
    for cameras in [2, 4, 8] {
        let config = common::config();
        assert!(
            config.optical_flow_detection_nonoverlap,
            "the tail selection only launches with the non-overlap pass on"
        );
        let calibration = widened(cameras);
        let options: FrontendOptions = FrontendOptions::default();
        let (builder, tracker, scanner) = slam_rs::gpu::gpu_backends::<Pattern51>(
            options.max_keypoints,
            config.optical_flow_levels as usize + 1,
            config.optical_flow_max_iterations as usize,
            config.optical_flow_max_recovered_dist2,
            cameras,
        )
        .unwrap();
        let mut flow = FrameToFrameOpticalFlow::with_backends(
            config,
            &calibration,
            options,
            builder,
            tracker,
            scanner,
        )
        .unwrap();

        // Forward and back over the three committed framesets, so consecutive
        // framesets are a small motion and the temporal pass has work.
        let frames: Vec<Vec<ImageU16>> = (0..3)
            .map(|frame| {
                (0..cameras)
                    .map(|camera| common::mio10_frame(frame, camera % 2))
                    .collect()
            })
            .collect();
        let order: [usize; 4] = [0, 1, 2, 1];
        let mut detected: usize = 0;
        let mut worst: usize = 0;
        for (index, frame) in order.iter().enumerate() {
            let ids_before: u64 = flow.last_keypoint_id();
            // Per frameset, and without draining: outstanding reservations
            // carry into the next one, and this only forgets the old maximum.
            slam_rs::gpu::seam::reset_queue_peak();
            flow.process_frame(
                index as i64 * 50_000_000,
                &frames[*frame],
                &PosePrediction::default(),
                &[],
            )
            .unwrap();
            let peak: usize = slam_rs::gpu::seam::queue_peak();
            worst = worst.max(peak);
            assert!(
                peak < slam_rs::gpu::CHANNEL_TASKS,
                "{cameras} cameras, frameset {index}: {peak} tasks reserved; \
                 leave room for the flush or the read"
            );
            if flow.last_keypoint_id() != ids_before {
                detected += 1;
            }
        }
        assert!(
            detected != 0,
            "{cameras} cameras: no frameset registered a corner, so neither \
             selection path ran"
        );
        assert!(
            worst != 0,
            "{cameras} cameras: nothing reserved a task, so the rig never \
             reached the device"
        );
        // Printed under `--nocapture`, because how much room is left is worth
        // reading when a stage is added: the bound holds, the margin moves.
        println!(
            "{cameras} cameras: {worst} of {} tasks reserved at the peak, \
             {detected} of {} framesets detected",
            slam_rs::gpu::CHANNEL_TASKS,
            order.len(),
        );
    }
}

/// Format the measured producer thread's delta per frameset.
fn seam_line(delta: slam_rs::gpu::seam::Snapshot, framesets: u64) -> String {
    let scale: f64 = framesets.max(1) as f64;
    let (launches, _) = (delta.launch.calls, delta.launch.nanos);
    let (uploads, upload_ns) = (delta.upload.calls, delta.upload.nanos);
    let (track_reads, track_ns) = (delta.read_track.calls, delta.read_track.nanos);
    let (detect_reads, detect_ns) = (delta.read_detect.calls, delta.read_detect.nanos);
    format!(
        "per frameset: {:.2} launches, {:.2} uploads ({:.3} ms), {:.2} reads ({:.3} ms) \
         = {:.2} tracker ({:.3} ms) + {:.2} detector ({:.3} ms)",
        launches as f64 / scale,
        uploads as f64 / scale,
        upload_ns as f64 / scale / 1e6,
        (track_reads + detect_reads) as f64 / scale,
        (track_ns + detect_ns) as f64 / scale / 1e6,
        track_reads as f64 / scale,
        track_ns as f64 / scale / 1e6,
        detect_reads as f64 / scale,
        detect_ns as f64 / scale / 1e6,
    )
}
