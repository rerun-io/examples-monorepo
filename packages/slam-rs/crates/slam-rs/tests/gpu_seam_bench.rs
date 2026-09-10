//! Where the GPU frontend's host time goes, over real MIO10 framesets.
//!
//! Not an assertion: a measurement rig. `SLAM_RS_SEAM_BENCH=<framesets>` runs
//! the whole frontend on the device and prints the four stage timers beside the
//! seam counters, so a change to the launch/wait shape can be read in one
//! process without the A/B harness. Without the variable it prints why and
//! passes.
//!
//! ```bash
//! SLAM_RS_SEAM_BENCH=300 cargo test --release --features gpu-wgpu \
//!   --test gpu_seam_bench -- --nocapture
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "gpu-wgpu")]

use slam_rs::frontend::flow::{FrameToFrameOpticalFlow, FrontendOptions, PosePrediction};
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::image::ImageU16;

mod common;

/// One committed MIO10 frame as the frontend takes it: 960x960, `u8 << 8`.
fn mio10_frame(frame: usize, camera: usize) -> ImageU16 {
    let pgm: common::Pgm = common::read_pgm(&common::fixtures().join("flow/frames"), frame, camera);
    let mut image: ImageU16 = ImageU16::zeros(pgm.width, pgm.height).unwrap();
    for y in 0..pgm.height {
        for x in 0..pgm.width {
            image.set(x, y, u16::from(pgm.pixels[y * pgm.width + x]) << 8);
        }
    }
    image
}

/// The median of a copy of `values`, which is what a stage timer is reported as.
fn median(values: &mut [u64]) -> f64 {
    values.sort_unstable();
    values[values.len() / 2] as f64 / 1e6
}

#[test]
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
                .map(|camera| mio10_frame(frame, camera % 2))
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
    for index in 0..count + warmup {
        if index == warmup {
            slam_rs::gpu::seam::reset();
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
        slam_rs::gpu::seam::line(count as u64),
    );
}
