//! Isolated CubeCL pool measurement; no other test may allocate on this client.
#![cfg(feature = "gpu-core")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use slam_rs::frontend::detect::CornerScan;
use slam_rs::frontend::patterns::{Pattern, Pattern51};
use slam_rs::frontend::tracker::{
    FlowResult, FlowTransforms, PatchTracker, PointsSoA, SourcePatches,
};
use slam_rs::gpu::{GpuCornerScan, GpuPatchTracker, GpuPatches, GpuPyramidBuilder, gpu_client};
use slam_rs::image::ImageU16;
use slam_rs::pyramid::PyramidBuilder;

mod common;

use common::gpu::{
    LEVELS, MAX_ITERATIONS, MAX_KEYPOINTS, MAX_RECOVERED_DIST2, band_at, guesses_at,
};
use common::{cornered_image, grid_positions};

/// The whole GPU path keeps CubeCL's pool bounded: it plateaus and holds flat.
///
/// **Bounded pool growth, not zero device allocations** — the distinction
/// matters and the earlier version of this test could not tell them apart.
/// CubeCL 0.10's only host-to-device write is `create*`
/// (`cubecl-runtime/src/client.rs`: `create_from_slice`, `create`, the tensor
/// forms, and `empty`; there is no write into an existing handle), so three
/// allocations are unavoidably per-frame: the frame upload per camera
/// (`GpuPyramidBuilder::build`), the positions buffer per patch build
/// (`GpuPatches::upload_staging`) and the transform buffer per tracking call
/// (`GpuPatchTracker::track`). What the design can promise is that the pool
/// they come out of stops growing, and that is what this measures.
///
/// Pool growth matters on shared-memory devices. The caller controls the
/// number of per-frame allocations and their sizes.
///
/// So the drive is the **whole** path — pyramid, corner scan and its band walk,
/// patch build and the KLT tracker, two cameras, 200 framesets — and the
/// assertion is that after a warm-up both the reserved bytes and the bytes in
/// use are *constant*, not merely under a ceiling. A ceiling alone passed while
/// each of the three per-frame allocations went unexercised.
#[test]
fn the_whole_gpu_path_holds_the_pool_flat() {
    const FRAMES: usize = 200;
    /// Framesets the pool may still be growing over.
    ///
    /// Ten frames allow the pool to settle before checking for leaks.
    const WARM_UP: usize = 10;
    /// Keep the existing reserved-byte ceiling; flatness is the stronger gate.
    const RESERVED_CEILING: u64 = 320 * 1024 * 1024;
    /// Half again the measured payload: **18.51 MiB** on wgpu, for two 960x960 four-level pyramids, the scanner's three
    /// per-camera buffers, two patch sets at a 1,024-keypoint capacity and the
    /// tracker's result buffers. Unlike the reserved figure this is the data
    /// itself, so it is the same on any device and a leak of any size shows in
    /// it.
    const IN_USE_CEILING: u64 = 32 * 1024 * 1024;

    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone(), Pattern51::OFFSETS);
    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(client.clone()).unwrap();
    scanner.share_level0(builder.level0_table());
    let mut tracker: GpuPatchTracker<Pattern51, _> = GpuPatchTracker::new(
        client.clone(),
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        1,
    )
    .unwrap();
    let mut patches: GpuPatches<Pattern51, _> = tracker.make_patches().unwrap();
    let mut result: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);

    // Two cameras of the same geometry, as a stereo rig is: the mixed-geometry
    // case is what `the_gpu_corner_scan_reads_the_pyramid_and_uploads_nothing`
    // drives, and one geometry is the harder test for a *leak*, because nothing
    // here can be blamed on a cache that keeps missing.
    let frames: [ImageU16; 2] = [cornered_image(960, 960), cornered_image(960, 960)];
    let mut pyramids: Vec<_> = frames
        .iter()
        .map(|frame| {
            builder
                .allocate(frame.width(), frame.height(), LEVELS)
                .unwrap()
        })
        .collect();
    let positions: PointsSoA = grid_positions(960);
    let guesses: FlowTransforms = guesses_at(&positions);

    let mut reserved: Vec<u64> = Vec::with_capacity(FRAMES);
    let mut in_use: Vec<u64> = Vec::with_capacity(FRAMES);
    for frame_index in 0..FRAMES {
        for (camera, frame) in frames.iter().enumerate() {
            builder.build(camera, frame, &mut pyramids[camera]).unwrap();
            scanner.scan(camera, frame).unwrap();
            // One band, so the download and the host-side walk run too.
            scanner.band(band_at(0, 0, 3, 44, 20)).unwrap();
        }
        // The patch build and the tracking call are the other two per-frame
        // allocations, and neither ran here before: camera 0's pyramid is the
        // source and camera 1's the target, which is one frame pair per
        // frameset through the whole KLT.
        let (previous, next) = pyramids.split_at_mut(1);
        patches.build(&previous[0], &positions, None).unwrap();
        tracker
            .track(&previous[0], &next[0], &patches, &guesses, &mut result)
            .unwrap();
        // `.unwrap()`, not `if let Ok`: a runtime that stops reporting its
        // memory usage would leave these at zero and this test — the only one
        // that would catch unbounded device growth — passing having measured
        // nothing.
        let usage = client.memory_usage().unwrap();
        reserved.push(usage.bytes_reserved);
        in_use.push(usage.bytes_in_use);
        if frame_index < 3 || frame_index == WARM_UP || frame_index + 1 == FRAMES {
            println!(
                "frameset {}: {} allocs, {:.2} MiB in use, {:.2} MiB reserved, {} tracked",
                frame_index + 1,
                usage.number_allocs,
                usage.bytes_in_use as f64 / (1024.0 * 1024.0),
                usage.bytes_reserved as f64 / (1024.0 * 1024.0),
                result.len(),
            );
        }
    }

    let plateau_reserved: u64 = reserved[WARM_UP];
    let plateau_in_use: u64 = in_use[WARM_UP];
    println!(
        "plateau after {WARM_UP} framesets: {:.2} MiB reserved, {:.2} MiB in use; \
         worst over {FRAMES}: {:.2} / {:.2} MiB",
        plateau_reserved as f64 / (1024.0 * 1024.0),
        plateau_in_use as f64 / (1024.0 * 1024.0),
        reserved.iter().copied().max().unwrap_or(0) as f64 / (1024.0 * 1024.0),
        in_use.iter().copied().max().unwrap_or(0) as f64 / (1024.0 * 1024.0),
    );
    assert!(
        plateau_reserved > 0 && plateau_in_use > 0,
        "the runtime reported no memory at all over {FRAMES} framesets"
    );
    for (frame_index, (&held, &used)) in
        reserved.iter().zip(in_use.iter()).enumerate().skip(WARM_UP)
    {
        assert_eq!(
            held,
            plateau_reserved,
            "reserved bytes moved at frameset {} ({held} against the plateau's \
             {plateau_reserved}), so the pool is still growing",
            frame_index + 1
        );
        assert_eq!(
            used,
            plateau_in_use,
            "bytes in use moved at frameset {} ({used} against the plateau's \
             {plateau_in_use}), so a per-frame buffer is not being freed",
            frame_index + 1
        );
    }
    assert!(
        plateau_reserved < RESERVED_CEILING,
        "CubeCL reserved {plateau_reserved} bytes over {FRAMES} framesets of the \
         whole path on two 960x960 cameras, against a ceiling of {RESERVED_CEILING}"
    );
    assert!(
        plateau_in_use < IN_USE_CEILING,
        "the whole path holds {plateau_in_use} bytes of device data, against a \
         ceiling of {IN_USE_CEILING}"
    );
}
