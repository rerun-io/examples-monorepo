//! `FrameToFrameOpticalFlow`: a refused frame or a backend error part-way
//! through one leaves the frontend as the last good frame left it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kornia_image::Image;
use slam_rs::frontend::flow::*;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::types::KeypointId;

mod common;

use common::flow::{failing_frontend, frontend};
use common::{FLOW_HEIGHT as HEIGHT, FLOW_WIDTH as WIDTH, dotted_image};

/// And it is refused transactionally: the frontend is as the last accepted
/// frame left it.
#[test]
fn a_frame_of_the_wrong_size_commits_nothing() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(1_000, &images, &PosePrediction::default(), &[])
        .unwrap();
    let before: FlowFrame = flow.frame().clone();
    let ids_before: u64 = flow.last_keypoint_id();

    let odd: [Image<u16, 1>; 2] = [
        slam_rs::image::zeros(WIDTH + 8, HEIGHT).unwrap(),
        slam_rs::image::zeros(WIDTH + 8, HEIGHT).unwrap(),
    ];
    assert!(
        flow.process_frame(2_000, &odd, &PosePrediction::default(), &[])
            .is_err()
    );
    assert_eq!(flow.t_ns(), Some(1_000));
    assert_eq!(flow.frame_counter(), 1);
    assert_eq!(flow.last_keypoint_id(), ids_before);
    assert_eq!(flow.frame(), &before);

    // And the frontend still tracks afterwards, from the frame it kept.
    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    let after: Vec<KeypointId> = flow
        .process_frame(3_000, &moved, &PosePrediction::default(), &[])
        .unwrap()
        .cameras[0]
        .ids
        .clone();
    assert_eq!(flow.frame_counter(), 2);
    let shared: usize = after
        .iter()
        .filter(|id| before.cameras[0].get(**id).is_some())
        .count();
    assert!(shared > 0, "the kept frame was not tracked against");
}

/// A frame the frontend refuses must leave it usable.
///
/// The review's sequence: a 1x1 first image is rejected, and the *next*,
/// valid frame used to take the tracking path against an empty previous
/// pyramid and panic. Nothing commits until every pyramid is built. The 1x1
/// frame is now refused one step earlier than it was — the calibration check
/// catches it before the pyramid builder ever sees it — and the sequence this
/// test is about is unchanged.
#[test]
fn a_rejected_frame_leaves_the_frontend_usable() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let tiny: [Image<u16, 1>; 2] = [
        slam_rs::image::zeros(1, 1).unwrap(),
        slam_rs::image::zeros(1, 1).unwrap(),
    ];
    let error = flow
        .process_frame(1, &tiny, &PosePrediction::default(), &[])
        .unwrap_err();
    assert_eq!(
        error,
        FrontendError::FrameSizeMismatch {
            camera: 0,
            expected_width: WIDTH,
            expected_height: HEIGHT,
            actual_width: 1,
            actual_height: 1,
        }
    );
    // Nothing moved: the clock is still before the first frame.
    assert_eq!(flow.t_ns(), None);
    assert_eq!(flow.frame_counter(), 0);
    assert_eq!(flow.last_keypoint_id(), 0);

    // The next valid frame is treated as the first, and the one after it
    // tracks against a pyramid that exists.
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let frame: &FlowFrame = flow
        .process_frame(2, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert!(!frame.cameras[0].is_empty());
    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(3, &moved, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.frame_counter(), 2);
}

/// A frameset of the wrong width is refused before anything commits too.
#[test]
fn a_frameset_of_the_wrong_width_commits_nothing() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let before: FlowFrame = flow.frame().clone();

    let short: [Image<u16, 1>; 1] = [dotted_image(1)];
    assert!(
        flow.process_frame(1, &short, &PosePrediction::default(), &[])
            .is_err()
    );
    assert_eq!(flow.t_ns(), Some(0));
    assert_eq!(flow.frame_counter(), 1);
    assert_eq!(flow.frame(), &before);

    // And the frontend still works afterwards.
    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(2, &moved, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.frame_counter(), 2);
}

/// A backend error part-way through frame 2 must leave the frontend exactly
/// as frame 1 left it, so frame 3 comes out of `{1, 2 fails, 3}` identical to
/// frame 3 of a clean `{1, 3}`.
///
/// The rotation used to commit before tracking, which paired frame 1's
/// keypoints with frame 2's pyramid on the next call.
#[test]
fn a_backend_error_leaves_the_frame_as_the_last_good_one() {
    let first: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let second: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    let third: [Image<u16, 1>; 2] = [dotted_image(2), dotted_image(2)];

    // The clean run skips the frame the other run fails on.
    let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    clean
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    clean
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();

    // Frame 1 makes one call (the stereo match), so call 2 is camera 0 of
    // frame 2: the failure lands after the first frame committed and before
    // the second one could.
    let mut faulty = failing_frontend(2);
    faulty
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    let after_first: FlowFrame = faulty.frame().clone();
    let ids_after_first: u64 = faulty.last_keypoint_id();
    let cells_after_first: Vec<i32> = faulty.cell_counts(0).to_vec();

    let error = faulty
        .process_frame(1, &second, &PosePrediction::default(), &[])
        .unwrap_err();
    assert!(matches!(error, FrontendError::Tracker(_)), "got {error}");

    // Nothing moved.
    assert_eq!(faulty.frame(), &after_first);
    assert_eq!(faulty.last_keypoint_id(), ids_after_first);
    assert_eq!(faulty.cell_counts(0), &cells_after_first[..]);
    assert_eq!(faulty.t_ns(), Some(0));
    assert_eq!(faulty.frame_counter(), 1);

    // And the next good frame is the one the clean run produced.
    faulty
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(faulty.frame(), clean.frame());
    assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
    assert_eq!(faulty.cell_counts(0), clean.cell_counts(0));
    assert_eq!(faulty.cell_counts(1), clean.cell_counts(1));
}

/// The same, with the failure on camera 1 — after camera 0 has already been
/// tracked and its keypoint map rewritten inside the same call.
#[test]
fn a_backend_error_after_the_first_camera_is_undone_too() {
    let first: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let second: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    let third: [Image<u16, 1>; 2] = [dotted_image(2), dotted_image(2)];

    let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    clean
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    clean
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();

    // Frame 1 makes one matching call, so frame 2's camera-1 track is call 3.
    let mut faulty = failing_frontend(3);
    faulty
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    assert!(
        faulty
            .process_frame(1, &second, &PosePrediction::default(), &[])
            .is_err()
    );
    faulty
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(faulty.frame(), clean.frame());
    assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
}

#[derive(Debug)]
struct RefuseAfterTracking {
    inner: slam_rs::frontend::detect::CpuCornerScan,
    refuse: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl slam_rs::frontend::detect::CornerScan for RefuseAfterTracking {
    fn scan(
        &mut self,
        camera: usize,
        image: &Image<u16, 1>,
    ) -> Result<(), slam_rs::frontend::detect::DetectError> {
        self.inner.scan(camera, image)
    }

    fn band(
        &mut self,
        request: slam_rs::frontend::detect::BandRequest,
    ) -> Result<&[slam_rs::frontend::detect::FastCorner], slam_rs::frontend::detect::DetectError>
    {
        self.inner.band(request)
    }

    fn take_cells(&mut self) -> Result<(), slam_rs::frontend::detect::DetectError> {
        if self.refuse.swap(false, std::sync::atomic::Ordering::SeqCst) {
            Err(slam_rs::frontend::detect::DetectError::NotScanned)
        } else {
            Ok(())
        }
    }
}

/// The failure arrives after all CPU backward stores have been overwritten.
/// A retry must track against the last committed image, with rebuilt templates.
#[test]
fn a_refused_frame_does_not_leave_uncommitted_backward_templates_in_the_cache() {
    use slam_rs::frontend::parallel::WorkPool;
    use slam_rs::frontend::tracker::CpuPatchTracker;
    use slam_rs::pyramid::CpuPyramidBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    for threads in [1, 4] {
        let config = common::flow_config();
        let options = FrontendOptions {
            threads,
            ..FrontendOptions::default()
        };
        let pool = WorkPool::new(threads).unwrap();
        let tracker = CpuPatchTracker::<Pattern51>::new(
            options.max_keypoints,
            config.optical_flow_levels as usize + 1,
            config.optical_flow_max_iterations as usize,
            config.optical_flow_max_recovered_dist2,
            pool.clone(),
        )
        .unwrap();
        let refuse = Arc::new(AtomicBool::new(false));
        let mut faulty = FrameToFrameOpticalFlow::with_stages(
            config,
            &common::flow_rig(2),
            options,
            slam_rs::frontend::stages::CpuStages::new(
                CpuPyramidBuilder::new(),
                tracker,
                slam_rs::frontend::detect::DetectorScratch::with_scanner(Box::new(
                    RefuseAfterTracking {
                        inner: Default::default(),
                        refuse: refuse.clone(),
                    },
                )),
            )
            .unwrap(),
            pool,
        )
        .unwrap();
        let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, options);
        for step in 0..6 {
            let images = [dotted_image(step), dotted_image(step + 1)];
            if step == 2 {
                let committed = faulty.frame().clone();
                refuse.store(true, Ordering::SeqCst);
                assert!(
                    faulty
                        .process_frame(step.into(), &images, &PosePrediction::default(), &[])
                        .is_err()
                );
                assert_eq!(faulty.frame(), &committed);
            } else {
                clean
                    .process_frame(step.into(), &images, &PosePrediction::default(), &[])
                    .unwrap();
                faulty
                    .process_frame(step.into(), &images, &PosePrediction::default(), &[])
                    .unwrap();
                assert_eq!(
                    faulty.frame(),
                    clean.frame(),
                    "frame {step}, threads {threads}"
                );
                assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
                for camera in 0..2 {
                    assert_eq!(faulty.cell_counts(camera), clean.cell_counts(camera));
                }
            }
        }
    }
}
