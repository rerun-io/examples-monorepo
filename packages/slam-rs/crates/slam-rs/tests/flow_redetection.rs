//! `FrameToFrameOpticalFlow`: the redetection gate (D75,
//! `port.redetect_survivor_ratio`), on the synthetic rig of `tests/common`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use slam_rs::config::VioConfig;
use slam_rs::frontend::detect::{Masks, Rect};
use slam_rs::frontend::flow::*;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::image::ImageU16;
use slam_rs::types::KeypointId;

mod common;

use common::flow::{failing_frontend_with_ratio, frontend};
use common::{
    FLOW_HEIGHT as HEIGHT, FLOW_WIDTH as WIDTH, dotted_image, flow_config as config,
    flow_rig as rig,
};

/// A frontend on the synthetic rig with the redetect gate set.
fn gated_frontend(cameras: usize, ratio: f32) -> FrameToFrameOpticalFlow<Pattern51> {
    let gated: VioConfig = VioConfig {
        port_redetect_survivor_ratio: ratio,
        ..config()
    };
    FrameToFrameOpticalFlow::new(gated, &rig(cameras), FrontendOptions::default()).unwrap()
}

/// At the default survivor ratio, detection runs every frameset (D75).
/// This control distinguishes gated detection from a scene with no new corners.
#[test]
fn the_default_config_detects_on_every_frameset() {
    assert_eq!(VioConfig::default().port_redetect_survivor_ratio, 0.0);
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let mut watermark: u64 = 0;
    for shift in 0..6 {
        let images: [ImageU16; 2] = [dotted_image(shift), dotted_image(shift)];
        flow.process_frame(i64::from(shift), &images, &PosePrediction::default(), &[])
            .unwrap();
        assert!(
            flow.last_keypoint_id() > watermark,
            "frameset {shift} detected nothing"
        );
        watermark = flow.last_keypoint_id();
    }
}

/// A ratio nothing can fall below detects once and then never again, which is
/// what says the gate is really the only thing deciding.
#[test]
fn a_survivor_ratio_nothing_reaches_detects_only_on_the_first_frameset() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = gated_frontend(2, 1e-6);
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let after_first: u64 = flow.last_keypoint_id();
    assert!(after_first > 0, "the first frameset must detect");

    for shift in 1..6 {
        let moved: [ImageU16; 2] = [dotted_image(shift), dotted_image(shift)];
        flow.process_frame(i64::from(shift), &moved, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(
            flow.last_keypoint_id(),
            after_first,
            "frameset {shift} detected while gated"
        );
    }
}

/// The gate is one decision for the whole rig, never a camera at a time: a
/// skipped frameset leaves camera 1 with no new ids either, because the
/// cross-camera match and the non-overlap pass are both inside `addPoints`.
#[test]
fn a_skipped_frameset_leaves_no_camera_detected() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = gated_frontend(3, 1e-6);
    let images: [ImageU16; 3] = [dotted_image(0), dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let mut known: Vec<Vec<KeypointId>> = flow
        .frame()
        .cameras
        .iter()
        .map(|camera| camera.ids.clone())
        .collect();
    assert!(known.iter().all(|ids| !ids.is_empty()));

    for shift in 1..4 {
        let moved: [ImageU16; 3] = [
            dotted_image(shift),
            dotted_image(shift),
            dotted_image(shift),
        ];
        flow.process_frame(i64::from(shift), &moved, &PosePrediction::default(), &[])
            .unwrap();
        for (camera, ids) in flow.frame().cameras.iter().enumerate() {
            let fresh: Vec<KeypointId> = ids
                .ids
                .iter()
                .filter(|id| !known[camera].contains(id))
                .copied()
                .collect();
            assert!(
                fresh.is_empty(),
                "camera {camera} gained {fresh:?} on a skipped frameset"
            );
        }
        known = flow
            .frame()
            .cameras
            .iter()
            .map(|camera| camera.ids.clone())
            .collect();
    }
}

/// Remove exact old positions while leaving other corners available to detect.
fn masks_leaving(frame: &FlowFrame, survivors: usize) -> [Masks; 1] {
    [Masks {
        masks: (survivors..frame.cameras[0].len())
            .map(|index| {
                let position = frame.cameras[0].transforms.translation(index);
                Rect {
                    x: position.x - 0.25,
                    y: position.y - 0.25,
                    w: 0.5,
                    h: 0.5,
                }
            })
            .collect(),
    }]
}

#[test]
fn redetection_resumes_only_below_the_survivor_threshold() {
    for offset in [-1, 0, 1] {
        let mut flow = gated_frontend(2, 0.5);
        let images = [dotted_image(0), dotted_image(0)];
        flow.process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        let count = flow.frame().cameras[0].len();
        assert!(count >= 4 && count % 2 == 0);
        let survivors = (count as isize / 2 + offset) as usize;
        let masks = masks_leaving(flow.frame(), survivors);
        let watermark = flow.last_keypoint_id();
        flow.process_frame(1, &images, &PosePrediction::default(), &masks)
            .unwrap();
        assert_eq!(
            flow.frame().cameras[0]
                .ids
                .iter()
                .filter(|id| id.0 < watermark)
                .count(),
            survivors
        );
        assert_eq!(
            flow.last_keypoint_id() > watermark,
            offset < 0,
            "offset {offset}"
        );
    }
}

#[test]
fn redetection_resumes_on_the_frameset_that_loses_every_track() {
    let mut flow = gated_frontend(2, 0.5);
    let images = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let watermark = flow.last_keypoint_id();
    assert!(watermark > 0);
    let masks = masks_leaving(flow.frame(), 0);
    flow.process_frame(1, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(!flow.frame().cameras[0].is_empty());
    assert!(
        flow.frame().cameras[0]
            .ids
            .iter()
            .all(|id| id.0 >= watermark)
    );
}

#[test]
fn redetection_uses_the_latest_post_detection_count() {
    let mut flow = gated_frontend(2, 0.5);
    let images = [dotted_image(0), dotted_image(0)];
    let initial_mask = [Masks {
        masks: vec![Rect {
            x: 60.0,
            y: 0.0,
            w: WIDTH as f32,
            h: HEIGHT as f32,
        }],
    }];
    flow.process_frame(0, &images, &PosePrediction::default(), &initial_mask)
        .unwrap();
    let initial = flow.frame().cameras[0].len();
    assert!(initial >= 2);
    let masks = masks_leaving(flow.frame(), 1);
    let watermark = flow.last_keypoint_id();
    flow.process_frame(1, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(flow.last_keypoint_id() > watermark);
    let replenished = flow.frame().cameras[0].len();
    assert!(replenished > 2 * initial);
    // Above the original baseline's threshold, below the replenished one's.
    let masks = masks_leaving(flow.frame(), initial);
    let watermark = flow.last_keypoint_id();
    flow.process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(flow.last_keypoint_id() > watermark);
    // Keeping all post-detection tracks must skip, even after replenishment.
    let watermark = flow.last_keypoint_id();
    flow.process_frame(3, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.last_keypoint_id(), watermark);
}

#[test]
fn redetection_resumes_after_an_empty_initial_detection() {
    let mut flow = gated_frontend(2, 0.5);
    let blank = [
        ImageU16::zeros(WIDTH, HEIGHT).unwrap(),
        ImageU16::zeros(WIDTH, HEIGHT).unwrap(),
    ];
    flow.process_frame(0, &blank, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.last_keypoint_id(), 0);
    let images = [dotted_image(0), dotted_image(0)];
    flow.process_frame(1, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert!(!flow.frame().cameras[0].is_empty());
    let masks = masks_leaving(flow.frame(), 0);
    let watermark = flow.last_keypoint_id();
    flow.process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(flow.last_keypoint_id() > watermark);
}

#[test]
fn redetection_keeps_its_baseline_after_a_rejected_frameset() {
    let images = [dotted_image(0), dotted_image(0)];
    let mut clean = gated_frontend(2, 0.5);
    let mut faulty = failing_frontend_with_ratio(3, 0.5);
    clean
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    faulty
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let committed = faulty.frame().clone();
    let watermark = faulty.last_keypoint_id();
    let masks = masks_leaving(&committed, 1);
    assert!(
        faulty
            .process_frame(1, &images, &PosePrediction::default(), &masks)
            .is_err()
    );
    assert_eq!(faulty.frame(), &committed);
    assert_eq!(faulty.last_keypoint_id(), watermark);
    // An unchanged retry skips: a zeroed baseline would incorrectly detect.
    faulty
        .process_frame(1, &images, &PosePrediction::default(), &[])
        .unwrap();
    clean
        .process_frame(1, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(faulty.last_keypoint_id(), watermark);
    let masks = masks_leaving(faulty.frame(), 1);
    faulty
        .process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    clean
        .process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(faulty.last_keypoint_id() > watermark);
    assert_eq!(faulty.frame(), clean.frame());
    assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
}
