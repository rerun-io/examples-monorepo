//! Re-anchoring on hand-built trajectories (PR f1).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use nalgebra::Vector3;

use super::*;

fn pose(x: f64, y: f64, yaw: f64) -> Se3<f64> {
    Se3::new(
        So3::exp(&Vector3::new(0.0, 0.0, yaw)),
        Vector3::new(x, y, 0.0),
    )
}

fn close(a: &Se3<f64>, b: &Se3<f64>, tolerance: f64) -> bool {
    (a.translation - b.translation).norm() < tolerance
        && (a.rotation.inverse() * b.rotation).log().norm() < tolerance
}

/// Raw frames every 10 ns along x with a sideways wiggle and a turning yaw,
/// so a frame is never on the line between its two keyframes.
fn raw_frames() -> Vec<(i64, Se3<f64>)> {
    (0..=10)
        .map(|i| {
            let x = i as f64;
            (10 * i, pose(x, 0.3 * (x * 1.3).sin(), 0.05 * x))
        })
        .collect()
}

#[test]
fn a_frame_between_two_corrected_keyframes_moves_by_the_relative_correction_not_a_lerp() {
    // B13: the keyframes at 0 and 100 ns both moved by the same correction.
    // The frame must keep its own wiggle; a lerp of the two keyframe poses would
    // put it on their straight line.
    let frames = raw_frames();
    let correction = pose(0.5, -2.0, 0.2);
    let keyframes: Vec<(i64, Se3<f64>)> = [0usize, 10]
        .iter()
        .map(|&row| (frames[row].0, correction * frames[row].1))
        .collect();
    let anchored = reanchor(&frames, &keyframes).unwrap();
    for (&(_, raw), result) in frames.iter().zip(&anchored) {
        assert!(close(result, &(correction * raw), 1e-12), "{result:?}");
    }
}

#[test]
fn two_sided_blends_the_two_corrections_by_time_and_hits_keyframes_exactly() {
    let frames = raw_frames();
    let left = pose(0.0, 0.0, 0.0);
    let right = pose(1.0, 0.5, 0.4);
    let keyframes = vec![
        (frames[2].0, left * frames[2].1),
        (frames[6].0, right * frames[6].1),
    ];
    let anchored = reanchor(&frames, &keyframes).unwrap();
    // Keyframes: their optimized pose, bit for bit.
    assert_eq!(anchored[2], keyframes[0].1);
    assert_eq!(anchored[6], keyframes[1].1);
    // Before the first / after the last keyframe: one-sided.
    assert!(close(&anchored[0], &(left * frames[0].1), 1e-12));
    assert!(close(&anchored[9], &(right * frames[9].1), 1e-12));
    // Frame 3 sits a quarter of the way from keyframe 2 to keyframe 6.
    let (l, r) = (left * frames[3].1, right * frames[3].1);
    let expected_t = l.translation * 0.75 + r.translation * 0.25;
    assert!((anchored[3].translation - expected_t).norm() < 1e-12);
    let expected_yaw = 0.05 * 3.0 + 0.25 * 0.4;
    assert!((anchored[3].rotation.log() - Vector3::new(0.0, 0.0, expected_yaw)).norm() < 1e-12);
}

#[test]
fn no_keyframe_returns_the_frames_and_bad_inputs_are_refused() {
    let frames = raw_frames();
    let anchored = reanchor(&frames, &[]).unwrap();
    assert!(anchored.iter().zip(&frames).all(|(a, f)| *a == f.1));
    assert_eq!(
        reanchor(&frames, &[(15, Se3::identity())]),
        Err(AnchorError::KeyframeNotAFrame { t_ns: 15 })
    );
    let mut unordered = frames.clone();
    unordered.swap(1, 2);
    assert!(matches!(
        reanchor(&unordered, &[]),
        Err(AnchorError::NotIncreasing { what: "frame", .. })
    ));
}

fn correction(epoch: u64, moved: &[(FrameId, Se3<f64>, Se3<f64>)]) -> MapCorrection {
    MapCorrection {
        epoch,
        poses: moved
            .iter()
            .map(|&(id, _, corrected)| (id, corrected))
            .collect(),
        raw: moved.iter().map(|&(id, raw, _)| (id, raw)).collect(),
    }
}

#[test]
fn online_is_identity_without_a_correction_and_one_sided_from_the_last_keyframe() {
    let board = Arc::new(CorrectionBoard::default());
    let mut anchor = OnlineAnchor::new(Arc::clone(&board));
    let raw = pose(3.0, 1.0, 0.3);
    // No keyframe, then no correction: the raw bits.
    assert_eq!(anchor.correct(&raw), raw);
    anchor.keyframe(100);
    assert_eq!(anchor.correct(&raw), raw);

    // Keyframe 100 moved by `delta`: a later frame moves with it.
    let kf_raw = pose(2.5, 0.8, 0.25);
    let delta = pose(-0.4, 0.2, -0.1);
    board.publish(correction(1, &[(100, kf_raw, delta * kf_raw)]));
    assert!(close(&anchor.correct(&raw), &(delta * raw), 1e-12));

    // A newer keyframe the correction did not move: one-sided from it, so the
    // older keyframe's correction no longer applies.
    anchor.keyframe(200);
    assert_eq!(anchor.correct(&raw), raw);
}

#[test]
fn online_reads_only_corrections_published_so_far_and_never_an_older_epoch() {
    let board = Arc::new(CorrectionBoard::default());
    let mut anchor = OnlineAnchor::new(Arc::clone(&board));
    anchor.keyframe(100);
    let raw = pose(1.0, 0.0, 0.0);
    let kf_raw = pose(0.9, 0.0, 0.0);
    // Before the back end publishes, the frame keeps its raw pose: causal.
    let before = anchor.correct(&raw);
    assert_eq!(before, raw);
    let first = pose(0.1, 0.0, 0.0);
    let second = pose(0.2, 0.0, 0.0);
    board.publish(correction(2, &[(100, kf_raw, second * kf_raw)]));
    // An older epoch arriving late is ignored.
    board.publish(correction(1, &[(100, kf_raw, first * kf_raw)]));
    assert_eq!(board.latest().epoch, 2);
    assert!(close(&anchor.correct(&raw), &(second * raw), 1e-12));
    // The answer given before the publish stays what it was.
    assert_eq!(before, raw);
}

#[test]
fn a_delta_needs_both_the_raw_and_the_corrected_pose() {
    let mut half = correction(1, &[(7, pose(1.0, 0.0, 0.0), pose(2.0, 0.0, 0.0))]);
    assert!(close(&half.delta(7).unwrap(), &pose(1.0, 0.0, 0.0), 1e-12));
    assert!(half.delta(8).is_none());
    half.raw = BTreeMap::new();
    assert!(half.delta(7).is_none());
}
