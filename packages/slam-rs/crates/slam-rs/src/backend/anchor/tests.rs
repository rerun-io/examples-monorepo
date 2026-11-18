//! Re-anchoring on hand-built trajectories (PR f1).

#![allow(clippy::unwrap_used)]

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

#[test]
fn keyframes_before_the_first_frame_are_left_out_and_later_strays_still_refused() {
    // An inertial bootstrap's window: keyframes at -20 and -10 ns reach the map, the first raw pose is at 0.
    let frames = raw_frames();
    let window = [(-20, pose(9.0, 9.0, 1.0)), (-10, pose(9.0, 9.0, 1.0))];
    let keyframes = [(20, pose(2.5, 0.0, 0.1)), (60, pose(6.5, 0.2, 0.3))];
    let with_window: Vec<(i64, Se3<f64>)> = window.iter().chain(&keyframes).copied().collect();
    assert_eq!(reanchor(&frames, &with_window).unwrap(), reanchor(&frames, &keyframes).unwrap());
    let stray: Vec<(i64, Se3<f64>)> = window.iter().copied().chain([(15, Se3::identity())]).collect();
    assert_eq!(reanchor(&frames, &stray), Err(AnchorError::KeyframeNotAFrame { t_ns: 15 }));
    // With no frame at all every keyframe is left out.
    assert!(reanchor(&[], &with_window).unwrap().is_empty());
}
