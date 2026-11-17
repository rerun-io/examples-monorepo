//! The reprojection pass's search (`searchInWindow`, `loop_closing.cpp:643-694`):
//! FAST-9 corners with OpenCV's scores and 3x3 non-maximum suppression in a
//! small window of the current keyframe's image, then the corner whose BRIEF
//! descriptor matches the landmark's.
//!
//! The image is only read: a redetected corner becomes a match of the caller's
//! local copy, never a keypoint of the keyframe (B8).

use kornia_image::Image;
use kornia_staging_imgproc::features::{
    BriefConfig, Descriptor256, MatchConfig, describe_keypoints_u16, match_mutual,
};
use kornia_staging_imgproc::interpolation::in_bounds_u16;
use nalgebra::Vector2;

/// OpenCV's 16-pixel Bresenham ring of radius 3, `(dx, dy)`, in circular order.
const RING: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Contiguous ring pixels a FAST-9 corner needs.
const ARC: usize = 9;

/// The OpenCV FAST score of the pixel at `(x, y)` of an 8-bit `side`-wide
/// patch, or `None` when it is no corner at `threshold`. A corner has nine
/// contiguous ring pixels all brighter than `centre + threshold` or all darker
/// than `centre - threshold`; its score is the largest threshold it would still
/// pass (`cv::FAST`'s `cornerScore<16>`).
fn corner_score(patch: &[u8], side: usize, x: usize, y: usize, threshold: u8) -> Option<u32> {
    let centre = i32::from(patch[y * side + x]);
    let diff: [i32; 16] = RING.map(|(dx, dy)| {
        centre - i32::from(patch[(y as i32 + dy) as usize * side + (x as i32 + dx) as usize])
    });
    // Largest min over any arc of nine, darker ring (diff > 0) or brighter (diff < 0).
    let mut best: i32 = 0;
    for start in 0..16 {
        let arc = (0..ARC).map(|k| diff[(start + k) % 16]);
        let darker = arc.clone().min().unwrap_or(0);
        let brighter = arc.map(|d| -d).min().unwrap_or(0);
        best = best.max(darker).max(brighter);
    }
    (best > i32::from(threshold)).then_some((best - 1) as u32)
}

/// FAST-9 corners of an 8-bit `side` x `side` patch, at least 3 pixels from
/// its edge, row-major, with OpenCV's non-maximum suppression: a corner stays
/// when its score beats all eight neighbours' (a non-corner scores 0).
pub(crate) fn fast_corners(patch: &[u8], side: usize, threshold: u8) -> Vec<[usize; 2]> {
    if side < 7 {
        return Vec::new();
    }
    let mut scores: Vec<Option<u32>> = vec![None; side * side];
    for y in 3..side - 3 {
        for x in 3..side - 3 {
            scores[y * side + x] = corner_score(patch, side, x, y, threshold);
        }
    }
    let mut corners: Vec<[usize; 2]> = Vec::new();
    for y in 3..side - 3 {
        for x in 3..side - 3 {
            let Some(score) = scores[y * side + x] else {
                continue;
            };
            let peak = (y - 1..=y + 1)
                .flat_map(|ny| (x - 1..=x + 1).map(move |nx| (nx, ny)))
                .filter(|&(nx, ny)| (nx, ny) != (x, y))
                .all(|(nx, ny)| score > scores[ny * side + nx].unwrap_or(0));
            if peak {
                corners.push([x, y]);
            }
        }
    }
    corners
}

/// What the window search needs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowSearch {
    /// Side of the window in pixels.
    pub side: u32,
    /// FAST threshold on 8-bit intensities.
    pub fast_threshold: u8,
    /// The descriptor test reading.
    pub brief: BriefConfig,
    /// Acceptance of the descriptor match.
    pub matching: MatchConfig,
}

/// The pixel near `center` whose descriptor matches `target`, if any.
///
/// The `side` x `side` window starts at `center - side / 2` (truncated to whole
/// pixels; RootSLAM adds the fractional part to every corner, a sub-pixel shift
/// this does not copy). FAST runs on the window's 8-bit intensities (`u16 >>
/// 8`, as Basalt's `detectKeypointsFAST`); corners closer than the BRIEF border
/// to the image edge get no descriptor. The best corner must pass `matching`
/// both ways.
pub(crate) fn search_window(
    image: &Image<u16, 1>,
    center: Vector2<f64>,
    target: &Descriptor256,
    search: &WindowSearch,
) -> Option<Vector2<f32>> {
    let half = f64::from(search.side) / 2.0;
    if !in_bounds_u16(image, center.x as f32, center.y as f32, half as f32) {
        return None;
    }
    let (x0, y0) = ((center.x - half) as usize, (center.y - half) as usize);
    let side = search.side as usize;
    let width = image.width();
    let pixels = image.as_slice();
    let patch: Vec<u8> = (0..side)
        .flat_map(|y| (0..side).map(move |x| (y, x)))
        .map(|(y, x)| (pixels[(y0 + y) * width + x0 + x] >> 8) as u8)
        .collect();
    let corners: Vec<[f32; 2]> = fast_corners(&patch, side, search.fast_threshold)
        .into_iter()
        .map(|[x, y]| [(x0 + x) as f32, (y0 + y) as f32])
        .collect();
    let described: Vec<([f32; 2], Descriptor256)> = corners
        .iter()
        .zip(describe_keypoints_u16(image, &corners, search.brief))
        .filter_map(|(&corner, described)| Some((corner, described?.descriptor)))
        .collect();
    let descriptors: Vec<Descriptor256> = described.iter().map(|(_, d)| *d).collect();
    let (_, index) =
        *match_mutual(std::slice::from_ref(target), &descriptors, search.matching).first()?;
    let [x, y] = described[index].0;
    Some(Vector2::new(x, y))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// A flat patch of `background` with one pixel of `dot` at (8, 8).
    fn dot_patch(background: u8, dot: u8) -> Vec<u8> {
        let mut patch = vec![background; 256];
        patch[8 * 16 + 8] = dot;
        patch
    }

    #[test]
    fn a_flat_patch_has_no_corner_and_a_dot_is_one() {
        assert!(fast_corners(&[50u8; 256], 16, 15).is_empty());
        assert_eq!(fast_corners(&dot_patch(20, 200), 16, 15), vec![[8, 8]]);
        assert_eq!(
            fast_corners(&dot_patch(200, 20), 16, 15),
            vec![[8, 8]],
            "a dark dot too"
        );
    }

    #[test]
    fn the_threshold_and_the_contrast_decide() {
        // Contrast 15 is not above threshold 15 (OpenCV's strict test).
        assert!(fast_corners(&dot_patch(30, 45), 16, 15).is_empty());
        assert_eq!(fast_corners(&dot_patch(30, 46), 16, 15), vec![[8, 8]]);
        assert_eq!(fast_corners(&dot_patch(30, 40), 16, 5), vec![[8, 8]]);
    }

    #[test]
    fn suppression_keeps_one_of_two_neighbouring_corners() {
        // A single bright pixel scores at its centre only; its 8 neighbours see
        // it on their ring at most three times, so they are not corners.
        let mut patch = vec![10u8; 256];
        patch[8 * 16 + 8] = 250;
        assert_eq!(fast_corners(&patch, 16, 15), vec![[8, 8]]);
        // Two equal peaks side by side both die (strictly greater, OpenCV).
        patch[8 * 16 + 9] = 250;
        let corners = fast_corners(&patch, 16, 15);
        assert!(
            !corners.contains(&[8, 8]) && !corners.contains(&[9, 8]),
            "{corners:?}"
        );
    }
}
