//! kornia's FAST candidate set is exactly `corner_score_9 > threshold` after an
//! in-block local-max filter.
//!
//! The one CPU-gate guard on the model the GPU corner kernel translates: it is
//! that equivalence which lets one dense score image answer every rung of the
//! detector's threshold ladder, and it is checked here against kornia itself at
//! five rungs and two widths, so a kornia bump that moved the candidate set
//! would fail before any GPU test ran.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::{FastCorner, Rect as KorniaRect, fast_detect_rect_u8};
use slam_rs::frontend::detect::{
    FAST_BORDER, FAST_FILTER_LANES, FAST_RING_COLUMN, FAST_RING_ROW, block_filter_end,
};

mod common;

use common::cornered_bytes;

/// `corner_score_9_scalar` (`kornia fast.rs:838`).
fn corner_score_9(gray: &[u8], width: usize, x: usize, y: usize) -> u8 {
    let center = i32::from(gray[y * width + x]);
    let mut dark = [0i32; 16];
    let mut bright = [0i32; 16];
    for k in 0..16 {
        let p = i32::from(
            gray[(y as i32 + FAST_RING_ROW[k]) as usize * width
                + (x as i32 + FAST_RING_COLUMN[k]) as usize],
        );
        dark[k] = (center - p).max(0);
        bright[k] = (p - center).max(0);
    }
    let mut dark_score = 0i32;
    let mut bright_score = 0i32;
    let mut k = 0;
    while k < 16 {
        let mut core_d = dark[(k + 1) & 15];
        let mut core_b = bright[(k + 1) & 15];
        for i in 2..=8 {
            core_d = core_d.min(dark[(k + i) & 15]);
            core_b = core_b.min(bright[(k + i) & 15]);
        }
        dark_score = dark_score
            .max(core_d.min(dark[k & 15]))
            .max(core_d.min(dark[(k + 9) & 15]));
        bright_score = bright_score
            .max(core_b.min(bright[k & 15]))
            .max(core_b.min(bright[(k + 9) & 15]));
        k += 2;
    }
    dark_score.max(bright_score) as u8
}

#[test]
fn the_model_reproduces_kornia() {
    for width in [960usize, 512] {
        let height = 240usize;
        // The same field the GPU corner tests run, so the two say the same
        // thing about the same pixels.
        let bytes = cornered_bytes(width, height);
        let gray: Image<u8, 1> =
            Image::from_size_slice(ImageSize { width, height }, &bytes).unwrap();
        let margin = FAST_BORDER;
        let col_end = width - margin;
        // kornia's block alignment and its unfiltered tail, from the one place
        // the port keeps them: the GPU kernel is launched off the same call.
        let (filtered_end, use_filter) = block_filter_end(width);

        let mut scores = vec![0u8; width * height];
        for y in margin..height - margin {
            for x in margin..col_end {
                scores[y * width + x] = corner_score_9(&bytes, width, x, y);
            }
        }

        for threshold in [40i32, 20, 10, 5, 1] {
            let mut model: Vec<(usize, usize, u8)> = Vec::new();
            for y in margin..height - margin {
                for x in margin..col_end {
                    let s = scores[y * width + x];
                    if i32::from(s) <= threshold {
                        continue;
                    }
                    let keep = if use_filter && x < filtered_end {
                        let lane = (x - margin) % FAST_FILTER_LANES;
                        let left = if lane == 0 {
                            0
                        } else {
                            scores[y * width + x - 1]
                        };
                        let right = if lane == FAST_FILTER_LANES - 1 {
                            0
                        } else {
                            scores[y * width + x + 1]
                        };
                        s > left && s > right
                    } else {
                        true
                    };
                    if keep {
                        model.push((y, x, s));
                    }
                }
            }
            let kornia: Vec<(usize, usize, u8)> = fast_detect_rect_u8(
                &gray,
                KorniaRect {
                    x: 0,
                    y: margin,
                    w: width,
                    h: height - 2 * margin,
                },
                threshold as f32,
                9,
                margin,
            )
            .into_iter()
            .map(|c: FastCorner| {
                (
                    c.xy[1] as usize,
                    c.xy[0] as usize,
                    (c.response * 255.0).round() as u8,
                )
            })
            .collect();
            assert_eq!(
                model.len(),
                kornia.len(),
                "width {width} threshold {threshold}: model {} corners, kornia {}",
                model.len(),
                kornia.len()
            );
            assert_eq!(model, kornia, "width {width} threshold {threshold}");
            println!(
                "width {width} threshold {threshold}: {} corners, identical",
                model.len()
            );
        }
    }
}
