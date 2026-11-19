//! Per-frame segmentation masks (people, the operator, the stick).
//!
//! The monorepo's mask layer stores a mask at the resolution it was computed at
//! (for example 512 x 512 from SAM) with a scale to the image's pixels. An image
//! pixel `(c, r)` is masked when the mask pixel that covers its centre is
//! nonzero: `mask[floor((r + 0.5) / sy), floor((c + 0.5) / sx)]`, with
//! `sx = image width / mask width` (the same for rows), computed in integers as
//! `(2c + 1) * mask width / (2 * image width)`. That is how the Rerun viewer
//! draws the scaled mask over the image, and with scale 1 it is the pixel itself.
//!
//! The frontend keeps no keypoint on a masked pixel
//! ([`crate::frontend::flow::TrackKillMask`]): a detected corner there is
//! dropped (its cell stays empty for that frameset), and so is a track, temporal
//! or stereo, whose new position lands there.

/// One camera's mask at its own resolution, row-major; nonzero = no features.
#[derive(Debug, Clone, Copy)]
pub struct PixelMask<'a> {
    /// Mask width in pixels.
    pub width: usize,
    /// Mask height in pixels.
    pub height: usize,
    /// `width * height` bytes.
    pub data: &'a [u8],
}

#[cfg(test)]
mod tests {
    use crate::frontend::flow::TrackKillMask;
    use proptest::prelude::*;

    /// The rule in the module docs, independently: the mask pixel `j` whose span `[j * s, (j + 1) * s)` holds the
    /// pixel centre `c + 0.5`, found by search with exact integer comparisons (`s = image / mask`). An f64 oracle is
    /// not exact: 98 / 20 = 4.9 has no binary form, and a centre on a span's edge then floors one pixel low.
    fn masked(mask: &TrackKillMask, x: f32, y: f32) -> bool {
        let covering = |pixel: usize, image: usize, size: usize| {
            (0..size)
                .find(|&j| {
                    2 * j * image <= (2 * pixel + 1) * size
                        && (2 * pixel + 1) * size < 2 * (j + 1) * image
                })
                .unwrap_or(size - 1)
        };
        let mask_column = covering(x.round() as usize, mask.image_width, mask.width);
        let mask_row = covering(y.round() as usize, mask.image_height, mask.height);
        mask.killed[mask_row * mask.width + mask_column] != 0
    }

    #[test]
    fn a_native_resolution_mask_is_read_through_its_scale() {
        // A 4x3 mask over a 16x12 image (scale 4): mask pixel (2, 1) covers image columns 8..11, rows 4..7.
        let mut killed = vec![0u8; 4 * 3];
        killed[4 + 2] = 1;
        let mask = TrackKillMask {
            width: 4,
            height: 3,
            image_width: 16,
            image_height: 12,
            killed,
        };
        assert!(mask.kills(8.0, 4.0) && mask.kills(11.4, 7.4));
        assert!(!mask.kills(7.4, 4.0) && !mask.kills(8.0, 3.4) && !mask.kills(11.6, 4.0));
        assert!(
            mask.kills(-1.0, 4.0) && mask.kills(16.0, 4.0),
            "off the image is dropped"
        );
    }

    proptest! {
        #[test]
        fn matches_the_pixel_rule(
            (width, height) in (1usize..30, 1usize..30),
            (image_width, image_height) in (30usize..200, 30usize..200),
            seed in any::<u64>(),
            points in prop::collection::vec((0.0f32..1.0, 0.0f32..1.0), 1..50),
        ) {
            let mut state = seed | 1;
            let killed: Vec<u8> = (0..width * height)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    u8::from(state % 3 == 0)
                })
                .collect();
            let mask = TrackKillMask { width, height, image_width, image_height, killed };
            for (u, v) in points {
                // Inside the image after rounding.
                let (x, y) = (u * (image_width as f32 - 1.0), v * (image_height as f32 - 1.0));
                prop_assert_eq!(mask.kills(x, y), masked(&mask, x, y), "({}, {})", x, y);
            }
        }
    }
}
