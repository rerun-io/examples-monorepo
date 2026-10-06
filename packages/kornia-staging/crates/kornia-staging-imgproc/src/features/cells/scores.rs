#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))] // NEON kernels check bounds before vector loads/stores.
//! Single-pass FAST-9 scoring and selection in each eligible cell.
use super::grid::{KEY_ROW_SHIFT, KEY_SCORE_SHIFT};
use super::{
    block_filter_end, CellSelect, Occupancy, EDGE_THRESHOLD, FAST_BORDER, FAST_FILTER_LANES,
    FAST_RING_COLUMN, FAST_RING_ROW, NO_CELL_WINNER,
};
use kornia_image::Image;

/// Reused across cells and frames; no per-row allocations or narrowed image.
#[derive(Default)]
pub(super) struct CellScores {
    scores: Vec<u8>,
    row: Vec<u8>,
}

impl CellScores {
    pub(super) fn select(
        &mut self,
        image: &Image<u16, 1>,
        select: &CellSelect,
        eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<u32>,
    ) {
        let grid = &select.grid;
        let (cells_x, cells_y) = grid.dimensions();
        out.clear();
        out.resize(cells_x * cells_y, NO_CELL_WINNER);
        let (width, height) = (image.width(), image.height());
        // Kornia clamps its threshold to u8. A threshold of 255 admits nothing.
        let threshold = select.threshold.clamp(0, 255) as u8;
        if threshold == 255 || grid.cell <= 2 * FAST_BORDER {
            return;
        }
        let (filtered_end, use_filter) = block_filter_end(width);
        let ring: [isize; 16] = std::array::from_fn(|k| {
            FAST_RING_ROW[k] as isize * image.width() as isize + FAST_RING_COLUMN[k] as isize
        });
        for (column, row) in grid.cells() {
            let index = row * cells_x + column;
            if let Some((occupancy, masked)) = eligibility {
                if masked[index]
                    || row >= occupancy.rows
                    || column >= occupancy.columns
                    || occupancy.counts[row * occupancy.columns + column] >= 1
                {
                    continue;
                }
            }
            let first_x = grid.x_start + column * grid.cell + FAST_BORDER;
            let first_y = grid.y_start + row * grid.cell + FAST_BORDER;
            let last_x =
                (first_x + grid.cell - 2 * FAST_BORDER).min(width.saturating_sub(FAST_BORDER));
            let last_y =
                (first_y + grid.cell - 2 * FAST_BORDER).min(height.saturating_sub(FAST_BORDER));
            if first_x >= last_x || first_y >= last_y {
                continue;
            }
            let pitch = last_x - first_x + 2;
            self.scores.resize(pitch * (last_y - first_y + 2), 0);
            self.scores.fill(0);
            // Only the wide-image block filter reads scores outside the cell.
            // Its blocks stay aligned to the image, including the scalar tail.
            let row_first = if use_filter {
                (first_x - 1).max(FAST_BORDER)
            } else {
                first_x
            };
            let row_last = if use_filter {
                (last_x + 1).min(width - FAST_BORDER)
            } else {
                last_x
            };
            self.row.resize(row_last - row_first, 0);
            for y in first_y..last_y {
                score_row(image, &ring, row_first, y, threshold, &mut self.row);
                for x in first_x..last_x {
                    let offset = x - row_first;
                    let score = self.row[offset];
                    if score <= threshold {
                        continue;
                    }
                    if use_filter && x < filtered_end {
                        let lane = (x - FAST_BORDER) % FAST_FILTER_LANES;
                        if (lane != 0 && score <= self.row[offset - 1])
                            || (lane != FAST_FILTER_LANES - 1 && score <= self.row[offset + 1])
                        {
                            continue;
                        }
                    }
                    self.scores[(y - first_y + 1) * pitch + x - first_x + 1] = score;
                }
            }
            let mut best = NO_CELL_WINNER;
            for y in first_y..last_y {
                for x in first_x..last_x {
                    let i = (y - first_y + 1) * pitch + x - first_x + 1;
                    let score = self.scores[i];
                    if score == 0 {
                        continue;
                    }
                    let key = ((255 - u32::from(score)) << KEY_SCORE_SHIFT)
                        | ((y as u32) << KEY_ROW_SHIFT)
                        | x as u32;
                    if key >= best {
                        continue;
                    }
                    // The zero rim is cell-local, as in suppress_non_maxima.
                    if [
                        i - pitch - 1,
                        i - pitch,
                        i - pitch + 1,
                        i - 1,
                        i + 1,
                        i + pitch - 1,
                        i + pitch,
                        i + pitch + 1,
                    ]
                    .iter()
                    .any(|&n| self.scores[n] >= score)
                    {
                        continue;
                    }
                    let (xf, yf) = (x as f32, y as f32);
                    let dx = xf - (width / 2) as f32;
                    let dy = yf - (height / 2) as f32;
                    if select.safe_radius != 0.0 && (dx * dx + dy * dy).sqrt() >= select.safe_radius
                    {
                        continue;
                    }
                    if crate::interpolation::in_bounds_u16(image, xf, yf, EDGE_THRESHOLD) {
                        best = key;
                    }
                }
            }
            out[index] = best;
        }
    }
}

fn score_row(
    image: &Image<u16, 1>,
    ring: &[isize; 16],
    first_x: usize,
    y: usize,
    threshold: u8,
    out: &mut [u8],
) {
    let pixels = image.as_slice();
    let width = image.width();
    #[cfg(target_arch = "aarch64")]
    if out.len() >= 16 {
        // CellSelect is public: establish memory bounds even for a grid whose
        // caller supplied overflowing origins. Do not rely on debug assertions.
        assert!(
            first_x >= FAST_BORDER
                && first_x
                    .checked_add(out.len())
                    .and_then(|end| end.checked_add(FAST_BORDER))
                    .is_some_and(|end| end <= width)
        );
        assert!(
            y >= FAST_BORDER
                && y.checked_add(FAST_BORDER)
                    .is_some_and(|end| end < image.height())
        );
        let mut offset = 0;
        loop {
            // SAFETY: every centre is within the radius-three image border.
            // Each load covers 16 centres plus the same ring offset, so it
            // remains within its image row, within the dense image.
            // The store covers exactly 16 entries in out. NEON is mandatory
            // on aarch64. The final block overlaps instead of reading past
            // the window or paying for a scalar tail of up to 15 pixels.
            unsafe {
                score_neon(
                    pixels.as_ptr().add(y * width + first_x + offset),
                    ring,
                    threshold,
                    out.as_mut_ptr().add(offset),
                );
            }
            if offset + 16 == out.len() {
                return;
            }
            offset = (offset + 16).min(out.len() - 16);
        }
    }
    for (lane, score) in out.iter_mut().enumerate() {
        *score = score_scalar(pixels, y * width + first_x + lane, ring, threshold);
    }
}

/// Sixteen independent FAST scores; u16 pixels are narrowed in registers.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn score_neon(base: *const u16, ring: &[isize; 16], threshold: u8, out: *mut u8) {
    use std::arch::aarch64::*;
    // SAFETY: score_row establishes the bounds of all ring loads and the store.
    unsafe {
        let load = |offset: isize| {
            let p = base.offset(offset);
            vcombine_u8(
                vshrn_n_u16::<8>(vld1q_u16(p)),
                vshrn_n_u16::<8>(vld1q_u16(p.add(8))),
            )
        };
        let center = load(0);
        let threshold = vdupq_n_u8(threshold);
        let zero = vdupq_n_u8(0);
        let mut pixels = [zero; 16];
        for k in [0, 4, 8, 12] {
            pixels[k] = load(ring[k]);
        }
        // Any arc of nine must include two adjacent cardinal points. If no
        // lane has such a pair, all sixteen scores are below the last rung.
        let mut possible = zero;
        for k in [0, 4, 8, 12] {
            let next = (k + 4) & 15;
            possible = vorrq_u8(
                possible,
                vcgtq_u8(
                    vqsubq_u8(center, vmaxq_u8(pixels[k], pixels[next])),
                    threshold,
                ),
            );
            possible = vorrq_u8(
                possible,
                vcgtq_u8(
                    vqsubq_u8(vminq_u8(pixels[k], pixels[next]), center),
                    threshold,
                ),
            );
        }
        if vmaxvq_u8(possible) == 0 {
            vst1q_u8(out, zero);
            return;
        }
        for k in 0..16 {
            if k % 4 != 0 {
                pixels[k] = load(ring[k]);
            }
        }
        // Move subtraction outside the extrema: min(center - p) is
        // center - max(p), and min(p - center) is min(p) - center. This needs
        // one ring, not separate dark/bright rings, and two saturated
        // subtractions at the end. Share pairs and quads between the eight
        // overlapping cores [k+1..k+8]; each core serves two nine-pixel arcs.
        let mut low2 = [zero; 8];
        let mut high2 = [zero; 8];
        for k in 0..8 {
            low2[k] = vminq_u8(pixels[2 * k + 1], pixels[(2 * k + 2) & 15]);
            high2[k] = vmaxq_u8(pixels[2 * k + 1], pixels[(2 * k + 2) & 15]);
        }
        let mut low4 = [zero; 8];
        let mut high4 = [zero; 8];
        for k in 0..8 {
            low4[k] = vminq_u8(low2[k], low2[(k + 1) & 7]);
            high4[k] = vmaxq_u8(high2[k], high2[(k + 1) & 7]);
        }
        let mut brightest_min = zero;
        let mut darkest_max = vdupq_n_u8(255);
        for k in 0..8 {
            let low8 = vminq_u8(low4[k], low4[(k + 2) & 7]);
            let high8 = vmaxq_u8(high4[k], high4[(k + 2) & 7]);
            let a = pixels[2 * k];
            let b = pixels[(2 * k + 9) & 15];
            brightest_min = vmaxq_u8(brightest_min, vminq_u8(low8, vmaxq_u8(a, b)));
            darkest_max = vminq_u8(darkest_max, vmaxq_u8(high8, vminq_u8(a, b)));
        }
        let score = vmaxq_u8(
            vqsubq_u8(brightest_min, center),
            vqsubq_u8(center, darkest_max),
        );
        vst1q_u8(out, vandq_u8(score, vcgtq_u8(score, threshold)));
    }
}

/// Raw FAST strength (OpenCV's integer score plus one), zero below the rung.
fn score_scalar(pixels: &[u16], index: usize, ring: &[isize; 16], threshold: u8) -> u8 {
    let center = (pixels[index] >> 8) as u8;
    let mut bright_mask = 0u32;
    let mut dark_mask = 0u32;
    let mut dark = [0u8; 16];
    let mut bright = [0u8; 16];
    for k in 0..16 {
        let p = (pixels[index.wrapping_add_signed(ring[k])] >> 8) as u8;
        dark[k] = center.saturating_sub(p);
        bright[k] = p.saturating_sub(center);
        dark_mask |= u32::from(dark[k] > threshold) << k;
        bright_mask |= u32::from(bright[k] > threshold) << k;
    }
    // An arc must contain nine consecutive set bits in either doubled ring.
    let mut d = dark_mask | (dark_mask << 16);
    let mut b = bright_mask | (bright_mask << 16);
    for shift in [1, 2, 4] {
        d &= d >> shift;
        b &= b >> shift;
    }
    if (d & (dark_mask | (dark_mask << 16)) >> 8) | (b & (bright_mask | (bright_mask << 16)) >> 8)
        == 0
    {
        return 0;
    }
    let mut score = 0;
    for k in (0..16).step_by(2) {
        let mut core_d = dark[(k + 1) & 15];
        let mut core_b = bright[(k + 1) & 15];
        for i in 2..=8 {
            core_d = core_d.min(dark[(k + i) & 15]);
            core_b = core_b.min(bright[(k + i) & 15]);
        }
        score = score
            .max(core_d.min(dark[k].max(dark[(k + 9) & 15])))
            .max(core_b.min(bright[k].max(bright[(k + 9) & 15])));
    }
    score
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// Compare every NEON lane, including overlapping final blocks, with the
    /// scalar scorer. Full u16 noise checks that only the high byte is read.
    #[test]
    #[allow(clippy::unwrap_used)]
    fn neon_scores_match_scalar_on_random_and_textured_pixels() {
        let random = crate::test_fixtures::random_image(131, 97, 0x5f71_4213);
        let mut textured = crate::test_fixtures::zeros(960, 960);
        for (index, pixel) in textured.as_slice_mut().iter_mut().enumerate() {
            let (x, y) = ((index % 960) as f32, (index / 960) as f32);
            let waves = 128.0 + 60.0 * (x / 11.0).sin() + 50.0 * (y / 7.0).cos();
            *pixel = (waves.clamp(0.0, 255.0) as u16) << 8;
        }
        for image in [&random, &textured] {
            let ring = std::array::from_fn(|k| {
                FAST_RING_ROW[k] as isize * image.width() as isize + FAST_RING_COLUMN[k] as isize
            });
            for threshold in [1, 5, 10, 40, 127, 254, 255] {
                for len in [1, 15, 16, 17, 31, 32, 44, 64] {
                    let mut scores = vec![0; len];
                    for y in (3..image.height() - 3).step_by(7) {
                        for first_x in (3..=image.width() - 3 - len).step_by(53) {
                            score_row(image, &ring, first_x, y, threshold, &mut scores);
                            for (lane, &score) in scores.iter().enumerate() {
                                assert_eq!(
                                    score,
                                    score_scalar(
                                        image.as_slice(),
                                        y * image.width() + first_x + lane,
                                        &ring,
                                        threshold
                                    ),
                                    "at ({}, {y}), threshold {threshold}, row length {len}",
                                    first_x + lane
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
