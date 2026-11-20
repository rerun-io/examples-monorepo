//! Per-frame segmentation masks (people, the operator, the stick) as the
//! frontend's per-frame cell masks, Basalt's `img->masks`.
//!
//! The monorepo's mask layer stores a mask at the resolution it was computed at
//! (for example 512 x 512 from SAM) with a scale to the image's pixels. An image
//! pixel `(c, r)` is masked when the mask pixel that covers its centre is
//! nonzero: `mask[floor((r + 0.5) / sy), floor((c + 0.5) / sx)]`, clamped into
//! the mask, with `sx = image width / mask width` (the same for rows). That is
//! how the Rerun viewer draws the scaled mask over the image, and with scale 1
//! it is the pixel itself.
//!
//! The frontend masks whole detection cells: a cell that holds one masked image
//! pixel gets no new keypoint, and a track that enters it is dropped. Whole
//! cells keep the one-corner-per-cell selection path (`cell_masks` in
//! kornia-staging), which refuses a rectangle that is not exactly one cell.
use kornia_image::ImageSize;
use kornia_staging_imgproc::features::{CellGrid, CellMasks, MaskRect};

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

/// For each image pixel index along one axis, the mask index that covers its
/// centre: `min(floor((i + 0.5) * mask / image), mask - 1)`.
fn covering_indices(image: usize, mask: usize) -> Vec<usize> {
    (0..image)
        // Integer form of floor((i + 0.5) / (image / mask)), exact for any size.
        .map(|i| ((2 * i + 1) * mask / (2 * image)).min(mask - 1))
        .collect()
}

/// The whole cells of `grid` that hold at least one masked pixel of an image
/// `image` in size, as one rectangle per cell.
///
/// Pixels outside the grid's cells (the centred grid's margins, less than one
/// cell at each edge) are not looked at: no cell is detected there.
///
/// # Panics
/// If `mask` is empty or `mask.data` is not `width * height` bytes; the caller
/// checks both.
pub fn masked_cells(mask: PixelMask<'_>, image: ImageSize, grid: &CellGrid) -> CellMasks {
    assert!(
        mask.width > 0
            && mask.height > 0
            && mask.width.checked_mul(mask.height) == Some(mask.data.len())
    );
    let columns: Vec<usize> = covering_indices(image.width, mask.width);
    let rows: Vec<usize> = covering_indices(image.height, mask.height);
    let cell: usize = grid.cell;
    // The distinct mask indices one cell's pixels sample along an axis. They are contiguous when the mask is no finer
    // than the image, and skip indices when it is finer.
    let sampled = |covering: &[usize], start: usize, end: usize| -> Vec<usize> {
        let mut indices: Vec<usize> = covering[start..(start + cell).min(end)].to_vec();
        indices.dedup();
        indices
    };
    let (grid_columns, grid_rows) = grid.dimensions();
    let cell_columns: Vec<Vec<usize>> = (0..grid_columns)
        .map(|column| sampled(&columns, grid.x_start + column * cell, image.width))
        .collect();
    let cell_rows: Vec<Vec<usize>> = (0..grid_rows)
        .map(|row| sampled(&rows, grid.y_start + row * cell, image.height))
        .collect();
    let mut out = CellMasks::default();
    for (column, row) in grid.cells() {
        let x: usize = grid.x_start + column * cell;
        let y: usize = grid.y_start + row * cell;
        let masked = cell_rows[row].iter().any(|&mask_row| {
            let values: &[u8] = &mask.data[mask_row * mask.width..(mask_row + 1) * mask.width];
            cell_columns[column]
                .iter()
                .any(|&mask_column| values[mask_column] != 0)
        });
        if masked {
            out.masks.push(MaskRect {
                x: x as f32,
                y: y as f32,
                w: cell as f32,
                h: cell as f32,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use proptest::prelude::*;

    fn grid(width: usize, height: usize, cell: usize) -> CellGrid {
        CellGrid::new(width, height, cell).unwrap()
    }

    /// The rule stated in the module docs, pixel by pixel, independently: the mask pixel `j` whose span
    /// `[j * s, (j + 1) * s)` holds the pixel centre, found by search with exact integer comparisons. An f64 oracle is
    /// not exact (98 / 20 = 4.9 has no binary form: a centre on a span's edge floors one pixel low).
    fn brute_force(mask: PixelMask<'_>, image: ImageSize, grid: &CellGrid) -> CellMasks {
        let covering = |pixel: usize, image: usize, size: usize| {
            (0..size)
                .find(|&j| {
                    2 * j * image <= (2 * pixel + 1) * size
                        && (2 * pixel + 1) * size < 2 * (j + 1) * image
                })
                .unwrap_or(size - 1)
        };
        let masked_pixel = |c: usize, r: usize| {
            let mc = covering(c, image.width, mask.width);
            let mr = covering(r, image.height, mask.height);
            mask.data[mr * mask.width + mc] != 0
        };
        let mut out = CellMasks::default();
        for (column, row) in grid.cells() {
            let (x, y) = (
                grid.x_start + column * grid.cell,
                grid.y_start + row * grid.cell,
            );
            let hit = (y..(y + grid.cell).min(image.height))
                .any(|r| (x..(x + grid.cell).min(image.width)).any(|c| masked_pixel(c, r)));
            if hit {
                out.masks.push(MaskRect {
                    x: x as f32,
                    y: y as f32,
                    w: grid.cell as f32,
                    h: grid.cell as f32,
                });
            }
        }
        out
    }

    #[test]
    fn full_size_mask_marks_the_cell_of_its_pixel() {
        let image = ImageSize {
            width: 100,
            height: 80,
        };
        let grid = grid(100, 80, 20);
        let mut data = vec![0u8; 100 * 80];
        // (45, 33) lies in the cell whose corner is (x_start + 2 * 20, y_start + 1 * 20).
        data[33 * 100 + 45] = 1;
        let cells = masked_cells(
            PixelMask {
                width: 100,
                height: 80,
                data: &data,
            },
            image,
            &grid,
        );
        assert_eq!(cells.masks.len(), 1);
        assert!(cells.in_bounds(45.0, 33.0));
        assert!(!cells.in_bounds(25.0, 33.0));
    }

    #[test]
    fn empty_mask_masks_nothing() {
        let image = ImageSize {
            width: 64,
            height: 64,
        };
        let data = vec![0u8; 16 * 16];
        let cells = masked_cells(
            PixelMask {
                width: 16,
                height: 16,
                data: &data,
            },
            image,
            &grid(64, 64, 16),
        );
        assert!(cells.masks.is_empty());
    }

    #[test]
    fn a_native_resolution_mask_is_read_through_its_scale() {
        // A 512 mask over a 1920 image (scale 3.75): the masked column 256 covers
        // image columns 960..963, which sit in one cell of a 30 px grid.
        let image = ImageSize {
            width: 1920,
            height: 1920,
        };
        let grid = grid(1920, 1920, 30);
        let mut data = vec![0u8; 512 * 512];
        data[256 * 512 + 256] = 255;
        let cells = masked_cells(
            PixelMask {
                width: 512,
                height: 512,
                data: &data,
            },
            image,
            &grid,
        );
        assert_eq!(
            cells,
            brute_force(
                PixelMask {
                    width: 512,
                    height: 512,
                    data: &data
                },
                image,
                &grid
            )
        );
        assert!(cells.in_bounds(961.0, 961.0));
        assert!(!cells.in_bounds(900.0, 961.0));
    }

    #[test]
    fn a_span_edge_on_a_non_binary_scale_is_exact() {
        // Scale 98 / 20 = 4.9 has no binary form: centres land on span edges.
        let image = ImageSize {
            width: 98,
            height: 50,
        };
        let grid = grid(98, 50, 10);
        let mut data = vec![0u8; 20 * 15];
        for (index, value) in data.iter_mut().enumerate() {
            *value = u8::from(index % 7 == 3);
        }
        let mask = PixelMask {
            width: 20,
            height: 15,
            data: &data,
        };
        assert_eq!(
            masked_cells(mask, image, &grid),
            brute_force(mask, image, &grid)
        );
    }

    #[test]
    fn a_finer_mask_masks_only_the_pixels_the_image_samples() {
        // An 80 px wide mask over a 40 px image: image column c samples mask column 2c + 1, so the
        // set column 20 is never sampled and masks nothing.
        let image = ImageSize {
            width: 40,
            height: 40,
        };
        let grid = grid(40, 40, 8);
        let mut data = vec![0u8; 80 * 40];
        data[20 * 80 + 20] = 1;
        let mask = PixelMask {
            width: 80,
            height: 40,
            data: &data,
        };
        assert_eq!(
            masked_cells(mask, image, &grid),
            brute_force(mask, image, &grid)
        );
        assert!(masked_cells(mask, image, &grid).masks.is_empty());
    }

    proptest! {
        #[test]
        fn matches_the_pixel_rule(
            (mask_w, mask_h) in (1usize..200, 1usize..200),
            (image_w, image_h) in (40usize..160, 40usize..160),
            cell in prop::sample::select(vec![8usize, 10, 16]),
            seed in any::<u64>(),
            density in 0u64..8,
        ) {
            // Sparse random masks: a dense one masks every cell and proves nothing.
            let mut state = seed | 1;
            let data: Vec<u8> = (0..mask_w * mask_h)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    u8::from(state % 64 < density)
                })
                .collect();
            let image = ImageSize { width: image_w, height: image_h };
            let grid = grid(image_w, image_h, cell);
            let mask = PixelMask { width: mask_w, height: mask_h, data: &data };
            prop_assert_eq!(masked_cells(mask, image, &grid), brute_force(mask, image, &grid));
        }
    }
}
