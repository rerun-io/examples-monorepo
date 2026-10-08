use crate::features::BandRequest;
/// One band of a 50-pixel cell grid, keyed the way
/// `detect_keypoints_with_cells` keys it: `row` is the grid row and `rung` the
/// place on the threshold ladder, and the cache is indexed by the pair.
pub fn band_at(row: usize, rung: usize, y: usize, rows: usize, threshold: i32) -> BandRequest {
    BandRequest {
        row,
        rung,
        y,
        rows,
        threshold,
    }
}
