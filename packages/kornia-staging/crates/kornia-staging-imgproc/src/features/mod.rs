//! Feature extraction and decoding.

/// Log-quadratic heatmap peak decoding.
mod heatmap;
pub use heatmap::{argmax_first, decode_peak_2d, refine_peak_log_quadratic};

mod cells;
pub use cells::{
    cell_select, detect_keypoints_with_cells, detect_prepared_keypoints_with_cells,
    opencv_corner_score, threshold_rungs, BandCache, BandRequest, CellGrid, CellGridError,
    CellMasks, CellSelect, CenteredCellConfig, CenteredCellError, CenteredCellKeypoints,
    CornerScan, CpuCornerScan, DetectorScratch, FastCorner, MaskRect, Occupancy, SelectionStatus,
    EDGE_THRESHOLD, FAST_BORDER, LOWEST_THRESHOLD_RUNG, MAX_CELLS,
};
