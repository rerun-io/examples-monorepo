//! Feature extraction and decoding.

/// Log-quadratic heatmap peak decoding.
mod heatmap;
pub use heatmap::{argmax_first, decode_peak_2d, refine_peak_log_quadratic};

/// Basalt's oriented BRIEF at given keypoints.
mod brief;
pub use brief::{
    describe_keypoints_u16, intensity_centroid_angle, BriefConfig, BriefSmoothing,
    OrientedDescriptor, BRIEF_BASALT_BORDER, BRIEF_HALF_PATCH,
};

/// 256-bit descriptors, Hamming distance and mutual matching.
mod hamming;
pub use hamming::{match_mutual, Descriptor256, MatchConfig};

#[cfg(test)]
mod brief_golden;

mod cells;
pub use cells::{
    cell_select, detect_keypoints_with_cells, detect_prepared_keypoints_with_cells,
    opencv_corner_score, threshold_rungs, BandRequest, CellGrid, CellGridError, CellMasks,
    CellSelect, CenteredCellConfig, CenteredCellError, CenteredCellKeypoints, CornerScan,
    CpuCornerScan, DetectorScratch, FastCorner, MaskRect, Occupancy, SelectionStatus,
    EDGE_THRESHOLD, FAST_BORDER, LOWEST_THRESHOLD_RUNG, MAX_CELLS,
};

/// Shared CPU/device implementation contract.
#[doc(hidden)]
pub mod backend {
    pub use super::cells::{
        block_filter_end, decode_cell_key, BandCache, CELL_KEY_LIMIT, FAST_FILTER_LANES,
        FAST_RING_COLUMN, FAST_RING_ROW, KEY_FIELD_MASK, KEY_ROW_SHIFT, KEY_SCORE_SHIFT,
        NO_CELL_WINNER,
    };
}
