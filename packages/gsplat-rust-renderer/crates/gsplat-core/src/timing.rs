//! Local stage timestamps for the renderer benchmark.
pub const STAGE_NAMES: [&str; 8] = [
    "project_forward",
    "depth_sort",
    "gather_scan",
    "project_visible",
    "map_intersections",
    "tile_sort",
    "tile_offsets",
    "rasterize",
];
/// Contiguous GPU intervals; projection includes indirect-dispatch preparation.
pub fn stage_queries(stage: usize) -> (usize, usize) {
    (if stage == 0 { 0 } else { stage + 1 }, stage + 2)
}
