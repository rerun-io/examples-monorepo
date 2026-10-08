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
fn stage_queries(stage: usize) -> (usize, usize) {
    (if stage == 0 { 0 } else { stage + 1 }, stage + 2)
}

/// Query slots written by one profiled frame.
pub const QUERY_COUNT: u32 = 10;
/// Convert monotonic completed-frame timestamps into the eight named stage durations.
pub fn stage_ms(ticks: &[u64], period_ns: f32) -> Result<[f64; 8], crate::Error> {
    if ticks.len() != QUERY_COUNT as usize
        || ticks[9] <= ticks[0]
        || ticks.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(crate::Error::Readback(format!(
            "invalid GPU stage timestamps: {ticks:?}"
        )));
    }
    Ok(std::array::from_fn(|i| {
        let (start, end) = stage_queries(i);
        (ticks[end] - ticks[start]) as f64 * (f64::from(period_ns) / 1e6)
    }))
}
