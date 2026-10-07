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

#[derive(Default)]
pub(crate) struct StageTimer(Option<wgpu::QuerySet>);
impl StageTimer {
    pub fn set(
        &mut self,
        device: &wgpu::Device,
        queries: Option<wgpu::QuerySet>,
    ) -> Result<(), crate::Error> {
        if let Some(q) = &queries
            && (!device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
                || !matches!(q.ty(), wgpu::QueryType::Timestamp)
                || q.count() < 10)
        {
            return Err(crate::Error::Input(
                "stage profiling requires TIMESTAMP_QUERY and ten timestamp slots",
            ));
        }
        self.0 = queries;
        Ok(())
    }
    pub fn pass(
        &self,
        start: Option<u32>,
        end: u32,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.0
            .as_ref()
            .map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: start,
                end_of_pass_write_index: Some(end),
            })
    }
}
