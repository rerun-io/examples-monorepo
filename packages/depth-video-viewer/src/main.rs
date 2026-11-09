//! Start the depth-video viewer.
use depth_video_viewer::run;

#[global_allocator]
static ALLOCATOR: re_viewer::external::re_memory::AccountingAllocator<std::alloc::System> =
    re_viewer::external::re_memory::AccountingAllocator::new(std::alloc::System);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run()
}
