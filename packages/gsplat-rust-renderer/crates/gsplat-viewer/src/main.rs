//! Start the custom viewer with compute-capable device limits.
use gsplat_viewer::application;

#[global_allocator]
static ALLOCATOR: re_viewer::external::re_memory::AccountingAllocator<std::alloc::System> =
    re_viewer::external::re_memory::AccountingAllocator::new(std::alloc::System);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    application::run().await
}
