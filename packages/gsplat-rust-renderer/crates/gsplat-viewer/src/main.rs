//! Start the custom viewer with compute-capable device limits.
use gsplat_viewer::application;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    application::run().await
}
