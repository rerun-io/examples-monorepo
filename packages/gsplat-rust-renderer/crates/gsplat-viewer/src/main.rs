//! Start the custom viewer with compute-capable device limits.
pub mod application;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    application::run().await
}
