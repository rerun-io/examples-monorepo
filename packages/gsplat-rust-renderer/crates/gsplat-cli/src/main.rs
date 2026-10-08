//! Render scenes, evaluate images, and compare renderer quality and speed.
mod eval_command;
use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::path::Path;
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Score matching rendered and ground-truth image directories.
    Eval(eval_command::Args),
}
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    let action = Args::parse().command;
    match action {
        Action::Eval(a) => eval_command::run(a).await,
    }
}
