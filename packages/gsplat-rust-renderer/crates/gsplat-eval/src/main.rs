//! Score paired PNG directories with an explicit metric convention.
use clap::{Parser, Subcommand};
use gsplat_eval::{Convention, evaluate_directories};
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Dirs {
        #[arg(long)]
        render: PathBuf,
        #[arg(long)]
        gt: PathBuf,
        #[arg(long, value_enum, default_value_t = Convention::Brush)]
        convention: Convention,
        #[arg(long)]
        lpips: bool,
        #[arg(long)]
        out: PathBuf,
    },
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Command::Dirs {
        render,
        gt,
        convention,
        lpips,
        out,
    } = Cli::parse().command;
    let result = evaluate_directories(&render, &gt, convention, lpips).await?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&result)?)?;
    println!(
        "{} views: PSNR {:.8}, SSIM {:.9}",
        result.views.len(),
        result.mean.psnr,
        result.mean.ssim
    );
    Ok(())
}
