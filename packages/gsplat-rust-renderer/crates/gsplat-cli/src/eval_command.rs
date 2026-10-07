//! Evaluate paired image directories with an explicit metric convention.
use gsplat_cli::{Convention, evaluate_directories};
use std::path::PathBuf;

#[derive(clap::Args)]
pub(crate) struct Args {
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
}
pub(crate) async fn run(args: Args) -> anyhow::Result<()> {
    let result = evaluate_directories(&args.render, &args.gt, args.convention, args.lpips).await?;
    super::write_json(&args.out, &result)?;
    println!(
        "{} views: PSNR {:.8}, SSIM {:.9}",
        result.views.len(),
        result.mean.psnr,
        result.mean.ssim
    );
    Ok(())
}
