//! Resolve a camera file or deterministic orbit for benchmark commands.
use anyhow::{Context, Result, ensure};
use glam::{Mat4, Vec2};
use gsplat_cli::{
    PlyScene, camera,
    camera::{CameraSpec, load_frames},
};
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub(super) struct CameraArgs {
    #[arg(long)]
    pub(super) ply: PathBuf,
    #[arg(long, default_value = "orbit:300")]
    pub(super) path: String,
    /// Keep every Nth camera after sorting input filenames.
    #[arg(long, default_value = "1")]
    pub(super) holdout_every: std::num::NonZeroUsize,
    #[arg(long, default_value = "native", value_parser = resolution)]
    pub(super) res: Resolution,
    #[arg(long, num_args = 2)]
    pub(super) radius: Option<Vec<f32>>,
    #[arg(long)]
    pub(super) elevation: Option<f32>,
    /// World direction above the orbit plane (default +Z).
    #[arg(long, num_args = 3, allow_hyphen_values = true)]
    pub(super) orbit_up: Option<Vec<f32>>,
    #[arg(long, value_delimiter = ',', num_args = 3, allow_hyphen_values = true)]
    pub(super) center: Option<Vec<f32>>,
}
#[derive(Clone, Debug)]
pub(super) struct Resolution(pub(super) Option<(u32, u32)>);
pub(super) fn resolution(value: &str) -> Result<Resolution> {
    if value == "native" {
        return Ok(Resolution(None));
    }
    let (w, h) = value
        .split_once('x')
        .context("resolution must be WxH or native")?;
    let (w, h) = (w.parse()?, h.parse()?);
    ensure!(w >= 11 && h >= 11, "resolution must be at least 11x11");
    Ok(Resolution(Some((w, h))))
}
pub(super) async fn cameras(args: &CameraArgs, scene: &PlyScene) -> Result<Vec<CameraSpec>> {
    let size = args.res.0;
    let path = if let Some(count) = args.path.strip_prefix("orbit:") {
        let count: std::num::NonZeroUsize =
            count.parse().context("orbit count must be positive")?;
        let (w, h) = size.unwrap_or((800, 800));
        let template = CameraSpec::from_nerf(Mat4::IDENTITY, 0.6911112, w, h);
        let r = args
            .radius
            .as_deref()
            .map(Vec2::from_slice)
            .unwrap_or(Vec2::splat(scene.extent * 2.5));
        ensure!(
            r.is_finite() && r.min_element() > 0.0,
            "invalid orbit radius"
        );
        let center = args
            .center
            .as_deref()
            .map(glam::Vec3::from_slice)
            .unwrap_or(scene.center);
        ensure!(center.is_finite(), "invalid orbit center");
        let up = args.orbit_up.as_deref().map(glam::Vec3::from_slice);
        ensure!(
            up.is_none_or(|up| up.is_finite() && up.length_squared() > 1e-12),
            "invalid orbit up-vector"
        );
        let path = camera::orbit(
            center,
            r,
            args.elevation.unwrap_or(scene.extent * 0.8),
            count.get(),
            &template,
            up,
        );
        for camera in &path {
            camera.validate()?;
        }
        path
    } else {
        let mut frames = load_frames(Path::new(&args.path), size).await?;
        if args.holdout_every.get() > 1 {
            frames.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        }
        frames
            .into_iter()
            .step_by(args.holdout_every.get())
            .map(|f| f.camera)
            .collect()
    };
    ensure!(!path.is_empty(), "empty camera path");
    for c in &path {
        ensure!(
            (c.width, c.height) == (path[0].width, path[0].height),
            "mixed camera resolutions"
        );
    }
    Ok(path)
}
