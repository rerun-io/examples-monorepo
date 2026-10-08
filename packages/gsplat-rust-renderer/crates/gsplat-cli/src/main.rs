//! Render scenes, evaluate images, and compare renderer quality and speed.
mod camera_args;
mod eval_command;
mod parity;
mod render_command;
mod score;
use anyhow::Result;
use camera_args::{CameraArgs, cameras};
use clap::{Parser, Subcommand};
use gsplat_cli::Provenance;
use serde::Serialize;
use std::path::Path;
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Render a PLY through the standalone GPU path.
    Render(render_command::Args),
    /// Score matching rendered and ground-truth image directories.
    Eval(eval_command::Args),
    /// Compare float renders against a reference renderer.
    Parity(parity::Args),
    /// Score a trained PLY without clipping or quantizing its float output.
    Score(score::ScoreArgs),
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
    let out = match &action {
        Action::Render(_) | Action::Eval(_) => None,
        Action::Parity(a) => Some(a.out.clone()),
        Action::Score(a) => Some(a.out.clone()),
    };
    let result = match action {
        Action::Render(a) => render_command::run(a).await,
        Action::Eval(a) => eval_command::run(a).await,
        Action::Parity(a) => parity::run(a).await,
        Action::Score(a) => score::run(a).await,
    };
    if let (Err(error), Some(out)) = (&result, out)
        && let Some(gsplat_cli::Unsupported(reason)) =
            error.downcast_ref::<gsplat_cli::Unsupported>()
    {
        #[derive(Serialize)]
        struct UnsupportedReport<'a> {
            status: &'static str,
            reason: &'a str,
            provenance: Provenance,
        }
        write_json(
            &out,
            &UnsupportedReport {
                status: "unsupported",
                reason,
                provenance: Provenance::capture(),
            },
        )?;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::camera_args::resolution;
    use super::*;
    use glam::Mat4;
    use gsplat_cli::PlyScene;
    use std::path::PathBuf;
    #[test]
    fn parity_accepts_archetype_and_plain_camera_paths_with_holdout() {
        use clap::CommandFactory as _;
        Args::command().debug_assert();
        for path in [None, Some(None), Some(Some("recording.rrd"))] {
            let mut cli = vec!["gsplat", "parity"];
            if let Some(recording) = path {
                cli.push("--archetype");
                if let Some(recording) = recording {
                    cli.push(recording);
                }
            }
            cli.extend([
                "--impl",
                "ours",
                "--ply",
                "scene.ply",
                "--path",
                "cameras.json",
                "--holdout-every",
                "8",
                "--out",
                "result.json",
            ]);
            let Action::Parity(parsed) = Args::try_parse_from(cli).unwrap().command else {
                panic!("expected parity command");
            };
            assert_eq!(parsed.archetype, path.map(|p| p.map(PathBuf::from)));
            assert_eq!(parsed.camera.path, "cameras.json");
            assert_eq!(parsed.camera.holdout_every.get(), 8);
        }
    }
    #[tokio::test]
    async fn orbit_up_places_cameras_above_the_scene_and_looks_at_focus() {
        let args = Args::try_parse_from([
            "gsplat",
            "parity",
            "--impl",
            "native",
            "--ply",
            "unused",
            "--path",
            "orbit:4",
            "--out",
            "unused",
            "--center",
            "1",
            "2",
            "3",
            "--radius",
            "3",
            "3",
            "--elevation",
            "1",
            "--orbit-up",
            "0",
            "-1",
            "0",
        ])
        .unwrap();
        let Action::Parity(args) = args.command else {
            panic!("parity command")
        };
        let scene = PlyScene {
            data: brush_serde::import::SplatData {
                means: vec![0.0; 3],
                rotations: None,
                log_scales: None,
                sh_coeffs: None,
                raw_opacities: None,
            },
            mode: gsplat_core::RenderMode::Default,
            center: glam::Vec3::ZERO,
            extent: 1.0,
        };
        let path = cameras(&args.camera, &scene).await.unwrap();
        let expected = [
            [4.0, 1.0, 3.0],
            [1.0, 1.0, 6.0],
            [-2.0, 1.0, 3.0],
            [1.0, 1.0, 0.0],
        ];
        for (camera, expected) in path.iter().zip(expected) {
            let pose = Mat4::from_cols_array_2d(&camera.world_from_camera).transpose();
            assert!((pose.w_axis.truncate() - glam::Vec3::from_array(expected)).length() < 1e-5);
            let target = pose
                .inverse()
                .transform_point3(glam::Vec3::new(1.0, 2.0, 3.0));
            assert!(target.x.abs() < 1e-5 && target.y.abs() < 1e-5 && target.z > 0.0);
        }
    }
    #[test]
    fn cli_rejects_unknown_oracle_and_invalid_resolution() {
        assert!(
            Args::try_parse_from([
                "gsplat", "parity", "--impl", "native", "--ply", "x", "--oracle", "typo", "--out",
                "y"
            ])
            .is_err()
        );
        assert_eq!(resolution("1920x1080").unwrap().0, Some((1920, 1080)));
        assert!(resolution("0x800").is_err());
    }
}
