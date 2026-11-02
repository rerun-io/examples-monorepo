//! NeRF, strict `CameraSpec` JSON, and COLMAP text/binary adapters.
use crate::camera::CameraSpec;
use anyhow::{Context as _, Result};
use gsplat_core::CameraModel;
use std::path::{Path, PathBuf};

pub struct CameraFrame {
    pub camera: CameraSpec,
    pub file_path: PathBuf,
}

#[derive(serde::Deserialize)]
struct NerfTransforms {
    camera_angle_x: f32,
    frames: Vec<NerfFrame>,
}

#[derive(serde::Deserialize)]
struct NerfFrame {
    file_path: PathBuf,
    transform_matrix: [[f32; 4]; 4],
}

pub(crate) fn nerf_frames(path: &Path, resize: Option<(u32, u32)>) -> Result<Vec<CameraFrame>> {
    let document: NerfTransforms = serde_json::from_reader(std::fs::File::open(path)?)?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    document
        .frames
        .into_iter()
        .map(|mut frame| {
            if frame.file_path.extension().is_none() {
                frame.file_path.set_extension("png");
            }
            let image = root.join(&frame.file_path);
            let (width, height) = match resize {
                Some(size) => size,
                None => image::image_dimensions(image)?,
            };
            let camera = CameraSpec::from_nerf(
                glam::Mat4::from_cols_array_2d(&frame.transform_matrix).transpose(),
                document.camera_angle_x,
                width,
                height,
            );
            Ok(CameraFrame {
                camera,
                file_path: frame.file_path,
            })
        })
        .collect()
}

/// Upstream Hexels camera record (2DGS `Camera`), as in its `contract.json` and our `cameras.json`.
#[derive(serde::Deserialize)]
struct UpstreamCamera {
    name: String,
    width: u32,
    height: u32,
    fov_x: f64,
    fov_y: f64,
    /// Row-vector convention: `[x y z 1] @ M` is the OpenCV camera point.
    world_view_transform: [[f64; 4]; 4],
}

#[derive(serde::Deserialize)]
struct UpstreamGeometry {
    test: Vec<UpstreamCamera>,
}

/// `cameras.json` (`cameras`) or the upstream `contract.json` (`geometry.test`).
#[derive(serde::Deserialize)]
struct UpstreamCameras {
    cameras: Option<Vec<UpstreamCamera>>,
    geometry: Option<UpstreamGeometry>,
}

/// Upstream intrinsics: focal from the field of view, principal point at the image centre.
fn upstream_frames(document: UpstreamCameras) -> Result<Vec<CameraFrame>> {
    let cameras = document
        .cameras
        .or_else(|| document.geometry.map(|g| g.test))
        .context("camera file has neither cameras nor geometry.test")?;
    cameras
        .into_iter()
        .map(|camera| {
            let view_from_world = glam::DMat4::from_cols_array_2d(&camera.world_view_transform);
            let focal = |size: u32, fov: f64| (f64::from(size) / (2.0 * (fov * 0.5).tan())) as f32;
            Ok(CameraFrame {
                camera: CameraSpec {
                    world_from_camera: view_from_world
                        .inverse()
                        .transpose()
                        .as_mat4()
                        .to_cols_array_2d(),
                    width: camera.width,
                    height: camera.height,
                    fx: focal(camera.width, camera.fov_x),
                    fy: focal(camera.height, camera.fov_y),
                    cx: camera.width as f32 / 2.0,
                    cy: camera.height as f32 / 2.0,
                    model: CameraModel::Pinhole,
                },
                file_path: PathBuf::from(camera.name),
            })
        })
        .collect()
}

/// A JSON file contains NeRF transforms, upstream Hexels cameras, or an array of strict
/// `CameraSpec` records. A directory is one COLMAP sparse model with cameras/images .bin or .txt.
pub async fn load_frames(path: &Path, resize: Option<(u32, u32)>) -> Result<Vec<CameraFrame>> {
    let mut frames = if path.is_dir() {
        colmap_frames(path).await?
    } else {
        let bytes = std::fs::read(path)?;
        if bytes.iter().copied().find(|c| !c.is_ascii_whitespace()) == Some(b'[') {
            serde_json::from_slice::<Vec<CameraSpec>>(&bytes)?
                .into_iter()
                .enumerate()
                .map(|(i, camera)| CameraFrame {
                    camera,
                    file_path: PathBuf::from(format!("{i:03}.png")),
                })
                .collect()
        } else if let Ok(document) = serde_json::from_slice::<UpstreamCameras>(&bytes)
            && (document.cameras.is_some() || document.geometry.is_some())
        {
            upstream_frames(document)?
        } else {
            nerf_frames(path, resize)?
        }
    };
    if frames.is_empty() {
        anyhow::bail!("empty camera path");
    }
    for frame in &mut frames {
        if let Some((w, h)) = resize
            && (frame.camera.width, frame.camera.height) != (w, h)
        {
            frame.camera = frame.camera.resized(w, h);
        }
        frame.camera.validate()?;
    }
    Ok(frames)
}

async fn colmap_frames(path: &Path) -> Result<Vec<CameraFrame>> {
    use colmap_reader::ColmapCameraModel as C;
    use tokio::io::BufReader;
    let binary = path.join("cameras.bin").exists();
    let extension = if binary { "bin" } else { "txt" };
    let cameras = colmap_reader::read_cameras(
        BufReader::new(tokio::fs::File::open(path.join(format!("cameras.{extension}"))).await?),
        binary,
    )
    .await?;
    let mut images = colmap_reader::read_images(
        BufReader::new(tokio::fs::File::open(path.join(format!("images.{extension}"))).await?),
        binary,
        false,
    )
    .await?;
    images.sort_by_key(|image| image.id);
    images
        .into_iter()
        .map(|image| {
            let camera = cameras
                .iter()
                .find(|c| c.id == image.camera_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(format!(
                        "{} references missing camera {}",
                        image.name, image.camera_id
                    ))
                })?;
            let p: Vec<f32> = camera.params.iter().map(|v| *v as f32).collect();
            let model = match camera.model {
                C::SimplePinhole | C::Pinhole => CameraModel::Pinhole,
                C::SimpleRadial | C::Radial | C::OpenCV | C::FullOpenCV => {
                    let [k1, k2, k3, k4, k5, k6, p1, p2] = match camera.model {
                        C::SimpleRadial => [p[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        C::Radial => [p[3], p[4], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        C::OpenCV => [p[4], p[5], 0.0, 0.0, 0.0, 0.0, p[6], p[7]],
                        _ => [p[4], p[5], p[8], p[9], p[10], p[11], p[6], p[7]],
                    };
                    CameraModel::RadialTangential8 {
                        k1,
                        k2,
                        k3,
                        k4,
                        k5,
                        k6,
                        p1,
                        p2,
                    }
                }
                C::SimpleRadialFisheye | C::RadialFisheye | C::OpenCvFishEye => {
                    let [k1, k2, k3, k4] = match camera.model {
                        C::SimpleRadialFisheye => [p[3], 0.0, 0.0, 0.0],
                        C::RadialFisheye => [p[3], p[4], 0.0, 0.0],
                        _ => [p[4], p[5], p[6], p[7]],
                    };
                    CameraModel::KannalaBrandt4 { k1, k2, k3, k4 }
                }
                C::ThinPrismFisheye => CameraModel::ThinPrismFisheye {
                    k1: p[4],
                    k2: p[5],
                    k3: p[8],
                    k4: p[9],
                    p1: p[6],
                    p2: p[7],
                    sx1: p[10],
                    sy1: p[11],
                },
                C::Fov => {
                    anyhow::bail!(
                        "COLMAP FOV lens is unsupported; convert it to a supported model"
                    );
                }
            };
            let (fx, fy) = camera.focal();
            let center = camera.principal_point();
            let pose = crate::camera::colmap_to_world(
                [image.quat.w, image.quat.x, image.quat.y, image.quat.z],
                image.tvec.to_array(),
            );
            let width = u32::try_from(camera.width).context("COLMAP width exceeds u32")?;
            let height = u32::try_from(camera.height).context("COLMAP height exceeds u32")?;
            Ok(CameraFrame {
                file_path: image.name.into(),
                camera: CameraSpec {
                    world_from_camera: pose.transpose().to_cols_array_2d(),
                    width,
                    height,
                    fx: fx as f32,
                    fy: fy as f32,
                    cx: center.x,
                    cy: center.y,
                    model,
                },
            })
        })
        .collect()
}
