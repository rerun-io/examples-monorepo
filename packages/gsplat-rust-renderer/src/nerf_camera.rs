//! NeRF Synthetic camera loader.
//!
//! Parses `transforms_*.json` files from the NeRF Synthetic dataset format.
//! Each JSON contains a global `camera_angle_x` (horizontal FOV) and a list
//! of frames, each with a `file_path` and 4×4 camera-to-world matrix.

use std::path::Path;

use glam::Mat4;
use serde::Deserialize;

use crate::gsplat_core::camera::camera_from_nerf_transform;
use crate::gsplat_core::types::CameraApproximation;

/// A single frame from a NeRF transforms JSON file.
pub struct NerfFrame {
    /// Relative path to the image file (e.g. `"./test/r_0"`), without extension.
    pub file_path: String,
    /// Camera parameters derived from the transform matrix and global FOV.
    pub camera: CameraApproximation,
}

#[derive(Deserialize)]
struct TransformsJson {
    camera_angle_x: f64,
    frames: Vec<FrameJson>,
}

#[derive(Deserialize)]
struct FrameJson {
    file_path: String,
    transform_matrix: [[f64; 4]; 4],
}

/// Load all cameras from a NeRF Synthetic transforms JSON.
///
/// Each frame's `transform_matrix` is a 4×4 camera-to-world (c2w) matrix.
/// This function inverts it to world-to-camera and derives the projection
/// matrix from `camera_angle_x` + image dimensions.
///
/// # Arguments
///
/// * `path` — Path to a `transforms_*.json` file
/// * `width` — Image width in pixels (800 for NeRF Synthetic)
/// * `height` — Image height in pixels (800 for NeRF Synthetic)
pub fn load_cameras(path: &Path, width: u32, height: u32) -> anyhow::Result<Vec<NerfFrame>> {
    let json_text = std::fs::read_to_string(path)?;
    let transforms: TransformsJson = serde_json::from_str(&json_text)?;
    let camera_angle_x = transforms.camera_angle_x as f32;

    let frames: Vec<NerfFrame> = transforms
        .frames
        .into_iter()
        .map(|frame| {
            let c2w = mat4_from_nested_array(&frame.transform_matrix);
            NerfFrame {
                file_path: frame.file_path,
                camera: camera_from_nerf_transform(c2w, camera_angle_x, width, height),
            }
        })
        .collect();

    Ok(frames)
}

/// Load a single camera by frame index.
pub fn load_camera(
    path: &Path,
    frame: usize,
    width: u32,
    height: u32,
) -> anyhow::Result<CameraApproximation> {
    let frames = load_cameras(path, width, height)?;
    frames
        .into_iter()
        .nth(frame)
        .map(|f| f.camera)
        .ok_or_else(|| anyhow::anyhow!("Frame index {frame} out of range"))
}

fn mat4_from_nested_array(arr: &[[f64; 4]; 4]) -> Mat4 {
    // The JSON stores row-major; glam::Mat4::from_cols_array_2d expects column-major.
    // So we transpose: each inner array is a row, but glam expects columns.
    Mat4::from_cols(
        glam::Vec4::new(
            arr[0][0] as f32,
            arr[1][0] as f32,
            arr[2][0] as f32,
            arr[3][0] as f32,
        ),
        glam::Vec4::new(
            arr[0][1] as f32,
            arr[1][1] as f32,
            arr[2][1] as f32,
            arr[3][1] as f32,
        ),
        glam::Vec4::new(
            arr[0][2] as f32,
            arr[1][2] as f32,
            arr[2][2] as f32,
            arr[3][2] as f32,
        ),
        glam::Vec4::new(
            arr[0][3] as f32,
            arr[1][3] as f32,
            arr[2][3] as f32,
            arr[3][3] as f32,
        ),
    )
}
