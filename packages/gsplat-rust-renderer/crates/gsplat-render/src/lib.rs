//! Shared file/camera boundary and headless renderer for the CLI and benchmark.
//! The render algorithms live in gsplat-core; this crate has no Rerun dependency.
pub mod camera;
mod camera_files;
mod renderer;
pub mod settings;
pub use renderer::{Output, PlyScene, Renderer, raw_splats};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Camera(#[from] camera::CameraError),
    #[error(transparent)]
    Core(#[from] gsplat_core::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
