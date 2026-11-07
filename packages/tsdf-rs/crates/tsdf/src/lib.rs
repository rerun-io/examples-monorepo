#![deny(missing_docs)]
//! Sparse colour TSDF fusion and marching-cubes mesh extraction.
//!
//! [`TsdfVolume`] fuses projective depth frames into 16-by-16-by-16 voxel
//! blocks allocated around observed surfaces (Curless–Levoy running means,
//! Open3D's truncation and voxel-center conventions), and extracts the zero
//! level set as a [`Mesh`] with per-vertex colors and TSDF-gradient normals:
//! all at once or as per-block meshes of the blocks a frame changed.
//!
//! Conventions: world units throughout (the voxel size, truncation, depths and
//! camera translations share them); depth is camera Z, not ray length; images
//! and matrices are row-major (see [`Camera`]). Results are deterministic and
//! independent of the Rayon thread count.

mod changes;
mod error;
mod extract;
mod frame;
mod geometry;
mod mesh;
mod tables;
mod volume;

pub use changes::{Delivery, DirtyMeshes};
pub use error::TsdfError;
pub use frame::{Camera, DepthFrame, DepthRange};
pub use mesh::Mesh;
pub use volume::{BlockKey, TsdfVolume};
