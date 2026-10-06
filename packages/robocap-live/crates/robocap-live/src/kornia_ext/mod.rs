//! Pieces kornia-rs lacks, written in its style (free functions over `kornia_image::Image`, `Result<_, ImageError>`, docs and
//! tests) so they can be upstreamed. Each module is listed in `packages/robocap-live/UPSTREAM.md` with its target crate.
#![deny(missing_docs)]

pub mod heatmap;
pub mod remap;
