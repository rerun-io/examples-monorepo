//! Shared CubeCL buffer layout and sampling operations.
pub mod layout;
pub mod sampling;
pub(crate) use sampling::launch_probe;

mod pyramid;
pub(crate) use layout::{Buffer, MAX_CUBES_PER_DIM};
pub(crate) use pyramid::{launch_ingest, launch_subsample, launch_subsample_batch};
pub(crate) use sampling::launch_copy_level0;
