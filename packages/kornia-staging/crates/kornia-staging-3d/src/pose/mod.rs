//! Pose geometry and bearing coordinates.
mod core;
pub mod gp3p;
mod stereographic;
mod triangulation_bearing;
pub use stereographic::*;
pub use triangulation_bearing::*;
