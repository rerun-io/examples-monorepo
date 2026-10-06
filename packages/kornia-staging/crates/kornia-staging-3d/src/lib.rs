#![deny(missing_docs)]
//! Staging for code destined for kornia-rs's kornia-3d crate.
//!
//! Modules mirror their upstream destination and use Kornia types and conventions.
//! Consumers depend on this crate; it must never depend on consumer packages.
//! When an item lands upstream, bump the dependency, swap imports, and delete it here.

pub mod camera;
/// Pose geometry and bearing coordinates.
pub mod pose;
