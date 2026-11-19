//! The Insta360 `.insv` reader's core, over kornia-staging-io's `.insv` parser (`kornia_staging_io::insta360`).
//!
//! [`profile`] holds the device profiles: what a clip does not carry (the IMU's rotation in the rig, the lever arm,
//! the frame-time model, the readout, the IMU noise), measured per body and video mode. [`rig`] applies the profile
//! to a clip: two lens cameras posed in the IMU frame, frame and IMU times on one clock, per-frame exposure, and the
//! lens tracks cut into GOPs and read as packed Annex B samples. The `insta360_io` Python module binds it
//! (`crates/insta360-io-py`) and turns it into Rerun chunks (`InsvReader`).
pub mod profile;
pub mod rig;
