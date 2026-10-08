//! robocap-live: live SLAM + hand tracking on the RoboCap cap (raw cameras + IMU in, Rerun out), and the same core replaying
//! an exported session ([`frame`] describes the dump directory, `sched/record.rs` the `--record` JSONL).
//!
//! Conventions (kornia-rs style):
//! - kornia-rs comes from the git revision pinned in `Cargo.toml`, the one slam-rs uses, so both share one `Image` type; no
//!   crates.io kornia crates beside it. Images are `kornia_image::Image<u8, 1>` (tight rows), shared as [`frame::Luma`].
//! - Image ops write into a destination (`fn op(src, dst, params) -> Result<(), ImageError>`), named output first with a dtype
//!   suffix (`gray_from_rgb`, `_u8`, `_f32`); parallel work is rayon over row chunks; SIMD kernels have NEON and scalar paths.
//! - Errors are `thiserror` enums per module, `anyhow` only in `main.rs`; no `unwrap`/`expect`/`panic!` in library code; every
//!   `unsafe` block has a `// SAFETY:` comment.
//! - The cap binary links neither GStreamer nor librknnrt nor librga: librknnrt is dlopened at run time and H.264 goes through
//!   `gst-launch-1.0` child processes; `scripts/build-arm.sh` checks the link-time dependencies and the glibc 2.34 floor.

pub mod capture;
pub mod downsample;
#[cfg(target_os = "linux")]
pub mod diagnostic;
pub mod frame;
pub mod hands;
pub mod layer;
pub mod log;
pub mod log_markers;
pub mod nets;
pub mod sched;
pub mod slam;
pub mod source;
