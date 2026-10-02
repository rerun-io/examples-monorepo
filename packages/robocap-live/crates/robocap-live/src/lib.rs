//! robocap-live: live SLAM + hand tracking on the RoboCap cap (raw cameras + IMU in, Rerun out), and the same core replaying
//! an exported session. See SPEC.md for the dump and record formats, and UPSTREAM.md for the pieces written in kornia-rs style
//! to be upstreamed.

pub mod downsample;
pub mod frame;
pub mod hands;
pub mod kornia_ext;
pub mod nets;
