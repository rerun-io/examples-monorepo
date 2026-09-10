//! The optical-flow frontend: patterns, patches, the SE(2) KLT tracker, grid FAST
//! detection and the frame-to-frame driver.
//!
//! Ported from `include/basalt/optical_flow/` and `src/utils/keypoints.cpp` of
//! the basalt VIO fork. Only `FrameToFrameOpticalFlow` is here: every shipped
//! config sets `optical_flow_type = "frame_to_frame"`, so `PatchOpticalFlow` and
//! `MultiscaleFrameToFrameOpticalFlow` are out of scope, and the recall
//! subsystem is off in every shipped config and leaks patches by design, so it is
//! left out too (trap 18).
//!
//! ```text
//!   patterns  Pattern52 / Pattern51                patterns.h
//!   se2       AffineCompact2f, SE2::exp           optical_flow.h:66, se2.hpp:609
//!   ldlt      Eigen's pivoted LDLT, 3x3           Cholesky/LDLT.h
//!   patch     OpticalFlowPatch, residual          patch.h
//!   tracker   PatchSoA, CpuPatchTracker           frame_to_frame_optical_flow.h:294-438
//!   detect    grid-cell FAST over kornia-rs       keypoints.cpp:132-205
//!   parallel  the explicit thread budget          decision D31
//! ```
//!
//! The whole frontend is `f32` on `u16` pixels (decision D05), the per-patch
//! buffers are structure-of-arrays with the patch index fast-varying, the
//! per-frame path preallocates, and every loop has a fixed bound with a converged
//! flag — the CubeCL seam rules from `cubecl-portability.md` §12, applied from
//! the first CPU commit so a GPU backend arrives as a new [`tracker::PatchTracker`]
//! rather than a rewrite.

pub mod cell;
pub mod detect;
pub mod flow;

pub mod ldlt;
pub mod parallel;
pub mod patch;
pub mod patterns;
pub mod se2;
pub mod tracker;
