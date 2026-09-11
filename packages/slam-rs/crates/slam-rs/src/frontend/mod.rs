//! Optical-flow frontend: sampling patterns, patch factors, SE(2) KLT tracking,
//! grid FAST detection and the frame-to-frame driver.
//! Only frame-to-frame flow and Pattern51 are selected by shipped configs.
//!
//! The frontend uses f32 arithmetic on u16 pixels (D05). Per-patch buffers put
//! patch index first in storage order, and frame buffers are reused. Fixed loop
//! bounds and explicit validity flags support CPU and GPU stage implementations.

pub mod cell;
pub mod detect;
pub mod flow;

pub mod ldlt;
pub mod parallel;
pub mod patch;
pub mod patterns;
pub mod se2;
pub mod tracker;
