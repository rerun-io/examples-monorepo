//! Rerun viewer extension that draws depth stored as AV1 10-bit monochrome video (`DepthVideoStream`)
//! in the built-in 2D and 3D spatial views.
//!
//! The archetype, decoder and visualizer are out-of-tree scaffolding for checking depth as video end to end;
//! they are deleted once `DepthVideoStream` ships in Rerun.
mod application;
mod archetype;
mod decoder;
mod visualizer;

pub use application::run;
