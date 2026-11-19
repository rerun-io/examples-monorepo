//! Insta360 dual-fisheye `.insv` clips: the two HEVC lens tracks plus the vendor trailer.
//!
//! The trailer sits at the end of the file (on recent bodies wrapped in a top-level `inst` box). It ends in a 72-byte
//! footer (32 zero bytes, `u32` trailer size, `u32` version, a 32-character magic), preceded by a 6-byte descriptor
//! `[format][id][u32 size]` of the index record, whose 10-byte slots `[id][format][u32 size][u32 offset]` locate every
//! other record relative to the trailer start. Only the index is read: on the X4 Air the records sit in 256 KiB slots
//! and walking descriptors backward from the footer fails. Record layouts follow telemetry-parser's `insta360`
//! module (<https://github.com/AdrianEddy/telemetry-parser>), checked against a real X4 Air clip.
//!
//! The factory lens strings the metadata carries are a camera format:
//! `kornia_staging_3d::camera::formats::insta360` parses them.
use crate::container::mp4::Mp4Error;
use crate::container::nal::NalError;
use kornia_staging_3d::camera::formats::insta360::LensStringError;

mod clip;
mod protobuf;
#[cfg(test)]
mod testing;
mod trailer;

pub use clip::{Insta360Clip, SampleReader};
pub use protobuf::Insta360Metadata;
pub use trailer::{read_trailer, Exposure, ImuSample, Trailer};

/// Why a clip could not be read.
#[derive(Debug, thiserror::Error)]
pub enum Insta360Error {
    /// The file could not be read.
    #[error("insv read failed: {0}")]
    Io(#[from] std::io::Error),
    /// The video container is malformed.
    #[error(transparent)]
    Mp4(#[from] Mp4Error),
    /// A video sample is malformed.
    #[error(transparent)]
    Nal(#[from] NalError),
    /// The trailer or a record is malformed or missing.
    #[error("malformed insv trailer: {0}")]
    Malformed(String),
    /// A factory lens string is malformed or names a lens the reader does not support.
    #[error(transparent)]
    LensString(#[from] LensStringError),
    /// The clip is well formed but its capture geometry is one the reader does not support.
    #[error("unsupported insv clip: {0}")]
    Unsupported(String),
}

fn malformed(what: impl Into<String>) -> Insta360Error {
    Insta360Error::Malformed(what.into())
}
