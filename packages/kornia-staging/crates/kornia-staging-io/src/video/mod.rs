//! Packet video input and output.
mod annex_b;
pub use annex_b::{has_nal, AccessUnit, AccessUnitSplitter};

#[cfg(all(unix, feature = "encoder"))]
mod process_encoder;
#[cfg(all(unix, feature = "encoder"))]
pub use process_encoder::{
    EncodedSample, EncoderConfig, EncoderStats, H264Encoder, VideoError,
};
