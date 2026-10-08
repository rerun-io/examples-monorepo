//! Packet video input and output.
mod annex_b;
pub use annex_b::{has_nal, AccessUnit, AccessUnitSplitter};

#[cfg(all(unix, feature = "encoder"))]
mod process_encoder;
#[cfg(all(unix, feature = "encoder"))]
pub use process_encoder::{EncodedSample, EncoderConfig, EncoderStats, H264Encoder, VideoError};

#[cfg(all(unix, feature = "decoder"))]
mod decoder;
#[cfg(all(unix, feature = "decoder"))]
pub use decoder::{decode_packets, validate_y_plane, ColorRange, DecodeError, PacketCodec};
