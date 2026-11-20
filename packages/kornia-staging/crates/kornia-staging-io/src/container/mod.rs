//! Container demuxing: ISO-BMFF (MP4) boxes and the per-codec sample framing a remux needs.
/// A minimal ISO-BMFF demuxer for H.264 and HEVC video tracks.
pub mod mp4;
/// H.264 and HEVC length-prefixed samples to Annex B access units.
pub mod nal;
