//! AV1 → full-bit-depth luma, and seeking within a stream of samples.
//!
//! Rerun's built-in AV1 decoder builds rav1d 8-bit only and truncates 10-bit luma (`re_video`'s
//! `decode/av1.rs`, TODO #7594). Here rav1d runs with 16-bit pixels on and the Y plane is copied as is,
//! so every 10-bit code comes back exact. Decoding is synchronous: a seek decodes from the nearest
//! keyframe, and playing forward continues from the last decoded sample.

use re_rav1d::dav1d;
use std::sync::Arc;

#[derive(Debug)]
pub enum DecodeError {
    Dav1d(dav1d::Error),
    NoPicture,
    BitDepth(usize),
    NoKeyframe(usize),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dav1d(err) => write!(f, "AV1 decoder: {err}"),
            Self::NoPicture => write!(f, "the AV1 sample produced no picture"),
            Self::BitDepth(bits) => write!(f, "unsupported luma bit depth {bits}"),
            Self::NoKeyframe(searched) => {
                write!(
                    f,
                    "no AV1 keyframe within {searched} samples before this one"
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<dav1d::Error> for DecodeError {
    fn from(err: dav1d::Error) -> Self {
        Self::Dav1d(err)
    }
}

/// One decoded depth frame: the luma codes, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LumaFrame {
    pub width: u32,
    pub height: u32,
    /// Coded bits per sample (8, 10 or 12).
    pub bit_depth: u8,
    pub codes: Vec<u16>,
}

/// rav1d with 16-bit pixels, handing back the Y plane of each picture.
pub struct LumaDecoder {
    decoder: dav1d::Decoder,
}

impl LumaDecoder {
    pub fn new() -> Result<Self, DecodeError> {
        let mut settings = dav1d::Settings::new();
        // Like re_video: prefer frames over error messages, and output each picture right away.
        settings.set_strict_std_compliance(false);
        settings.set_max_frame_delay(1);
        Ok(Self {
            decoder: dav1d::Decoder::with_settings(&settings)?,
        })
    }

    /// Drop all decoder state; the next sample must be a keyframe.
    pub fn reset(&mut self) {
        self.decoder.flush();
    }

    /// Decode one temporal unit and return the picture it shows.
    pub fn decode(&mut self, sample: &[u8]) -> Result<LumaFrame, DecodeError> {
        re_tracing::profile_function!();
        let mut shown = None;
        match self.decoder.send_data(sample.to_vec(), None, None, None) {
            Ok(()) => {}
            Err(dav1d::Error::Again) => loop {
                // The decoder is full: drain pictures until it takes the rest of the data.
                self.drain(&mut shown)?;
                match self.decoder.send_pending_data() {
                    Ok(()) => break,
                    Err(dav1d::Error::Again) => {}
                    Err(err) => return Err(err.into()),
                }
            },
            Err(err) => return Err(err.into()),
        }
        self.drain(&mut shown)?;
        let picture = shown.ok_or(DecodeError::NoPicture)?;
        luma_frame(&picture)
    }

    fn drain(&mut self, shown: &mut Option<dav1d::Picture>) -> Result<(), DecodeError> {
        loop {
            match self.decoder.get_picture() {
                Ok(picture) => *shown = Some(picture),
                Err(dav1d::Error::Again) => return Ok(()),
                Err(err) => return Err(err.into()),
            }
        }
    }
}

/// Copy the Y plane without its row padding; 10/12-bit samples are native-endian `u16` in rav1d.
fn luma_frame(picture: &dav1d::Picture) -> Result<LumaFrame, DecodeError> {
    let bits = picture
        .bits_per_component()
        .map_or_else(|| picture.bit_depth(), |bpc| bpc.0);
    let width = picture.width() as usize;
    let height = picture.height() as usize;
    let stride = picture.stride(dav1d::PlanarImageComponent::Y) as usize;
    let plane = picture.plane(dav1d::PlanarImageComponent::Y);
    let rows = plane.chunks(stride).take(height);
    let codes: Vec<u16> = match bits {
        8 => rows
            .flat_map(|row| row[..width].iter().map(|&v| u16::from(v)))
            .collect(),
        9..=16 => rows
            .flat_map(|row| {
                row[..2 * width]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| u16::from_ne_bytes(b))
            })
            .collect(),
        _ => return Err(DecodeError::BitDepth(bits)),
    };
    Ok(LumaFrame {
        width: picture.width(),
        height: picture.height(),
        bit_depth: bits as u8,
        codes,
    })
}

/// One encoded sample of the stream, as the visualizer reads it from the store.
pub struct Sample<K> {
    /// Where the sample sits on the timeline; the visualizer uses (time, row id).
    pub key: K,
    pub data: Vec<u8>,
    pub is_keyframe: bool,
}

/// The longest run of samples a seek may decode before it gives up looking for a keyframe.
pub const MAX_GOP: usize = 1000;

/// A decoder that shows any sample of one stream: it remembers the last decoded sample, so playing
/// forward decodes one sample per frame, and a seek decodes from the keyframe at or before the target.
pub struct StreamDecoder<K> {
    decoder: LumaDecoder,
    last: Option<(K, Arc<LumaFrame>)>,
}

impl<K: Copy + Eq> StreamDecoder<K> {
    pub fn new() -> Result<Self, DecodeError> {
        Ok(Self {
            decoder: LumaDecoder::new()?,
            last: None,
        })
    }

    /// The frame `target` shows. `previous(key)` returns the sample before `key` on the timeline.
    pub fn frame(
        &mut self,
        target: Sample<K>,
        mut previous: impl FnMut(K) -> Option<Sample<K>>,
    ) -> Result<Arc<LumaFrame>, DecodeError> {
        if let Some((key, frame)) = &self.last
            && *key == target.key
        {
            return Ok(frame.clone());
        }
        let last_key = self.last.take().map(|(key, _)| key);

        // Walk back to the first sample that can start the decode: a keyframe (restart there),
        // or the sample after the one decoded last (continue from the decoder's state).
        let mut pending = vec![target];
        let restart = loop {
            let oldest = &pending[pending.len() - 1];
            if oldest.is_keyframe {
                break true;
            }
            match previous(oldest.key) {
                Some(sample) if Some(sample.key) == last_key => break false,
                Some(sample) if pending.len() < MAX_GOP => pending.push(sample),
                _ => return Err(DecodeError::NoKeyframe(pending.len())),
            }
        };

        if restart {
            self.decoder.reset();
        }
        let mut frame = None;
        let key = pending[0].key;
        for sample in pending.iter().rev() {
            match self.decoder.decode(&sample.data) {
                Ok(decoded) => frame = Some(decoded),
                Err(err) => {
                    // Unknown reference state: the next request restarts at a keyframe.
                    self.decoder.reset();
                    return Err(err);
                }
            }
        }
        let frame = Arc::new(frame.ok_or(DecodeError::NoPicture)?);
        self.last = Some((key, frame.clone()));
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 24 frames of 32x16 AV1 Main 10-bit monochrome lossless, keyframes at 0, 10 and 20, made with
    /// `depth_video_viewer.encode.encode_av1` (libaom); the codes are `(37 x + 101 y + 53 f) % 1024`.
    const STREAM: &[u8] = include_bytes!("testdata/ramp_32x16_24f_gop10.ivf");
    const FRAMES: usize = 24;

    fn expected(f: usize) -> Vec<u16> {
        (0..16)
            .flat_map(|y| (0..32).map(move |x| ((37 * x + 101 * y + 53 * f) % 1024) as u16))
            .collect()
    }

    fn samples() -> Vec<Vec<u8>> {
        let mut pos = usize::from(u16::from_le_bytes([STREAM[6], STREAM[7]]));
        let mut out = Vec::new();
        while pos < STREAM.len() {
            let size = u32::from_le_bytes(STREAM[pos..pos + 4].try_into().unwrap()) as usize;
            out.push(STREAM[pos + 12..pos + 12 + size].to_vec());
            pos += 12 + size;
        }
        out
    }

    fn sample(samples: &[Vec<u8>], i: usize) -> Sample<usize> {
        Sample {
            key: i,
            data: samples[i].clone(),
            is_keyframe: re_video::is_start_of_gop(&samples[i], re_video::VideoCodec::AV1).unwrap(),
        }
    }

    #[test]
    fn decodes_the_exact_10_bit_codes() {
        let samples = samples();
        assert_eq!(samples.len(), FRAMES);
        let mut decoder = LumaDecoder::new().unwrap();
        for (f, data) in samples.iter().enumerate() {
            let frame = decoder.decode(data).unwrap();
            assert_eq!((frame.width, frame.height, frame.bit_depth), (32, 16, 10));
            assert_eq!(frame.codes, expected(f), "frame {f}");
        }
    }

    #[test]
    fn keyframes_are_every_tenth_sample() {
        let samples = samples();
        let keyframes: Vec<_> = (0..FRAMES)
            .filter(|&i| sample(&samples, i).is_keyframe)
            .collect();
        assert_eq!(keyframes, [0, 10, 20]);
    }

    /// Forward steps, a jump into another GOP, a backward seek and a repeat all show the right frame.
    #[test]
    fn seeks_show_the_requested_frame() {
        let samples = samples();
        let mut decoder = StreamDecoder::new().unwrap();
        for target in [15, 16, 19, 23, 3, 3, 12, 0, 9] {
            let mut visited = 0;
            let frame = decoder
                .frame(sample(&samples, target), |key| {
                    visited += 1;
                    key.checked_sub(1).map(|i| sample(&samples, i))
                })
                .unwrap();
            assert_eq!(frame.codes, expected(target), "frame {target}");
            assert!(
                visited <= 10,
                "frame {target}: walked back {visited} samples"
            );
        }
    }

    #[test]
    fn a_stream_without_keyframe_is_an_error() {
        let samples = samples();
        let mut decoder = StreamDecoder::new().unwrap();
        let result = decoder.frame(sample(&samples, 5), |key| {
            (key > 1).then(|| sample(&samples, key - 1))
        });
        assert!(matches!(result, Err(DecodeError::NoKeyframe(_))));
    }
}
