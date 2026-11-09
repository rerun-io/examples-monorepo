//! AV1 → full-bit-depth luma, and seeking within a stream of samples.
//!
//! Out-of-tree scaffolding: deleted once `DepthVideoStream` ships in Rerun, where `re_video` would gain a raw-luma
//! decode output instead.
//!
//! rav1d decodes 10-bit streams in every build; stock `re_video` then right-shifts the samples to 8 bits
//! (`decode/av1.rs:295-297`, TODO #7594). Here the Y plane is copied as is, so every 10-bit code comes back exact.
//! Decoding is synchronous: a seek decodes from the nearest keyframe, and playing forward continues from the last
//! decoded sample.

use re_rav1d::dav1d;
use re_viewer::external::re_sdk_types::encodings::Blob;
use std::sync::Arc;

// rav1d's 16-bit planes are native-endian and `L16` textures are little-endian, so the rows are copied unchanged.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "needs a little-endian target"
);

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("AV1 decoder: {0}")]
    Dav1d(#[from] dav1d::Error),
    #[error("the AV1 sample produced no picture")]
    NoPicture,
    #[error("an AV1 sample of zero bytes")]
    EmptySample,
    #[error("unsupported luma bit depth {0}")]
    BitDepth(usize),
    #[error("no AV1 keyframe within {0} samples before this one")]
    NoKeyframe(usize),
}

/// One decoded depth frame: the coded luma values as packed little-endian `u16` rows (an `L16` image).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LumaFrame {
    pub width: u32,
    pub height: u32,
    /// Coded bits per sample (8, 10 or 12).
    pub bit_depth: u8,
    pub data: Blob,
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
    pub fn decode<D: AsRef<[u8]> + Send + Sync + 'static>(
        &mut self,
        sample: D,
    ) -> Result<dav1d::Picture, DecodeError> {
        re_tracing::profile_function!();
        // rav1d aborts the process on an empty buffer.
        if sample.as_ref().is_empty() {
            return Err(DecodeError::EmptySample);
        }
        let mut shown = None;
        match self.decoder.send_data(sample, None, None, None) {
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
        shown.ok_or(DecodeError::NoPicture)
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

/// Copy the Y plane without its row padding; 8-bit values are widened to `u16`.
pub fn luma_frame(picture: &dav1d::Picture) -> Result<LumaFrame, DecodeError> {
    let bits = picture
        .bits_per_component()
        .map_or_else(|| picture.bit_depth(), |bpc| bpc.0);
    if !(8..=16).contains(&bits) {
        return Err(DecodeError::BitDepth(bits));
    }
    let width = picture.width() as usize;
    let height = picture.height() as usize;
    let stride = picture.stride(dav1d::PlanarImageComponent::Y) as usize;
    let plane = picture.plane(dav1d::PlanarImageComponent::Y);
    let mut data = Vec::with_capacity(2 * width * height);
    for row in plane.chunks(stride).take(height) {
        if bits == 8 {
            data.extend(
                row[..width]
                    .iter()
                    .flat_map(|&v| u16::from(v).to_le_bytes()),
            );
        } else {
            data.extend_from_slice(&row[..2 * width]);
        }
    }
    Ok(LumaFrame {
        width: picture.width(),
        height: picture.height(),
        bit_depth: bits as u8,
        data: data.into(),
    })
}

/// One encoded sample of a stream. A stream is a slice of these in decode order: sorted by time, then row id.
pub struct Sample<K, D> {
    pub key: K,
    pub data: D,
}

/// The sample that shows at `time`: the last one at or before it. Several samples may share a time point.
pub fn latest_at<T: Ord + Copy, R, D>(samples: &[Sample<(T, R), D>], time: T) -> Option<usize> {
    samples
        .partition_point(|sample| sample.key.0 <= time)
        .checked_sub(1)
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

    /// The last decoded frame, the one `frame` returns again for the same sample.
    pub fn last(&self) -> Option<&LumaFrame> {
        self.last.as_ref().map(|(_, frame)| frame.as_ref())
    }

    /// The frame that `samples[target]` shows.
    pub fn frame<D: AsRef<[u8]> + Clone + Send + Sync + 'static>(
        &mut self,
        samples: &[Sample<K, D>],
        target: usize,
    ) -> Result<Arc<LumaFrame>, DecodeError> {
        if let Some((key, frame)) = &self.last
            && *key == samples[target].key
        {
            return Ok(frame.clone());
        }
        let last_key = self.last.take().map(|(key, _)| key);

        // Walk back to the first sample that can start the decode: a keyframe (restart there),
        // or the sample after the one decoded last (continue from the decoder's state).
        let mut start = target;
        let restart = loop {
            let is_keyframe =
                re_video::is_start_of_gop(samples[start].data.as_ref(), re_video::VideoCodec::AV1)
                    .unwrap_or(false);
            if is_keyframe {
                break true;
            }
            if start == 0 || target - start + 1 >= MAX_GOP {
                return Err(DecodeError::NoKeyframe(target - start + 1));
            }
            if Some(samples[start - 1].key) == last_key {
                break false;
            }
            start -= 1;
        };

        if restart {
            self.decoder.reset();
        }
        // Only the target's picture is copied out; the samples before it only build the reference state.
        let mut picture = None;
        for sample in &samples[start..=target] {
            match self.decoder.decode(sample.data.clone()) {
                Ok(decoded) => picture = Some(decoded),
                Err(err) => {
                    // Unknown reference state: the next request restarts at a keyframe.
                    self.decoder.reset();
                    return Err(err);
                }
            }
        }
        let frame = Arc::new(luma_frame(&picture.ok_or(DecodeError::NoPicture)?)?);
        self.last = Some((samples[target].key, frame.clone()));
        Ok(frame)
    }
}

/// The test stream: 24 frames of 32x16 AV1 Main 10-bit monochrome lossless, keyframes at 0, 10 and 20, made with
/// `depth_video_viewer.encode.encode_av1` (libaom); the codes are `(37 x + 101 y + 53 f) % 1024`.
#[cfg(test)]
pub mod testdata {
    use super::LumaFrame;

    const STREAM: &[u8] = include_bytes!("testdata/ramp_32x16_24f_gop10.ivf");
    pub const FRAMES: usize = 24;

    pub fn expected(f: usize) -> Vec<u16> {
        (0..16)
            .flat_map(|y| (0..32).map(move |x| ((37 * x + 101 * y + 53 * f) % 1024) as u16))
            .collect()
    }

    pub fn codes(frame: &LumaFrame) -> Vec<u16> {
        frame
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect()
    }

    /// The temporal units of the IVF file, in order.
    pub fn units() -> Vec<Vec<u8>> {
        let mut pos = usize::from(u16::from_le_bytes([STREAM[6], STREAM[7]]));
        let mut out = Vec::new();
        while pos < STREAM.len() {
            let size = u32::from_le_bytes(STREAM[pos..pos + 4].try_into().unwrap()) as usize;
            out.push(STREAM[pos + 12..pos + 12 + size].to_vec());
            pos += 12 + size;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testdata::{FRAMES, codes, expected, units};
    use super::*;

    /// The fixture's samples with `key(frame index)` as keys.
    fn samples<K>(key: impl Fn(usize) -> K) -> Vec<Sample<K, Vec<u8>>> {
        units()
            .into_iter()
            .enumerate()
            .map(|(f, data)| Sample { key: key(f), data })
            .collect()
    }

    #[test]
    fn decodes_the_exact_10_bit_codes() {
        let samples = samples(|f| f);
        assert_eq!(samples.len(), FRAMES);
        let mut decoder = LumaDecoder::new().unwrap();
        for (f, sample) in samples.iter().enumerate() {
            let frame = luma_frame(&decoder.decode(sample.data.clone()).unwrap()).unwrap();
            assert_eq!((frame.width, frame.height, frame.bit_depth), (32, 16, 10));
            assert_eq!(codes(&frame), expected(f), "frame {f}");
        }
    }

    /// Forward steps, a jump into another GOP, a backward seek and a repeat all show the right frame.
    #[test]
    fn seeks_show_the_requested_frame() {
        let samples = samples(|f| f);
        let mut decoder = StreamDecoder::new().unwrap();
        for target in [15, 16, 19, 23, 3, 3, 12, 0, 9] {
            let frame = decoder.frame(&samples, target).unwrap();
            assert_eq!(codes(&frame), expected(target), "frame {target}");
        }
    }

    /// Two samples per time point, keyframes sharing their time with the next frame: the time cursor
    /// shows the later row, and decoding it still starts at the keyframe of the same time.
    #[test]
    fn samples_sharing_a_time_point() {
        let samples = samples(|f| (f / 2, f));
        let mut decoder = StreamDecoder::new().unwrap();
        for (time, frame) in [(5, 11), (0, 1), (10, 21), (6, 13)] {
            let target = latest_at(&samples, time).unwrap();
            assert_eq!(target, frame);
            let decoded = decoder.frame(&samples, target).unwrap();
            assert_eq!(codes(&decoded), expected(frame), "time {time}");
        }
        assert_eq!(latest_at(&samples[1..], 0), Some(0));
        assert!(latest_at(&samples[2..], 0).is_none());
    }

    #[test]
    fn a_stream_without_keyframe_is_an_error() {
        let samples = samples(|f| f);
        let mut decoder = StreamDecoder::new().unwrap();
        let result = decoder.frame(&samples[1..], 4);
        assert!(matches!(result, Err(DecodeError::NoKeyframe(5))));
    }
}
