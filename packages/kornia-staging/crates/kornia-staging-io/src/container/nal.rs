//! H.264 and HEVC sample conversion from MP4's length-prefixed form to Annex B.
use super::mp4::VideoCodec;

/// Why a sample could not be converted.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NalError {
    /// A NAL length prefix runs past the end of the sample, or a NAL is empty.
    #[error("NAL length prefix at byte {at} overruns a {len}-byte sample")]
    Truncated {
        /// Offset of the bad prefix.
        at: usize,
        /// Sample length.
        len: usize,
    },
    /// The length prefix size is not 1, 2 or 4 bytes.
    #[error("unsupported NAL length size {0}")]
    LengthSize(u8),
}

const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The `nal_unit_type` of a NAL: 5 bits for H.264 (ITU-T H.264 section 7.3.1), 6 bits for HEVC (H.265 section 7.3.1.2).
pub fn nal_type(codec: VideoCodec, nal: &[u8]) -> u8 {
    let first = nal.first().copied().unwrap_or(0);
    match codec {
        VideoCodec::H264 => first & 0x1f,
        VideoCodec::H265 => (first >> 1) & 0x3f,
    }
}

/// Whether a NAL type starts a picture a decoder can begin at: an IDR slice (H.264 type 5) or an IRAP (HEVC 16-23).
fn is_random_access(codec: VideoCodec, t: u8) -> bool {
    match codec {
        VideoCodec::H264 => t == 5,
        VideoCodec::H265 => (16..=23).contains(&t),
    }
}

/// The access-unit-delimiter NAL type of a codec.
fn aud(codec: VideoCodec) -> u8 {
    match codec {
        VideoCodec::H264 => 9,
        VideoCodec::H265 => 35,
    }
}

/// Convert one length-prefixed H.264 or HEVC sample into an Annex B access unit, appended to `out`.
///
/// A random-access sample (an H.264 IDR, an HEVC IRAP) gets the out-of-band `parameter_sets` (SPS and PPS from
/// `avcC`, or VPS, SPS and PPS from `hvcC`) in front of its first non-AUD NAL unless it carries every one of their
/// types in band, so a decoder can start at any keyframe — what Rerun's `VideoStream` needs. In-band sets follow the
/// inserted ones and so stay the active definitions.
///
/// Returns whether the sample is a random-access picture.
///
/// ```
/// use kornia_staging_io::container::mp4::VideoCodec;
/// use kornia_staging_io::container::nal::sample_to_annex_b;
/// let sample = [0, 0, 0, 2, 0x26, 0x01]; // one HEVC IDR_W_RADL NAL
/// let mut out = Vec::new();
/// assert!(sample_to_annex_b(VideoCodec::H265, &sample, 4, &[vec![0x42, 0x01]], &mut out).unwrap());
/// assert_eq!(out, [0, 0, 0, 1, 0x42, 0x01, 0, 0, 0, 1, 0x26, 0x01]);
/// ```
pub fn sample_to_annex_b(
    codec: VideoCodec,
    sample: &[u8],
    nal_length_size: u8,
    parameter_sets: &[Vec<u8>],
    out: &mut Vec<u8>,
) -> Result<bool, NalError> {
    let n = usize::from(nal_length_size);
    if !matches!(n, 1 | 2 | 4) {
        return Err(NalError::LengthSize(nal_length_size));
    }
    let mut nals: Vec<&[u8]> = Vec::new();
    let mut at = 0usize;
    while at < sample.len() {
        let prefix = sample.get(at..at + n).ok_or(NalError::Truncated {
            at,
            len: sample.len(),
        })?;
        let len = prefix
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        let nal = sample
            .get(at + n..at + n + len)
            .filter(|nal| !nal.is_empty())
            .ok_or(NalError::Truncated {
                at,
                len: sample.len(),
            })?;
        nals.push(nal);
        at += n + len;
    }
    let aud = aud(codec);
    let irap = nals
        .iter()
        .any(|nal| is_random_access(codec, nal_type(codec, nal)));
    let in_band = parameter_sets.iter().all(|set| {
        nals.iter()
            .any(|nal| nal_type(codec, nal) == nal_type(codec, set))
    });
    let mut inserted = !irap || in_band;
    for nal in nals {
        if !inserted && nal_type(codec, nal) != aud {
            for set in parameter_sets {
                out.extend_from_slice(&START_CODE);
                out.extend_from_slice(set);
            }
            inserted = true;
        }
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(nal);
    }
    Ok(irap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefixed(nals: &[&[u8]], n: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&(nal.len() as u32).to_be_bytes()[4 - n..]);
            out.extend_from_slice(nal);
        }
        out
    }

    fn annex_b(nals: &[&[u8]]) -> Vec<u8> {
        nals.iter()
            .flat_map(|nal| START_CODE.iter().chain(nal.iter()).copied())
            .collect()
    }

    const VPS: &[u8] = &[0x40, 0x01, 1];
    const SPS: &[u8] = &[0x42, 0x01, 2];
    const PPS: &[u8] = &[0x44, 0x01, 3];
    const IDR: &[u8] = &[0x26, 0x01, 0xaf];
    const TRAIL: &[u8] = &[0x02, 0x01, 0xd0];
    const AUD_NAL: &[u8] = &[0x46, 0x01, 0x50];

    fn sets() -> Vec<Vec<u8>> {
        vec![VPS.to_vec(), SPS.to_vec(), PPS.to_vec()]
    }

    #[test]
    fn an_irap_sample_gets_the_parameter_sets_after_its_aud() {
        let mut out = Vec::new();
        assert!(sample_to_annex_b(
            VideoCodec::H265,
            &prefixed(&[AUD_NAL, IDR, IDR], 4),
            4,
            &sets(),
            &mut out
        )
        .unwrap());
        assert_eq!(out, annex_b(&[AUD_NAL, VPS, SPS, PPS, IDR, IDR]));
    }

    #[test]
    fn a_non_irap_sample_and_an_irap_with_in_band_sets_are_only_reframed() {
        for (nals, irap) in [
            (vec![TRAIL, TRAIL], false),
            (vec![VPS, SPS, PPS, IDR], true),
        ] {
            let mut out = Vec::new();
            assert_eq!(
                sample_to_annex_b(VideoCodec::H265, &prefixed(&nals, 2), 2, &sets(), &mut out)
                    .unwrap(),
                irap
            );
            assert_eq!(out, annex_b(&nals));
        }
    }

    #[test]
    fn an_h264_idr_gets_its_sps_and_pps_and_a_p_slice_does_not() {
        let (sps, pps): (&[u8], &[u8]) = (&[0x67, 0x64, 0x28], &[0x68, 0xee]);
        let (aud, idr, p_slice): (&[u8], &[u8], &[u8]) =
            (&[0x09, 0xf0], &[0x65, 0x88], &[0x41, 0x9a]);
        let h264_sets = vec![sps.to_vec(), pps.to_vec()];
        let mut out = Vec::new();
        assert!(sample_to_annex_b(
            VideoCodec::H264,
            &prefixed(&[aud, idr], 4),
            4,
            &h264_sets,
            &mut out
        )
        .unwrap());
        assert_eq!(out, annex_b(&[aud, sps, pps, idr]));
        let mut out = Vec::new();
        assert!(!sample_to_annex_b(
            VideoCodec::H264,
            &prefixed(&[p_slice], 4),
            4,
            &h264_sets,
            &mut out
        )
        .unwrap());
        assert_eq!(out, annex_b(&[p_slice]));
        // 0x26 is an HEVC IDR but an H.264 type-6 SEI: the codec decides.
        assert!(!sample_to_annex_b(
            VideoCodec::H264,
            &prefixed(&[IDR], 4),
            4,
            &h264_sets,
            &mut Vec::new()
        )
        .unwrap());
    }

    #[test]
    fn a_keyframe_missing_some_parameter_sets_in_band_gets_the_out_of_band_ones_first() {
        // An H.264 IDR carrying its SPS but not its PPS; the in-band SPS follows and so stays the active one.
        let (sps, pps, in_band_sps): (&[u8], &[u8], &[u8]) =
            (&[0x67, 0x64, 0x28], &[0x68, 0xee], &[0x67, 0x64, 0x29]);
        let idr: &[u8] = &[0x65, 0x88];
        let mut out = Vec::new();
        assert!(sample_to_annex_b(
            VideoCodec::H264,
            &prefixed(&[in_band_sps, idr], 4),
            4,
            &[sps.to_vec(), pps.to_vec()],
            &mut out
        )
        .unwrap());
        assert_eq!(out, annex_b(&[sps, pps, in_band_sps, idr]));
        // An HEVC IRAP with SPS and PPS in band but no VPS.
        let mut out = Vec::new();
        sample_to_annex_b(
            VideoCodec::H265,
            &prefixed(&[SPS, PPS, IDR], 4),
            4,
            &sets(),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, annex_b(&[VPS, SPS, PPS, SPS, PPS, IDR]));
    }

    #[test]
    fn a_length_prefix_that_overruns_the_sample_is_an_error() {
        let mut sample = prefixed(&[TRAIL], 4);
        sample[3] = 9;
        assert_eq!(
            sample_to_annex_b(VideoCodec::H265, &sample, 4, &sets(), &mut Vec::new()),
            Err(NalError::Truncated { at: 0, len: 7 })
        );
        assert_eq!(
            sample_to_annex_b(VideoCodec::H265, &sample, 3, &sets(), &mut Vec::new()),
            Err(NalError::LengthSize(3))
        );
    }
}
