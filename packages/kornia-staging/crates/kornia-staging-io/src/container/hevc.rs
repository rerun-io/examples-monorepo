//! HEVC sample conversion from MP4's length-prefixed form to Annex B.

/// Why a sample could not be converted.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HevcError {
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
const AUD: u8 = 35;

/// The 6-bit HEVC `nal_unit_type` (ITU-T H.265 section 7.3.1.2).
pub fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map_or(0, |b| (b >> 1) & 0x3f)
}

/// Convert one length-prefixed HEVC sample into an Annex B access unit, appended to `out`.
///
/// An IRAP sample (`nal_unit_type` 16-23) gets the out-of-band `parameter_sets` (VPS, SPS, PPS from `hvcC`) in front
/// of its first non-AUD NAL unless it already carries an SPS in band, so a decoder can start at any keyframe — what
/// Rerun's `VideoStream` needs for H.265.
///
/// Returns whether the sample holds an IRAP picture.
///
/// ```
/// use kornia_staging_io::container::hevc::sample_to_annex_b;
/// let sample = [0, 0, 0, 2, 0x26, 0x01]; // one IDR_W_RADL NAL
/// let mut out = Vec::new();
/// assert!(sample_to_annex_b(&sample, 4, &[vec![0x42, 0x01]], &mut out).unwrap());
/// assert_eq!(out, [0, 0, 0, 1, 0x42, 0x01, 0, 0, 0, 1, 0x26, 0x01]);
/// ```
pub fn sample_to_annex_b(
    sample: &[u8],
    nal_length_size: u8,
    parameter_sets: &[Vec<u8>],
    out: &mut Vec<u8>,
) -> Result<bool, HevcError> {
    let n = usize::from(nal_length_size);
    if !matches!(n, 1 | 2 | 4) {
        return Err(HevcError::LengthSize(nal_length_size));
    }
    let mut nals: Vec<&[u8]> = Vec::new();
    let mut at = 0usize;
    while at < sample.len() {
        let prefix = sample.get(at..at + n).ok_or(HevcError::Truncated {
            at,
            len: sample.len(),
        })?;
        let len = prefix
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        let nal = sample
            .get(at + n..at + n + len)
            .filter(|nal| !nal.is_empty())
            .ok_or(HevcError::Truncated {
                at,
                len: sample.len(),
            })?;
        nals.push(nal);
        at += n + len;
    }
    let irap = nals.iter().any(|nal| (16..=23).contains(&nal_type(nal)));
    let in_band = nals.iter().any(|nal| nal_type(nal) == 33);
    let mut inserted = !irap || in_band;
    for nal in nals {
        if !inserted && nal_type(nal) != AUD {
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
        assert!(
            sample_to_annex_b(&prefixed(&[AUD_NAL, IDR, IDR], 4), 4, &sets(), &mut out).unwrap()
        );
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
                sample_to_annex_b(&prefixed(&nals, 2), 2, &sets(), &mut out).unwrap(),
                irap
            );
            assert_eq!(out, annex_b(&nals));
        }
    }

    #[test]
    fn a_length_prefix_that_overruns_the_sample_is_an_error() {
        let mut sample = prefixed(&[TRAIL], 4);
        sample[3] = 9;
        assert_eq!(
            sample_to_annex_b(&sample, 4, &sets(), &mut Vec::new()),
            Err(HevcError::Truncated { at: 0, len: 7 })
        );
        assert_eq!(
            sample_to_annex_b(&sample, 3, &sets(), &mut Vec::new()),
            Err(HevcError::LengthSize(3))
        );
    }
}
