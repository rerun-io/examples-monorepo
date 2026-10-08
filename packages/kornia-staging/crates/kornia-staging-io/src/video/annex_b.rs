//! Annex-B access unit boundary heuristic.
/// One H.264 access unit (one frame) in Annex-B form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessUnit {
    /// The access unit's bytes, start codes included, owned by the access unit.
    pub data: Vec<u8>,
    /// Whether it holds an IDR slice (with `header-mode=each-idr`, also the SPS and PPS).
    pub keyframe: bool,
}

/// Splits an H.264 Annex-B byte stream into access units as bytes arrive.
///
/// An access unit ends where the next begins: at an access unit delimiter, SPS, PPS or SEI NAL, or at a slice whose
/// `first_mb_in_slice` is 0, once the current unit holds a slice (H.264 section 7.4.1.2.3). The last unit of a stream is
/// complete only at its end ([`AccessUnitSplitter::finish`]), so a live stream is one frame behind.
/// This is a NAL-boundary heuristic for encoder output with ordered pictures, not a full H.264
/// picture parser: it does not resolve arbitrary slice groups, field pictures or reordered timestamps.
///
/// ```
/// use kornia_staging_io::video::AccessUnitSplitter;
/// let mut splitter = AccessUnitSplitter::new();
/// splitter.push(&[0, 0, 1, 0x65, 0x80], &mut Vec::new());
/// assert!(splitter.finish().unwrap().keyframe);
/// ```
#[derive(Debug, Default)]
pub struct AccessUnitSplitter {
    buf: Vec<u8>,
    scan_from: usize,
    has_slice: bool,
    keyframe: bool,
}

impl AccessUnitSplitter {
    /// A splitter with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes of the stream.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The next bytes of the Annex-B stream, any length.
    /// * `out` - Receives every access unit these bytes complete, in stream order.
    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<AccessUnit>) {
        self.buf.extend_from_slice(bytes);
        let mut i = self.scan_from;
        // A start code is 00 00 01; the NAL header byte and the first slice-header byte follow it.
        while i + 4 < self.buf.len() {
            if !(self.buf[i] == 0 && self.buf[i + 1] == 0 && self.buf[i + 2] == 1) {
                i += 1;
                continue;
            }
            let nal_type = self.buf[i + 3] & 0x1f;
            let is_slice = nal_type == 1 || nal_type == 5;
            // first_mb_in_slice is ue(v); its value 0 is the single bit 1.
            let first_slice = is_slice && self.buf[i + 4] & 0x80 != 0;
            let starts_unit = matches!(nal_type, 6..=9 | 14..=18) || first_slice;
            if starts_unit && self.has_slice {
                // A four-byte start code's leading zero belongs to the new unit.
                let start = if i > 0 && self.buf[i - 1] == 0 {
                    i - 1
                } else {
                    i
                };
                let data: Vec<u8> = self.buf.drain(..start).collect();
                out.push(AccessUnit {
                    data,
                    keyframe: self.keyframe,
                });
                self.has_slice = false;
                self.keyframe = false;
                i -= start;
            }
            if is_slice {
                self.has_slice = true;
                self.keyframe |= nal_type == 5;
            }
            i += 3;
        }
        self.scan_from = i;
    }

    /// End of stream: the last access unit, if the buffer holds a slice.
    pub fn finish(&mut self) -> Option<AccessUnit> {
        self.scan_from = 0;
        let has_slice = std::mem::take(&mut self.has_slice);
        let keyframe = std::mem::take(&mut self.keyframe);
        let data = std::mem::take(&mut self.buf);
        (has_slice && !data.is_empty()).then_some(AccessUnit { data, keyframe })
    }
}

/// Whether an Annex-B buffer contains a NAL unit of `kind` (5 = IDR slice, 7 = SPS, 8 = PPS).
// A four-byte start code includes the three-byte suffix. Emulation prevention
// keeps this delimiter out of an encoded NAL payload.
pub fn has_nal(bytes: &[u8], kind: u8) -> bool {
    bytes
        .windows(4)
        .any(|w| w[0] == 0 && w[1] == 0 && w[2] == 1 && w[3] & 0x1f == kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nal(kind: u8, first_slice: bool, payload: &[u8], four_byte: bool) -> Vec<u8> {
        let mut out = if four_byte {
            vec![0, 0, 0, 1]
        } else {
            vec![0, 0, 1]
        };
        out.push(0x60 | kind);
        if kind == 1 || kind == 5 {
            out.push(if first_slice { 0x88 } else { 0x08 });
        }
        out.extend_from_slice(payload);
        out
    }

    fn stream() -> (Vec<u8>, Vec<AccessUnit>) {
        let idr = [
            nal(9, false, &[0x10], true),
            nal(7, false, &[1, 2, 3], true),
            nal(8, false, &[4, 5], true),
            nal(5, true, &[9; 40], true),
        ]
        .concat();
        let p1 = [nal(1, true, &[7; 20], true), nal(1, false, &[7; 10], false)].concat();
        let p2 = [
            nal(1, true, &[8; 25], true),
            nal(12, false, &[0xff; 5], false),
        ]
        .concat();
        let all = [idr.clone(), p1.clone(), p2.clone()].concat();
        let expected = vec![
            AccessUnit {
                data: idr,
                keyframe: true,
            },
            AccessUnit {
                data: p1,
                keyframe: false,
            },
            AccessUnit {
                data: p2,
                keyframe: false,
            },
        ];
        (all, expected)
    }

    #[test]
    fn the_splitter_cuts_access_units_at_aud_sps_and_first_slices_for_any_chunking() {
        let (all, expected) = stream();
        for chunk in [1, 2, 3, 5, 7, 64, all.len()] {
            let mut splitter = AccessUnitSplitter::new();
            let mut units = Vec::new();
            for piece in all.chunks(chunk) {
                splitter.push(piece, &mut units);
            }
            units.extend(splitter.finish());
            assert_eq!(units, expected, "chunk size {chunk}");
        }
    }

    #[test]
    fn a_second_slice_of_the_same_picture_and_filler_stay_in_their_unit() {
        let (all, expected) = stream();
        let mut splitter = AccessUnitSplitter::new();
        let mut units = Vec::new();
        splitter.push(&all, &mut units);
        // The last unit is held until the stream ends.
        assert_eq!(units.len(), 2);
        assert_eq!(splitter.finish(), Some(expected[2].clone()));
        assert!(
            has_nal(&expected[0].data, 7)
                && has_nal(&expected[0].data, 8)
                && has_nal(&expected[0].data, 5)
        );
        assert!(!has_nal(&expected[1].data, 5));
    }

    #[test]
    fn an_empty_or_slice_less_stream_yields_nothing() {
        let mut splitter = AccessUnitSplitter::new();
        let mut units = Vec::new();
        splitter.push(&nal(7, false, &[1, 2, 3], true), &mut units);
        assert!(units.is_empty());
        assert_eq!(splitter.finish(), None);
    }
}
