//! The metadata record (record 1, `ExtraMetadata`): a protobuf message read with a minimal wire decoder.
use super::{malformed, Insta360Error};
use kornia_staging_3d::camera::formats::insta360::WindowCrop;

/// The metadata protobuf fields the reader uses (record 1, `ExtraMetadata`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Insta360Metadata {
    /// Field 2, e.g. `Insta360 X4 Air`.
    pub camera_type: String,
    /// Field 3, e.g. `v1.2.7_build1`.
    pub firmware: String,
    /// Field 5, the V1 lens string.
    pub offset_v1: Option<String>,
    /// Field 53, the V2 lens string.
    pub offset_v2: Option<String>,
    /// Field 54, the V3 lens string (Mei sphere with Brown terms).
    pub offset_v3: Option<String>,
    /// Field 111, the V6 lens string (Mei sphere with radial, tangential and thin-prism terms).
    pub offset_v6: Option<String>,
    /// Field 27.
    pub window_crop: Option<WindowCrop>,
    /// Field 19 (or 44), the per-lens picture size.
    pub resolution: Option<[u32; 2]>,
    /// Field 20, nominal frames per second.
    pub frame_rate: Option<f64>,
    /// Field 24, the first video frame's time on the trailer clock, microseconds.
    pub first_frame_timestamp_us: i64,
    /// Field 25, rolling-shutter readout time, milliseconds.
    pub rolling_shutter_ms: Option<f64>,
    /// Field 28, the vendor's gyro timestamp shift, milliseconds (subtracted from IMU times by the vendor's tools).
    pub gyro_timestamp_ms: Option<f64>,
    /// Field 62: IMU items are raw 16-bit counts (20 bytes) instead of doubles (56 bytes).
    pub is_raw_gyro: bool,
    /// Field 65.1, accelerometer full scale, g.
    pub acc_range_g: Option<f64>,
    /// Field 65.2, gyroscope full scale, degrees per second.
    pub gyro_range_dps: Option<f64>,
    /// Field 131 `stream_type` (4 = two tracks in reverse order).
    pub stream_type: u64,
    /// Field 80 `pano_record_multi_track_order` (1 = track 0 holds stream 10, i.e. lens 1).
    pub multi_track_order: u64,
}

impl Insta360Metadata {
    /// Video track index of each lens: reversed exactly when the clip states `Track0Stream10` (or, without that
    /// field, `DualStreamTrackReverse`), the rule telemetry-parser documents for the vendor's player.
    pub fn lens_track(&self) -> [usize; 2] {
        let reversed = match self.multi_track_order {
            1 => true,
            2 => false,
            _ => self.stream_type == 4,
        };
        if reversed {
            [1, 0]
        } else {
            [0, 1]
        }
    }
}

/// One protobuf wire value.
enum Wire<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl Wire<'_> {
    /// A numeric field whatever its wire type: varints as integers, fixed widths as IEEE doubles and floats.
    fn number(&self) -> Option<f64> {
        match *self {
            Wire::Varint(v) => Some(v as i64 as f64),
            Wire::Fixed64(v) => Some(f64::from_bits(v)),
            Wire::Fixed32(v) => Some(f64::from(f32::from_bits(v))),
            Wire::Bytes(_) => None,
        }
    }
}

/// Decode one protobuf message into `(field number, value)` pairs (protobuf encoding spec, wire types 0, 1, 2, 5).
fn protobuf_fields(b: &[u8]) -> Result<Vec<(u64, Wire<'_>)>, Insta360Error> {
    let varint = |at: &mut usize| -> Result<u64, Insta360Error> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *b
                .get(*at)
                .ok_or_else(|| malformed("protobuf varint past the end"))?;
            *at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(malformed("protobuf varint longer than 10 bytes"))
    };
    let fixed = |at: &mut usize, n: usize| -> Result<u64, Insta360Error> {
        let bytes = b
            .get(*at..*at + n)
            .ok_or_else(|| malformed("protobuf fixed field past the end"))?;
        *at += n;
        Ok(bytes
            .iter()
            .rev()
            .fold(0u64, |acc, &x| (acc << 8) | u64::from(x)))
    };
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < b.len() {
        let key = varint(&mut at)?;
        let value = match key & 7 {
            0 => Wire::Varint(varint(&mut at)?),
            1 => Wire::Fixed64(fixed(&mut at, 8)?),
            2 => {
                let len = usize::try_from(varint(&mut at)?)
                    .map_err(|_| malformed("protobuf length overflow"))?;
                let bytes = b
                    .get(at..at.saturating_add(len))
                    .ok_or_else(|| malformed("protobuf bytes past the end"))?;
                at += len;
                Wire::Bytes(bytes)
            }
            5 => Wire::Fixed32(fixed(&mut at, 4)? as u32),
            wire => return Err(malformed(format!("protobuf wire type {wire}"))),
        };
        out.push((key >> 3, value));
    }
    Ok(out)
}

pub(super) fn parse_metadata(b: &[u8]) -> Result<Insta360Metadata, Insta360Error> {
    let mut m = Insta360Metadata::default();
    let mut first_frame = None;
    let text = |w: &Wire| match w {
        Wire::Bytes(s) => Some(String::from_utf8_lossy(s).into_owned()),
        _ => None,
    };
    // Fields 1 to 6 of a submessage, 0 when absent; every caller reads only these, so higher field numbers are
    // ignored and no allocation depends on a field number from the file.
    let submessage = |w: &Wire| -> Result<[f64; 6], Insta360Error> {
        let mut values = [0.0; 6];
        let Wire::Bytes(s) = w else {
            return Ok(values);
        };
        for (n, v) in protobuf_fields(s)? {
            if let Some(slot) = (n as usize).checked_sub(1).and_then(|i| values.get_mut(i)) {
                *slot = v.number().unwrap_or(0.0);
            }
        }
        Ok(values)
    };
    for (field, value) in protobuf_fields(b)? {
        match field {
            2 => m.camera_type = text(&value).unwrap_or_default(),
            3 => m.firmware = text(&value).unwrap_or_default(),
            5 => m.offset_v1 = text(&value),
            53 => m.offset_v2 = text(&value),
            54 => m.offset_v3 = text(&value),
            111 => m.offset_v6 = text(&value),
            19 | 44 => {
                let v = submessage(&value)?;
                if v[0] > 0.0 && v[1] > 0.0 {
                    m.resolution = Some([v[0] as u32, v[1] as u32]);
                }
            }
            20 => m.frame_rate = value.number(),
            24 => {
                if let Wire::Varint(v) = value {
                    first_frame = Some(v as i64);
                }
            }
            25 => m.rolling_shutter_ms = value.number(),
            27 => {
                let v = submessage(&value)?;
                if v[..4].iter().all(|&x| x > 0.0) {
                    m.window_crop = Some(WindowCrop {
                        src: [v[0] as u32, v[1] as u32],
                        dst: [v[2] as u32, v[3] as u32],
                        offset: [v[4] as i64, v[5] as i64],
                    });
                }
            }
            28 => m.gyro_timestamp_ms = value.number(),
            62 => m.is_raw_gyro = value.number().is_some_and(|v| v != 0.0),
            65 => {
                let v = submessage(&value)?;
                m.acc_range_g = Some(v[0]).filter(|&x| x > 0.0);
                m.gyro_range_dps = Some(v[1]).filter(|&x| x > 0.0);
            }
            80 => m.multi_track_order = value.number().unwrap_or(0.0) as u64,
            131 => m.stream_type = value.number().unwrap_or(0.0) as u64,
            _ => {}
        }
    }
    m.first_frame_timestamp_us =
        first_frame.ok_or_else(|| malformed("metadata has no first_frame_timestamp (field 24)"))?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::super::testing::{bytes_field, metadata, varint_field};
    use super::*;

    #[test]
    fn a_huge_submessage_field_number_allocates_nothing_and_is_ignored() {
        // Field numbers 2^40 and 2^28 inside the resolution, crop and range submessages: no allocation sized by
        // them (a panic or a multi-gigabyte buffer before), and the fields 1 to 6 still read.
        let mut m = metadata(true);
        for field in [19, 27, 65] {
            let mut sub = Vec::new();
            varint_field(1, 1920, &mut sub);
            varint_field(2, 1920, &mut sub);
            varint_field(1 << 40, 7, &mut sub);
            varint_field(1 << 28, 7, &mut sub);
            bytes_field(field, &sub, &mut m);
        }
        let parsed = parse_metadata(&m).unwrap();
        assert_eq!(parsed.resolution, Some([1920, 1920]));
        assert_eq!(
            (parsed.acc_range_g, parsed.gyro_range_dps),
            (Some(1920.0), Some(1920.0))
        );
    }
}
