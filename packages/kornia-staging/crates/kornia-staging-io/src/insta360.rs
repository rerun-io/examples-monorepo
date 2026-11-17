//! Insta360 dual-fisheye `.insv` clips: the two HEVC lens tracks plus the vendor trailer.
//!
//! The trailer sits at the end of the file (on recent bodies wrapped in a top-level `inst` box). It ends in a 72-byte
//! footer (32 zero bytes, `u32` trailer size, `u32` version, a 32-character magic), preceded by a 6-byte descriptor
//! `[format][id][u32 size]` of the index record, whose 10-byte slots `[id][format][u32 size][u32 offset]` locate every
//! other record relative to the trailer start. Only the index is read: on the X4 Air the records sit in 256 KiB slots
//! and walking descriptors backward from the footer fails. Record layouts follow telemetry-parser's `insta360`
//! module (<https://github.com/AdrianEddy/telemetry-parser>), checked against a real X4 Air clip.
use crate::container::hevc::{sample_to_annex_b, HevcError};
use crate::container::mp4::{read_hevc_tracks, Mp4Error, VideoTrack};
use kornia_staging_3d::camera::insta360::{mul, transpose, vendor_rotation};
pub use kornia_staging_3d::camera::insta360::{LensCalibration, LENS_HALF_FIELD_RAD};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 32] = b"8db42d694ccc418790edff439fe026bf";
const FOOTER: u64 = 72;
const RECORD_METADATA: u8 = 1;
const RECORD_IMU: u8 = 3;
const RECORD_EXPOSURE_VIN0: u8 = 4;
const RECORD_EXPOSURE_VIN1: u8 = 12;
const STANDARD_GRAVITY: f64 = 9.80665;

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
    Hevc(#[from] HevcError),
    /// The trailer, a record or a calibration string is malformed or missing.
    #[error("malformed insv trailer: {0}")]
    Malformed(String),
}

fn malformed(what: impl Into<String>) -> Insta360Error {
    Insta360Error::Malformed(what.into())
}

/// `window_crop_info`: the sensor window a calibration string describes and the part of it the picture holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCrop {
    /// Source window width and height, pixels.
    pub src: [u32; 2],
    /// Recorded window width and height, pixels.
    pub dst: [u32; 2],
    /// Crop offset, pixels.
    pub offset: [i64; 2],
}

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

fn parse_metadata(b: &[u8]) -> Result<Insta360Metadata, Insta360Error> {
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

/// One IMU sample in SI units on the trailer clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuSample {
    /// Time on the trailer clock (the clock of `first_frame_timestamp`), microseconds, as stored.
    pub t_us: i64,
    /// Specific force, m/s^2, in the IMU's own axes.
    pub accel_m_s2: [f64; 3],
    /// Angular rate, rad/s, in the IMU's own axes.
    pub gyro_rad_s: [f64; 3],
}

/// One exposure-record entry: when a frame was taken and for how long.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exposure {
    /// Time on the trailer clock, microseconds.
    pub t_us: i64,
    /// Exposure (shutter) time, seconds.
    pub exposure_s: f64,
}

/// The trailer's decoded records.
#[derive(Clone, Debug, PartialEq)]
pub struct Trailer {
    /// Record 1.
    pub metadata: Insta360Metadata,
    /// Record 3, scaled to SI units.
    pub imu: Vec<ImuSample>,
    /// Record 4, the first sensor's (`vin0`) exposures.
    pub exposures_vin0: Vec<Exposure>,
    /// Record 12, the second sensor's (`vin1`) exposures; empty when absent.
    pub exposures_vin1: Vec<Exposure>,
}

fn le(b: &[u8], at: usize, n: usize) -> u64 {
    b[at..at + n]
        .iter()
        .rev()
        .fold(0u64, |acc, &x| (acc << 8) | u64::from(x))
}

fn items(record: &[u8], width: usize, what: &str) -> Result<usize, Insta360Error> {
    if !record.len().is_multiple_of(width) {
        return Err(malformed(format!(
            "{what} record of {} bytes is not a whole number of {width}-byte items",
            record.len()
        )));
    }
    Ok(record.len() / width)
}

fn parse_imu(record: &[u8], m: &Insta360Metadata) -> Result<Vec<ImuSample>, Insta360Error> {
    if m.is_raw_gyro {
        // u64 t_us, then six u16 counts offset by 32768: accel xyz, gyro xyz; full scale from field 65.
        let acc = m
            .acc_range_g
            .ok_or_else(|| malformed("raw IMU without an accelerometer range (field 65)"))?
            / 32768.0
            * STANDARD_GRAVITY;
        let gyr = m
            .gyro_range_dps
            .ok_or_else(|| malformed("raw IMU without a gyroscope range (field 65)"))?
            / 32768.0
            * std::f64::consts::PI
            / 180.0;
        let count = items(record, 20, "raw IMU")?;
        Ok((0..count)
            .map(|i| {
                let b = &record[i * 20..];
                let v = |k: usize| le(b, 8 + 2 * k, 2) as f64 - 32768.0;
                ImuSample {
                    t_us: le(b, 0, 8) as i64,
                    accel_m_s2: [v(0) * acc, v(1) * acc, v(2) * acc],
                    gyro_rad_s: [v(3) * gyr, v(4) * gyr, v(5) * gyr],
                }
            })
            .collect())
    } else {
        // u64 t_ms, then six f64: accel in g, gyro in rad/s.
        let count = items(record, 56, "float IMU")?;
        (0..count)
            .map(|i| {
                let b = &record[i * 56..];
                let v = |k: usize| f64::from_bits(le(b, 8 + 8 * k, 8));
                Ok(ImuSample {
                    t_us: (le(b, 0, 8) as i64)
                        .checked_mul(1000)
                        .ok_or_else(|| malformed("float IMU time overflows microseconds"))?,
                    accel_m_s2: [
                        v(0) * STANDARD_GRAVITY,
                        v(1) * STANDARD_GRAVITY,
                        v(2) * STANDARD_GRAVITY,
                    ],
                    gyro_rad_s: [v(3), v(4), v(5)],
                })
            })
            .collect()
    }
}

fn parse_exposures(record: &[u8]) -> Result<Vec<Exposure>, Insta360Error> {
    let count = items(record, 16, "exposure")?;
    Ok((0..count)
        .map(|i| Exposure {
            t_us: le(record, 16 * i, 8) as i64,
            exposure_s: f64::from_bits(le(record, 16 * i + 8, 8)),
        })
        .collect())
}

/// Read the trailer at the end of a clip through its index record.
pub fn read_trailer<R: Read + Seek>(reader: &mut R) -> Result<Trailer, Insta360Error> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    if file_len < FOOTER + 6 {
        return Err(malformed("file shorter than the trailer footer"));
    }
    let mut read_at = |at: u64, len: usize| -> Result<Vec<u8>, Insta360Error> {
        let mut buf = vec![0u8; len];
        reader.seek(SeekFrom::Start(at))?;
        reader.read_exact(&mut buf)?;
        Ok(buf)
    };
    let footer = read_at(file_len - FOOTER, FOOTER as usize)?;
    if &footer[40..72] != MAGIC {
        return Err(malformed(
            "no Insta360 trailer magic at the end of the file",
        ));
    }
    let trailer_size = le(&footer, 32, 4);
    let start = file_len
        .checked_sub(trailer_size)
        .ok_or_else(|| malformed(format!("trailer size {trailer_size} exceeds the file")))?;
    let descriptor = read_at(file_len - FOOTER - 6, 6)?;
    let (id, index_size) = (descriptor[1], le(&descriptor, 2, 4));
    if id != 0 || !index_size.is_multiple_of(10) || index_size + FOOTER + 6 > trailer_size {
        return Err(malformed(format!(
            "index descriptor id {id} size {index_size}"
        )));
    }
    let index = read_at(file_len - FOOTER - 6 - index_size, index_size as usize)?;
    let mut records: [Option<Vec<u8>>; 3] = [None, None, None];
    let mut metadata = None;
    for slot in index.chunks_exact(10) {
        let (id, size, offset) = (slot[0], le(slot, 2, 4), le(slot, 6, 4));
        let wanted = [
            RECORD_METADATA,
            RECORD_IMU,
            RECORD_EXPOSURE_VIN0,
            RECORD_EXPOSURE_VIN1,
        ];
        if id == 0 || !wanted.contains(&id) {
            continue;
        }
        if offset + size > trailer_size {
            return Err(malformed(format!(
                "record {id} at {offset}+{size} overruns the {trailer_size}-byte trailer"
            )));
        }
        let bytes = read_at(start + offset, size as usize)?;
        match id {
            RECORD_METADATA => metadata = Some(parse_metadata(&bytes)?),
            RECORD_IMU => records[0] = Some(bytes),
            RECORD_EXPOSURE_VIN0 => records[1] = Some(bytes),
            _ => records[2] = Some(bytes),
        }
    }
    let metadata = metadata.ok_or_else(|| malformed("no metadata record"))?;
    let [imu, vin0, vin1] = records;
    Ok(Trailer {
        imu: parse_imu(&imu.ok_or_else(|| malformed("no IMU record"))?, &metadata)?,
        exposures_vin0: parse_exposures(&vin0.ok_or_else(|| malformed("no exposure record"))?)?,
        exposures_vin1: vin1
            .map(|r| parse_exposures(&r))
            .transpose()?
            .unwrap_or_default(),
        metadata,
    })
}

/// Parse a V3 or V6 lens string and state it for a `track_size` picture.
///
/// Tokens: the lens count, then per lens `xi fx fy cx cy yaw pitch roll tx ty tz <coefficients> W H type`, then one
/// tail token `(version << 16) | (mirror << 8) | lens class`. `W x H` is the calibration frame holding every lens
/// side by side. The window crop maps a lens's frame onto the recorded window about pixel centres (telemetry-parser
/// `rig.rs`, `Offset::windowed`), then the window is scaled to the track: `u = (u' + 0.5) s - 0.5`, `s = track/dst`.
/// Rotations use the vendor's `R_lens<-world = Ry(pitch + 90) Rz(yaw) Rx(roll)`, with an extra `Rz(180)` on lens 0
/// of a two-lens body; the translations are taken as already in lens 0's frame (telemetry-parser `rig.rs`).
pub fn parse_lens_string(
    text: &str,
    crop: Option<WindowCrop>,
    track_size: [u32; 2],
) -> Result<Vec<LensCalibration>, Insta360Error> {
    let tokens: Vec<f64> = text
        .split('_')
        .map(|t| {
            t.parse::<f64>()
                .map_err(|_| malformed(format!("lens string token {t:?} is not a number")))
        })
        .collect::<Result<_, _>>()?;
    let (&count, rest) = tokens
        .split_first()
        .ok_or_else(|| malformed("empty lens string"))?;
    let (&tail, body) = rest
        .split_last()
        .ok_or_else(|| malformed("lens string has no tail token"))?;
    let version = (tail as u64 >> 16) as u8;
    let (n_coeffs, n_radial, n_tangential) = match version {
        3 => (5, 3, 2),
        6 => (13, 5, 4),
        v => {
            return Err(malformed(format!(
                "lens string version {v} has no typed reader (V3 and V6 do)"
            )))
        }
    };
    let per_lens = 11 + n_coeffs + 3;
    if count.fract() != 0.0 || !(1.0..=8.0).contains(&count) {
        return Err(malformed(format!(
            "V{version} lens string states {count} lenses, not a whole number from 1 to 8"
        )));
    }
    let count = count as usize;
    if body.len() != count * per_lens {
        return Err(malformed(format!(
            "V{version} lens string: {} tokens for {count} lenses of {per_lens}",
            body.len()
        )));
    }
    let mut lenses = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    for (i, t) in body.chunks_exact(per_lens).enumerate() {
        let frame = [t[per_lens - 3] / count as f64, t[per_lens - 2]];
        let (mut cx, mut cy, mut fx, mut fy) = (t[3] - frame[0] * i as f64, t[4], t[1], t[2]);
        let mut window = frame;
        if let Some(c) = crop {
            let src = [f64::from(c.src[0]), f64::from(c.src[1])];
            let dst = [f64::from(c.dst[0]), f64::from(c.dst[1])];
            let scale = src[0] / frame[0];
            cx = scale * (cx + 0.5) - 0.5 - (src[0] - dst[0]) * 0.5 - c.offset[0] as f64;
            cy = scale * (cy + 0.5) - 0.5 - (src[1] - dst[1]) * 0.5 - c.offset[1] as f64;
            (fx, fy, window) = (fx * scale, fy * scale, dst);
        }
        let s = [
            f64::from(track_size[0]) / window[0],
            f64::from(track_size[1]) / window[1],
        ];
        let ypr_deg = [t[5], t[6], t[7]];
        rotations.push(vendor_rotation(ypr_deg, count == 2 && i == 0));
        let coeffs = &t[11..11 + n_coeffs];
        lenses.push(LensCalibration {
            version,
            xi: t[0],
            fx: fx * s[0],
            fy: fy * s[1],
            cx: (cx + 0.5) * s[0] - 0.5,
            cy: (cy + 0.5) * s[1] - 0.5,
            radial: coeffs[..n_radial].to_vec(),
            tangential: coeffs[n_radial..n_radial + n_tangential].to_vec(),
            prism: coeffs[n_radial + n_tangential..].to_vec(),
            ypr_deg,
            translation_m: [t[8], t[9], t[10]],
            lens_from_rig_rotation: [[0.0; 3]; 3],
            lens_from_rig_translation_m: [0.0; 3],
            pano_from_lens_rotation: [[0.0; 3]; 3],
            pano_centre_m: [0.0; 3],
        });
    }
    // Rig frame: lens 0's axes, origin at the mean of the lens centres. lens_from_rig = (R_i R_0^T, -R (c_i - o)).
    let origin: [f64; 3] = std::array::from_fn(|k| {
        lenses.iter().map(|l| l.translation_m[k]).sum::<f64>() / count as f64
    });
    let r0t = transpose(&rotations[0]);
    let pano_base = vendor_rotation([0.0, 0.0, 90.0], count == 2);
    for (lens, r) in lenses.iter_mut().zip(&rotations) {
        lens.pano_from_lens_rotation = mul(&pano_base, &transpose(r));
        let rot = mul(r, &r0t);
        let d: [f64; 3] = std::array::from_fn(|k| lens.translation_m[k] - origin[k]);
        lens.lens_from_rig_translation_m =
            std::array::from_fn(|row| -(0..3).map(|k| rot[row][k] * d[k]).sum::<f64>());
        lens.lens_from_rig_rotation = rot;
    }
    let pano_from_lens0 = lenses[0].pano_from_lens_rotation;
    for lens in &mut lenses {
        let d: [f64; 3] = std::array::from_fn(|k| lens.translation_m[k] - origin[k]);
        lens.pano_centre_m =
            std::array::from_fn(|row| (0..3).map(|k| pano_from_lens0[row][k] * d[k]).sum::<f64>());
    }
    Ok(lenses)
}

/// An Insta360 dual-fisheye clip: trailer, factory lenses and the two HEVC lens tracks.
#[derive(Clone, Debug)]
pub struct Insta360Clip {
    /// File the clip was read from.
    pub path: PathBuf,
    /// Metadata, IMU and exposure records.
    pub trailer: Trailer,
    /// V3 factory calibration per lens, in lens order, at the track resolution with the window crop applied.
    pub lenses: Vec<LensCalibration>,
    /// V6 factory calibration per lens when the clip carries it; empty otherwise.
    pub lenses_v6: Vec<LensCalibration>,
    /// Video track index (into [`Insta360Clip::tracks`]) of each lens.
    pub lens_track: [usize; 2],
    /// HEVC video tracks in file order.
    pub tracks: Vec<VideoTrack>,
    /// Index of the exposure entry that is frame 0: the entry stamped `first_frame_timestamp`.
    pub first_frame_entry: usize,
}

impl Insta360Clip {
    /// Read the container index and the trailer. Samples stay on disk; read them with [`Insta360Clip::sample_reader`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Insta360Error> {
        let path = path.as_ref().to_path_buf();
        let mut file = BufReader::new(File::open(&path)?);
        let tracks = read_hevc_tracks(&mut file)?;
        let trailer = read_trailer(&mut file)?;
        if tracks.len() != 2 {
            return Err(malformed(format!(
                "expected 2 HEVC lens tracks, found {}",
                tracks.len()
            )));
        }
        let m = &trailer.metadata;
        let size = [u32::from(tracks[0].width), u32::from(tracks[0].height)];
        let v3 = m
            .offset_v3
            .as_deref()
            .ok_or_else(|| malformed("metadata has no V3 lens string (field 54)"))?;
        let lenses = parse_lens_string(v3, m.window_crop, size)?;
        let lenses_v6 = m
            .offset_v6
            .as_deref()
            .map(|s| parse_lens_string(s, m.window_crop, size))
            .transpose()?
            .unwrap_or_default();
        if lenses.len() != 2 {
            return Err(malformed(format!(
                "expected 2 lenses, the V3 string has {}",
                lenses.len()
            )));
        }
        let first = m.first_frame_timestamp_us;
        let exposures = &trailer.exposures_vin0;
        // In i128, so times anywhere in i64 cannot overflow the difference.
        let distance = |i: usize| (i128::from(exposures[i].t_us) - i128::from(first)).abs();
        let nearest = (0..exposures.len()).min_by_key(|&i| distance(i));
        let period_us = 1e6 / m.frame_rate.unwrap_or(30.0);
        let first_frame_entry = nearest
            .filter(|&i| (distance(i) as f64) < 0.5 * period_us)
            .ok_or_else(|| {
                malformed(format!(
                    "no exposure entry within half a frame of first_frame_timestamp {first}"
                ))
            })?;
        let frames = tracks.iter().map(|t| t.samples.len()).max().unwrap_or(0);
        if exposures.len() < first_frame_entry + frames {
            return Err(malformed(format!(
                "{} exposure entries after frame 0 for {frames} frames",
                exposures.len() - first_frame_entry
            )));
        }
        Ok(Self {
            path,
            lens_track: m.lens_track(),
            trailer,
            lenses,
            lenses_v6,
            tracks,
            first_frame_entry,
        })
    }

    /// Exposure entries of the video frames of one sensor record (frame `i` is entry `first_frame_entry + i`).
    pub fn frame_exposures(&self, vin1: bool) -> &[Exposure] {
        let record = if vin1 {
            &self.trailer.exposures_vin1
        } else {
            &self.trailer.exposures_vin0
        };
        let frames = self
            .tracks
            .iter()
            .map(|t| t.samples.len())
            .max()
            .unwrap_or(0);
        record
            .get(self.first_frame_entry..self.first_frame_entry + frames)
            .unwrap_or(&[])
    }

    /// Open the file for reading samples.
    pub fn sample_reader(&self) -> Result<SampleReader, Insta360Error> {
        Ok(SampleReader {
            file: File::open(&self.path)?,
            scratch: Vec::new(),
        })
    }
}

/// Reads video samples of a clip as Annex B access units, one at a time.
#[derive(Debug)]
pub struct SampleReader {
    file: File,
    scratch: Vec<u8>,
}

impl SampleReader {
    /// Read sample `index` of `track`, convert it to Annex B with parameter sets on IRAP samples, and append it to
    /// `out`. Returns whether the sample is an IRAP picture.
    pub fn read_annex_b(
        &mut self,
        track: &VideoTrack,
        index: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool, Insta360Error> {
        let sample = track
            .samples
            .get(index)
            .ok_or_else(|| malformed(format!("sample {index} of {}", track.samples.len())))?;
        self.scratch.resize(sample.size as usize, 0);
        self.file.seek(SeekFrom::Start(sample.offset))?;
        self.file.read_exact(&mut self.scratch)?;
        Ok(sample_to_annex_b(
            &self.scratch,
            track.nal_length_size,
            &track.parameter_sets,
            out,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // The real X4 Air strings, from the clip VID_20261116_125200_00_003.insv.
    include!("../tests/common/x4air_lens.rs");

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn the_v3_string_gives_the_recon_lens_numbers_at_1920() {
        let lenses = parse_lens_string(V3, Some(CROP), [1920, 1920]).unwrap();
        let [l0, l1] = [&lenses[0], &lenses[1]];
        assert!(
            close(l0.cx, 960.242, 1e-3) && close(l0.cy, 965.681, 1e-3),
            "{} {}",
            l0.cx,
            l0.cy
        );
        assert!(
            close(l1.cx, 966.509, 1e-3) && close(l1.cy, 965.847, 1e-3),
            "{} {}",
            l1.cx,
            l1.cy
        );
        assert!(close(l0.px_per_rad(), 553.5, 0.05) && close(l1.px_per_rad(), 557.7, 0.05));
        assert_eq!(
            (l0.radial.len(), l0.tangential.len(), l0.prism.len()),
            (3, 2, 0)
        );
        // Image radius at 100 deg: 972.2 and 980.1 px.
        assert!(
            close(l0.radius_px(100f64.to_radians()), 972.2, 0.1)
                && close(l1.radius_px(100f64.to_radians()), 980.1, 0.1)
        );
        // Lens 0 has the rig's axes; lens 1 is turned 179.91 deg; the centres are 33.1 mm apart on the optical axis.
        let r = l1.lens_from_rig_rotation;
        let angle = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        assert!(close(angle, 179.91, 0.01), "{angle}");
        assert!(close(r[0][0], -0.99982, 1e-4) && close(r[2][2], -1.0, 1e-4));
        assert!(l0
            .lens_from_rig_rotation
            .iter()
            .enumerate()
            .all(|(i, row)| row.iter().enumerate().all(|(j, &v)| close(
                v,
                f64::from(u8::from(i == j)),
                1e-12
            ))));
        // Lens 0 sits at +z 16.56 mm from the midpoint, i.e. its lens_from_rig translation is -z.
        let t0 = l0.lens_from_rig_translation_m;
        assert!(
            close(t0[2], -0.0165615, 1e-6) && close(t0[0], 0.0004025, 1e-7),
            "{t0:?}"
        );
        let centre_distance = (0..3).map(|k| (2.0 * t0[k]).powi(2)).sum::<f64>().sqrt();
        assert!(close(centre_distance, 0.0331, 1e-4), "{centre_distance}");
    }

    #[test]
    fn the_v6_string_keeps_tangential_and_prism_terms() {
        let lenses = parse_lens_string(V6, Some(CROP), [1920, 1920]).unwrap();
        assert_eq!(
            (
                lenses[0].version,
                lenses[0].radial.len(),
                lenses[0].tangential.len(),
                lenses[0].prism.len()
            ),
            (6, 5, 4, 4)
        );
        assert!(close(lenses[0].cx, 958.292, 1e-3) && close(lenses[1].cy, 964.396, 1e-3));
    }

    #[test]
    fn without_the_window_crop_the_centre_is_128_calibration_pixels_off() {
        let lenses = parse_lens_string(V3, None, [1920, 1920]).unwrap();
        // 1920/7680 instead of 1920/7424: the centre lands 128 px * 0.25 further from the origin than with the crop.
        assert!(close(lenses[0].cx, (3842.37 + 0.5) * 0.25 - 0.5, 1e-9));
    }

    #[test]
    fn malformed_lens_strings_are_errors_not_panics() {
        // The last three state 1e30, 2.5 and -1 lenses: the count must be a whole number from 1 to 8.
        for text in [
            "",
            "2_1_2",
            "2_x_197632",
            "1_2.3_197632",
            "2_1_2_3_132096",
            "1e30_1_197632",
            "2.5_1_197632",
            "-1_1_197632",
        ] {
            assert!(
                parse_lens_string(text, None, [1920, 1920]).is_err(),
                "{text}"
            );
        }
    }

    // ---- a synthetic trailer --------------------------------------------------------------------------------

    fn key(field: u64, wire: u64, out: &mut Vec<u8>) {
        varint((field << 3) | wire, out);
    }

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    fn bytes_field(field: u64, b: &[u8], out: &mut Vec<u8>) {
        key(field, 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }

    fn varint_field(field: u64, v: u64, out: &mut Vec<u8>) {
        key(field, 0, out);
        varint(v, out);
    }

    fn metadata(raw: bool) -> Vec<u8> {
        let mut m = Vec::new();
        bytes_field(2, b"Insta360 X4 Air", &mut m);
        bytes_field(3, b"v1.2.7_build1", &mut m);
        let mut res = Vec::new();
        varint_field(1, 1920, &mut res);
        varint_field(2, 1920, &mut res);
        bytes_field(19, &res, &mut m);
        varint_field(20, 30, &mut m);
        varint_field(24, 1_000_200, &mut m);
        key(25, 1, &mut m);
        m.extend_from_slice(&15.882978f64.to_le_bytes());
        let mut crop = Vec::new();
        for (i, v) in [7680u64, 7680, 7424, 7424, 0, 0].into_iter().enumerate() {
            varint_field(i as u64 + 1, v, &mut crop);
        }
        bytes_field(27, &crop, &mut m);
        key(28, 1, &mut m);
        m.extend_from_slice(&1.6f64.to_le_bytes());
        bytes_field(54, V3.as_bytes(), &mut m);
        varint_field(62, u64::from(raw), &mut m);
        let mut ranges = Vec::new();
        varint_field(1, 32, &mut ranges);
        varint_field(2, 2000, &mut ranges);
        bytes_field(65, &ranges, &mut m);
        varint_field(80, 1, &mut m);
        varint_field(131, 4, &mut m);
        m
    }

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

    /// A file of `prefix` junk bytes then a trailer holding the given records in 256-byte slots.
    fn trailer_file(records: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let mut body = vec![0u8; 64];
        let mut index = vec![0u8; 310];
        for (id, record) in records {
            let offset = body.len() as u32;
            body.extend_from_slice(record);
            body.resize(body.len().div_ceil(256) * 256, 0xee);
            let slot = &mut index[10 * usize::from(*id)..10 * usize::from(*id) + 10];
            slot[0] = *id;
            slot[2..6].copy_from_slice(&(record.len() as u32).to_le_bytes());
            slot[6..10].copy_from_slice(&offset.to_le_bytes());
        }
        body.extend_from_slice(&index);
        body.extend_from_slice(&[0, 0]);
        body.extend_from_slice(&310u32.to_le_bytes());
        let size = (body.len() + 72) as u32;
        body.extend_from_slice(&[0u8; 32]);
        body.extend_from_slice(&size.to_le_bytes());
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(MAGIC);
        let mut file = vec![0x55u8; 1000];
        file.extend(body);
        file
    }

    fn exposures(times: &[u64]) -> Vec<u8> {
        times
            .iter()
            .flat_map(|&t| {
                let mut item = t.to_le_bytes().to_vec();
                item.extend(((t % 7) as f64 / 100.0).to_le_bytes());
                item
            })
            .collect()
    }

    #[test]
    fn the_index_table_locates_metadata_raw_imu_and_exposures() {
        let mut imu = Vec::new();
        for (t, counts) in [(
            1_000_000u64,
            [
                32768u16 + 1024,
                32768,
                32768 - 1024,
                32768 + 16384,
                32768,
                32768,
            ],
        )] {
            imu.extend_from_slice(&t.to_le_bytes());
            for c in counts {
                imu.extend_from_slice(&c.to_le_bytes());
            }
        }
        let file = trailer_file(&[
            (1, metadata(true)),
            (3, imu),
            (4, exposures(&[1_000_100, 1_000_200, 1_033_567])),
            (12, exposures(&[1_000_234])),
        ]);
        let trailer = read_trailer(&mut Cursor::new(file)).unwrap();
        let m = &trailer.metadata;
        assert_eq!(
            (m.camera_type.as_str(), m.firmware.as_str()),
            ("Insta360 X4 Air", "v1.2.7_build1")
        );
        assert_eq!(
            (m.first_frame_timestamp_us, m.resolution, m.frame_rate),
            (1_000_200, Some([1920, 1920]), Some(30.0))
        );
        assert_eq!(
            (m.rolling_shutter_ms, m.gyro_timestamp_ms, m.is_raw_gyro),
            (Some(15.882978), Some(1.6), true)
        );
        assert_eq!(m.window_crop, Some(CROP));
        assert_eq!(m.lens_track(), [1, 0]);
        let s = trailer.imu[0];
        assert_eq!(s.t_us, 1_000_000);
        // 1024 counts of +-32 g is 1 g; 16384 counts of +-2000 dps is 1000 dps.
        assert!(close(s.accel_m_s2[0], 9.80665, 1e-12) && close(s.accel_m_s2[2], -9.80665, 1e-12));
        assert!(close(s.gyro_rad_s[0], 1000f64.to_radians(), 1e-12));
        assert_eq!(trailer.exposures_vin0.len(), 3);
        assert_eq!(
            trailer.exposures_vin0[1],
            Exposure {
                t_us: 1_000_200,
                exposure_s: 0.05
            }
        );
        assert_eq!(trailer.exposures_vin1[0].t_us, 1_000_234);
    }

    #[test]
    fn float_imu_items_are_56_bytes_in_g_and_rad_s() {
        let mut imu = 5u64.to_le_bytes().to_vec();
        for v in [0.5f64, 0.0, -1.0, 0.1, 0.2, 0.3] {
            imu.extend_from_slice(&v.to_le_bytes());
        }
        let file = trailer_file(&[(1, metadata(false)), (3, imu), (4, exposures(&[1_000_200]))]);
        let s = read_trailer(&mut Cursor::new(file)).unwrap().imu[0];
        assert_eq!(s.t_us, 5000);
        assert!(
            close(s.accel_m_s2[0], 0.5 * 9.80665, 1e-12) && close(s.accel_m_s2[2], -9.80665, 1e-12)
        );
        assert_eq!(s.gyro_rad_s, [0.1, 0.2, 0.3]);
        // A raw-width record with the float flag is refused, not misread.
        let file = trailer_file(&[
            (1, metadata(false)),
            (3, vec![0u8; 20]),
            (4, exposures(&[1])),
        ]);
        assert!(read_trailer(&mut Cursor::new(file)).is_err());
        // A time of 2^62 ms overflows i64 microseconds: an error, not a panic or a wrapped time.
        let mut imu = (1u64 << 62).to_le_bytes().to_vec();
        imu.extend_from_slice(&[0u8; 48]);
        let file = trailer_file(&[(1, metadata(false)), (3, imu), (4, exposures(&[1]))]);
        assert!(matches!(
            read_trailer(&mut Cursor::new(file)),
            Err(Insta360Error::Malformed(_))
        ));
    }

    #[test]
    fn a_missing_magic_or_a_bad_index_is_an_error() {
        let mut file = trailer_file(&[(1, metadata(true)), (3, Vec::new()), (4, Vec::new())]);
        let n = file.len();
        assert!(read_trailer(&mut Cursor::new(file.clone())).is_ok());
        file[n - 78 + 2] = 0xff; // index size
        assert!(read_trailer(&mut Cursor::new(file.clone())).is_err());
        file[n - 1] = b'x';
        assert!(read_trailer(&mut Cursor::new(file)).is_err());
        assert!(read_trailer(&mut Cursor::new(vec![0u8; 10])).is_err());
    }

    #[test]
    fn frame_zero_is_the_exposure_entry_stamped_first_frame_timestamp() {
        use crate::container::mp4::testing::{bx, trak, TrackSpec};
        const PARAMETER_SETS: &[(u8, &[u8])] = &[(32, &[0x40, 0x01])];
        let spec = |id| TrackSpec {
            track_id: id,
            sizes: &[6, 6],
            chunk_offsets: &[8, 14],
            stsc: &[[1, 1, 1]],
            stts: &[[2, 1001]],
            ctts: &[[2, 0]],
            stss: Some(&[1]),
            media_time: 0,
            co64: false,
            parameter_sets: PARAMETER_SETS,
        };
        let mut file = bx(b"mdat", &[0, 0, 0, 2, 0x26, 0x01, 0, 0, 0, 2, 0x02, 0x01]);
        let mut moov = trak(&spec(1));
        moov.extend(trak(&spec(2)));
        file.extend(bx(b"moov", &moov));
        let trailer = trailer_file(&[
            (1, metadata(true)),
            (3, Vec::new()),
            (
                4,
                exposures(&[1_000_100, 1_000_150, 1_000_200, 1_033_567, 1_066_934]),
            ),
        ]);
        let video = file.clone();
        file.extend(bx(b"inst", &trailer[1000..]));
        let dir = std::env::temp_dir().join(format!("insta360-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.insv");
        std::fs::write(&path, &file).unwrap();
        let clip = Insta360Clip::open(&path).unwrap();
        assert_eq!(clip.first_frame_entry, 2);
        assert_eq!(
            clip.frame_exposures(false)
                .iter()
                .map(|e| e.t_us)
                .collect::<Vec<_>>(),
            vec![1_000_200, 1_033_567]
        );
        assert_eq!(clip.lens_track, [1, 0]);
        let mut reader = clip.sample_reader().unwrap();
        let mut out = Vec::new();
        assert!(reader.read_annex_b(&clip.tracks[0], 0, &mut out).unwrap());
        assert_eq!(out, [0, 0, 0, 1, 0x40, 0x01, 0, 0, 0, 1, 0x26, 0x01]);
        // A first-frame time of i64::MIN and an exposure at i64::MAX: the distance between them does not overflow,
        // and the clip is refused because no exposure is near the first frame.
        let mut far_metadata = metadata(true);
        varint_field(24, 1 << 63, &mut far_metadata);
        let far = trailer_file(&[
            (1, far_metadata),
            (3, Vec::new()),
            (4, exposures(&[i64::MAX as u64])),
        ]);
        let mut far_file = video;
        far_file.extend(bx(b"inst", &far[1000..]));
        std::fs::write(&path, &far_file).unwrap();
        assert!(matches!(
            Insta360Clip::open(&path),
            Err(Insta360Error::Malformed(_))
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
