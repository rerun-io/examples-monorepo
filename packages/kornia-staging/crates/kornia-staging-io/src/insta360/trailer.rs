//! The trailer's index and its IMU and exposure records.
use super::protobuf::{parse_metadata, Insta360Metadata};
use super::{malformed, Insta360Error};
use std::io::{Read, Seek, SeekFrom};

pub(super) const MAGIC: &[u8; 32] = b"8db42d694ccc418790edff439fe026bf";
const FOOTER: u64 = 72;
const RECORD_METADATA: u8 = 1;
const RECORD_IMU: u8 = 3;
const RECORD_EXPOSURE_VIN0: u8 = 4;
const RECORD_EXPOSURE_VIN1: u8 = 12;
const STANDARD_GRAVITY: f64 = 9.80665;

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

/// The `N` bytes at `at` of a slice the caller has sized.
fn bytes<const N: usize>(b: &[u8], at: usize) -> [u8; N] {
    std::array::from_fn(|k| b[at + k])
}

/// A record as whole `N`-byte items.
fn items<'a, const N: usize>(record: &'a [u8], what: &str) -> Result<&'a [[u8; N]], Insta360Error> {
    let (items, rest) = record.as_chunks::<N>();
    if !rest.is_empty() {
        return Err(malformed(format!(
            "{what} record of {} bytes is not a whole number of {N}-byte items",
            record.len()
        )));
    }
    Ok(items)
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
        Ok(items::<20>(record, "raw IMU")?
            .iter()
            .map(|b| {
                let v = |k: usize| f64::from(u16::from_le_bytes(bytes(b, 8 + 2 * k))) - 32768.0;
                ImuSample {
                    t_us: u64::from_le_bytes(bytes(b, 0)) as i64,
                    accel_m_s2: [v(0) * acc, v(1) * acc, v(2) * acc],
                    gyro_rad_s: [v(3) * gyr, v(4) * gyr, v(5) * gyr],
                }
            })
            .collect())
    } else {
        // u64 t_ms, then six f64: accel in g, gyro in rad/s.
        items::<56>(record, "float IMU")?
            .iter()
            .map(|b| {
                let v = |k: usize| f64::from_le_bytes(bytes(b, 8 + 8 * k));
                Ok(ImuSample {
                    t_us: (u64::from_le_bytes(bytes(b, 0)) as i64)
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
    Ok(items::<16>(record, "exposure")?
        .iter()
        .map(|b| Exposure {
            t_us: u64::from_le_bytes(bytes(b, 0)) as i64,
            exposure_s: f64::from_le_bytes(bytes(b, 8)),
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
    let trailer_size = u64::from(u32::from_le_bytes(bytes(&footer, 32)));
    let start = file_len
        .checked_sub(trailer_size)
        .ok_or_else(|| malformed(format!("trailer size {trailer_size} exceeds the file")))?;
    let descriptor = read_at(file_len - FOOTER - 6, 6)?;
    let (id, index_size) = (
        descriptor[1],
        u64::from(u32::from_le_bytes(bytes(&descriptor, 2))),
    );
    if id != 0 || !index_size.is_multiple_of(10) || index_size + FOOTER + 6 > trailer_size {
        return Err(malformed(format!(
            "index descriptor id {id} size {index_size}"
        )));
    }
    let index = read_at(file_len - FOOTER - 6 - index_size, index_size as usize)?;
    let mut records: [Option<Vec<u8>>; 3] = [None, None, None];
    let mut metadata = None;
    for slot in index.as_chunks::<10>().0 {
        let (id, size, offset) = (
            slot[0],
            u64::from(u32::from_le_bytes(bytes(slot, 2))),
            u64::from(u32::from_le_bytes(bytes(slot, 6))),
        );
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

#[cfg(test)]
mod tests {
    use super::super::testing::{exposures, metadata, trailer_file, CROP};
    use super::*;
    use std::io::Cursor;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
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
}
