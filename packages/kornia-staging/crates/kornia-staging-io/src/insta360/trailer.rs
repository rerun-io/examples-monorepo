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

/// Read the trailer at the end of a clip, through its index record or, on bodies that write none, its descriptor chain.
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
    // Two layouts. Recent bodies (X4 and later) close the trailer with an index record (id 0) whose 10-byte slots
    // `[id][format][u32 size][u32 offset]` locate every record; on the X4 Air the records sit in 256 KiB slots, so
    // only the index finds them. Older bodies (ONE X2, ONE R/RS, X3) write no index: each record is followed by its
    // 6-byte descriptor `[format][id][u32 size]`, and the reader walks them backward from the footer.
    let wanted = [
        RECORD_METADATA,
        RECORD_IMU,
        RECORD_EXPOSURE_VIN0,
        RECORD_EXPOSURE_VIN1,
    ];
    let mut located: Vec<(u8, u64, u64)> = Vec::new();
    let descriptor = read_at(file_len - FOOTER - 6, 6)?;
    let last_id = descriptor[1];
    if last_id == 0 {
        let index_size = u64::from(u32::from_le_bytes(bytes(&descriptor, 2)));
        if !index_size.is_multiple_of(10) || index_size + FOOTER + 6 > trailer_size {
            return Err(malformed(format!("index record of {index_size} bytes")));
        }
        let index = read_at(file_len - FOOTER - 6 - index_size, index_size as usize)?;
        for slot in index.as_chunks::<10>().0 {
            let id = slot[0];
            let size = u64::from(u32::from_le_bytes(bytes(slot, 2)));
            let offset = u64::from(u32::from_le_bytes(bytes(slot, 6)));
            if wanted.contains(&id) {
                located.push((id, start + offset, size));
            }
        }
    } else {
        // `end` is the trailer-relative end of the next descriptor to read.
        let mut end = trailer_size - FOOTER;
        while end >= 6 {
            let descriptor = read_at(start + end - 6, 6)?;
            let (id, size) = (
                descriptor[1],
                u64::from(u32::from_le_bytes(bytes(&descriptor, 2))),
            );
            let Some(record_start) = (end - 6).checked_sub(size) else {
                return Err(malformed(format!(
                    "record {id} of {size} bytes runs past the start of the {trailer_size}-byte trailer"
                )));
            };
            if wanted.contains(&id) {
                located.push((id, start + record_start, size));
            }
            end = record_start;
        }
    }
    let mut records: [Option<Vec<u8>>; 3] = [None, None, None];
    let mut metadata = None;
    for (id, at, size) in located {
        if at + size > start + trailer_size {
            return Err(malformed(format!(
                "record {id} at {}+{size} overruns the {trailer_size}-byte trailer",
                at - start
            )));
        }
        let bytes = read_at(at, size as usize)?;
        match id {
            RECORD_METADATA => metadata = Some(parse_metadata(&bytes)?),
            RECORD_IMU => records[0] = Some(bytes),
            RECORD_EXPOSURE_VIN0 => records[1] = Some(bytes),
            _ => records[2] = Some(bytes),
        }
    }
    let mut metadata = metadata.ok_or_else(|| malformed("no metadata record"))?;
    let [imu, vin0, vin1] = records;
    let mut exposures_vin0 =
        parse_exposures(&vin0.ok_or_else(|| malformed("no exposure record"))?)?;
    let mut exposures_vin1 = vin1
        .map(|r| parse_exposures(&r))
        .transpose()?
        .unwrap_or_default();
    // Most bodies stamp exposures and the first frame in microseconds; the ONE X2 stamps them in milliseconds (33 or
    // 34 apart at 30 fps). No video runs at 1000 frames a second, so a median spacing under 1000 means milliseconds.
    let mut spacing: Vec<i64> = exposures_vin0
        .windows(2)
        .map(|w| w[1].t_us - w[0].t_us)
        .collect();
    spacing.sort_unstable();
    if spacing
        .get(spacing.len() / 2)
        .is_some_and(|&median| median < MILLISECOND_STAMPS_BELOW)
    {
        for e in exposures_vin0.iter_mut().chain(exposures_vin1.iter_mut()) {
            e.t_us = e
                .t_us
                .checked_mul(1000)
                .ok_or_else(|| malformed("a millisecond exposure stamp overflows microseconds"))?;
        }
        metadata.first_frame_timestamp_us = metadata
            .first_frame_timestamp_us
            .checked_mul(1000)
            .ok_or_else(|| malformed("a millisecond first-frame stamp overflows microseconds"))?;
    }
    Ok(Trailer {
        imu: parse_imu(&imu.ok_or_else(|| malformed("no IMU record"))?, &metadata)?,
        exposures_vin0,
        exposures_vin1,
        metadata,
    })
}

/// A median exposure spacing below this many stamp units is in milliseconds, not microseconds.
const MILLISECOND_STAMPS_BELOW: i64 = 1000;

#[cfg(test)]
mod tests {
    use super::super::testing::{
        chained_trailer_file, exposures, metadata, trailer_file, varint_field, CROP,
    };
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
    fn a_trailer_without_an_index_is_read_through_its_descriptor_chain() {
        let records = [
            (1, metadata(true)),
            (3, vec![0u8; 40]),
            (2, vec![7u8; 33]), // a thumbnail the reader skips
            (4, exposures(&[1_000_100, 1_000_200])),
            (12, exposures(&[1_000_234])),
        ];
        let chained = read_trailer(&mut Cursor::new(chained_trailer_file(&records))).unwrap();
        assert_eq!(
            chained,
            read_trailer(&mut Cursor::new(trailer_file(&records))).unwrap()
        );
        assert_eq!((chained.imu.len(), chained.exposures_vin0.len()), (2, 2));
        // A descriptor whose record would start before the trailer is refused.
        let mut file = chained_trailer_file(&records);
        let n = file.len();
        file[n - 72 - 4..n - 72].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            read_trailer(&mut Cursor::new(file)),
            Err(Insta360Error::Malformed(_))
        ));
    }

    #[test]
    fn millisecond_exposure_stamps_are_read_as_microseconds() {
        // A ONE X2 trailer: frames 33-34 ms apart, stated in milliseconds, the first frame too.
        let mut m = metadata(false);
        varint_field(24, 153_676, &mut m);
        let records = [
            (1, m),
            (3, Vec::new()),
            (4, exposures(&[153_687, 153_720, 153_754])),
            (12, exposures(&[153_688])),
        ];
        let trailer = read_trailer(&mut Cursor::new(chained_trailer_file(&records))).unwrap();
        let t: Vec<i64> = trailer.exposures_vin0.iter().map(|e| e.t_us).collect();
        assert_eq!(t, [153_687_000, 153_720_000, 153_754_000]);
        assert_eq!(trailer.exposures_vin1[0].t_us, 153_688_000);
        assert_eq!(trailer.metadata.first_frame_timestamp_us, 153_676_000);
        // Microsecond stamps 33 367 apart stay as they are.
        let records = [
            (1, metadata(false)),
            (3, Vec::new()),
            (4, exposures(&[1_000_200, 1_033_567, 1_066_934])),
        ];
        let trailer = read_trailer(&mut Cursor::new(trailer_file(&records))).unwrap();
        assert_eq!(
            (
                trailer.exposures_vin0[1].t_us,
                trailer.metadata.first_frame_timestamp_us
            ),
            (1_033_567, 1_000_200)
        );
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
