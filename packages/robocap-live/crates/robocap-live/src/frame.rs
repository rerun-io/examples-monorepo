//! Shared types: camera indices, framesets, IMU samples, the rig, and the `robocap-live-dump/1` replay format.
//!
//! Every module uses these types (SPEC.md "The dump format" describes the files). Images are kornia-rs
//! [`Image<u8, 1>`] behind an [`Arc`], so a frameset is cheap to hand from capture to SLAM, hands and logging. Following
//! kornia-slam's conventions: integer-nanosecond time (slam-rs's `i64` ns), frame metadata shaped like sensor-rt's `FrameMeta`,
//! combined gyro + accel IMU samples shaped like kornia-sensors' `ImuMeasurement`, and `thiserror` errors (no `anyhow` in the library).
#![deny(missing_docs)]

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kornia_image::{Image, ImageError, ImageSize};
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use serde::{Deserialize, Serialize};

/// Errors of the shared types and the dump format.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// Reading or writing a file failed.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
    /// A JSON file did not parse into its type.
    #[error("{path}: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// The parse error.
        source: serde_json::Error,
    },
    /// An image could not be built from the data.
    #[error("{0}")]
    Image(#[from] ImageError),
    /// The data broke the format (magic, version, sizes, camera names).
    #[error("invalid data: {0}")]
    Invalid(String),
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> FrameError + '_ {
    move |source| FrameError::Io { path: path.to_path_buf(), source }
}

fn invalid(message: impl Into<String>) -> FrameError {
    FrameError::Invalid(message.into())
}

pub use robocap_types::{CAMERA_NAMES, CameraFrame, FULL_SIZE, FrameMeta, Frameset, ImuSample, Luma, NUM_CAMERAS, SLAM_CAMERAS, SMALL_SIZE, SourceEvent};

/// One calibrated camera of the rig (catalog values at 1920x1080).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigCamera {
    /// One of [`CAMERA_NAMES`].
    pub name: String,
    /// Image width, pixels.
    pub width: u32,
    /// Image height, pixels.
    pub height: u32,
    /// Row-major 4x4, metres: camera from rig (`/world/rig_00`).
    pub cam_from_rig: [[f64; 4]; 4],
    /// (fx, fy) pixels.
    pub focal: [f64; 2],
    /// (cx, cy) pixels.
    pub principal: [f64; 2],
    /// `[k1..k6, p1, p2]` in simplecv's Fisheye62 order (handtrack `CameraRig.fisheye62`); `None` = pinhole.
    /// RoboCap's KB4 calibration has k5 = k6 = p1 = p2 = 0, which is kornia-3d's `FisheyeCamera`.
    pub fisheye62: Option<[f64; 8]>,
}

/// The six cameras in index order, plus where they came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rig {
    /// The cameras in index order ([`CAMERA_NAMES`]).
    pub cameras: Vec<RigCamera>,
    /// E.g. "catalog rerun+http://<host>:9994 robocap s66".
    pub source: String,
    /// "cap_a" or "cap_b".
    pub device: String,
}

impl Rig {
    /// Read a dump's `rig.json`.
    ///
    /// # Errors
    ///
    /// [`FrameError::Io`] / [`FrameError::Json`] when the file cannot be read or parsed; [`FrameError::Invalid`] unless it
    /// holds the six cameras in [`CAMERA_NAMES`] order.
    pub fn load(path: &Path) -> Result<Self, FrameError> {
        let text = std::fs::read_to_string(path).map_err(io_error(path))?;
        let rig: Rig = serde_json::from_str(&text).map_err(|source| FrameError::Json { path: path.to_path_buf(), source })?;
        if rig.cameras.len() != NUM_CAMERAS {
            return Err(invalid(format!("{}: {} cameras, expected {NUM_CAMERAS}", path.display(), rig.cameras.len())));
        }
        for (camera, name) in rig.cameras.iter().zip(CAMERA_NAMES) {
            if camera.name != name {
                return Err(invalid(format!("{}: camera {} where {name} was expected", path.display(), camera.name)));
            }
        }
        Ok(rig)
    }
}

/// `meta.json` of a dump directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DumpMeta {
    /// Always [`DUMP_FORMAT`].
    pub format: String,
    /// Where the frames came from (e.g. the catalog URL and dataset).
    pub source: String,
    /// The catalog segment (or recording) name.
    pub segment: String,
    /// "cap_a" or "cap_b".
    pub device: String,
    /// Framesets in `frames.bin`.
    pub frames: u64,
    /// Frame width, pixels.
    pub width: u32,
    /// Frame height, pixels.
    pub height: u32,
    /// Camera names in index order.
    pub cameras: Vec<String>,
    /// Time of the first frameset, nanoseconds.
    pub first_t_ns: i64,
    /// Time of the last frameset, nanoseconds.
    pub last_t_ns: i64,
}

impl DumpMeta {
    /// Read the `meta.json` of the dump directory `dir`.
    ///
    /// # Errors
    ///
    /// [`FrameError::Io`] / [`FrameError::Json`] when it cannot be read or parsed; [`FrameError::Invalid`] for a format other
    /// than [`DUMP_FORMAT`].
    pub fn load(dir: &Path) -> Result<Self, FrameError> {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path).map_err(io_error(&path))?;
        let meta: DumpMeta = serde_json::from_str(&text).map_err(|source| FrameError::Json { path: path.clone(), source })?;
        if meta.format != DUMP_FORMAT {
            return Err(invalid(format!("{}: format {}, expected {DUMP_FORMAT}", path.display(), meta.format)));
        }
        Ok(meta)
    }

    /// The frames' size.
    pub fn size(&self) -> ImageSize {
        ImageSize { width: self.width as usize, height: self.height as usize }
    }
}

/// The dump format's name and version, in [`DumpMeta::format`].
pub const DUMP_FORMAT: &str = "robocap-live-dump/1";
/// The files of a dump directory besides `meta.json` and `frames.bin`; `reference_world_from_rig.bin` is optional.
pub const DUMP_SIDE_FILES: [&str; 3] = ["rig.json", "imu.bin", "reference_world_from_rig.bin"];
const FRAME_MAGIC: [u8; 4] = *b"RLF1";
/// Size of [`FrameHeader`] on disk.
pub const FRAME_HEADER_BYTES: usize = 80;
/// Size of one IMU record on disk: `i64 t_ns`, `f64 gyro[3]`, `f64 accel[3]`.
pub const IMU_RECORD_BYTES: usize = 56;

/// The 80-byte little-endian header in front of each frameset in `frames.bin` (normative):
/// `magic "RLF1"`, `u32 version (1)`, `u64 index`, `i64 t_ns`, `u8 present_mask`, `7 x u8 0`, `i64 cam_t_ns[6]` (0 when absent).
/// Then, for each present camera in index order, `width * height` luma bytes (stride = width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// The frameset's index.
    pub index: u64,
    /// The frameset's time, nanoseconds.
    pub t_ns: i64,
    /// Bit `c` set when camera `c`'s frame follows.
    pub present_mask: u8,
    /// Each camera's capture time, nanoseconds (0 when absent).
    pub cam_t_ns: [i64; NUM_CAMERAS],
}

impl FrameHeader {
    /// The header's 80 bytes.
    pub fn to_bytes(&self) -> [u8; FRAME_HEADER_BYTES] {
        let mut out = [0u8; FRAME_HEADER_BYTES];
        out[0..4].copy_from_slice(&FRAME_MAGIC);
        out[4..8].copy_from_slice(&1u32.to_le_bytes());
        out[8..16].copy_from_slice(&self.index.to_le_bytes());
        out[16..24].copy_from_slice(&self.t_ns.to_le_bytes());
        out[24] = self.present_mask;
        for (camera, t) in self.cam_t_ns.iter().enumerate() {
            out[32 + 8 * camera..40 + 8 * camera].copy_from_slice(&t.to_le_bytes());
        }
        out
    }

    /// Parse a header.
    ///
    /// # Errors
    ///
    /// [`FrameError::Invalid`] without the `RLF1` magic or for a version other than 1.
    pub fn from_bytes(bytes: &[u8; FRAME_HEADER_BYTES]) -> Result<Self, FrameError> {
        if bytes[0..4] != FRAME_MAGIC {
            return Err(invalid("frame record without the RLF1 magic"));
        }
        let version = u32::from_le_bytes(le4(&bytes[4..8]));
        if version != 1 {
            return Err(invalid(format!("frame record version {version}, expected 1")));
        }
        let mut cam_t_ns = [0i64; NUM_CAMERAS];
        for (camera, t) in cam_t_ns.iter_mut().enumerate() {
            *t = i64::from_le_bytes(le8(&bytes[32 + 8 * camera..40 + 8 * camera]));
        }
        Ok(Self { index: u64::from_le_bytes(le8(&bytes[8..16])), t_ns: i64::from_le_bytes(le8(&bytes[16..24])), present_mask: bytes[24], cam_t_ns })
    }
}

fn le4(bytes: &[u8]) -> [u8; 4] {
    let mut out = [0u8; 4];
    out.copy_from_slice(&bytes[..4]);
    out
}

fn le8(bytes: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&bytes[..8]);
    out
}

/// Encode one IMU sample as its 56-byte record.
pub fn imu_to_bytes(sample: &ImuSample) -> [u8; IMU_RECORD_BYTES] {
    let mut out = [0u8; IMU_RECORD_BYTES];
    out[0..8].copy_from_slice(&sample.t_ns.to_le_bytes());
    for (axis, value) in sample.gyro.iter().chain(sample.accel.iter()).enumerate() {
        out[8 + 8 * axis..16 + 8 * axis].copy_from_slice(&value.to_le_bytes());
    }
    out
}

/// Decode one 56-byte IMU record.
pub fn imu_from_bytes(bytes: &[u8; IMU_RECORD_BYTES]) -> ImuSample {
    let value = |axis: usize| f64::from_le_bytes(le8(&bytes[8 + 8 * axis..16 + 8 * axis]));
    ImuSample { t_ns: i64::from_le_bytes(le8(&bytes[0..8])), gyro: [value(0), value(1), value(2)], accel: [value(3), value(4), value(5)] }
}

/// Sequential writer of a dump's `frames.bin` (Python writes dumps too; this one is for tests and tools).
pub struct FrameWriter {
    out: BufWriter<File>,
    size: ImageSize,
    path: PathBuf,
}

impl FrameWriter {
    /// Create (truncate) `path` for frames of `size`.
    ///
    /// # Errors
    ///
    /// [`FrameError::Io`] when the file cannot be created.
    pub fn create(path: &Path, size: ImageSize) -> Result<Self, FrameError> {
        Ok(Self { out: BufWriter::new(File::create(path).map_err(io_error(path))?), size, path: path.to_path_buf() })
    }

    /// Append one frameset (header, then each present camera's luma). A refused frameset writes nothing.
    ///
    /// # Errors
    ///
    /// [`FrameError::Invalid`] for a frame of another size or a [`FrameMeta::turned_180`] frame (dump v1 frames are upright);
    /// [`FrameError::Io`] when writing fails.
    pub fn write(&mut self, frameset: &Frameset) -> Result<(), FrameError> {
        let mut header = FrameHeader { index: frameset.index, t_ns: frameset.t_ns, present_mask: 0, cam_t_ns: [0; NUM_CAMERAS] };
        for (camera, frame) in frameset.cameras.iter().enumerate() {
            if let Some(frame) = frame {
                if frame.full.size() != self.size {
                    return Err(invalid(format!("camera {camera}: image {:?}, dump is {:?}", frame.full.size(), self.size)));
                }
                if frame.meta.turned_180 {
                    return Err(invalid(format!("camera {camera}: a turned frame (dump v1 frames are upright)")));
                }
                header.present_mask |= 1 << camera;
                header.cam_t_ns[camera] = frame.meta.pts_ns;
            }
        }
        self.out.write_all(&header.to_bytes()).map_err(io_error(&self.path))?;
        for frame in frameset.cameras.iter().flatten() {
            self.out.write_all(frame.full.as_slice()).map_err(io_error(&self.path))?;
        }
        Ok(())
    }

    /// Flush and close the file.
    ///
    /// # Errors
    ///
    /// [`FrameError::Io`] when the flush fails.
    pub fn finish(mut self) -> Result<(), FrameError> {
        self.out.flush().map_err(io_error(&self.path))
    }
}

/// Sequential reader of a dump's `frames.bin`.
pub struct FrameReader {
    input: BufReader<File>,
    size: ImageSize,
    path: PathBuf,
}

impl FrameReader {
    /// Open `path`, whose frames are `size`.
    ///
    /// # Errors
    ///
    /// [`FrameError::Io`] when the file cannot be opened.
    pub fn open(path: &Path, size: ImageSize) -> Result<Self, FrameError> {
        Ok(Self { input: BufReader::with_capacity(1 << 22, File::open(path).map_err(io_error(path))?), size, path: path.to_path_buf() })
    }

    /// The next frameset, or `None` at the end of the file.
    ///
    /// # Errors
    ///
    /// [`FrameError::Io`] for a truncated record or a read error, [`FrameError::Invalid`] for a bad header.
    pub fn next_frameset(&mut self) -> Result<Option<Frameset>, FrameError> {
        if self.input.fill_buf().map_err(io_error(&self.path))?.is_empty() {
            return Ok(None);
        }
        let mut header_bytes = [0u8; FRAME_HEADER_BYTES];
        self.input.read_exact(&mut header_bytes).map_err(io_error(&self.path))?;
        let header = FrameHeader::from_bytes(&header_bytes)?;
        let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
        for (camera, slot) in cameras.iter_mut().enumerate() {
            if header.present_mask & (1 << camera) != 0 {
                let mut data = vec![0u8; self.size.width * self.size.height];
                self.input.read_exact(&mut data).map_err(io_error(&self.path))?;
                let meta = FrameMeta { seq: header.index, pts_ns: header.cam_t_ns[camera], source_id: camera as u32, turned_180: false };
                *slot = Some(CameraFrame { meta, full: Arc::new(Image::new(self.size, data)?) });
            }
        }
        Ok(Some(Frameset { index: header.index, t_ns: header.t_ns, cameras }))
    }
}

/// Write `framesets` (640x360 frames) as a small copy of the dump `src` in the directory `out`: `frames.bin`, `src`'s `meta.json`
/// with the small size, the frameset count, the last frameset's time and `note` after its source, and every one of
/// [`DUMP_SIDE_FILES`] that `src` has.
///
/// # Returns
///
/// The number of framesets written.
///
/// # Errors
///
/// [`FrameError::Io`] / [`FrameError::Json`] when a file cannot be read or written, [`FrameError::Invalid`] for a frame that is
/// not 640x360 or a `src` that is not a dump.
pub fn write_small_dump(src: &Path, out: &Path, framesets: impl IntoIterator<Item = Frameset>, note: &str) -> Result<u64, FrameError> {
    let meta = DumpMeta::load(src)?;
    std::fs::create_dir_all(out).map_err(io_error(out))?;
    let mut writer = FrameWriter::create(&out.join("frames.bin"), SMALL_SIZE)?;
    let (mut frames, mut last_t_ns) = (0u64, meta.last_t_ns);
    for frameset in framesets {
        writer.write(&frameset)?;
        frames += 1;
        last_t_ns = frameset.t_ns;
    }
    writer.finish()?;
    let small = DumpMeta {
        width: SMALL_SIZE.width as u32,
        height: SMALL_SIZE.height as u32,
        frames,
        source: format!("{} ({note})", meta.source),
        last_t_ns,
        ..meta
    };
    let path = out.join("meta.json");
    let text = serde_json::to_string(&small).map_err(|source| FrameError::Json { path: path.clone(), source })?;
    std::fs::write(&path, text).map_err(io_error(&path))?;
    for name in DUMP_SIDE_FILES {
        if src.join(name).exists() {
            std::fs::copy(src.join(name), out.join(name)).map_err(io_error(&src.join(name)))?;
        }
    }
    Ok(frames)
}

/// Read a whole `imu.bin` (combined IMU0 records, time order).
///
/// # Errors
///
/// [`FrameError::Io`] when the file cannot be read, [`FrameError::Invalid`] when its size is not a whole number of records.
pub fn read_imu(path: &Path) -> Result<Vec<ImuSample>, FrameError> {
    let bytes = std::fs::read(path).map_err(io_error(path))?;
    if bytes.len() % IMU_RECORD_BYTES != 0 {
        return Err(invalid(format!("{}: {} bytes is not a whole number of {IMU_RECORD_BYTES}-byte records", path.display(), bytes.len())));
    }
    Ok(bytes.chunks_exact(IMU_RECORD_BYTES).map(|chunk| {
        let mut record = [0u8; IMU_RECORD_BYTES];
        record.copy_from_slice(chunk);
        imu_from_bytes(&record)
    }).collect())
}

/// `[tx, ty, tz, qx, qy, qz, qw]` -> isometry.
pub fn isometry_from_array(pose: &[f64; 7]) -> Isometry3<f64> {
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(pose[6], pose[3], pose[4], pose[5]));
    Isometry3::from_parts(Translation3::new(pose[0], pose[1], pose[2]), rotation)
}

/// Row-major 4x4 -> isometry, or `None` when it is not finite. The rotation goes through a quaternion (Shepperd's method,
/// closed form), which also re-normalises a slightly non-orthonormal block. (`Rotation3::from_matrix` iterates until it
/// converges and never returns on a NaN matrix: a lossless s66-full replay hung on the dump's 7 NaN reference poses.)
pub fn isometry_from_matrix(m: &[f64; 16]) -> Option<Isometry3<f64>> {
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    let block = nalgebra::Matrix3::new(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
    let rotation = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(block));
    rotation.coords.iter().all(|v| v.is_finite()).then(|| Isometry3::from_parts(Translation3::new(m[3], m[7], m[11]), rotation))
}

/// Isometry -> row-major 4x4.
pub fn matrix_from_isometry(pose: &Isometry3<f64>) -> [f64; 16] {
    let m = pose.to_homogeneous();
    std::array::from_fn(|i| m[(i / 4, i % 4)])
}

#[cfg(test)]
mod tests {
    #[test]
    fn poses_convert_between_arrays_matrices_and_isometries() {
        let pose = isometry_from_array(&[1.0, 2.0, 3.0, 0.0, 0.0, (0.5f64).sin(), (0.5f64).cos()]);
        let matrix = matrix_from_isometry(&pose);
        assert!((matrix[3] - 1.0).abs() < 1e-12 && (matrix[7] - 2.0).abs() < 1e-12 && (matrix[11] - 3.0).abs() < 1e-12);
        assert!((matrix[0] - 1.0f64.cos()).abs() < 1e-12 && (matrix[4] - 1.0f64.sin()).abs() < 1e-12);
        let back = isometry_from_matrix(&matrix).unwrap_or_else(Isometry3::identity);
        assert!(isometry_from_matrix(&[f64::NAN; 16]).is_none(), "a NaN pose is refused, not iterated on");
        assert!((back.inverse() * pose).to_homogeneous().iter().zip(Isometry3::<f64>::identity().to_homogeneous().iter()).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    use super::*;

    #[test]
    fn only_eof_before_a_header_is_clean() -> Result<(), FrameError> {
        let path = std::env::temp_dir().join(format!("robocap-live-truncated-frame-{}.bin", std::process::id()));
        let size = ImageSize { width: 8, height: 4 };
        std::fs::write(&path, []).map_err(io_error(&path))?;
        assert!(FrameReader::open(&path, size)?.next_frameset()?.is_none());
        for length in 1..FRAME_HEADER_BYTES {
            std::fs::write(&path, vec![0; length]).map_err(io_error(&path))?;
            assert!(matches!(FrameReader::open(&path, size)?.next_frameset(), Err(FrameError::Io { .. })), "header length {length}");
        }
        let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
        cameras[0] = Some(CameraFrame {
            meta: FrameMeta { seq: 0, pts_ns: 0, source_id: 0, turned_180: false },
            full: Arc::new(Image::new(size, vec![7; 32])?),
        });
        let mut writer = FrameWriter::create(&path, size)?;
        writer.write(&Frameset { index: 0, t_ns: 0, cameras })?;
        writer.finish()?;
        let bytes = std::fs::read(&path).map_err(io_error(&path))?;
        for length in FRAME_HEADER_BYTES..bytes.len() {
            std::fs::write(&path, &bytes[..length]).map_err(io_error(&path))?;
            assert!(matches!(FrameReader::open(&path, size)?.next_frameset(), Err(FrameError::Io { .. })), "record length {length}");
        }
        std::fs::remove_file(&path).map_err(io_error(&path))?;
        Ok(())
    }

    #[test]
    fn a_turned_frame_is_refused_before_anything_is_written() -> Result<(), FrameError> {
        let path = std::env::temp_dir().join(format!("robocap-live-turned-frame-{}.bin", std::process::id()));
        let size = ImageSize { width: 8, height: 4 };
        let frameset = |index: u64, turned_180: bool| -> Result<Frameset, FrameError> {
            let frame = |source_id: u32, turned_180: bool| -> Result<CameraFrame, FrameError> {
                let meta = FrameMeta { seq: index, pts_ns: 10 + i64::from(source_id), source_id, turned_180 };
                Ok(CameraFrame { meta, full: Arc::new(Image::new(size, vec![source_id as u8; 32])?) })
            };
            let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
            cameras[0] = Some(frame(0, false)?);
            cameras[3] = Some(frame(3, turned_180)?);
            Ok(Frameset { index, t_ns: 10, cameras })
        };
        let mut writer = FrameWriter::create(&path, size)?;
        writer.write(&frameset(0, false)?)?;
        assert!(matches!(writer.write(&frameset(1, true)?), Err(FrameError::Invalid(_))), "camera 3 is turned");
        writer.finish()?;
        let mut reader = FrameReader::open(&path, size)?;
        assert_eq!(reader.next_frameset()?.map(|frameset| frameset.index), Some(0));
        assert!(reader.next_frameset()?.is_none(), "the refused frameset left no partial record");
        std::fs::remove_file(&path).map_err(io_error(&path))?;
        Ok(())
    }

    #[test]
    fn a_small_dump_keeps_the_meta_and_copies_the_side_files_it_finds() -> Result<(), FrameError> {
        let src = std::env::temp_dir().join(format!("robocap-live-small-src-{}", std::process::id()));
        let out = std::env::temp_dir().join(format!("robocap-live-small-out-{}", std::process::id()));
        std::fs::create_dir_all(&src).map_err(io_error(&src))?;
        let meta = DumpMeta {
            format: DUMP_FORMAT.into(),
            source: "test".into(),
            segment: "s".into(),
            device: "cap_a".into(),
            frames: 9,
            width: 1920,
            height: 1080,
            cameras: CAMERA_NAMES.iter().map(|name| name.to_string()).collect(),
            first_t_ns: 10,
            last_t_ns: 90,
        };
        std::fs::write(src.join("meta.json"), serde_json::to_string(&meta).map_err(|e| invalid(e.to_string()))?).map_err(io_error(&src))?;
        std::fs::write(src.join("imu.bin"), imu_to_bytes(&ImuSample { t_ns: 5, gyro: [0.0; 3], accel: [0.0; 3] })).map_err(io_error(&src))?;
        let frameset = |index: u64, size: ImageSize| -> Result<Frameset, FrameError> {
            let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
            cameras[4] = Some(CameraFrame { meta: FrameMeta { seq: index, pts_ns: 20 + index as i64, source_id: 4, turned_180: false }, full: Arc::new(Image::from_size_val(size, 3)?) });
            Ok(Frameset { index, t_ns: 20 + index as i64, cameras })
        };
        assert_eq!(write_small_dump(&src, &out, [frameset(1, SMALL_SIZE)?, frameset(2, SMALL_SIZE)?], "note")?, 2);
        let small = DumpMeta::load(&out)?;
        assert_eq!((small.size(), small.frames, small.first_t_ns, small.last_t_ns, small.source.as_str()), (SMALL_SIZE, 2, 10, 22, "test (note)"));
        assert!(out.join("imu.bin").exists() && !out.join("rig.json").exists() && !out.join("reference_world_from_rig.bin").exists());
        let back = FrameReader::open(&out.join("frames.bin"), SMALL_SIZE)?.next_frameset()?.ok_or_else(|| invalid("no frameset"))?;
        assert_eq!(back.cameras[4].as_ref().map(|f| (f.meta.pts_ns, f.full.as_slice()[0])), Some((21, 3)));
        assert!(write_small_dump(&src, &out, [frameset(1, FULL_SIZE)?], "note").is_err(), "a full-size frame");
        for dir in [&src, &out] {
            std::fs::remove_dir_all(dir).map_err(io_error(dir))?;
        }
        Ok(())
    }

    #[test]
    fn a_frameset_and_imu_samples_round_trip_through_the_dump_format() -> Result<(), FrameError> {
        let dir = std::env::temp_dir().join(format!("robocap-live-frame-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(io_error(&dir))?;
        let size = ImageSize { width: 8, height: 4 };
        let image = |value: u8| -> Result<Luma, FrameError> { Ok(Arc::new(Image::new(size, vec![value; 32])?)) };
        let meta = |pts_ns: i64, source_id: u32| FrameMeta { seq: 7, pts_ns, source_id, turned_180: false };
        let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
        cameras[1] = Some(CameraFrame { meta: meta(1_000_010, 1), full: image(11)? });
        cameras[4] = Some(CameraFrame { meta: meta(1_000_050, 4), full: image(44)? });
        let frameset = Frameset { index: 7, t_ns: 1_000_010, cameras };
        let mut writer = FrameWriter::create(&dir.join("frames.bin"), size)?;
        writer.write(&frameset)?;
        writer.finish()?;
        let mut reader = FrameReader::open(&dir.join("frames.bin"), size)?;
        let back = reader.next_frameset()?.ok_or_else(|| invalid("expected one frameset"))?;
        assert_eq!((back.index, back.t_ns), (7, 1_000_010));
        assert!(back.cameras[0].is_none() && back.cameras[1].is_some() && back.cameras[4].is_some());
        assert_eq!(back.cameras[4].as_ref().map(|f| (f.meta.pts_ns, f.meta.source_id, f.full.as_slice()[0])), Some((1_000_050, 4, 44)));
        assert!(reader.next_frameset()?.is_none());
        let sample = ImuSample { t_ns: -5, gyro: [0.01, -0.02, 0.03], accel: [0.5, -9.81, 1e-3] };
        assert_eq!(imu_from_bytes(&imu_to_bytes(&sample)), sample);
        std::fs::write(dir.join("imu.bin"), [imu_to_bytes(&sample), imu_to_bytes(&sample)].concat()).map_err(io_error(&dir))?;
        assert_eq!(read_imu(&dir.join("imu.bin"))?, vec![sample, sample]);
        std::fs::remove_dir_all(&dir).map_err(io_error(&dir))?;
        Ok(())
    }
}
