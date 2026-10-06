//! RoboCap device capture: the six rkisp mainpath cameras (V4L2 multi-planar NV12), the frame trigger, and IMU0 over IIO.
//!
//! Adapted, with attribution, from draft PR #270 (`packages/slam-rs/crates/robocap-recorder` at 271ce643: `native/camera.c`,
//! `src/camera.rs`, `src/device.rs`, `src/iio.rs`, `src/device_profile.rs`), the code that was verified on Cap A and Cap B. This
//! crate copies what it needs instead of depending on `robocap-recorder`, which pins rerun 0.37 and links GStreamer.
//!
//! The ownership rules from PR #270 hold: the vendor recorder must be stopped first (`robocap-panel handoff`), every
//! changed IIO attribute is restored in reverse order on drop, and the frame trigger is stopped on drop.
#![deny(missing_docs)]

#[cfg(target_os = "linux")]
pub mod camera;
#[cfg(target_os = "linux")]
pub mod device;
pub mod iio;
pub mod imu;

use std::path::PathBuf;

use serde::Deserialize;

/// Errors of device capture.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// Shared IIO decoder or buffer-owner failure.
    #[error(transparent)]
    Iio(#[from] kornia_staging_sensor_iio::IioError),
    /// A system call or file access failed.
    #[error("{what}: {source}")]
    Io {
        /// What was being done.
        what: String,
        /// The OS error.
        source: std::io::Error,
    },
    /// A sysfs attribute could not be read or written.
    #[error("{path}: {message}")]
    Attribute {
        /// The attribute path.
        path: PathBuf,
        /// What went wrong.
        message: String,
    },
    /// The device is not in the state capture requires (wrong device, owned by the vendor, unexpected format).
    #[error("{0}")]
    Device(String),
    /// A sample or frame violated a timing rule (clock guard, sequence).
    #[error("{0}")]
    Timing(String),
    /// A frame could not be wrapped as an image.
    #[error("{0}")]
    Image(#[from] kornia_image::ImageError),
}

impl CaptureError {
    /// An [`CaptureError::Io`] from the last OS error.
    pub fn last_os_error(what: impl Into<String>) -> Self {
        Self::Io {
            what: what.into(),
            source: std::io::Error::last_os_error(),
        }
    }
}

/// The rkisp mainpath of each camera, in camera-index order ([`crate::frame::CAMERA_NAMES`]; PR #270's `CAMERAS`).
pub const CAMERA_DEVICES: [&str; crate::frame::NUM_CAMERAS] = [
    "/dev/video75",
    "/dev/video111",
    "/dev/video84",
    "/dev/video66",
    "/dev/video102",
    "/dev/video93",
];

/// The vendor's camera layout, which its recorder (`omni-specs.bin`) reads.
pub const ROBOT_JSON: &str = "/userdata/robot.json";
/// The vendor's `camera_position` of each camera index.
pub const VENDOR_POSITIONS: [&str; crate::frame::NUM_CAMERAS] = [
    "left-front",
    "right-front",
    "left-eye",
    "right-eye",
    "left",
    "right",
];

/// The fields of one `robot.json` camera entry that capture reads; the vendor's other fields are ignored.
#[derive(Deserialize)]
struct VendorCamera {
    /// The V4L2 node.
    name: String,
    /// One of [`VENDOR_POSITIONS`] (or a camera we do not capture).
    camera_position: String,
    /// The OpenCV rotate code; only an absent field means no rotation.
    #[serde(default = "no_rotation")]
    rotate: i64,
    #[serde(default)]
    mirror: bool,
    #[serde(default)]
    flip: bool,
}

/// OpenCV's "no rotation" code, the vendor's default for an absent `rotate`.
fn no_rotation() -> i64 {
    -1
}

/// Which cameras the vendor recorder turns 180 degrees before it encodes them, from `robot.json`'s per-camera `rotate` (an
/// OpenCV rotate code: -1 none, 1 `ROTATE_180`), `mirror` and `flip`. Each cap has its own file: on Cap B both eye cameras are
/// rotate 1 (2026-10-01); Cap A's live eye frames match its recordings, so its file most likely says -1 (not read yet). The
/// recordings, the catalog and the calibrations are upright; the mainpaths deliver the sensors' readout, so a turned camera
/// arrives upside down here. The three compose in the group of size-preserving turns, where mirror and flip together are the
/// 180-degree turn; the downsample and the crop maps implement only the turn, so a mirror or flip alone is refused.
///
/// # Errors
///
/// [`CaptureError::Device`] for malformed JSON, a camera entry (an entry with a `camera_position`) whose `name`, `rotate` (an
/// integer), `mirror` or `flip` (bools) has the wrong type, a camera missing, a camera on another node than [`CAMERA_DEVICES`], a
/// rotate code other than -1 or 1 (0 and 2 are 90-degree turns, which would change the image size), or a mirror or flip alone.
pub fn vendor_turned_180(text: &str) -> Result<[bool; crate::frame::NUM_CAMERAS], CaptureError> {
    let bad = |message: String| CaptureError::Device(format!("{ROBOT_JSON}: {message}"));
    let root: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
    let cameras: Vec<VendorCamera> = root
        .iter()
        .filter(|(_, entry)| entry.get("camera_position").is_some())
        .map(|(key, entry)| {
            VendorCamera::deserialize(entry).map_err(|e| bad(format!("{key}: {e}")))
        })
        .collect::<Result<_, _>>()?;
    let mut out = [false; crate::frame::NUM_CAMERAS];
    for (camera, position) in VENDOR_POSITIONS.iter().enumerate() {
        let entry = cameras
            .iter()
            .find(|entry| entry.camera_position == *position)
            .ok_or_else(|| bad(format!("no {position} camera")))?;
        if entry.name != CAMERA_DEVICES[camera] {
            return Err(bad(format!(
                "{position} is {}, we capture {}",
                entry.name, CAMERA_DEVICES[camera]
            )));
        }
        let turned = match entry.rotate {
            -1 => false,
            1 => true,
            other => {
                return Err(bad(format!(
                    "{position}: rotate {other} is not supported (only -1 and 1; 0 and 2 turn 90 degrees)"
                )));
            }
        };
        let (mirror, flip) = (entry.mirror, entry.flip);
        if mirror != flip {
            return Err(bad(format!(
                "{position}: mirror {mirror} with flip {flip} (a one-axis flip) is not supported"
            )));
        }
        out[camera] = turned != mirror;
    }
    Ok(out)
}

/// IIO device index of IMU0's gyro (`icm42688-gyro`).
pub const IMU0_GYRO_IIO: u8 = 1;
/// IIO device index of IMU0's accelerometer (`icm42688-accel`).
pub const IMU0_ACCEL_IIO: u8 = 2;
/// IMU0 gyro scale measured in the basalt fork, rad/s per count (sysfs `in_anglvel_scale` reads the same).
pub const GYRO_SCALE: f64 = 0.000266316;
/// IMU0 accel scale, m/s^2 per count (sysfs `in_accel_scale`).
pub const ACCEL_SCALE: f64 = 0.001197101;
/// DataForge's RoboCap IMU time shift (`packages/dataforge/dataforge/datasets/robocap.py`: `times = raw - CAMERA_TO_IMU_OFFSET_NS`),
/// the live source's default [`crate::source::live::LiveConfig::imu_time_offset_ns`].
pub const DATAFORGE_IMU_TIME_OFFSET_NS: i64 = -14_902_432;
/// Where `scripts/deploy.sh` installs the cap's `rig.json`.
pub const DEFAULT_RIG_PATH: &str = "/root/robocap-live/rig.json";

/// The two caps live capture may run on (PR #270's `DeviceProfile`: hostname + device-tree serial). Both have the same camera,
/// trigger and IIO layout, and each has its own factory calibration: [`Cap::slam_calibration`] for SLAM, and the `rig.json`
/// whose `device` is [`Cap::device`] for the hands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cap {
    /// `robocap_f403b0`, serial `f408193e6447b3b0`.
    A,
    /// `robocap_fe62fa`, serial `fe6fede545c972fa`.
    B,
}

impl Cap {
    /// Identify a cap from its hostname and device-tree serial.
    pub fn identify(hostname: &str, serial: &str) -> Option<Self> {
        match (hostname, serial) {
            ("robocap_f403b0", "f408193e6447b3b0") => Some(Self::A),
            ("robocap_fe62fa", "fe6fede545c972fa") => Some(Self::B),
            _ => None,
        }
    }

    /// The cap whose [`Cap::device`] is `device` (a rig's or a dump's `device`).
    pub fn from_device(device: &str) -> Option<Self> {
        [Self::A, Self::B]
            .into_iter()
            .find(|cap| cap.device() == device)
    }

    /// The cap's name in `rig.json` and a dump's `meta.json` (`device`).
    pub fn device(self) -> &'static str {
        match self {
            Self::A => "cap_a",
            Self::B => "cap_b",
        }
    }

    /// The cap's factory 4-camera SLAM calibration at 640x360 (Basalt JSON).
    pub fn slam_calibration(self) -> &'static str {
        match self {
            Self::A => crate::slam::CAP_A_CALIBRATION,
            Self::B => crate::slam::CAP_B_CALIBRATION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cap B's /userdata/robot.json camera entries (2026-10-01), the fields we read plus one we do not; `left_eye_rotate` is
    /// the left eye's `rotate` as JSON text (`None`: no `rotate` field).
    fn robot_json(left_eye_node: &str, left_eye_rotate: Option<&str>) -> String {
        let camera = |k: usize, node: &str, rotate: Option<&str>, position: &str| {
            let rotate = rotate
                .map(|rotate| format!(r#""rotate": {rotate}, "#))
                .unwrap_or_default();
            format!(
                r#""video{k}": {{"name": "{node}", "mirror": false, {rotate}"flip": false, "fps": 30, "camera_position": "{position}"}}"#
            )
        };
        format!(
            "{{\"imu\": [], {}, {}, {}, {}, {}, {}}}",
            camera(0, "/dev/video66", Some("1"), "right-eye"),
            camera(1, "/dev/video75", Some("-1"), "left-front"),
            camera(2, left_eye_node, left_eye_rotate, "left-eye"),
            camera(3, "/dev/video93", Some("-1"), "right"),
            camera(4, "/dev/video102", Some("-1"), "left"),
            camera(5, "/dev/video111", Some("-1"), "right-front"),
        )
    }

    #[test]
    fn the_vendor_turns_the_eye_cameras_180_degrees() -> Result<(), CaptureError> {
        // camera order: left_front, right_front, left_eye, right_eye, left, right
        assert_eq!(
            vendor_turned_180(&robot_json("/dev/video84", Some("1")))?,
            [false, false, true, true, false, false]
        );
        Ok(())
    }

    #[test]
    fn only_an_absent_rotate_is_upright_and_a_rotate_that_is_not_an_integer_is_refused()
    -> Result<(), CaptureError> {
        let upright_left_eye = [false, false, false, true, false, false];
        assert_eq!(
            vendor_turned_180(&robot_json("/dev/video84", None))?,
            upright_left_eye,
            "absent"
        );
        assert_eq!(
            vendor_turned_180(&robot_json("/dev/video84", Some("-1")))?,
            upright_left_eye
        );
        assert!(vendor_turned_180(&robot_json("/dev/video84", Some("1")))?[2]);
        for malformed in ["\"1\"", "null", "1.5", "1.0", "true", "[1]"] {
            assert!(
                vendor_turned_180(&robot_json("/dev/video84", Some(malformed))).is_err(),
                "rotate {malformed}"
            );
        }
        for unsupported in ["0", "2", "180", "-2"] {
            assert!(
                vendor_turned_180(&robot_json("/dev/video84", Some(unsupported))).is_err(),
                "rotate {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_vendor_layout_we_do_not_capture_is_refused() {
        assert!(
            vendor_turned_180(&robot_json("/dev/video85", Some("1"))).is_err(),
            "left-eye on another node than we capture"
        );
        assert!(
            vendor_turned_180("{\"video0\": {}}").is_err(),
            "cameras missing"
        );
    }

    #[test]
    fn mirror_and_flip_compose_with_the_turn_and_a_one_axis_flip_is_refused()
    -> Result<(), CaptureError> {
        // The fixture's first entry is right_eye, rotate 1.
        let mirrored = robot_json("/dev/video84", Some("1")).replacen(
            "\"mirror\": false",
            "\"mirror\": true",
            1,
        );
        assert!(
            vendor_turned_180(&mirrored).is_err(),
            "a mirror alone is not a turn"
        );
        assert!(
            vendor_turned_180(&mirrored.replacen("\"mirror\": true", "\"mirror\": 1", 1)).is_err(),
            "mirror is a bool"
        );
        let both = mirrored.replacen("\"flip\": false", "\"flip\": true", 1);
        assert_eq!(
            vendor_turned_180(&both)?,
            [false, false, true, false, false, false],
            "the turn, mirrored and flipped, is upright"
        );
        Ok(())
    }
}
