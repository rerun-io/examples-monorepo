use std::ops::Range;

use crate::{CalibrationSource, MotionKind};
use anyhow::{Result, bail};

/// Capture layouts verified against each cap's vendor camera settings.
#[derive(Clone, Copy)]
pub enum DeviceProfile {
    CapA,
    CapB,
}

impl DeviceProfile {
    pub fn identify(hostname: &str, serial: &str) -> Result<Self> {
        match (hostname, serial) {
            ("robocap_f403b0", "f408193e6447b3b0") => Ok(Self::CapA),
            ("robocap_fe62fa", "fe6fede545c972fa") => Ok(Self::CapB),
            _ => bail!("capture requires a verified hostname and device serial"),
        }
    }

    /// Identify the cap this process runs on from the kernel hostname and the
    /// device-tree serial (NUL-terminated).
    pub fn from_this_device() -> Result<Self> {
        let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")?;
        let serial = std::fs::read_to_string("/proc/device-tree/serial-number")?;
        Self::identify(hostname.trim(), serial.trim_end_matches('\0'))
    }

    pub fn serial(self) -> &'static str {
        match self {
            Self::CapA => "f408193e6447b3b0",
            Self::CapB => "fe6fede545c972fa",
        }
    }

    pub fn calibration(self) -> CalibrationSource {
        CalibrationSource {
            device_serial: Self::CapA.serial().into(),
            placeholder: matches!(self, Self::CapB),
            document: include_str!("../../../configs/robocap_calib_downscale3.json").into(),
        }
    }
}

/// Cortex-A76 cores reserved for the SLAM worker on both caps; capture stays
/// under the normal scheduler.
pub const SLAM_CPUS: Range<usize> = 4..8;
/// Recorded streams: the cameras, then every sensor channel in `SENSORS` order.
pub const STREAMS: usize = CAMERAS.len() + SENSORS.len();

/// Capture geometry shared by the camera and SLAM paths.
pub const FRAME_WIDTH: usize = 1920;
pub const FRAME_HEIGHT: usize = 1080;
pub const SLAM_DOWNSCALE: usize = 3;
pub const SLAM_PIXELS: usize = (FRAME_WIDTH / SLAM_DOWNSCALE) * (FRAME_HEIGHT / SLAM_DOWNSCALE);

/// Area-average the full-resolution NV12 luma plane for the estimator.
#[cfg(feature = "live-slam")]
pub fn slam_luma(nv12: &[u8]) -> Result<Vec<u8>> {
    let mut pixels = vec![0; SLAM_PIXELS];
    kornia_staging_imgproc::resize::resize_area_u8_into::<1>(
        nv12,
        (FRAME_WIDTH, FRAME_HEIGHT),
        FRAME_WIDTH,
        &mut pixels,
        (FRAME_WIDTH / SLAM_DOWNSCALE, FRAME_HEIGHT / SLAM_DOWNSCALE),
    )?;
    Ok(pixels)
}

#[derive(Clone, Copy)]
pub struct SensorChannel {
    pub iio_index: u8,
    pub device: u8,
    pub kind: MotionKind,
    pub prefix: &'static str,
    /// Dataforge entity the samples are logged under.
    pub entity: &'static str,
}

pub const SENSORS: [SensorChannel; 7] = [
    SensorChannel {
        iio_index: 1,
        device: 0,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        entity: "/world/rig_00/imu_00/gyro",
    },
    SensorChannel {
        iio_index: 2,
        device: 0,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        entity: "/world/rig_00/imu_00/accel",
    },
    SensorChannel {
        iio_index: 3,
        device: 1,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        entity: "/world/rig_00/imu_01/gyro",
    },
    SensorChannel {
        iio_index: 4,
        device: 1,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        entity: "/world/rig_00/imu_01/accel",
    },
    SensorChannel {
        iio_index: 5,
        device: 2,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        entity: "/world/rig_00/imu_02/gyro",
    },
    SensorChannel {
        iio_index: 6,
        device: 2,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        entity: "/world/rig_00/imu_02/accel",
    },
    SensorChannel {
        iio_index: 7,
        device: 0,
        kind: MotionKind::Mag,
        prefix: "in_magn",
        entity: "/world/rig_00/mag_00",
    },
];

#[derive(Clone, Copy)]
pub struct CameraSpec {
    pub name: &'static str,
    pub path: &'static str,
    /// Dataforge video entity the encoded samples are logged under.
    pub video_entity: &'static str,
}

pub const CAMERAS: [CameraSpec; 6] = [
    CameraSpec {
        name: "left_front",
        path: "/dev/video75",
        video_entity: "/world/rig_00/cam_00/pinhole/video",
    },
    CameraSpec {
        name: "right_front",
        path: "/dev/video111",
        video_entity: "/world/rig_00/cam_01/pinhole/video",
    },
    CameraSpec {
        name: "left_eye",
        path: "/dev/video84",
        video_entity: "/world/rig_00/cam_02/pinhole/video",
    },
    CameraSpec {
        name: "right_eye",
        path: "/dev/video66",
        video_entity: "/world/rig_00/cam_03/pinhole/video",
    },
    CameraSpec {
        name: "left",
        path: "/dev/video102",
        video_entity: "/world/rig_00/cam_04/pinhole/video",
    },
    CameraSpec {
        name: "right",
        path: "/dev/video93",
        video_entity: "/world/rig_00/cam_05/pinhole/video",
    },
];
