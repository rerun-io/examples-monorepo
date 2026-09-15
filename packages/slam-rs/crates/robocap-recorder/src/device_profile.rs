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

/// Capture geometry shared by the Rust camera and SLAM paths.
pub const FRAME_WIDTH: usize = 1920;
pub const FRAME_HEIGHT: usize = 1080;
pub const SLAM_DOWNSCALE: usize = 3;
/// Pixels per downsampled SLAM image.
pub const SLAM_PIXELS: usize = (FRAME_WIDTH / SLAM_DOWNSCALE) * (FRAME_HEIGHT / SLAM_DOWNSCALE);

#[derive(Clone, Copy)]
pub struct SensorChannel {
    pub iio_index: u8,
    pub device: u8,
    pub kind: MotionKind,
    pub prefix: &'static str,
    pub packet_size: usize,
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
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 2,
        device: 0,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        entity: "/world/rig_00/imu_00/accel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 3,
        device: 1,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        entity: "/world/rig_00/imu_01/gyro",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 4,
        device: 1,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        entity: "/world/rig_00/imu_01/accel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 5,
        device: 2,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        entity: "/world/rig_00/imu_02/gyro",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 6,
        device: 2,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        entity: "/world/rig_00/imu_02/accel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 7,
        device: 0,
        kind: MotionKind::Mag,
        prefix: "in_magn",
        entity: "/world/rig_00/mag_00",
        packet_size: 24,
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

/// Area-average the full-resolution NV12 luma plane for the estimator.
pub fn slam_luma(nv12: &[u8]) -> Vec<u8> {
    let width = FRAME_WIDTH / SLAM_DOWNSCALE;
    let height = FRAME_HEIGHT / SLAM_DOWNSCALE;
    let area = (SLAM_DOWNSCALE * SLAM_DOWNSCALE) as u16;
    let mut pixels = vec![0; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0_u16;
            for dy in 0..SLAM_DOWNSCALE {
                let at = (y * SLAM_DOWNSCALE + dy) * FRAME_WIDTH + x * SLAM_DOWNSCALE;
                sum += nv12[at..at + SLAM_DOWNSCALE]
                    .iter()
                    .map(|&v| u16::from(v))
                    .sum::<u16>();
            }
            pixels[y * width + x] = ((sum + area / 2) / area) as u8;
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    #[test]
    fn slam_luma_preserves_flat_planes_and_rounds_block_averages() {
        let mut plane = vec![27; 1920 * 1080];
        assert_eq!(super::slam_luma(&plane), vec![27; 640 * 360]);
        for y in 0..1080 {
            for x in 0..1920 {
                let k = (x / 3 % 200) as u8;
                plane[y * 1920 + x] = k + if y % 3 == 0 && x % 3 == 0 { 4 } else { 0 };
            }
        }
        let pixels = super::slam_luma(&plane);
        assert_eq!(pixels.len(), 640 * 360);
        for row in pixels.chunks_exact(640) {
            assert_eq!(row, (0..640).map(|x| (x % 200) as u8).collect::<Vec<_>>());
        }
        // One more count crosses the existing nearest-integer threshold.
        plane[0] = 5;
        assert_eq!(super::slam_luma(&plane)[0], 1);
    }
}
