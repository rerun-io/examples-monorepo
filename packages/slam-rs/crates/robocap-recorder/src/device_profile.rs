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

    pub fn serial(self) -> &'static str {
        match self {
            Self::CapA => "f408193e6447b3b0",
            Self::CapB => "fe6fede545c972fa",
        }
    }

    /// Canonical order: left front, right front, left eye, right eye, left, right.
    pub fn camera_paths(self) -> [&'static str; 6] {
        CAMERAS.map(|camera| camera.path)
    }

    pub fn slam_cpus(self) -> Option<std::ops::Range<usize>> {
        match self {
            Self::CapA | Self::CapB => Some(4..8),
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
}

pub const SENSORS: [SensorChannel; 7] = [
    SensorChannel {
        iio_index: 1,
        device: 0,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 2,
        device: 0,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 3,
        device: 1,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 4,
        device: 1,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 5,
        device: 2,
        kind: MotionKind::Gyro,
        prefix: "in_anglvel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 6,
        device: 2,
        kind: MotionKind::Accel,
        prefix: "in_accel",
        packet_size: 16,
    },
    SensorChannel {
        iio_index: 7,
        device: 0,
        kind: MotionKind::Mag,
        prefix: "in_magn",
        packet_size: 24,
    },
];

#[derive(Clone, Copy)]
pub struct CameraSpec {
    pub name: &'static str,
    pub path: &'static str,
}

pub const CAMERAS: [CameraSpec; 6] = [
    CameraSpec {
        name: "left_front",
        path: "/dev/video75",
    },
    CameraSpec {
        name: "right_front",
        path: "/dev/video111",
    },
    CameraSpec {
        name: "left_eye",
        path: "/dev/video84",
    },
    CameraSpec {
        name: "right_eye",
        path: "/dev/video66",
    },
    CameraSpec {
        name: "left",
        path: "/dev/video102",
    },
    CameraSpec {
        name: "right",
        path: "/dev/video93",
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
