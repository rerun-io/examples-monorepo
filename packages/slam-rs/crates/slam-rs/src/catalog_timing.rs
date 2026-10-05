//! Frameset association and IMU pairing shared by catalog readers.

/// Camera selection, decoding and clock rules for a supported catalog rig.
#[derive(Debug, Clone, Copy)]
pub struct RigProfile {
    pub camera_names: [&'static str; 4],
    pub rig_cameras: usize,
    pub codec: [u8; 4],
    pub tolerance_ns: i64,
    pub nominal_fps: i64,
    pub downscale: u32,
    pub interpolate_accel: bool,
    pub video_time_is_absolute: bool,
}

pub const ROBOCAP: RigProfile = RigProfile {
    camera_names: ["left_front", "right_front", "left", "right"],
    rig_cameras: 6,
    codec: *b"avc1",
    tolerance_ns: 1_000_000,
    nominal_fps: 30,
    downscale: 3,
    interpolate_accel: true,
    video_time_is_absolute: true,
};
pub const MSD_G2: RigProfile = RigProfile {
    camera_names: ["cam0", "cam1", "cam2", "cam3"],
    rig_cameras: 4,
    codec: *b"av01",
    tolerance_ns: 0,
    nominal_fps: 54,
    downscale: 1,
    interpolate_accel: false,
    video_time_is_absolute: false,
};

/// One gyroscope sample with acceleration on the same timestamp.
#[derive(Debug, PartialEq)]
pub struct ImuRow {
    pub t_ns: i64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
}
