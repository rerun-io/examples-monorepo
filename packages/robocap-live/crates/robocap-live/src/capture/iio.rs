//! RoboCap IMU device profiles.

/// The IMU channel an IIO device carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionKind {
    /// `icm42688-gyro`, `in_anglvel_*`.
    Gyro,
    /// `icm42688-accel`, `in_accel_*`.
    Accel,
}

impl MotionKind {
    /// The sysfs channel prefix.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Gyro => "in_anglvel",
            Self::Accel => "in_accel",
        }
    }

    /// The driver name this kind requires.
    pub fn driver_name(self) -> &'static str {
        match self {
            Self::Gyro => "icm42688-gyro",
            Self::Accel => "icm42688-accel",
        }
    }
}
