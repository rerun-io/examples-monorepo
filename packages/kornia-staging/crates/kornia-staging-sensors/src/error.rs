/// Sensor input or source-policy failure.
#[derive(Debug, Clone, Copy, thiserror::Error, PartialEq, Eq)]
pub enum SensorError {
    /// An invalid construction parameter.
    #[error("invalid sensor configuration: {0}")]
    InvalidConfig(&'static str),
    /// Noise covariance contains a nonfinite or negative entry.
    #[error("IMU noise covariance must be finite and nonnegative")]
    InvalidNoise,
    /// Input samples or bias values contain non-finite numbers.
    #[error("non-finite IMU input")]
    NonFiniteInput,
    /// Finite inputs overflowed during integration.
    #[error("non-finite IMU integration result")]
    NonFiniteResult,
    /// A sample fails to strictly follow the integrated interval. Duplicate timestamps
    /// would give zero duration and a singular measurement.
    #[error("imu sample at {timestamp_ns} ns does not follow the integrated state at {previous_timestamp_ns} ns")]
    NonMonotonicSample {
        /// End of what has been integrated so far, relative to the start time.
        previous_timestamp_ns: i64,
        /// Timestamp of the rejected sample, on the same clock.
        timestamp_ns: i64,
    },
    /// A timestamp difference does not fit in an `i64`.
    #[error("timestamps {a_ns} and {b_ns} ns are too far apart to subtract")]
    TimestampOverflow {
        /// Left operand.
        a_ns: i64,
        /// Right operand.
        b_ns: i64,
    },
    /// An acceleration interpolation bracket exceeds the configured maximum.
    #[error("accelerometer gap of {gap_ns} ns")]
    AccelGap {
        /// Actual bracket width, nanoseconds.
        gap_ns: u64,
    },
    /// A per-channel IMU queue reached its configured capacity.
    #[error("IMU backlog exceeded its bound")]
    ImuQueueFull,
    /// A frame names a slot outside the runtime rig.
    #[error("camera {camera_slot} is outside a rig of {cameras} cameras")]
    InvalidCamera {
        /// Supplied camera index.
        camera_slot: usize,
        /// Configured number of cameras.
        cameras: usize,
    },
}
